# Kaizen Eye 2 - decisions log (verification evidence)

Plan: `C:\Users\Vijay Chikkala\.claude\plans\so-now-we-are-effervescent-tulip.md`. Newest entries at the bottom.

## B0 - preflight + scaffold (2026-09-27, ~01:27-02:00 IST)

### Done
- Legacy Expo work snapshotted locally: branch `legacy/expo-photo-mode`, commit `4f23cc0`, tag `legacy-expo-v1`; work continues on branch `native/kaizen-eye-2` (no push, nothing else committed).
- Toolchain copied out of the baseline project into `native/` (wrapper, daemon-JVM properties, reference copies in `native/third_party/`), so the old `C:\...\AndroidStudioProjects\KaizenEye` folder is no longer needed.
- `native/` scaffold builds: `:core` (pure Kotlin/JVM, 2 JUnit tests pass) + `:app` (Compose hello-world + `DevicePassport`), Gradle 9.5.0 / AGP 9.3.3 / Kotlin 2.2.10 / compileSdk 36.1.
- FastVLM-0.5B `qualcomm.sm8850.litertlm` downloaded (942,374,912 bytes, sha256 `0EB6860D...B482`, recorded in `native/models.lock.json`).
- LiteRT v2.2.0 NPU runtime libs for Hexagon v81 extracted (`libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`; SHA-256 in `native/third_party/litert-npu/SHA256SUMS.txt`).
- `tools/apk-audit.ps1` PASS on the debug APK (105.7 MB): 14 arm64 libs (all 16 KB-aligned except the 32-bit Hexagon skel, which the DSP loads), `extractNativeLibs=true`, `zipalign -P 16` OK, merged permissions = CAMERA + VIBRATE + AndroidX signature permission only (no INTERNET, no network-state).

### Dependency issues found and fixed by the resolution audit (not in the docs)
1. **Java 21 NIO on Windows needs a short TEMP path.** The Gradle daemon failed with `Unable to establish loopback connection` (`UnixDomainSockets.connect: Invalid argument`) because this tool session's TEMP path is > ~100 chars (AF_UNIX path limit). `native/build.ps1` now sets `TEMP/TMP=D:\kz-tmp`.
2. **Daemon keeps piped output open.** A piped caller waits until the Gradle daemon exits. `native/build.ps1` now writes the build to `native/dist/logs/last-build.log` (file redirect) and prints the tail.
3. **LiteRT 2.2.0 duplicate namespace.** `litert` and `litert-api` both use manifest namespace `com.google.ai.edge.litert`; AGP 9 rejects that (`processDebugMainManifest`). Fixed with `android.uniquePackageNames=false` in `native/gradle.properties` (AGP 9.3.3 still honours it; it degrades to a warning).
4. **WorkManager arrives transitively** via `litert -> litert-api -> ai-delivery -> asset-delivery -> androidx.work:work-runtime:2.9.1`, adding ACCESS_NETWORK_STATE / WAKE_LOCK / RECEIVE_BOOT_COMPLETED and a startup initializer. Excluded (`exclude(group = "androidx.work")` in `app/build.gradle.kts`) plus `tools:node="remove"` for litert's FOREGROUND_SERVICE permissions; the audit now enforces an explicit permission allow-list.
5. Resolution audit (`docs/verification/deps-app-debugRuntimeClasspath.txt`): all upgrades land on the pinned set (activity 1.13.0, core 1.18.0, lifecycle 2.10.0, coroutines 1.10.2, kotlin-stdlib 2.2.21 via LiteRT-LM's kotlin-reflect 2.2.21, guava 33.5.0-android, serialization 1.9.0); no compileSdk-37 artifact was pulled in.

### Still open in B0 (needs the phone)
- Phone is `unauthorized` in adb (serial `RZCT41JSLGB`; an earlier serial `10BFAU15AT000XR` also appeared). Pending: accept the USB-debugging RSA prompt, then install the APK, pull `device-passport.json` (SoC string, Camera2 limits, thermal/battery APIs) and run the dummy `adb push` test into `/sdcard/Android/data/com.kaizeneye.v2/files/` (gate G0).
- Legacy baseline measurements (the user's 0:10-0:45 task).
- Helpers H1 (Kotlin core) and H2 (Python/models) are NOT started yet (held until the user says go).

## Phone-free build-out (2026-09-27, ~02:10-04:05 IST)

Everything that can be built and verified without the phone. No git commits, no push (working-tree files only).
Five helpers ran in parallel on disjoint files (H1a Twin maths, H1b vision/tracker, H2a numpy reference + goldens,
H2b model export + A/B tooling, H3 Android runtime); the main session wrote the app and integrated.

### Delivered
- **Spec:** `docs/verification/twin-spec.md` (single source of truth; every algorithm, parameter and golden section).
- **`:core`** (pure Kotlin/JVM): legacy PatchCore port; stats/binary16/Clopper-Pearson/AUROC; image ops, empty-sheet
  mask, morphology, connected components, geometry, crop square, patch coverage; k-NN, scoring (smoothed max,
  top-1 % mean, top-k views, rotation helpers), teach (keyframes, coreset, leave-segment-out tau, identity/geometry
  gates), verdicts + borderline voting, calibration + certificate, Twin store (`twin.json` + `.f16`), negatives
  library; tracker + photo-eye/steady-hold triggers, thermal governor, latency stats, explanation facts/guard/
  templates, replay comparator.
- **`:runtime`** (Android library): LiteRT 2.2.0 backbone + k-NN graph runners; accelerator benchmark in a `:probe`
  process with crash guard, decision cache and honest badges; SHA-256-pinned model locator + SAF import; FastVLM
  service in a `:vlm` process (NPU -> GPU -> CPU, template fallback). See `native/runtime/README.md`.
- **`:app`**: Home / Teach / Inspect-Calibrate-Replay / Certificate / Negatives / Telemetry / Readiness / Clips /
  Self-test; CameraX (Camera2 preset, locks, torch, analysis fps cap) + clip recorder + deterministic replay source
  (MP4 and JPEG sequences); sessions (sheet + flicker check, teach, live line, negatives, calibration); JSONL + soak
  CSV logs; thermal monitor + governor; reject alarm; offline VLM with location guard; eval export
  (`docs/verification/export-format.md`); one-shot self-test (`--es selftest all|quick|accel|replay|vlm`).
- **Python:** `tools/lab/twin_ref.py` (independent numpy reference), `make_golden_twin.py` -> `testdata/golden_twin.json`,
  `test_twin_ref.py`; `tools/dinov2/*` (DINOv2-S/14@448 export: ONNX + fp32/fp16w TFLite, k-NN graph P1024 D384);
  `tools/lab/twin_eval.py` (24-variant A/B harness), `make_synthetic_export.py`, `make_replay_synth.py`
  (`testdata/replay_synth`, the bundled self-test clip).

### Verified (laptop + x86_64 emulator; none of this is a phone measurement)
- JUnit: `:core` 162/162 (all golden_twin.json sections + legacy goldens: coreset indices exact, max rel. err 9.4e-6),
  `:runtime` 58/58 (incl. all 8 model SHA-256 pins against the real files), `:app` 9/9. Python `test_twin_ref.py` 26/26.
- `apk-audit.ps1`: PASS - 125.5 MB, arm64 only, 16 KB aligned, permissions CAMERA + VIBRATE only; Play asset-pack
  services/receiver removed from the merged manifest; only the two runtime services (`:probe`, `:vlm`, not exported).
  APK copy: `native/dist/apk/KaizenEye2-phonefree-debug.apk` (sha256 `1B106094...D93C`).
- DINOv2 export parity vs PyTorch: fp32 TFLite cosine 1.000000, fp16w 0.99999 (min patch 0.99999); R18 ONNX vs
  shipped TFLite cosine 1.0.
- **Emulator (Android 16 x86_64, ARM translation), quick self-test PASS:** offline proof PASS; core maths on device
  PASS (131 golden checks reproduce the laptop numbers on ART); replay regression PASS (mechanics: 10/10 parts
  triggered at the expected times, 0 missing / 0 extra) using the DEBUG Kotlin feature extractor, because LiteRT's
  XNNPACK CPU delegate crashes under the emulator's ARM translation (SIGSEGV in
  `TfLiteXNNPackDelegateCreateWithThreadpool`; expected to be an emulator limitation - it is LiteRT's standard CPU path
  on arm64 phones - but it is the first thing to check in the phone self-test).
- Emulator UI walk-through: teach-from-clip (16 keyframes, build 0.8 s), paced replay with live overlay (boxes riding
  the parts, PASS / NOT THE ENROLLED PART / REJECT with heat-map peak, reject card with template sentence), Telemetry
  stage timers (fast loop p50 3.0 / p95 8.2 ms per frame even under translation), live CameraX preview + sheet capture.

### Fixed during integration (found by building / the emulator)
1. Crash-loop risk: after the probe records a native crash, the app retried the same model in-process at start-up.
   `EngineHolder.autoLoad` now leaves a marker around model loading and never auto-retries a load that killed the app.
2. Replay overlay lagged the preview (preview refreshed every 150 ms, boxes every 50 ms) -> preview now rendered from
   the same frame as the overlay.
3. Photo-eye axis was a sensor-axis preference (wrong for landscape clips) -> now "parts move across / down / held
   still" in screen terms, resolved per frame from the rotation (portrait camera and landscape clips both correct).
4. Flicker check flagged static horizontal texture as banding -> now uses the temporal drift of the row profile, trusts
   the static ripple only on a uniform sheet, and warns when the background is not plain.
5. `native/tools/with-lock.ps1` mis-parsed `--out`-style arguments under `-File` -> reads `$args` verbatim.

### Decisions / spec questions (details at the top of the named source files)
- tau_id rule string with negatives = "midpoint"; negative globals and kfCov are rounded through binary16 (H1a, H2a).
- Tracker: missed frames neither count toward nor reset the LINE downstream run; LINE is checked before STEADY; the
  firing frame is buffered before triggers; buffering stops once judged (H1b, H2a - goldens agree).
- Governor cooldown keeps running while the target stays below the level; headroom compared in float32 (H1b).
- Calibration positives include sanity-ok samples that failed identity (literal spec reading); certificate gate
  counts are over sanity-ok samples (H2a, H1a).
- DINOv2 spec in the app = the fp16-weight file only (it is also the CPU accuracy reference; 44 MB, adb-pushed).
- Runtime crash guard softens system/low-memory kills of the probe to one strike (two = crashed) (H3).

### Open risks / to check on the phone
- Fill feature (spec 2.7) depends on rotation for small parts (60x20 px rectangle at 30 deg: 0.937 vs 1.0). Teach video
  rotation widens the band; if the phone shows "shape" false rejects on rotated parts, switch to a moment-based fill.
- Synthetic A/B only (NOT evidence): R18 default caught 8/15 synthetic defects, CANONICAL rotation 15/15; DINOv2 let
  look-alikes pass identity (sim 0.80-0.83 vs tau_id 0.797). Decide challengers only on real held-out captures.
- Keyframe globals are binary16-rounded and not re-normalised (identity sims can read 1.00001) - harmless, noted.
- Replay "latency" includes judge-queue wait when frames are fed faster than real time; the live claim uses camera
  frames (JSONL also logs `judgeMs` per part).
- Everything in `native/runtime/README.md` "must be verified on the phone": SOC string, NPU speed-up, JIT time,
  probe lifecycle, k-NN graph binding, VLM backend/latency, ADSP path in all three processes.

## Usability fixes after the first hands-on try (2026-09-27, ~04:10-04:40 IST)

The user reported "a bunch of boxes and no verdict". Reproduced on the emulator with a non-plain background (61 REFRAME
"verdicts" on background specks). Changes (APK `native/dist/apk/KaizenEye2-phonefree-debug.apk`, sha256 `60C21B87...48BE`,
audit PASS, all JVM tests green):
1. Every camera screen (Teach, Inspect/Calibrate, Negatives) now starts with an explicit **Step 1: clear the sheet ->
   LEARN SHEET**; the sheet is never learned silently (it was, at Inspect start, even with a part or hand in view).
2. Learning the sheet first lets auto exposure / white balance / focus settle (20 frames), then **locks** them for the
   rest of that screen (`CameraController.lockForInspection`), so a part entering the view cannot shift the background.
3. Only part-sized blobs (>= 1 % of the analysis frame, the TOO_SMALL limit) are tracked, judged and drawn (app-level
   filter before the core tracker; spec 11.1 unchanged). Border-touching and tiny blobs are no longer drawn.
4. Plain-language hints on Inspect ("too many blobs -> re-learn", "something touches the edge", "slide it across the
   dashed line", "keep it still"), a Re-learn sheet button, and on Teach a live box that turns green ("part ✓") only
   when the part is usable, with the reason otherwise.
5. Headroom-only thermal level L2 is labelled "near the thermal limit (headroom x)", not "phone is hot".

## Accuracy package v2 - look-alikes, circle-to-select, OOM (2026-09-27, ~05:00-07:10 IST)
The other earbud case still PASSed after teaching one. Full analysis, numbers and limits: accuracy-v2.md.
- Circle to select (Roi.kt, CircleLayer): the circled part is the only candidate (glare-split pieces merged), the ROI follows the part while teaching; on Inspect a circle judges that part at once, then restricts the view. Tests: RoiSelectTest.
- Look-alike fix: DINOv2 + CANONICAL rotation + FIT gate + other Twins / Negatives as known wrong objects; score gate factor 1.4 -> 1.25.
- Memory: largeHeap, at most 40 embedded teach frames, at most 24 keyframes. (User error: "Failed to allocate 50331664 byte allocation ... growth limit 268435456" = 32 DINOv2 keyframe maps.)
- Slider renamed Tolerance, 0.5-2.5x. Weak teaches are called out on the Armed card.
- Tests: core 171 -> 181 (FitGateTest, CanonicalCropTest, RealTwinFitTest skipped without data), app RoiSelectTest; goldens unchanged.
- NOT verified on the phone: live canonical judging latency (telemetry stage snapshot), end-to-end look-alike rejection on live frames.

## Final fine-tune pass before the evaluation (2026-09-27, ~07:15-07:45 IST)
Four read-only reviews (pipeline, UI/camera, core, runtime) then fixes; no architecture change, no new feature. Tests 244/244.
- Core: a NaN/infinite score can no longer PASS (REFRAME "SCORE_FAILED"); the fit gate never passes a NaN fit; the vote only
  applies to DEFECTs, re-checks the chosen crop's fit gate and carries its orientation (heat map / explanation no longer
  mirrored); calibration replaces the fit-gate samples with the VALID ones (a look-alike shown during calibration cannot
  loosen the gate); teach fails on all-zero LSO; segments clamped >= 0; TwinStore restores the old Twin if a save fails and a
  damaged fit.json degrades to "no fit gate" instead of an unloadable Twin; negatives maps validated + written atomically;
  exited-unjudged counts only confirmed tracks; canonical square clamped to the frame, MULTIPLE checked on the rotated
  square, round parts (aspect >= 0.85) cropped at theta = 0 (unstable principal axis).
- Pipeline: leaving any camera screen resets the sheet/teach/circle state (system back during a recording used to build and
  activate a partial Twin on the next screen); an engine reload stops every session that holds the old engine; failed
  reload keeps the old engine; ensure() never double-loads; the self-test no longer switches the app backbone; flicker now
  re-learns the sheet at 10 ms instead of changing exposure after the sheet was learned (and never persists the locks);
  replay threads guarded (no stale session hijacks Inspect); calibration drains the judge queue before finishing; judge
  failures show as REFRAME "error" (never dropped); the best SANE snapshot is judged; whole-view tracking merges glare-split
  pieces like the circle does; fit negatives applied to the CURRENT active Twin (no lost calibration); batch jobs never
  beep/explain/write reject files; uncaught coroutine errors are logged instead of killing the app.
- Runtime: NaN features answer NaN (graphs stay enabled); one accelerator error is retried before the CPU fallback;
  an accelerator must beat CPU on the median too; the crash-guard marker is cleared on a cancelled load; the app-level
  "crashed natively - not retried" block is gone (the runtime guard already strikes the candidate).
- UI declutter: Home = model/offline chips, active part card (plain calibration sentence, look-alike protection line),
  Wrong objects / Calibrate / Certificate, teach, other parts (delete with confirm), one "Advanced" toggle for clips,
  telemetry, readiness, self-test; INSPECT/TEACH disabled until the model is loaded; Inspect HUD = PASS / REJECT /
  NOT THE PART; motion chips, zoom on Inspect/Negatives, tau/LSO/fps/p95/governor jargon removed from operator screens;
  fixed reject-card slot (camera view no longer jumps); tolerance persisted on release + certificate-void warning;
  buttons never wrap; keyboard no longer hides the part name (imePadding); camera permission re-binds the preview;
  reject beep on the ALARM stream (audible in silent / DND); Negatives screen shows one step at a time.
- Second pass (verification review of the fixes): sheet state reset when a camera screen is left; engine reload clears the
  sheet and re-shows LEARN SHEET; heat map anchored on the judged snapshot; whole-view piece merge only joins pieces
  <= 0.5x the biggest candidate, carries the merged label map and is off for the spec pipeline (parity); a failed reload
  restores the previous backbone choice and shows the error on Home; only the newest fit job applies; a part is never
  counted twice after a judge failure; the theta = 0 rule for round parts bumps preprocessVersion to 2 (accuracy mode), so
  parts taught before 07:45 IST must be taught again; interactive Replay keeps its reject card; Teach panel scrolls.
- DINOv2 accelerator retry (07:58 IST): registering the plain fp32 export (`dinov2_s14_448_fp32.tflite`, 88 MB, pushed) as
  the first variant lets the GPU delegate compile it: **GPU 57 ms (2.1x CPU), chosen**. NPU (QNN 2.49) still runs it at
  184-187 ms = no speed-up (the ViT falls back inside the delegate), so the honest rule keeps it off. The fp16-weight file
  fails GPU compile (dequantize ops). ResNet18 stays the only backbone that runs on the NPU (10.9 ms; GPU 4.7 ms wins).
