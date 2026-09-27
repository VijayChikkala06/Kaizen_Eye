# Kaizen Eye 2 — phone-day runbook

Everything that could be built without the phone is in `native/` (see `docs/verification/decisions.md`, section
"Phone-free build-out"). This is the ordered list of what to do once the iQOO 15 (SM8850) is on the desk. Each step
says what to run and which evidence file it produces. Paths on the phone are under
`/sdcard/Android/data/com.kaizeneye.v2/files/` (adb-pullable, no root).

## 0. Build + audit on the laptop (5 min)
```
powershell -File native\build.ps1 :core:test :runtime:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug
powershell -File native\tools\apk-audit.ps1 -Apk native\app\build\outputs\apk\debug\app-debug.apk
```
Expect: all JVM tests green, `AUDIT PASS` (arm64 only, 16 KB aligned, CAMERA + VIBRATE only, no INTERNET).

## 0b. Optional: emulator smoke test (no phone, already passed once — see decisions.md)
The `KaizenTest` AVD (Android 16 x86_64) runs the arm64 APK through ARM translation, but LiteRT's XNNPACK crashes
there, so use the pure-Kotlin DEBUG feature extractor (never for claims):
```
D:\AndroidSdk\emulator\emulator.exe -avd KaizenTest -no-window -no-audio -no-snapshot -memory 3072 -gpu swiftshader_indirect
adb -s emulator-5554 install -r -g native\app\build\outputs\apk\debug\app-debug.apk
adb -s emulator-5554 shell am start -n com.kaizeneye.v2/.MainActivity --es backbone debug --es selftest quick
```
Expect offline PASS, core maths PASS (131 checks), replay regression PASS (mechanics only). Stop Gradle daemons first
(`native\build.ps1 --stop`) — the emulator needs ~3 GB. On the PHONE the default backbone is `r18`; if a debug run
ever changed it, `--es backbone r18` switches back.

## 1. Connect + install (G0)
1. Plug in only the demo phone; accept the USB-debugging RSA prompt; enable "Install via USB" if OriginOS shows it.
2. `adb devices` → `device`. `adb install -r native\app\build\outputs\apk\debug\app-debug.apk` (never uninstall: Twins,
   decisions and weights live in app data).
3. Push the VLM: `adb push native\dist\downloads\FastVLM-0.5B.qualcomm.sm8850.litertlm /sdcard/Android/data/com.kaizeneye.v2/files/models/`
   (the app verifies SHA-256 `0EB6860D…B482`; Readiness → CHECK MODELS).
4. Launch once, grant CAMERA.

## 2. One-shot self-test (no hands)
```
adb shell am start -n com.kaizeneye.v2/.MainActivity --es selftest all
adb pull /sdcard/Android/data/com.kaizeneye.v2/files/selftest/selftest.json docs/verification/selftest-device.json
```
Read it in this order:
- **offline proof** must be PASS (no INTERNET, only CAMERA + VIBRATE).
- **device passport**: record `Build.SOC_MODEL` (the NPU checker accepts anything starting with `SM8850`), Camera2
  exposure/ISO ranges, `getThermalHeadroom` availability, battery current sign.
- **core maths on device**: must PASS (same numbers on ART as on the JVM goldens).
- **backbone accelerators**: which candidate won, cosine/rel-err vs CPU, p50/p95 per candidate, speed-up vs CPU.
  This is gate **G1a/G1b**. If NPU is "requested — no speed-up" or failed: see §4.
- **k-NN graph accuracy**: max rel err ≤ 1e-3.
- **replay regression**: the bundled synthetic clip (teach + 10 parts) must reproduce `expected.json` (borderline
  entries accept any verdict).
- **VLM explanation**: backend (NPU/GPU/CPU) + ms; "unavailable" is acceptable (template explanations are used).

## 3. Camera tuning (plan B1a) — Telemetry + Teach screens
1. Phone on the stand (flash is never used — even room or lamp light) ~25 cm over the matte sheet. Readiness → note exposure/ISO ranges from the passport.
2. Teach → LEARN SHEET: the flicker check prints `no flicker` or `FLICKER → use 10 ms exposure`.
3. Default preset is AE-auto + torch; for the moving line switch to manual 0.5–1 ms exposure, raise ISO until the part
   is well exposed; lock AWB; lock focus once converged. (Values are persisted.)
4. Ruler: measure mm/px at the working distance (for the line-speed claim).

## 4. NPU bake-off (plan B1b) — only if the self-test did not already show an NPU speed-up
- Arm A (LiteRT 2.2.0 JIT, in the app): Telemetry → Re-benchmark all; read the candidate table. Check logcat tag
  `KaizenRuntime` for dispatch/QNN load errors (missing `libQnnHtpPrepare`/Ir/Saver, `ADSP_LIBRARY_PATH`).
- Arms C (LiteRT 1.4.0 + qnn delegate, int8) and B (ORT-QNN, `tools/dinov2/out/backbone_r18_320.onnx`) are spike
  modules still to be created on the day if A fails (plan B1b; one runtime family per APK).
- No NPU by the time box → the app runs GPU/CPU with an honest badge; record it in `decisions.md`.

## 5. Teach, negatives, calibrate (plan B3, gate G3)
Every camera screen (Teach, Inspect, Calibrate, Negatives) starts with an explicit **LEARN SHEET** on the empty sheet;
nothing is learned silently. Step-by-step operator walkthrough: `how-to-test-on-phone.md`.
1. Teach: LEARN SHEET → name → START RECORDING (12 s, turn the part slowly) → Armed (target ≤ 30 s from the tap).
   Repeat ×5 for the teach-time claim (single operator).
2. Negatives: LEARN SHEET, then show 30–50 different WRONG objects, ½ s steady hold each. Never reuse them as test objects.
3. Calibrate: LEARN SHEET, then present ≥ 40 DISTINCT good parts (not the taught one) → FINISH CALIBRATION → Certificate.
4. Held-out tests (G3): ≥ 20 good parts never used before (≥ 90 % PASS), ≥ 20 wrong objects not in the negatives
   (≥ 90 % NOT THE ENROLLED PART, look-alikes separately), ≥ 10 seeded defects (≥ 80 % DEFECT). Clopper–Pearson
   bounds for all three go into `claims.csv`.
5. Record labelled clips of every set (Clips → label → REC) so the numbers can be re-run offline.

## 6. Live line (plan B4, gate G4)
Inspect: LEARN SHEET (locks exposure/WB/focus), "Parts move: across", then 50 parts at ≈ 0.3 m/s across the dashed photo-eye line → 50 verdicts, 0 double counts, every seeded defect
beeps; p95 trigger→verdict in the HUD and in the JSONL log. `adb pull …/files/logs`.
Replay the same clip (Clips → Fast) → identical verdict list.

## 7. Offline A/B on the laptop (plan "A/B")
Record labelled clips (Clips: `teach`, `good`, `defect`, `wrong`, `rotated`, `lookalike`, `calib`, `negatives`; each
starting with ½ s of empty sheet), then Clips → **Export eval set**. It replays them through the identical pipeline
(teach from the newest `teach` clip, negatives from `negatives` clips, everything else judged) and writes
`…/files/exports/<twin>_<stamp>/` (`docs/verification/export-format.md`). `adb pull` it, then
`.venv\Scripts\python tools\lab\twin_eval.py --export <dir> --out <dir>\eval` (add `--backbone dinov2` for B5).
A failed live teach can be redone from its recording with Clips → **Teach**.
Adopt a challenger (TOP1_MEAN, TOPK_VIEWS, rotation) only if it wins on held-out AUROC.

## 8. Telemetry + soak (plan B7, gate G7)
Telemetry → START SOAK LOG, run Inspect 10 min with the torch on, STOP. Check: 0 crashes, ≤ 10 % fps drop, the
governor log. First verify the battery-current sign (plugged vs unplugged) before quoting any watts.
CPU↔accelerator A/B: Telemetry → Force CPU / Accelerator, note both p50s.

## 9. Rehearse (plan B8, gate G8)
Airplane mode ON, cold start ×2, full demo ≤ 6 min, backup screen recording. Export the Twin (Readiness) as backup.

## Evidence checklist → `docs/verification/claims.csv`
selftest-device.json · device-passport.json · soak CSV · JSONL session logs · G3 counts with CP bounds ·
teach-time runs · twin_eval report · rehearsal notes.
