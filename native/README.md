# Kaizen Eye 2 — native Android app

Learn a part from one 12 s video, lock onto the object, judge every part once on a live hand-fed line (PASS / DEFECT /
NOT THE ENROLLED PART / REFRAME) with a beep and a red box that rides the part, print a statistically bounded
false-alarm certificate, and explain rejects with an offline VLM — all on the phone, no network permission.

Plan: `C:\Users\Vijay Chikkala\.claude\plans\so-now-we-are-effervescent-tulip.md` · algorithm spec:
`docs/verification/twin-spec.md` · decisions/evidence: `docs/verification/decisions.md` · phone-day steps:
`docs/verification/phone-day-runbook.md`.

## Modules
| module | what | tests |
|---|---|---|
| `:core` (pure Kotlin/JVM) | all maths: legacy PatchCore port, image ops, empty-sheet mask + connected components + geometry, crop/patch coverage, k-NN, scoring, teach (keyframes, coreset, leave-segment-out τ, identity/geometry gates), verdicts + voting, calibration + Clopper–Pearson certificate, Twin store (`twin.json` + `.f16`), tracker + photo-eye/steady-hold triggers, thermal governor, explanation guard/templates, replay comparator | JUnit vs `testdata/golden_core.json`, `golden_app.json`, `golden_twin.json` (numpy reference `tools/lab/twin_ref.py`) + synthetic trajectories |
| `:runtime` (Android library) | LiteRT 2.2.0 backbone/k-NN runners, accelerator probe in a `:probe` process with crash guard + decision cache + honest badges, model locator (SHA-256 pinned), FastVLM service in a `:vlm` process | JVM unit tests of the decision rules and k-NN bank preparation |
| `:app` | Compose UI (Home, Teach, Inspect/Calibrate/Replay, Certificate, Negatives, Telemetry, Readiness, Clips, Self-test), CameraX pipeline + replay source + recorder, sessions (sheet, teach, line, negatives), JSONL/CSV logs, thermal monitor, alarm, eval export, self-test | JVM unit tests (flicker check, …) |

## Build (always through the wrapper: JBR 21, short TEMP, one build at a time)
```
powershell -File native\build.ps1 :core:test :runtime:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug
powershell -File native\tools\apk-audit.ps1 -Apk native\app\build\outputs\apk\debug\app-debug.apk
```
Pins that matter (do not bump casually): Gradle 9.5.0, AGP 9.3.3 (built-in Kotlin 2.2.10), compileSdk 36.1, LiteRT 2.2.0,
LiteRT-LM 0.16.1, qnn-runtime 2.49.0, Compose BOM 2026.02.01, CameraX 1.6.2.

## On the phone
- Install: `adb install -r native\app\build\outputs\apk\debug\app-debug.apk` (never uninstall — Twins live in app data).
- Big models: `adb push <file> /sdcard/Android/data/com.kaizeneye.v2/files/models/` (FastVLM 942 MB; DINOv2 optional).
- Self-test: `adb shell am start -n com.kaizeneye.v2/.MainActivity --es selftest all` →
  `/sdcard/Android/data/com.kaizeneye.v2/files/selftest/selftest.json`.
- Logs (JSONL per session, soak CSV): `/sdcard/Android/data/com.kaizeneye.v2/files/logs/`.
- Clips (labelled recordings for replay/eval): `…/files/clips/`; eval export for `tools/lab/twin_eval.py`: `…/files/exports/`.

## Processes
`main` (UI, camera, vision LiteRT) · `:probe` (accelerator benchmark; a native crash there only marks that candidate) ·
`:vlm` (LiteRT-LM FastVLM — its JNI library embeds its own LiteRT, so it never shares a process with the vision stack).
`Application.onCreate` is empty on purpose (it runs in every process).
