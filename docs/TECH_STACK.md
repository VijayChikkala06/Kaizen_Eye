# Kaizen Eye 2 — Complete Tech Stack

*Every technology, version and reason. Versions are taken from the project's own build files (`native/gradle/libs.versions.toml`, `native/*/build.gradle.kts`, `tools/requirements*.txt`) and from measurements on the target phone.*

---

## 1. At a glance

| Area | Choice |
|---|---|
| **Platform** | Native **Android**, single-Activity app, portrait, **arm64-v8a only**, `minSdk 31`, `targetSdk 36`, `compileSdk 36.1` |
| **Language** | **Kotlin 2.2.10** (app, runtime, core); Python 3.11 for laptop tooling |
| **UI** | **Jetpack Compose** (BOM 2026.02.01) + Material 3, own small navigator (no navigation library) |
| **Camera** | **CameraX 1.6.2** (core, camera2, lifecycle, view, video) + Camera2 interop |
| **AI runtime** | **LiteRT 2.2.0** `CompiledModel` (CPU / GPU / NPU) + **Qualcomm QNN runtime 2.49.0** |
| **Vision model** | **DINOv2-S/14 @ 448 px** (default) · **ResNet18 @ 320 px** (fast alternative) |
| **Explanations (optional)** | **FastVLM-0.5B** via **LiteRT-LM 0.16.1**, in its own process |
| **Build** | **Gradle 9.5.0**, **Android Gradle Plugin 9.3.3**, JBR/JDK 21 toolchain |
| **Verification** | JUnit 4 (244 tests), independent **numpy reference** implementation, on-device self-test, APK audit script |
| **Target device** | iQOO 15 (model I2501), Snapdragon **SM8850** |
| **Network** | **None** — no INTERNET permission in the shipped manifest |

---

## 2. Project structure (three Gradle modules)

| Module | Type | Responsibility | Depends on |
|---|---|---|---|
| **`:core`** | Pure Kotlin/JVM library (JVM 17) | All the maths: segmentation, tracking, teach, scoring, verdicts, calibration, storage formats, explanation guard | kotlinx-serialization only — **no Android code**, so it runs and is tested on a laptop |
| **`:runtime`** | Android library | Loading/benchmarking models on NPU/GPU/CPU, k-NN engine, model file location + SHA-256 checks, crash guard, on-device VLM client/service | LiteRT, LiteRT-LM, QNN runtime, coroutines |
| **`:app`** | Android application | Camera, Compose UI, pipeline sessions (sheet / teach / inspect / negatives / replay), alarms, telemetry, self-test | `:core`, `:runtime`, CameraX, Compose |

**Three Android processes**, for stability:

| Process | Purpose | Why separate |
|---|---|---|
| main | UI, camera, pipeline, judging | — |
| `:probe` | Benchmarks NPU/GPU/CPU candidates | A driver crash while testing an accelerator cannot take the app down; the runtime records strikes and avoids the crashing candidate |
| `:vlm` | On-device language model (~1 GB) | The LiteRT-LM library embeds its own LiteRT and must not share a process with the vision stack |

---

## 3. Languages, build and toolchain

| Item | Version / detail |
|---|---|
| Kotlin | 2.2.10 (built into AGP 9.3.3) |
| Kotlin serialization plugin + library | 2.2.10 / kotlinx-serialization-json 1.9.0 |
| Coroutines | kotlinx-coroutines-android 1.10.2 |
| Android Gradle Plugin | 9.3.3 |
| Gradle | 9.5.0 (wrapper) |
| JDK | Android Studio JBR 21 for Gradle; `:core` compiles to JVM 17 |
| Compose compiler | Kotlin Compose plugin 2.2.10 |
| Build wrapper | `native/build.ps1` — machine-wide build lock, short temp dir, per-run log in `native/dist/logs` |
| APK audit | `native/tools/apk-audit.ps1` — verifies permissions (CAMERA + VIBRATE only), 16 KB page alignment, native libs |
| Output | Debug APK ≈ 132 MB (models included), `arm64-v8a` |

---

## 4. Android / UI layer

| Component | Version | Used for |
|---|---|---|
| Jetpack Compose (BOM) | 2026.02.01 | All screens (Home, Teach, Inspect, Wrong objects, Certificate, Clips, Telemetry, Readiness, Self-test) |
| Material 3 | via BOM | Buttons, sliders, dialogs |
| Activity Compose | 1.13.0 | Single activity, back handling, permission launcher |
| Lifecycle (runtime, runtime-compose) | 2.10.0 | `collectAsStateWithLifecycle` for pipeline state |
| Core KTX | 1.18.0 | Utilities |
| Custom `Navigator` | — | Tiny back stack instead of a navigation library (predictable back behaviour, fewer dependencies) |
| Canvas overlays | — | Bounding boxes, heat-map cells (rotated for canonical crops), photo-eye line, finger-circle "doodle" |

---

## 5. Camera layer

| Component | Detail |
|---|---|
| **CameraX 1.6.2** | Preview + ImageAnalysis (RGBA_8888, 16:9, keep-only-latest) + VideoCapture (labelled clips) |
| **Camera2 interop** | Capture-request options: manual/locked exposure, AE/AWB lock, fixed focus distance, frame-rate range, **flash always off** |
| **Zoom** | `cameraControl.setZoomRatio` (up to 5×) — only the zoomed view is analysed |
| **Sheet learning** | Camera unlocked → AE/AWB/AF settle → sheet learned → settings **locked** for the session |
| **Flicker check** | Detects mains-light banding; switches to a 10 ms exposure and re-learns the sheet |
| **Replay source** | Feeds recorded clips/JPEG sequences through the *identical* pipeline (regression tests, evaluation export) |

---

## 6. Computer-vision pipeline (`:core`, pure Kotlin)

| Stage | Method |
|---|---|
| Background model | Per-channel **median / MAD** colour model of the empty sheet; slow adaptation while nothing is in view |
| Foreground | Colour distance test (`d² > kσ²`), 3×3 open/close morphology, 8-connected components |
| Shape features | Fill, aspect, Hu moment, solidity, area (± 3σ band with floors) |
| Tracking | Constant-velocity tracker, gated assignment, **photo-eye (line-crossing)** and **steady-hold (½ s)** triggers, best-crop buffer, one verdict per part |
| Crop | Square crop with 10 % margin; **canonical rotation** (long axis horizontal, judged at θ and θ+π, square fitted to the rotated extent); round parts cropped unrotated |
| Circle-to-select | Finger-drawn polygon (point-in-polygon, merge of glare-split pieces, following the part while teaching) |
| Sharpness | Variance of the Laplacian (frame quality, blur warning) |

---

## 7. AI models

| Model | Role | Details |
|---|---|---|
| **DINOv2-S/14** (Meta, Apache-2.0) | Default backbone | 448 px input → 32×32 patch grid × 384 dims; exported from PyTorch (own minimal implementation, checked against an independent float64 numpy version); **fp32 file (88 MB) runs on the GPU**, fp16-weight file (44 MB) as CPU fallback |
| **ResNet18** (torchvision, ImageNet weights) | Fast alternative backbone | 320 px input → 40×40 × 128 dims (layers 2+3 features); fp32 and int8 files bundled in the APK |
| **k-NN matmul graphs** | Nearest-neighbour search | Small LiteRT graphs (`knn_p1024_d384_k2400`, `knn_p1600_d128_k{800,1600,2400}`) with masking; exact float64 CPU fallback; agree to ≤ 1e-3 |
| **FastVLM-0.5B** (Apple research licence) | Optional one-sentence reject explanation | Runs offline via LiteRT-LM in the `:vlm` process; a **location guard** rejects sentences that contradict the measured defect location; template text is the fallback |

**Measured on the target phone (SM8850):**

| Model | CPU | GPU | NPU | Chosen |
|---|---|---|---|---|
| DINOv2 fp32 | 120 ms | **57 ms (2.1×)** | 187 ms (no speed-up) | GPU |
| DINOv2 fp16w | 124 ms | fails to compile | 184 ms | CPU |
| ResNet18 fp32 | 22.6 ms | **4.7 ms (4.8×)** | 10.9 ms (int8: 5.7 ms) | GPU |
| k-NN engine | 2.6 ms (CPU) | — | — | CPU |

---

## 8. Inference runtime and accelerators (`:runtime`)

| Component | Version | Role |
|---|---|---|
| **LiteRT** (`com.google.ai.edge.litert`) | 2.2.0 | `CompiledModel` API: one interface for CPU, GPU and NPU |
| **Qualcomm QNN runtime** | 2.49.0 | HTP (NPU) dispatch + compiler plugin, `ADSP_LIBRARY_PATH` set for all processes |
| **LiteRT-LM** (`litertlm-android`) | 0.16.1 | On-device language model engine (0.17.x needs Kotlin 2.4, so pinned lower) |
| Honest-badge rules | own | An accelerator is used and claimed only if its time ≤ 0.9 × CPU (best and median); decisions cached per device fingerprint |
| Crash guard | own | Strike counting via `ApplicationExitInfo`; crashed candidates are skipped; cancelled loads clear their markers |
| Model locator | own | Looks in adb-push folder → imported → bundled assets; **SHA-256 pinned** for every model |

---

## 9. The algorithm ("Visual Twin") — technical summary

| Step | Technique |
|---|---|
| Teach | ~8 fps frames for 12 s → keep sharp frames (P40) → **greedy k-centre** keyframes (16–24) → 5 time segments → pooled patch bank → **coreset** (10 %, 256–2400 rows, greedy k-centre) → binary16 rounding |
| Anomaly score | **PatchCore-style** nearest-neighbour distance per patch → masked 3×3 smoothing → max over the part (**defect score**) |
| Whole-part fit | Mean nearest-patch distance over the part core (**FIT gate**) — separates look-alike objects |
| Thresholds | τ from **leave-segment-out** scores × 1.25; τ_id from positive/negative similarity midpoint; τ_fit from same-part fits × 1.10 (midpoint to known wrong objects: other parts + "Wrong objects" library) |
| Verdict order | Sanity → identity → shape → FIT → score (→ borderline vote of up to 3 crops) → PASS / REJECT / NOT THE PART / REFRAME |
| Calibration | 40 good parts → order-statistic bound α = 1 − 0.05^(1/m), **Clopper–Pearson** intervals per gate → certificate |
| Storage | `twin.json` + `.f16` matrices (`KZF16v1` format) + `fit.json`; atomic saves (`.tmp` → rename); **pipeline fingerprint** (SHA-256 of canonical JSON) refuses mismatched models |
| Explanations | Facts line + template sentence, optional VLM sentence guarded by location check |

---

## 10. Telemetry, safety and power

| Feature | Implementation |
|---|---|
| Thermal governor | Android thermal status + headroom → levels L0/L1/L2 (fps cap, voting off, explanations paused) |
| Idle detector | Drops frame-rate when nothing is in view |
| Logging | JSONL per session (verdicts, scores, timings, sheet, governor) in the app's external files dir |
| Soak recorder | CSV every 2 s (thermal, battery °C, power, fps, p95) |
| Alerts | `ToneGenerator` on the **alarm stream** + `Vibrator` (escalates after 3 rejects) |
| Offline proof | No INTERNET permission, checked by audit script; Readiness screen shows it |
| Device passport | JSON of SoC, camera characteristics, exposure/ISO ranges |

---

## 11. Verification and tooling (laptop side)

| Tool | Version | Purpose |
|---|---|---|
| JUnit | 4.13.2 | 244 tests across `:core`, `:runtime`, `:app` (goldens, gates, tracker, calibration, storage, circle selection, canonical crops) |
| **`tools/lab/twin_ref.py`** | numpy 2.4 | **Independent reference** of the whole Twin maths, written only from the spec; generates the golden files the Kotlin tests check |
| `tools/lab/twin_eval.py`, `real_twin_accuracy.py` | numpy / scipy / Pillow | Offline A/B evaluation and the real-capture accuracy study |
| `tools/dinov2/` | PyTorch 2.13, torchvision 0.28, ONNX 1.23, onnxruntime 1.30, ai-edge-litert 2.2 | Model export (own DINOv2 implementation), ONNX/TFLite conversion, parity checks (cosine 1.0 fp32) |
| `tools/convert_backbone.py`, `make_knn_model.py` | litert-torch / onnx2tf | Build the backbone and k-NN `.tflite` files |
| On-device self-test | own | 131 core checks on ART, accelerator parity + latency, k-NN graph accuracy, replay regression on a bundled synthetic clip, offline/permission proof |
| `docs/verification/` | — | Spec (`twin-spec.md`), decisions log, accuracy study, runbook, phone test guide |

---

## 12. Licences and third-party components

| Component | Licence | Note |
|---|---|---|
| DINOv2 | Apache-2.0 | Free for commercial use |
| ResNet18 (torchvision) | BSD-3 | ImageNet-pretrained weights |
| LiteRT, CameraX, Compose, AndroidX | Apache-2.0 | |
| Qualcomm QNN runtime | Qualcomm licence (redistributable binaries via Maven) | Required for the NPU path |
| FastVLM-0.5B | Apple research licence | Fine for a competition/demo; **a commercial release would need another licence or a swap** |
| Kotlin, kotlinx libraries | Apache-2.0 | |

---

## 13. History note (useful if asked "did you start over?")

The first version (`mobile/`, Expo / React Native) worked from photos with an on-device model. It was **kept frozen as a fallback** and the product was **rebuilt natively in Kotlin** to get live video, direct camera control, accelerator access and process isolation — none of which the cross-platform version could provide.
