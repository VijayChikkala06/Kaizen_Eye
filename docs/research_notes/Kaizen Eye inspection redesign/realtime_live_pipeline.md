# Real-time live-camera defect inspection on an Android phone (Kaizen Eye live pipeline)

_Scope: a live inspection loop for Kaizen Eye on an iQOO 15 (Snapdragon 8 Elite Gen 5, offline). The loop covers capture, part detection, tracking, one decision per part, on-screen marking, beep/vibration, counters and logging, plus a phone-only demo. Researched 2026-09-26. Every latency number below names its device/chip and runtime. None of them was measured on an iQOO 15. "Search-result summary" means the claim came from a search snippet and the page itself was not fetched and read._

---

## Q1. What end-to-end latency and throughput can a Snapdragon 8 Elite / 8 Elite Gen 5 phone achieve?

### Takeaway
The nano detectors are no longer the bottleneck. Qualcomm AI Hub profiles YOLO11n, YOLOv8n and YOLO26n at 640×640 at roughly 0.6–3.5 ms on the Snapdragon 8 Elite Gen 5 NPU (model only). Ultralytics measured a full YOLO26n call (pre-processing + inference + post-processing) on a Snapdragon 8 Elite Gen 5 phone at 10.7 ms on the NPU (QNN), 15.8 ms on the GPU (LiteRT) and 52.2 ms on the CPU. A 30 fps detect-and-track loop therefore fits comfortably, and 60 fps is plausible on the NPU or with classical segmentation. The real bottleneck is Kaizen Eye's per-part anomaly scoring (about 140–440 ms today). It has to run once per part, off the frame loop.

### Cited Findings
#### Detector latency, model only (Qualcomm AI Hub profiling, NPU, 640×640 input)
- **YOLO11-N** ("YOLOv11-Detection", 2.64 M params, AGPL-3.0)
  - Snapdragon 8 Elite Gen 5: TFLite float **1.614 ms**, TFLite w8a8 **0.732 ms**, QNN_DLC w8a16 **1.634 ms**.
  - Snapdragon 8 Elite: TFLite float **1.977 ms**, w8a8 **0.805 ms**.
  - Snapdragon 8 Gen 3: TFLite float **2.622 ms**, w8a8 **1.097 ms**.
  - Model size: float 10.1 MB, w8a8 2.83 MB.
  - Sources: [HF qualcomm/YOLOv11-Detection](https://huggingface.co/qualcomm/YOLOv11-Detection); [AI Hub model page](https://aihub.qualcomm.com/models/yolov11_det)
- **YOLOv8-N** (3.18 M params, AGPL-3.0) on "Snapdragon 8 Elite Gen 5 For Galaxy Mobile", NPU:
  - TFLite float **1.485 ms**, TFLite w8a8 **0.582 ms**.
  - QNN_DLC float **1.754 ms**, QNN_DLC w8a8 **0.587 ms**.
  - ONNX float **2.654 ms**, ONNX w8a8 **0.629 ms**.
  - Source: [HF qualcomm/YOLOv8-Detection](https://huggingface.co/qualcomm/YOLOv8-Detection)
- **YOLO26-N** (2.4 M params, AGPL-3.0, NMS-free, "NMS-Free Real-Time Object Detection for Edge Devices"), NPU:
  - "Snapdragon 8 Elite Gen 5 For Galaxy Mobile": TFLite float **2.295 ms**, QNN_DLC float **2.188 ms**, QNN_DLC w8a16 **1.939 ms**, ONNX float **3.461 ms**.
  - "Snapdragon 8 Elite For Galaxy Mobile": TFLite float **3.079 ms**, QNN_DLC w8a16 **2.158 ms**.
  - Sources: [HF qualcomm/YOLO26-Detection](https://huggingface.co/qualcomm/YOLO26-Detection); [AI Hub YOLO26 page](https://aihub.qualcomm.com/models/yolo26_det)

#### Detector latency, app level, measured on a Snapdragon 8 Elite Gen 5 phone
- **Test setup.** Ultralytics benchmarked YOLO26n detection at 640 on a **Xiaomi 17** (Snapdragon 8 Elite Gen 5 / SM8850, 12 GB LPDDR5X, Android 16). They used the Ultralytics Flutter plugin 0.6.10 and report the "mean of 15 runs after 3 warmup runs on `bus.jpg`".
- **CPU and GPU results** ([Ultralytics LiteRT integration docs](https://docs.ultralytics.com/integrations/litert)):
  - CPU (LiteRT, w8a32): **52.2 ms** = 1.8 pre / 48.1 inference / 2.4 post.
  - GPU (LiteRT OpenCL, w8a32): **15.8 ms** = 2.3 / 8.9 / 4.6.
- **NPU result** ([Ultralytics QNN integration docs](https://docs.ultralytics.com/integrations/qnn)):
  - Same device and harness. Runtime is QNN HTP, W8A16, through the ONNX Runtime QNN Execution Provider loading an HTP v81 context binary.
  - Latency **10.7 ms** = 1.8 / 6.7 / 2.2.
  - Export command: `model.export(format="qnn", name="81", imgsz=640)`.
  - Context-binary generation "runs on an x64 host" and does not support macOS.
- **Packaging risk.** An open Ultralytics issue is titled "YOLO11n TFLite GPU delegate fails on Android — GpuDelegateFactory$Options missing from LiteRT packaging". This matters for a native LiteRT GPU build — [ultralytics/ultralytics#24267](https://github.com/ultralytics/ultralytics/issues/24267) (title only; issue body not read)

#### Can the app reach the NPU?
- **LiteRT Qualcomm accelerator** ([Google Developers Blog](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)):
  - Built on QNN, it replaces the older TFLite QNN delegate.
  - Google claims "up to a 100x speedup over CPU and a 10x speedup over GPU".
  - On Snapdragon 8 Elite Gen 5, "over 56 models run in under 5ms with the NPU, while only 13 models achieve that on the CPU". The suite has 72 models, 64 of which delegate fully, using 90 supported ops.
  - API: `CompiledModel.create(..., CompiledModel.Options(Accelerator.NPU, Accelerator.GPU))`.
  - Deployment is either AOT (AI Pack via Play) or on-device compilation from a `.tflite` in assets.
- The LiteRT docs list "Snapdragon 8 Elite Gen 5 Mobile Platform (SM8850)" as supported, with both AOT and on-device compilation through `CompiledModel` — [LiteRT Qualcomm NPU docs](https://developers.google.com/edge/litert/next/qualcomm)
- Qualcomm markets the 8 Elite Gen 5 Hexagon NPU as **37% faster** than the previous generation — [Qualcomm product page](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5); [GSMArena](https://m.gsmarena.com/hz5/qualcomm_snapdragon_8_elite_gen_5_soc_features_specs-news-69656.php) (search-result summaries)
- **react-native-fast-tflite**, the app's current runtime, has no QNN/NPU delegate. On Android it offers only `['android-gpu']` and `['nnapi']`, and warns: "NNAPI is deprecated on Android 15. GPU delegate is preferred." — [react-native-fast-tflite README](https://github.com/mrousavy/react-native-fast-tflite)

#### Camera pipeline latency and frame budgets
- **Glass-to-glass latency** ([inovex](https://www.inovex.de/de/blog/the-glass-to-glass-latency-on-android/)):
  - Measured on a Pixel 2 (camera 30 fps, display 60 Hz): **"80 milliseconds ± 25 milliseconds"**.
  - Breakdown: exposure 0–33 ms, sensor→app **24 ms** (OnImageAvailable), app→display ~33 ms, display scan-out 0–16 ms.
  - Last sensor pixel → first display pixel is 57 ms; first sensor pixel → last display pixel is 106 ms.
- CameraX says to analyze each frame "preferably within the given frame rate time limit (for example, less than 32ms for 30 fps case)". For analyzers under 16 ms at 60 fps, "either operating mode provides a smooth overall experience" — [CameraX Image analysis](https://developer.android.com/media/camera/camerax/analyze)
- CameraX team (Scott Nien): "if the next frame comes and your Analyzer hasn't finished the previous frame and closed the image, then frames will be dropped" — [camerax-developers thread](https://groups.google.com/a/android.com/g/camerax-developers/c/M4cAi_6U7wc)

#### Cost references for tracking and anomaly scoring
- SORT (Kalman filter + Hungarian algorithm) "updates at a rate of 260 Hz which is over 20x faster than other state-of-the-art trackers". That was measured on the MOT benchmark with a 2016 desktop CPU — [Bewley et al., arXiv 1602.00763](https://arxiv.org/abs/1602.00763)
- ByteTrack reports "30 FPS running speed on a single V100 GPU", detector included — [arXiv 2110.06864](https://arxiv.org/abs/2110.06864)
- EfficientAD reports "a latency of two milliseconds and a throughput of six hundred images per second". Its feature extractor processes an image in under 1 ms "on a modern GPU" (desktop GPU, not mobile) — [arXiv 2303.14535](https://arxiv.org/abs/2303.14535)
- Kaizen Eye today takes about 40–90 ms for the ResNet18 embedding plus about 100–350 ms for the JS kNN, per photo. These are the user's figures; I did not measure them.

### Inferences
- **Budget with app-level numbers, not AI Hub numbers.** For YOLO26n on the NPU, AI Hub shows about 1.9 ms model-only (QNN_DLC w8a16). Ultralytics' in-app call took 6.7 ms of inference and 10.7 ms in total. That is a 3–5× gap from runtime overhead, tensor copies and pre/post-processing.
- **Per-frame latency budget at 30 fps (33.3 ms).** Numbers are estimates assembled from the citations above:

| Stage | Budget | Basis |
|---|---|---|
| Sensor → frame in analyzer/worklet | ~24 ms latency (does not limit throughput) | Pixel 2 [inovex](https://www.inovex.de/de/blog/the-glass-to-glass-latency-on-android/) |
| Resize / YUV→RGB to 320–640 px | ~2 ms | Ultralytics pre-processing 1.8–2.3 ms on SD 8 Elite Gen 5 |
| Detection A: threshold + contours at 320×180 | ~1–4 ms (estimate, not measured) | No benchmark found (see Gaps) |
| Detection B: YOLO nano 640, GPU (fast-tflite / LiteRT GPU) | ~9 ms inference + ~5 ms post | Ultralytics YOLO26n GPU, 15.8 ms total |
| Detection B': YOLO nano 640, NPU (LiteRT QNN / ORT QNN) | ~7 ms inference + ~2 ms post | Ultralytics YOLO26n NPU, 10.7 ms total |
| Tracker update (≤10 objects) | <1 ms | SORT ran at 260 Hz on much larger MOT scenes |
| Trigger-line test, counters | ≪1 ms | trivial arithmetic |
| Overlay to screen | next vsync; ~33 ms app→display | Pixel 2 [inovex](https://www.inovex.de/de/blog/the-glass-to-glass-latency-on-android/) |
| **Per-part, async:** crop + embed + kNN | 140–440 ms today; target ≤50–110 ms | user figures; target assumes kNN moves to native/GPU |

- **FPS targets for the iQOO 15.**
  - Classical segmentation path: 30 fps is safe and 60 fps is likely (about 4–8 ms of compute per frame).
  - YOLO nano on the fast-tflite GPU delegate: 30 fps is safe (about 17–20 ms per frame). 60 fps is borderline, because 15.8 ms total is about equal to the 16.7 ms budget.
  - YOLO on the NPU through a native LiteRT/ORT-QNN path: 60 fps is plausible (about 12–14 ms per frame).
  - Dropping the YOLO input to 320×320 should cut its compute roughly 4× (inference; not measured).
- **Event latency, from part crossing the trigger line to red box + beep.**
  - Today: about 24 + 15 + (140–440) ms ≈ **180–480 ms**, plus audio output latency.
  - With the anomaly step optimized to 50–110 ms: about **90–150 ms**, plus audio latency. To a human operator that feels "instant".
- **Throughput ceiling if anomaly scoring runs serially.** Maximum parts/min = 60,000 ÷ scoring ms:
  - 440 ms → about **136 ppm**.
  - 140 ms → about **430 ppm**.
  - 50 ms → about **1,200 ppm**.
  - For comparison, industry cites 1,600 ppm small-part lines and smart cameras rated "up to 5,000 parts per minute" (see Q5).
- **The detection loop itself is not the constraint.** Even at 30 fps the detector sees each part many times (see the Q5 frames-per-part table). The priority work is making anomaly scoring fast and asynchronous.

### Gaps
- No published latency for any YOLO model on the iQOO 15 specifically. All figures come from Qualcomm reference devices ("For Galaxy" SKUs) or a Xiaomi 17 with the same SM8850 chip.
- No published fast-tflite (TFLite GPU delegate) latency for YOLO on Snapdragon 8 Elite Gen 5. I assume it is close to Ultralytics' LiteRT-GPU figure, but that is not verified.
- No sustained or thermal-throttled FPS data for a 10–15 minute demo. An Android thermal benchmark harness for YOLOv10n/11n exists, but I did not read its numbers: [EdgeSurveillanceBenchmark](https://github.com/Zeeshndev/EdgeSurveillanceBenchmark).
- The only verified glass-to-glass figure is from a Pixel 2 (2017). A search snippet claimed 30–60 ms for "a typical phone or laptop in 2026" ([forasoft](https://www.forasoft.com/learn/video-streaming/articles-streaming/latency-glass-to-glass-explained)), but I did not verify it.
- No data found on ToneGenerator / Android audio output latency on the iQOO 15.
- No mobile (Snapdragon) benchmark found for EfficientAD or PatchCore-style kNN.

---

## Q2. What capture settings and frame-analysis architecture should the live loop use?

### Takeaway
There are two viable stacks.
- **(a) Stay in the existing Expo/React Native app.** Replace the live path's `expo-camera` with **VisionCamera V5** (Nitro) frame processors, then add react-native-fast-tflite (GPU), the GPU resizer, and a Skia/SharedValue overlay.
- **(b) Use the native Kotlin project.** Build on CameraX `ImageAnalysis` (KEEP_ONLY_LATEST, YUV) with Camera2Interop for manual exposure and fixed fps, and LiteRT `CompiledModel` (NPU → GPU fallback).

Either way the capture settings are the same:
- fixed 30 fps (60 if supported);
- short manual exposure (~0.5–1 ms) or AE lock;
- locked AF and AWB;
- no stabilization, HDR or low-light boost;
- detection on a downscaled frame, with the part crop taken from the higher-resolution frame.

### Cited Findings
#### Local project context
- The RN app `kaizen-eye` runs Expo ~57.0.25 and react-native 0.86.3, with `expo-camera` ~57.0.5, `react-native-fast-tflite` ^3.0.1, `react-native-nitro-modules` ^0.37.1 and `jpeg-js` — local file `D:\Projects\Kaizen_Eye\mobile\package.json`
- There is also a native Kotlin/Compose Android project at `C:\Users\Vijay Chikkala\AndroidStudioProjects\KaizenEye`, with `app/src/main/assets/feature_extractor.tflite` — local file listing

#### VisionCamera V5 (React Native)
- **Release and features** ([Margelo blog](https://margelo.com/blog/whats-new-in-visioncamera-v5)):
  - Published **April 16, 2026** as a full Nitro Modules rewrite.
  - Frame processors run on `react-native-worklets` and can drive Reanimated SharedValues.
  - Pixel formats: `'yuv'` and `'native'`.
  - The GPU resizer (`react-native-vision-camera-resizer`) handles "YUV -> RGB conversion, pixel packing, data type conversions, and cropping", with a "~5x performance speedup" over the CPU path.
  - Constraints API, e.g. `{ fps: 60 }`, plus resolution selection.
  - Manual exposure duration, ISO, and focus position (0 = closest … 1 = furthest).
  - Exposure lock with min/max duration and ISO ranges, and white-balance gains.
- **AsyncRunner:** "If the AsyncRunner is currently busy, it will immediately return false and not schedule the work", and the frame must then be disposed. It needs `react-native-vision-camera-worklets` and `react-native-worklets` — [VisionCamera docs: frame-processor tips](https://visioncamera.margelo.com/docs/guides/frame-processors-tips)
- **Performance guidance** ([VisionCamera Performance](https://visioncamera.margelo.com/docs/performance)):
  - The native format is YUV. RGB "introduces additional overhead and consumes more memory".
  - Higher FPS "uses more memory and could heat up the battery".
  - Video stabilization "introduces capture latency"; Video HDR adds computation.
  - Binned formats "use significantly less bandwidth".
  - Prefer the single `'wide-angle'` device.
  - Toggle `isActive` instead of remounting.
- "Lower FPS gives the Camera more time to expose each Frame". Low Light Boost lets the pipeline "automatically extend exposure times (and effectively drop frame rate)". For moving parts it must stay off — [Low Light Boost](https://visioncamera.margelo.com/docs/low-light-boost); [Constraints API](https://react-native-vision-camera.com/docs/guides/formats) (search-result summary)
- **fast-tflite with VisionCamera** ([README](https://github.com/mrousavy/react-native-fast-tflite)):
  - Works inside VisionCamera frame processors.
  - "Unlike v4, VisionCamera v5 no longer requires boxing the model with `NitroModules.box()`".
  - APIs: `model.runSync([...])` and async `model.run([...])`.
  - The Expo config plugin option `enableAndroidGpuLibraries: true` adds `libOpenCL.so` for the GPU delegate.
- react-native-fast-opencv runs inside VisionCamera frame processors. Its real-time example downscales frames to **320×180** with `react-native-vision-camera-resizer` and draws results with `react-native-vision-camera-skia` — [FastOpenCV real-time detection](https://lukaszkurantdev.github.io/react-native-fast-opencv/examples/realtimedetection)
- Skia Frame Processors "allow using @Shopify/react-native-skia to draw on a Frame in realtime" (V3/V4 docs) — [VisionCamera Skia Frame Processors](https://react-native-vision-camera.com/docs/guides/skia-frame-processors) (search-result summary)
- The V5 (Nitro) ML Kit plugin currently covers text recognition and barcode scanning only, "with more ML Kit features on the roadmap". There is no object-detection plugin for V5 yet — [react-native-vision-camera-mlkit](https://github.com/pedrol2b/react-native-vision-camera-mlkit)

#### CameraX ImageAnalysis (native Kotlin)
- **ImageAnalysis behaviour** ([CameraX Image analysis](https://developer.android.com/media/camera/camerax/analyze)):
  - `STRATEGY_KEEP_ONLY_LATEST` is the default: effectively "a queue with a depth of one", and a new image overwrites the cached one.
  - `STRATEGY_BLOCK_PRODUCER` blocks "across the entire camera device scope", including Preview.
  - Default output is `YUV_420_888`; `RGBA_8888` is optional and converted by CameraX.
  - You must call `ImageProxy.close()`.
  - The docs' example uses `setTargetResolution(Size(1280, 720))`.
- **Fixed frame rate.** Use `Camera2Interop.Extender(builder).setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30))`. The CameraX team advises to "query CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES first" — [camerax-developers thread](https://groups.google.com/a/android.com/g/camerax-developers/c/M4cAi_6U7wc). A community gist shows (60, 60) combined with AE_MODE_OFF — [gist](https://gist.github.com/magdamiu/b9caca8de54648307d8051a55e999cde) (search-result summary). CameraX may override `setTargetFrameRate()` to stay compatible with other bound use cases (search-result summary, same sources).
- **Manual exposure through Camera2Interop:** set `CONTROL_AE_MODE_OFF`, then `SENSOR_SENSITIVITY` (ISO) and `SENSOR_EXPOSURE_TIME` (nanoseconds) — [camerax-developers: manual exposure](https://groups.google.com/a/android.com/g/camerax-developers/c/jAuYkLc1wLg) (search-result summary)
- **3A semantics** ([AOSP 3A modes](https://source.android.com/docs/core/camera/camera3_3Amodes)):
  - AE_MODE_OFF: "the user controls exposure, gain, frame duration, and flash", through SENSOR_EXPOSURE_TIME, SENSOR_SENSITIVITY and SENSOR_FRAME_DURATION.
  - `CONTROL_AE_LOCK` freezes exposure.
  - `CONTROL_AE_TARGET_FPS_RANGE`: "The AE routine cannot change the frame rate to be outside these bounds".
  - `CONTROL_AE_ANTIBANDING_MODE` offers 50 Hz, 60 Hz or AUTO.
  - AF_MODE_OFF means the app controls lens position.
  - `CONTROL_AWB_LOCK` freezes white balance.

### Inferences
- **Recommended capture preset "Real parts" (moving objects):**
  - Stream: YUV at 1280×720 (or 1920×1080 if crops need more detail), at a fixed 30 fps, or 60 fps if the capabilities list includes it.
  - Exposure: AE off, 0.5–1.0 ms, ISO raised as needed, **torch on** as a steady light source near the lens.
  - Focus and colour: AF locked after focusing on the table/belt plane (or a fixed manual focus distance), and AWB locked.
  - Off: stabilization, HDR, low-light boost.
  - Why lock 3A: a new part entering the frame changes brightness and contrast, so AE/AF would hunt, and the anomaly model's "good" statistics would drift.
- **Recommended preset "Screen demo" (filming a monitor):**
  - AE on, then locked once it converges, with exposure at or above 1/60–1/120 s to avoid refresh/PWM banding (see Q7).
  - AF and AWB locked.
- **Two-rate architecture (both stacks):**
  - **Fast loop, every frame, under 16–33 ms:** downscale → detect → track → trigger-line test → overlay state.
  - **Slow path, once per part, async:** on the trigger event, crop the part from the highest-resolution buffer available (or keep the best-centred crop over the last N frames) → anomaly model → decision → overlay colour, beep and vibration, counters, log.
  - The slow path needs its **own FIFO job queue**, because VisionCamera's AsyncRunner *drops* work when busy. A dropped job would mean a part is never judged.
- **RN path specifics:**
  - The frame-processor worklet does resize, detection and a tracker written in plain JS in the worklet.
  - Per-part crops (e.g. 224×224) are copied out and pushed to the queue; `model.run` async executes the embedding.
  - Results reach the UI through SharedValues or Skia drawing.
  - The main risk is that fast-tflite has **no NPU**, but the GPU budget at 30 fps is still sufficient (Q1).
  - The existing 100–350 ms JS kNN is too slow for many parts per second. Move it into native code or into the TFLite graph, e.g. a matmul of patch features against a constant memory bank, then min-reduce, on the GPU. This is an engineering suggestion, not sourced.
- **Native path specifics:**
  - Camera2Interop gives fully documented control of manual exposure and fps.
  - LiteRT `CompiledModel` gives NPU access, and ToneGenerator and Vibrator are direct calls.
  - The drawback is rewriting the UI and kNN code that exists in RN. In a 10-hour window, whichever codebase already holds the working PatchCore pipeline should win.

### Gaps
- Unknown iQOO 15 Camera2 capabilities:
  - whether `MANUAL_SENSOR` is advertised;
  - the exposure-time and ISO ranges;
  - whether a (60, 60) AE fps range exists for YUV analysis streams;
  - whether the OEM exposes the main sensor to third-party apps at full capability.
  These must be queried at runtime through CameraCharacteristics or VisionCamera device info.
- Exact VisionCamera V5 prop and constraint names for exposure duration, ISO and locks. My only source is blog-level; I did not read the API reference.
- Compatibility of Expo SDK 57 with the VisionCamera V5 config plugin (dev client) was not verified.
- I did not check whether `expo-camera` 57 exposes any per-frame processing hook.
- Not documented: how LiteRT's QNN runtime libraries are delivered to a **sideloaded, offline** APK, versus Play for On-device AI. The [LiteRT Qualcomm page](https://developers.google.com/edge/litert/next/qualcomm) does not say.

---

## Q3. What is the simplest robust part detector that needs no training (background model + contours, saliency, class-agnostic YOLO, trigger zone + foreground mask)?

### Takeaway
On a plain, contrasting background (a belt or an A3 sheet), a classical **foreground mask + contours** detector at about 320×180, restricted to an ROI band around the trigger line, is the simplest, fastest and training-free option.

When the phone is **handheld**, build the mask by colour/brightness thresholding against the known background, not with a learned background model. Background subtraction assumes a static camera.

The other options have drawbacks:
- A COCO-pretrained YOLO is only a fallback, useful if the parts happen to resemble COCO objects.
- ML Kit Object Detection & Tracking is class-agnostic and gives tracking IDs. But it can need "30 or more frames" before its first detection, is capped at five objects, and its page says it is an unbundled, downloaded library. That makes it risky for fast parts and offline use.

### Cited Findings
- **Background subtraction** ([OpenCV-Python tutorial](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html)):
  - It is presented for static-camera scenes such as visitor counters and traffic monitoring.
  - MOG2 picks the number of Gaussians per pixel automatically, has better "adaptibility to varying scenes due illumination changes", and offers shadow detection at a computational cost.
  - GMG "uses first few (120 by default) frames for background modelling".
  - The current OpenCV 4.x tutorial ([docs.opencv.org](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)) returned HTTP 403 to my fetcher, so it is not quoted.
- **MOG2 cost:** search summaries describe MOG2 as heavy for constrained edge devices. One comparison on a Raspberry Pi 3 (768×576 video) took **40.96 s** for MOG2 against **17.63 s** for CNT for the whole clip. Per-frame time is not given — [BackgroundSubtractorCNT](https://github.com/sagi-z/BackgroundSubtractorCNT) (search-result summary)
- **FastOpenCV real-time pipeline** ([FastOpenCV example](https://lukaszkurantdev.github.io/react-native-fast-opencv/examples/realtimedetection)):
  1. Resize to **320×180**.
  2. `cvtColor` BGR→HSV.
  3. `inRange`.
  4. `findContours` (RETR_TREE, CHAIN_APPROX_SIMPLE).
  5. `contourArea` > 3000.
  6. `boundingRect`.
  
  Objects are host objects reclaimed by GC, and manual disposal is recommended to limit peak memory.
- **ML Kit Object Detection & Tracking** ([ML Kit ODT Android](https://developers.google.com/ml-kit/vision/object-detection/android); [ML Kit ODT overview](https://developers.google.com/ml-kit/vision/object-detection)):
  - Detects "up to five objects" and assigns tracking IDs in STREAM_MODE.
  - In STREAM_MODE it "might produce incomplete results … on the first few invocations" and "might need to process 30 or more frames, depending on device performance, before it detects the first object".
  - Google's tips: "Disable classification if you don't need it"; "don't use multiple object detection, as most devices won't be able to produce adequate framerates"; use CameraX `STRATEGY_KEEP_ONLY_LATEST`.
  - Gradle dependency `com.google.mlkit:object-detection:17.0.2`. The page states "This API uses an unbundled library that must be downloaded before use".
  - The overview calls the model "intended for use in real-time applications, even on lower-end devices".
- **Industrial precedent for image-based self-triggering:**
  - "Some vision systems used with objects in continuous motion can operate without a trigger using a method called self-triggering, which typically operates by monitoring portions of captured images for a change in brightness or color that indicates the presence of an object" — [US 9092841, Method and apparatus for visual detection and inspection of objects](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9092841) (search-result summary)
  - e-con Systems' self-trigger mode defines "sensing and capturing zones within the image sensor's active pixel array" — [e-con Systems See3CAM self-trigger](https://www.e-consystems.com/blog/camera/technology/how-see3cam_37cugms-self-trigger-mode-ensures-seamless-automated-capture/) (search-result summary)
- YOLO nano latencies are in Q1: 0.6–3.5 ms model-only on the NPU, 10.7–15.8 ms app-level on NPU/GPU, SD 8 Elite Gen 5.

### Inferences
- **Detector options compared** (latency estimates for the SD 8 Elite Gen 5):

| Option | Training? | Works handheld? | Latency (est.) | Pros | Cons |
|---|---|---|---|---|---|
| **Colour/brightness threshold (HSV `inRange` or Otsu) + morphology + contours, 320×180, ROI band** | No | **Yes**, the background colour is known | ~1–4 ms (unmeasured) | Deterministic, instant, easy to explain; any part shape | Needs a plain contrasting background and steady light; touching parts merge (split with distance transform or area rules) |
| Background model (MOG2/KNN) + contours | No | **No**, camera motion puts the whole scene in the foreground | Higher than thresholding (per MOG2 cost reports) | Handles textured belts when the phone is fixed | Warm-up; ghosts from parts that stop; breaks when the phone moves |
| Frame differencing against a reference "empty" frame | No | Only after global-motion compensation | ~1–3 ms | Very simple | Same handheld problem; lighting drift |
| COCO YOLO11n / YOLO26n as a class-agnostic detector (take every box above a low confidence) | No | Yes | 10.7–15.8 ms app-level | Robust to background clutter | Misses parts unlike COCO classes; AGPL-3.0 licence; export and packaging work |
| ML Kit ODT (STREAM_MODE, single object, no classification) | No | Yes | "real-time" (no number) | Tracking IDs built in; class-agnostic | 30+ frame warm-up; max 5 objects; multi-object mode is slow; downloaded library conflicts with offline; no VisionCamera V5 plugin |
| 1-class YOLO fine-tuned on ~100 labelled frames | Yes, minutes to hours | Yes | 10.7–15.8 ms | Best robustness | Labelling time eats into the 10-hour window |

- **Recommendation for a 10-hour build:**
  - Primary: the "trigger zone + foreground mask" detector, computed by thresholding against a known plain background.
    - Calibrate the background colour or threshold once with an "empty scene" tap.
    - Run it only in an ROI band (e.g. ±25% of frame width around the trigger line) to save time and reduce false blobs.
    - Filter blobs by area and aspect ratio, and ignore blobs touching the frame edge (partial parts).
  - Optional fallback: YOLO26n/YOLO11n with class filtering disabled, for cluttered scenes.
  - The same blob mask gives a clean part crop and silhouette for the anomaly model, which helps PatchCore ignore the background.
- The ML Kit ODT warm-up of "30 or more frames" (about 1 s at 30 fps) is longer than a fast part spends in view (Q5 table). Treat it as unsuitable for the conveyor use case.

### Gaps
- No measured per-frame latency on Snapdragon for the OpenCV threshold+contour pipeline or for MOG2. The 1–4 ms figure is my estimate.
- Open-vocabulary detectors (YOLO-World, YOLOE) on the mobile NPU were not researched in this pass.
- It is ambiguous whether ML Kit ODT's model is bundled or downloaded. The fetched page text says "unbundled library that must be downloaded", but the artifact is a normal Gradle dependency. Verify in airplane mode on the device before relying on it.

---

## Q4. How should tracking and the single-decision logic work so each part is counted and judged exactly once (line crossing, multiple parts in frame), including per-part crop and anomaly scoring?

### Takeaway
Use a lightweight SORT-style tracker: constant-velocity prediction plus IoU or centroid-distance gating, then Hungarian or greedy matching. Add a **virtual photo-eye**: a line across the belt direction.
- A track is judged **once**, when its anchor point has been confirmed on the downstream side for N consecutive frames. These are the supervision `LineZone` semantics.
- The decision is stored on the track, so the red box and heat map keep following the defective piece until it leaves the frame.
- Anomaly scoring runs asynchronously on the best crop, through a FIFO queue.

### Cited Findings
- SORT is a "rudimentary combination of familiar techniques such as the Kalman Filter and Hungarian algorithm", runs at 260 Hz, and notes that "changing the detector can improve tracking by up to 18.9%" (detection quality dominates) — [arXiv 1602.00763](https://arxiv.org/abs/1602.00763)
- ByteTrack does "tracking by associating almost every detection box instead of only the high score ones". Low-score boxes rescue occluded or degraded objects. It reached "80.3 MOTA, 77.3 IDF1 and 63.1 HOTA" on MOT17 at 30 FPS on a V100 — [arXiv 2110.06864](https://arxiv.org/abs/2110.06864)
- **supervision `LineZone` counting semantics** ([supervision LineZone docs](https://supervision.roboflow.com/latest/detection/tools/line_zone/)):
  - It requires a `tracker_id`; without one it warns and skips counting.
  - `triggering_anchors` default to the four box corners and can be set to, e.g., the centre.
  - It keeps a per-tracker "confirmed crossing side", so a crossing counts only on a transition to a different confirmed side.
  - `minimum_crossing_threshold`: "Detection needs to be seen on the other side of the line for this many consecutive observations to be considered as having crossed the line".
  - `in_count` / `out_count` are kept per direction.
  - Stale tracking history is evicted after enough absent frames.
- **Industrial analogues:**
  - Reject tracking uses FIFO buffers indexed by encoder position rather than fixed delays, so it stays accurate "regardless of line speed variance" — [Elementary ML triggering guide](https://www.elementaryml.com/blog/the-complete-guide-to-machine-vision-triggering-for-high-speed-image-acquisition)
  - A photoelectric sensor detects the part's leading edge, the controller counts encoder pulses, and the camera and light fire when the count says the part is in position — [British Encoder: machine vision](https://encoder.co.uk/applications/by-industry/machine-vision/); [Quality Magazine: encoder input](https://www.qualitymag.com/articles/91113-encoder-input-improves-part-inspection) (search-result summaries)
- VisionCamera V5's AsyncRunner returns `false` and does not schedule work when busy — [VisionCamera docs](https://visioncamera.margelo.com/docs/guides/frame-processors-tips)
- EfficientAD (student–teacher plus autoencoder, lightweight feature extractor) reaches 2 ms latency and 600 img/s on a desktop GPU. It is a candidate if PatchCore kNN stays too slow — [arXiv 2303.14535](https://arxiv.org/abs/2303.14535)

### Inferences
- **Per-track state machine:**
  1. `TENTATIVE` (fewer than `min_hits`=2–3 matches)
  2. `TRACKED`
  3. `IN_ZONE` (the anchor is within the ROI band; crop candidates are buffered)
  4. `CROSSED` (anchor downstream for ≥`minimum_crossing_threshold`=2–3 frames), which enqueues one anomaly job
  5. `JUDGED_PASS` / `JUDGED_FAIL` (the decision is frozen)
  6. `EXITED` (after `max_age` missed frames or leaving the frame)
  
  Only tracks that were seen **upstream** of the line and then crossed are judged. A track first seen already downstream (e.g. an ID switch after the line) is never judged, which prevents double counting.
- **Which crop to score.** While the part is `IN_ZONE`, keep the best candidate crop. Prefer the one nearest the frame centre, which has least lens distortion, is fully inside the frame and has the highest sharpness (variance of Laplacian). Score that single crop at the crossing. Optionally score 2–3 crops and take the max or median for robustness, at 2–3× the cost.
- **Association at speed.** Per-frame displacement is v ÷ fps. At 1 m/s and 30 fps that is 33 mm, about 71 px at 0.47 mm/px (see Q5). This can exceed the part's own size, so consecutive boxes have **IoU = 0** and pure IoU matching fails. Instead:
  - predict each track's position with its velocity along the belt (Kalman or a simple α-β filter), as SORT does;
  - gate by centroid distance to the prediction (e.g. under 0.5 × the expected displacement plus the part size);
  - run at 60 fps for fast demos.
  
  With few parts in view (≤10), greedy nearest-neighbour matching is sufficient and trivially fast.
- **Multiple parts in frame.** Each part has its own track and its own queue entry. Anomaly jobs run FIFO. The overlay shows "judging…" (amber outline) until the result arrives, then green or red. If the queue grows past a threshold, show a "line too fast" warning; that is the equivalent of an industrial overrun alarm.
- **Anomaly scoring speed-ups** (engineering suggestions, not benchmarked):
  1. Run the ResNet18 embedding on the GPU (fast-tflite `android-gpu`) or NPU (LiteRT) with a fixed 224-px crop.
  2. Replace the JS kNN with a native or GPU kNN. Options: a TFLite graph computing ‖f‖² − 2f·Mᵀ + ‖M‖² against a coreset memory bank M, followed by a min-reduce; or a C++/Kotlin brute-force loop over a coreset of about 1–10k vectors.
  3. Shrink the coreset.
  4. Cache the threshold computed on "good" parts captured **live, in motion, with the same exposure preset**. Motion blur and rolling-shutter shear make live crops differ from still photos, which could otherwise cause false rejects.
- **Heat map on the defective piece.** Keep the PatchCore patch-score map (e.g. 28×28), colour-map it, and draw it with alpha inside the tracked box. The box moves with the track, so the heat map "rides" the part. The part's rotation is assumed small on a belt or slide.

### Gaps
- No published mobile latency for PatchCore kNN or EfficientAD on the Snapdragon 8 Elite (Gen 5).
- No published benchmark of SORT, ByteTrack or centroid trackers on Android. Cost is inferred as negligible from the desktop 260 Hz figure.

---

## Q5. How does motion blur limit shutter time, how do industrial systems trigger inspection, and what are typical conveyor speeds and parts per minute?

### Takeaway
Blur in pixels = speed × exposure time ÷ (field of view ÷ pixels). Machine-vision practice keeps blur at about 0.5–1 px, and "0.5 to 2 pixels is usually no longer visible to the eye".

For a phone about 30 cm above the parts (≈300 mm field of view across 640 analysis pixels ≈ 0.47 mm/px, an assumption):
- a 0.3 m/s hand slide needs an exposure of about **≤1.5 ms**;
- a 0.5 m/s belt (≈100 FPM) needs about **≤0.9 ms**.

A 1/60 s auto-exposure would smear those parts by 5–8 mm (10–18 px).

Industry freezes motion with microsecond exposures and strobes, and triggers from photo-eyes and encoders (hardware, deterministic). Packaging belts typically run 20–200 FPM (0.1–1 m/s) and bottle lines up to 400 FPM (~2 m/s). A current AI smart camera (Cognex In-Sight 3900) is reported at up to 5,000 parts/min.

### Cited Findings
#### Motion-blur math
- **Vision-Doctor formula and worked example** ([Vision-Doctor: exposure time of area scan cameras](https://www.vision-doctor.com/en/camera/exposure-time-area-scan-camera.html)):
  - Exposure time = pixel size (FOV ÷ pixels in the direction of motion) ÷ object speed.
  - Example: 120 mm FOV ÷ 2464 px = 0.0487 mm/px. At 600 mm/s, **81 µs** gives 1 px of blur.
  - "A blur of 0.5 to 2 pixels is usually no longer visible to the eye, but still has a negative effect on the accuracy of the measurement results".
- Blur in pixels = part velocity × exposure time × number of pixels ÷ FOV in the direction of motion — [1stVision](https://www.1stvision.com/machine-vision-solutions/2018/06/industrial-camera-exposure-time.html) (search-result summary)
- To avoid blur, "the object does not move more than 0.5 pixel during exposure time" — [VA Imaging: how to avoid motion blur](https://va-imaging.com/en-us/blogs/machine-vision-knowledge-center/how-to-avoid-motion-blur) (search-result summary)
- High-speed example: "if an object moves at 10 m/s and the acceptable blur tolerance is 0.125 mm, the maximum permissible exposure time is limited to 12.5 μs". Freezing motion combines a minimal exposure with a high-intensity strobe fired "precisely during the camera's exposure window" — [Elementary ML triggering guide](https://www.elementaryml.com/blog/the-complete-guide-to-machine-vision-triggering-for-high-speed-image-acquisition)
- Android can set exposure manually. With AE_MODE_OFF the app sets SENSOR_EXPOSURE_TIME (ns), SENSOR_SENSITIVITY and SENSOR_FRAME_DURATION — [AOSP 3A modes](https://source.android.com/docs/core/camera/camera3_3Amodes)

#### Rolling shutter (phone sensors)
- "Nearly all CMOS sensors in smartphones and consumer cameras today use rolling shutter". The top and bottom rows differ in capture time by "several to tens of milliseconds", which skews or smears fast-moving objects. Readout time grows with resolution mode — [wolfcrow: rolling shutter guide](https://wolfcrow.com/the-essential-guide-to-rolling-shutter-what-you-need-to-know/); [Wikipedia: Rolling shutter](https://en.wikipedia.org/wiki/Rolling_shutter) (search-result summaries)

#### How industrial systems trigger
- **Trigger types and timing** ([Elementary ML triggering guide](https://www.elementaryml.com/blog/the-complete-guide-to-machine-vision-triggering-for-high-speed-image-acquisition)):
  - "External (Hardware) Triggering is the industry standard for production environments".
  - Software triggers suffer "unpredictable delays caused by operating system processes, kernel overhead, and software stack latency".
  - Photoelectric sensors give basic presence detection.
  - Encoders give "distance-based acquisition (independent of line speed variations)".
  - Trigger *latency* is a fixed delay that can be compensated; *jitter* is unpredictable variation.
  - TTL trigger cables are reliable under 15 m; RS-422/644 up to about 60 m.
- **Classic sequence:** a photo-eye detects the leading edge, encoder pulses are counted, and the camera and strobe fire at the target count. Reject gates are timed from encoder travel — [British Encoder](https://encoder.co.uk/applications/by-industry/machine-vision/); [Encoder Products: machine vision](https://www.encoder.com/machine-vision); [Quality Magazine](https://www.qualitymag.com/articles/91113-encoder-input-improves-part-inspection) (search-result summaries)
- **Sensorless "self-triggering"** watches image regions for brightness or colour change — [US 9092841](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9092841); [e-con Systems](https://www.e-consystems.com/blog/camera/technology/how-see3cam_37cugms-self-trigger-mode-ensures-seamless-automated-capture/) (search-result summaries)

#### Line speeds and parts per minute
- **Packaging conveyor speeds** ([PackagingConveyor.com engineering guide](https://packagingconveyor.com/resources/conveyor-design-guide/conveyor-engineering/)):
  - "Packaging conveyors typically run between 20 FPM and 200 FPM".
  - Manual pack stations: 10–30 FPM.
  - Automated case conveying: 60–150 FPM.
  - High-speed bottle and container lines: 100–400 FPM.
  - Throughput formula: **Products/min = V ÷ P**, with V = belt speed and P = pitch (product length + gap).
- **Cognex In-Sight 3900** is reported to inspect "up to 5,000 parts per minute", built on Qualcomm Dragonwing, at "four times the speed of previous Cognex vision systems" — [Control.com news](https://control.com/news/cognex-launches-a-high-speed-ai-powered-machine-vision-system/); [Cognex In-Sight 3900 page](https://www.cognex.com/en/products/2d-machine-vision-systems/in-sight-3900) (search-result summaries; the Cognex press release returned HTTP 403). It is a useful pitch line: industrial smart cameras now run on Qualcomm silicon too.
- **Blog example:** caps 1.1 in in diameter at 1,600 ppm need about 1,763 in/min of feed conveyor (≈0.75 m/s) — [Industrial Monitor Direct KB](https://industrialmonitordirect.com/blogs/knowledgebase/cognex-keyence-vision-system-1600-part-identification-guide) (search-result summary; vendor blog)

### Inferences
- **Worked blur table for the phone.**
  - Assumption: 300 mm FOV along the motion (phone about 25–35 cm above the table, main wide lens; not measured).
  - Analysis frame 640 px wide → **0.47 mm/px**. Crop source 1920 px → **0.156 mm/px**.

| Speed | Exposure for 1 px blur @640 px | Exposure for 1 px blur @1920 px | Blur at 1/60 s (16.7 ms) | Frames in view @30 fps | Displacement per frame @30 fps |
|---|---|---|---|---|---|
| 0.1 m/s (≈20 FPM) | 4.7 ms | 1.6 ms | 1.7 mm ≈ 3.5 px | ~90 | 3.3 mm ≈ 7 px |
| 0.3 m/s (hand slide, ≈60 FPM) | 1.6 ms | 0.52 ms | 5 mm ≈ 11 px | ~30 | 10 mm ≈ 21 px |
| 0.5 m/s (≈100 FPM) | 0.94 ms | 0.31 ms | 8.3 mm ≈ 18 px | ~18 | 17 mm ≈ 35 px |
| 1.0 m/s (≈200 FPM) | 0.47 ms | 0.16 ms | 16.7 mm ≈ 35 px | ~9 | 33 mm ≈ 71 px |
| 2.0 m/s (≈400 FPM) | 0.23 ms | 0.08 ms | 33 mm ≈ 71 px | ~4.5 | 67 mm ≈ 142 px |

- **Practical phone settings:**
  - Exposure: 1/1000–1/2000 s (0.5–1 ms), which allows about 1–2 px of blur at analysis resolution up to about 0.5 m/s.
  - Light: that exposure admits about 4 stops less light than 1/60 s. Make it up with ISO and the **phone torch**, plus bright room light.
  - Enrol and calibrate "good" parts at the *same* settings, so noise and blur are part of the "normal" model.
  - Speed limit: around 1 m/s the anomaly crops (at 1920 px) blur by about 3–6 px at 0.5–1 ms, so fine scratches will be lost. Present the demo honestly as a ≤0.5 m/s small-part line.
- **Rolling-shutter shear.** Shear ≈ v × readout time. For example, 0.5 m/s × 10–20 ms = 5–10 mm between the top and bottom of the frame (the readout range is assumed; iQOO 15 data not found). A part shears in proportion to its own height, so small parts distort less. Keep the part's travel consistent in one image direction, and enrol "good" parts in motion.
- **Industrial ↔ phone mapping:**
  - photo-eye ↔ virtual trigger line on the tracked centroid;
  - encoder ↔ tracker velocity estimate (pixels/frame → mm/s once the scale is known);
  - strobe ↔ torch plus short exposure;
  - reject gate / shift register ↔ per-track decision plus a beep at crossing time.
  
  Describe this in the pitch as "image-based self-triggering", an established industrial concept.
- **Demo throughput.** At 0.3 m/s with a 10–15 cm pitch, V ÷ P = 18 m/min ÷ 0.1–0.15 m = **120–180 parts/min**, i.e. 2–3 parts/s. That leaves 330–500 ms of per-part scoring budget if scoring runs serially. Today's 140–440 ms fits only marginally, which is another reason to optimize the kNN.
- **Industrial scale for context.** 1,600 ppm leaves 37.5 ms per part (60,000 ÷ 1,600); 5,000 ppm leaves 12 ms per part. A phone doing about 50–100 ms per-part scoring maps to roughly 600–1,200 ppm if scoring runs serially.

### Gaps
- Unknown for the iQOO 15: minimum exposure time, ISO range and sensor readout time.
- No authoritative cross-industry survey of typical small-part ppm was found. I have only a vendor-blog example (1,600 ppm) and a vendor maximum (5,000 ppm).
- The 300 mm FOV / 30 cm height is an assumption. Measure the real mm/px on the device with a ruler in frame.

---

## Q6. What operator UI/UX should the app use (marking the defective piece, andon-style alerts, sound, vibration, counters, SPC, logging)?

### Takeaway
Follow high-performance HMI (ISA-101-style) practice:
- a neutral or grey UI where **colour appears only for abnormal states**: red = reject/fault, amber = warning, green = running;
- a big reject flash on the tracked part;
- a short, distinctive tone plus vibration **only on rejects**, with an escalation tone for consecutive rejects;
- rolling counters (parts/min, reject %) and a p-chart of the reject fraction;
- an append-only per-part log.

Draw boxes on (or extrapolated to) the analyzed frame so the mark stays on the moving defective piece.

### Cited Findings
- ISA-101 high-performance HMI guidance (summaries of the standard):
  - "The background color should be light gray"; a gray base (60–70% gray) creates a neutral field.
  - "Foreground colors should be kept to a minimum and used sparingly to indicate abnormal situations".
  - Yellow/orange mean warnings or deviations; "Red is reserved exclusively for critical conditions".
  - Sources: [Industrial Monitor Direct: ISA-101 colour strategy](https://industrialmonitordirect.com/blogs/knowledgebase/isa-101-high-performance-hmi-design-principles-color-strategy); [Control.com: Going Gray](https://control.com/technical-articles/going-gray/); [Emerson white paper](https://www.emersonautomationexperts.com/2023/industrial-internet-things/up-your-productivity-and-safety-with-high-performance-hmi-design-white-paper/) (search-result summaries)
- Andon / stack-light conventions:
  - green = running normally;
  - yellow/amber = action needed / warning;
  - red = stop, error, or "a call for help";
  - blue = often a quality issue or maintenance;
  - white = changeover or custom.
  - Sources: [Stack-Light.com: andon color codes](https://stack-light.com/blogs/andons-stack-lights/andon-color-codes-2); [AllAboutLean: stack lights](https://www.allaboutlean.com/stack-lights/); [Wolf Automation: andon lights](https://www.wolfautomation.com/products/andon-lights/) (search-result summaries)
- **Android `ToneGenerator`:**
  - Constructor `ToneGenerator(streamType, volume)`, with volume from MIN_VOLUME (0) to MAX_VOLUME (100).
  - `startTone(toneType, durationMs)` plays for the lesser of `durationMs` and the tone's own length.
  - Useful tones:
    - `TONE_PROP_BEEP`: "general beep: 400Hz+1200Hz, 35ms ON"
    - `TONE_PROP_BEEP2`: "general double beep: twice 400Hz+1200Hz, 35ms ON, 200ms OFF, 35ms ON"
    - `TONE_PROP_ACK`: "1200Hz, 100ms ON, 100ms OFF 2 bursts"
    - `TONE_PROP_NACK`: "300Hz+400Hz+500Hz, 400ms ON"
    - `TONE_CDMA_ALERT_CALL_GUARD`: "{1319Hz 125ms ON, 125ms OFF} 3 times"
    - `TONE_SUP_ERROR`: "950Hz+1400Hz+1800Hz, 330ms ON, 1s OFF…"
    - `TONE_CDMA_ABBR_ALERT`: "1150Hz+770Hz 400ms ON"
  - Sources: [Android ToneGenerator reference (mirror)](https://emanual.github.io/Android-docs/reference/android/media/ToneGenerator.html); official page: [developer.android.com ToneGenerator](https://developer.android.com/reference/android/media/ToneGenerator) (the official page did not render for my fetcher; the constants match the mirror)
- **React Native `Vibration`** ([React Native Vibration docs](https://reactnative.dev/docs/vibration)):
  - `Vibration.vibrate(pattern, repeat)`. On Android, a number is a duration in ms (default 400 ms). In an array, "odd indices" are vibration durations and even indices are separation times.
  - Needs `android.permission.VIBRATE`.
  - `cancel()` stops a repeating pattern.
- **NIST p-chart for proportion defective** ([NIST/SEMATECH e-Handbook 6.3.3.2](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc332.htm)):
  - UCL = p + 3√(p(1−p)/n); centre line = p; LCL = p − 3√(p(1−p)/n). Use p̄ when p is unknown.
  - The handbook example uses subgroups of n = 50.
  - Assumptions: a stable process, independent units, and a conforming/nonconforming classification.
- VisionCamera V5 frame processors can update Reanimated SharedValues "for instant feedback" — [Margelo blog](https://margelo.com/blog/whats-new-in-visioncamera-v5). Skia can draw on the frame itself — [VisionCamera Skia FP](https://react-native-vision-camera.com/docs/guides/skia-frame-processors)
- The app→display path adds about 33 ms, and the whole glass-to-glass path about 80 ± 25 ms, on a Pixel 2 — [inovex](https://www.inovex.de/de/blog/the-glass-to-glass-latency-on-android/)

### Inferences
- **Overlay design:**
  - Full-screen camera.
  - A thin trigger line; its colour stays neutral until a part crosses.
  - Per-part box that changes with state:
    - thin grey while tracking;
    - amber dashed while "judging";
    - green thin outline for PASS, fading after about 300 ms;
    - **thick red box + translucent red fill + heat map + "REJECT #042 score 0.83"** for FAIL, kept until the part leaves the frame.
  - Screen-edge **andon border flash** in red for about 500 ms on each reject.
  - A persistent top status bar coloured like a stack light:
    - green = running;
    - amber = recent reject rate above the p-chart UCL, or the queue is lagging;
    - red = N consecutive rejects (stop-the-line).
- **Keep the mark on the part.** The overlay shows results from a frame that is about 24 ms or more old, plus processing time. At 0.5 m/s, 50 ms of lag is a 25 mm offset, so a box would visibly trail the part. Either draw on the analyzed frame with a Skia frame processor, so frame and box are always consistent, or **extrapolate** the box using track velocity × measured latency.
- **Sound design:**
  - Silence on PASS, to avoid alarm fatigue in line with the "colour/alerts only for abnormal" principle.
  - On FAIL: `TONE_CDMA_ALERT_CALL_GUARD` (1319 Hz ×3 bursts, attention-grabbing) or `TONE_PROP_BEEP2` (short double beep for high rates), at volume 80–100.
  - Stop-the-line (e.g. 3 consecutive rejects): `TONE_SUP_ERROR` plus a long vibration.
  - Vibration pattern for FAIL: `[0, 120, 80, 120]`, i.e. wait 0, vibrate 120, pause 80, vibrate 120 ms (odd indices vibrate).
  - In RN, ToneGenerator needs a tiny native (Expo) module, or use a pre-loaded short WAV. Pre-create the ToneGenerator once; don't create it per event.
- **Counters:**
  - Total, pass, reject.
  - **Parts/min** from crossing timestamps in a sliding 60 s window.
  - **Reject rate %** overall and over the last 50 parts.
  - Per-part **decision latency** (trigger → verdict ms) and **FPS**. This "latency HUD" is persuasive for judges.
- **SPC:** a p-chart of reject fraction in subgroups of n = 20–50 parts (NIST form), drawn as a small sparkline with UCL/LCL. It turns amber when a point exceeds the UCL, the "process drifted" signal.
- **Logging** (append-only JSONL/CSV plus thumbnails): one record per judged part with
  - ISO timestamp, track ID, direction, frame index;
  - bounding box, score, threshold, verdict;
  - per-stage ms (detect, track, crop, embed, kNN);
  - crop JPEG path, and the heat-map PNG for rejects.
  
  Add a one-tap export via the Android share sheet. The log also backs the "reject gallery" screen for after the demo.

### Gaps
- No access to the ISA-101 standard text (paywalled). Colour rules come from secondary summaries.
- No source on ToneGenerator start latency or audibility in noisy halls. Test on the device.

---

## Q7. How can the live system be demoed convincingly with a phone only, and what are the pitfalls (moiré, screen-refresh flicker, auto-exposure)?

### Takeaway
The most convincing phone-only demo is **live camera over real small parts** on a plain contrasting sheet. Either slide the parts across by hand, or sweep the phone steadily over a row of parts so they "flow" through the frame. Use the torch, locked AE/AF/AWB and a short exposure.

Pointing the phone at a laptop playing a conveyor video gives an "infinite conveyor", but it brings three problems:
- moiré;
- refresh/PWM banding, which needs an exposure of about 1/60 s, directly conflicting with motion freezing;
- auto-exposure hunting.

A clearly labelled **"REPLAY" mode** that feeds a video file into the same pipeline is the deterministic backup, not the headline.

### Cited Findings
- **Moiré and screen flicker** ([Divoom: reduce moiré](https://divoom.com/blogs/setup-ideas/reduce-moire-filming-pixel-led-screen); [RemoveMoire blog](https://removemoire.com/blog/prevent-moire-photos-screens); [Wikipedia: Flicker (screen)](https://en.wikipedia.org/wiki/Flicker_(screen)) — search-result summaries, vendor and blog sources):
  - Moiré is spatial interference between the display's pixel grid and the camera sensor's grid, independent of refresh rate.
  - Phone mitigations: "Switch to your 2x, 3x, or 5x lens and stand further back"; tilt the camera slightly; pull focus slightly off the screen.
  - Flicker/banding: at very fast shutter speeds (e.g. 1/4000 s) the camera captures the screen mid-refresh or mid-PWM cycle, producing dark bands. Use 1/30–1/60 s, or match the refresh or mains frequency (1/50, 1/60, 1/100).
- Android exposes anti-banding modes (50 Hz / 60 Hz / AUTO) and AE/AF/AWB locks — [AOSP 3A modes](https://source.android.com/docs/core/camera/camera3_3Amodes)
- Rolling shutter skews fast-moving content by "several to tens of milliseconds" of top-to-bottom capture difference — [wolfcrow](https://wolfcrow.com/the-essential-guide-to-rolling-shutter-what-you-need-to-know/) (search-result summary)
- Background subtraction is framed for static cameras — [OpenCV-Python tutorial](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html)
- VisionCamera Low Light Boost lengthens exposure and drops the frame rate — [VisionCamera Low Light Boost](https://visioncamera.margelo.com/docs/low-light-boost)
- **Public footage and datasets:**
  - **IPAD** (Industrial Process Anomaly Detection) is a *video* dataset: 16 industrial devices, more than 6 hours of synthetic and real footage, including conveyor belts, lift tables, cutting and drilling machines, grippers and cranes. Its anomalies are *process* anomalies with periodic structure, not part defects — [arXiv 2404.15033](https://arxiv.org/abs/2404.15033); [project page](https://ljf1113.github.io/IPAD_VAD/)
  - **MVTec AD** is CC BY-NC-SA 4.0, "not allowed … for commercial purposes" — [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) (search-result summary)
  - **VisA** (9,621 normal + 1,200 anomalous images) is reported as CC BY 4.0 (search-result summary; licence not verified at the official repository).
  - Royalty-free conveyor clips on Pexels, e.g. [Plastic Bottle Factory](https://www.pexels.com/video/plastic-bottle-factory-4177966/), [Factory Conveyor Belt](https://www.pexels.com/video/factory-conveyor-belt-4473250/), [Food on Conveyor Belt](https://www.pexels.com/video/food-on-conveyor-belt-in-factory-9936784/), [Industrial Conveyor System](https://www.pexels.com/video/industrial-conveyor-system-in-factory-35069365/), and the search pages for [conveyor](https://www.pexels.com/search/videos/conveyor/) and [production line](https://www.pexels.com/search/videos/production%20line/).
- React Native Skia's `useVideo` loads video frames from local files or the app bundle for drawing, a possible frame source for a replay mode — [React Native Skia: Video](https://shopify.github.io/react-native-skia/docs/video/) (search-result summary)

### Inferences
- **Demo A, recommended headline: "live hand-fed line".**
  - Props: an A3 sheet in a colour contrasting with the parts (black for shiny metal or white parts, white for dark parts).
  - Parts: 10–20 identical small parts such as washers, bottle caps, M8 nuts, coins, or chocolates/biscuits (edible defects are easy to make). Deliberately damage 3–5 of them with a scratch, a chip, or a marker dot.
  - Setup: hold the phone about 25–35 cm above the sheet, or brace your elbows on the table.
  - Camera: torch ON, "Real parts" preset (0.5–1 ms, AE/AF/AWB locked, fixed fps).
  - Action: slide the parts one after another across the trigger line at walking-hand speed (~0.2–0.4 m/s).
  - Show: every part gets a box and ID; rejects flash red with a beep and vibration; the counter and parts/min update; the log gallery at the end.
  - Variant: **sweep the phone** over a pre-laid row of parts. Relative motion is identical, and the part-flow direction is controllable. This only works with the threshold/contour detector, not background subtraction.
  - Enrol "good" parts live, in the same preset, just before the demo, e.g. by sliding 10 good parts in "teach" mode.
- **Demo B, "infinite conveyor": phone aimed at a laptop playing a conveyor video.**
  - Settings: AE converged then **locked**, exposure 1/60 s (60 Hz laptop panels) or 1/100–1/120 s; zoom 2×; stand back; tilt slightly off-axis; max laptop brightness in a dimmer room; play the clip at 0.5× so there is less baked-in blur.
  - Pitfalls:
    - moiré patterns can be mistaken for "texture anomalies" by PatchCore, so enrol the "good" memory bank *through the screen* too;
    - banding still appears at short exposures;
    - screen glare;
    - 30 fps capture of a 60 Hz display duplicates or tears frames;
    - the stock clip may not show distinct, top-down, separable parts.
  - A good source clip is a *synthetic* conveyor video built from VisA or MVTec images scrolled over a plain background at a known speed. It has known ground truth (which items are defective), so the app's hit rate can be shown on stage. Mind MVTec's non-commercial licence; VisA is reportedly CC BY.
- **Demo C, fallback: "REPLAY (video file as live feed)".**
  - Decode an MP4 (native MediaCodec, or pre-extracted frames) and push frames through *exactly the same* detect→track→judge code.
  - It is deterministic and immune to moiré, flicker and hand shake.
  - Show an on-screen **"REPLAY" badge** (vs "LIVE CAMERA"). The judges require a live feed, so replay must be presented honestly as a regression/backup mode.
- **Other pitfalls:**
  - Handheld shake moves every part in image space. Tracking handles this, but a static trigger line may be crossed back and forth. Use `minimum_crossing_threshold` of 2–3 frames and direction-aware counting.
  - Thermal throttling over a long demo: warm up for 1–2 minutes, keep resolution and fps modest, and show the FPS HUD.
  - Mains-lit rooms plus short exposure can band. The torch is a steady source; use it.
  - Auto-exposure changes as dark or bright parts enter will shift anomaly scores. Lock 3A.
- **Judge-facing story:** "Industrial cameras use a photo-eye and an encoder to fire the camera. Kaizen Eye uses an image-based virtual photo-eye and a tracker that estimates belt speed, so it needs no hardware. Each part is judged exactly once at the line, and the reject stays marked while the part travels." Back it with the latency HUD and the "frames seen per part" counter.

### Gaps
- No measured data found on inspection accuracy or false-reject rate when filming a monitor versus real parts.
- Not checked: the IPAD dataset licence, and whether any clip shows discrete parts with part-level defects (it targets process anomalies).
- The specific Pexels clips were not reviewed for a top-down view, part separation or length; the Pexels licence text was not fetched.
- The frame-rate and CPU cost of Skia `useVideo`, or of a MediaCodec→analysis replay path in RN, was not benchmarked.
