# Platform feasibility: a 10-hour, offline, NPU-accelerated Android live-inspection app with an on-device LLM/VLM (as of Sept 2026)

Scope: choose between (a) the current Expo/React Native app plus react-native-vision-camera frame processing and react-native-fast-tflite, (b) native Android in Kotlin, (c) Flutter, and (d) a hybrid (a native camera and inference module inside the RN app). Criteria: time to a working demo, camera throughput, NPU access, maintained samples, Windows build risk, bundling 1-3 GB of LLM weights, and LLM/VLM SDK integration. Target phone: iQOO 15 (Snapdragon 8 Elite Gen 5 / SM8850).

Local context used (not web sources):
- Project README: `D:\Projects\Kaizen_Eye\README.md`.
- Backbone: ResNet18 → `backbone_r18_320.tflite`. Input is float32 `[1,320,320,3]`. Output is `[1,40,40,128]`, which is 1,600 patch vectors.
- Scoring: PatchCore kNN in TypeScript, checked against `testdata/golden_core.json`.
- int8 quantisation kept AUROC within noise (0.898 vs 0.905) and ran 3.2x faster on CPU.
- Windows fix already in the README: `mklink /J D:\AndroidSdk` works around a user name that contains a space.

---

## 1. Option (a): keep Expo/RN and add VisionCamera frame processing + fast-tflite. Status, speed, compatibility, NPU reach

### Takeaway
VisionCamera V5 (Nitro, April 2026) works and fast-tflite supports it. But on Android, fast-tflite offers only the `android-gpu` and `nnapi` delegates. There is no QNN delegate, and NNAPI is deprecated since Android 15. So the Hexagon NPU cannot be reached reliably without writing native code. V5 is also new: there are Android HardwareBuffer/resizer crash reports, and it needs react-native-worklets 0.13 or later for RN 0.86. The GPU path is realistic. The NPU path and a fast JS kNN scorer are not.

### Cited Findings
- **Release and architecture.**
  - VisionCamera V5 was published April 16, 2026. It was "fully rewritten to Nitro Modules".
  - Nitro calls are "~15x faster than with Turbo-Modules, or ~60x faster than Expo-Modules".
  - Native frame-processor plugins are now Nitro Modules.
  - It uses `react-native-worklets` instead of `react-native-worklets-core`.
  - Source: [Margelo blog](https://margelo.com/blog/whats-new-in-visioncamera-v5)
- **Packages and install.** V5 is split into separate packages: core, `-worklets`, `-resizer` (GPU-accelerated, "~5x performance speedup" vs the CPU/SIMD alternative), `-skia`, `-barcode-scanner` and `-location`. Install needs `react-native-nitro-modules` and `react-native-nitro-image`. — [Margelo blog](https://margelo.com/blog/whats-new-in-visioncamera-v5)
- **Frame output API.**
  - In V5, `useFrameProcessor` became `useFrameOutput`. It needs `react-native-vision-camera-worklets` plus `react-native-worklets`, and runs `onFrame` on a parallel worklet runtime.
  - Frames must be `frame.dispose()`d. Otherwise the buffer pool fills, "the pipeline stalls and subsequent frames will be dropped".
  - Pixel formats are `yuv` (about 2.6x less bandwidth than RGB), `rgb` (conversion overhead) and `native`.
  - Source: [VisionCamera docs – Frame Output](https://visioncamera.margelo.com/docs/guides/frame-processors)
- **No thread hop.** Frame processors run synchronously on the camera thread with no thread hop or serialisation. Without worklets, a 4K frame would be "~33MB per Frame, or ~1GB/s". SWM calls worklets "still young". No fps or latency numbers are given. — [Software Mansion blog, Jan 29 2026](https://swmansion.com/blog/behind-the-scenes-of-react-native-multithreading-vision-camera-v5-x-react-native-worklets-a102c37b32ae/)
- **Worklets and RN 0.86.**
  - react-native-worklets 0.13.x and 0.14.x support RN 0.86 and 0.87. 0.12.x supports RN 0.84 to 0.87.
  - Only the New Architecture (Fabric) is supported.
  - Source: [Worklets compatibility table](https://docs.swmansion.com/react-native-worklets/docs/guides/compatibility/)
- **Expo.** Expo SDK 55 and later always run on the New Architecture. — [Expo docs (search snippet)](https://docs.expo.dev/guides/new-architecture/)
- **Known Android issues in V5:**
  - With `react-native-vision-camera-resizer`, apps "get stuck and crash after some time", with repeated "A resource failed to call HardwareBuffer.close" warnings and SIGABRT. Reported on Android 16, RN 0.83.4, VC 5.0.1. `dropFramesWhileBusy` was already on. The issue is marked a duplicate of a Reanimated issue (#9317), and no fix is confirmed. — [VC issue #3824](https://github.com/mrousavy/react-native-vision-camera/issues/3824)
  - VC 5.0.11 (May 27, 2026) fixed "closing HardwareBuffer Java references" and "ByteBuffer reading from stale offsets". VC 5.1.0 (July 1, 2026) reports YUV range on Android. — [VC releases](https://github.com/mrousavy/react-native-vision-camera/releases) / [v5.1.0](https://github.com/mrousavy/react-native-vision-camera/releases/tag/v5.1.0) (search snippet)
  - VC 5.0.9 breaks react-native-video on Android because of an androidx.media3 version clash. — [react-native-video #4900](https://github.com/TheWidlarzGroup/react-native-video/issues/4900)
  - A search snippet reported 5.2.3 as the latest version. — [npm](https://www.npmjs.com/package/react-native-vision-camera) (not independently verified)
- **fast-tflite on Android.**
  - Delegates are exactly `'android-gpu'` and `'nnapi'`. The README says: "NNAPI is deprecated on Android 15. GPU delegate is preferred."
  - The Expo plugin option is `enableAndroidGpuLibraries`.
  - The V5 example uses `useResizer({width, height, channelOrder:'rgb', dataType:'uint8'})` and `resizer.resize(frame)`.
  - No fps or ms figures are published.
  - Source: [fast-tflite README](https://raw.githubusercontent.com/mrousavy/react-native-fast-tflite/main/README.md)
- **fast-tflite features.** It is built on Nitro with "zero-copy ArrayBuffers". With VC V5, models no longer need `NitroModules.box()` because worklets can access HybridObjects directly. — [fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite)
- **NNAPI status.** The NNAPI NDK API is deprecated starting in Android 15, and Google points to LiteRT with its GPU delegate. The NNAPI HAL and vendor drivers "aren't affected" by the deprecation.
  - [NNAPI migration guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)
  - [AOSP NNAPI drivers](https://source.android.com/docs/core/interaction/neural-networks)
  - Both via search snippets.
- **Vendor route to the NPU.** Qualcomm's route to the NPU from LiteRT is its own QNN LiteRT delegate on Maven, not NNAPI. — [Google: Qualcomm NPU with LiteRT](https://developers.google.com/edge/litert/android/npu/qualcomm)
- **Current app** (local README):
  - fast-tflite 3.0.1 (LiteRT 1.4.0) and expo-camera, taking photos only.
  - The backbone output is 1,600 × 128-d patches per frame.
  - Enrolment uses ~20 frames and a coreset ratio of 0.05 with minK 256.
  - Source: `D:\Projects\Kaizen_Eye\README.md`

### Inferences
- **The kNN scorer must leave JS.** With 20 enrolment frames × 1,600 patches, N = 32,000 and the bank k = max(256, 0.05·N) = 1,600. One live frame then needs about 1,600 × 1,600 × 128 ≈ 3.3×10^8 multiply-adds, plus the whole-image check. JS in a Hermes worklet has no SIMD, so this is likely at least 1 s per frame, which is not real time. The scorer would have to become a native Nitro HybridObject in C++ or Kotlin, or a second TFLite graph. That adds NDK/CMake work on Windows, which is where this team has already been hit.
- **NPU through fast-tflite would mean forking it.** You would have to link `qnn-litert-delegate` into fast-tflite's native code, or rely on NNAPI. Whether Qualcomm still ships a capable NNAPI driver for SM8850 is unverified (see Gaps). Realistically, option (a) gives GPU (Adreno 840) acceleration, not the NPU.
- **Fixed cost of adopting VisionCamera V5 on RN 0.86:** VC5, nitro-modules, nitro-image, worklets 0.13 or 0.14, vision-camera-worklets and vision-camera-resizer. That is six or more native packages, each compiling C++ with CMake. On Windows that multiplies the path and toolchain risk (see section 8).

### Gaps
- No published fps or latency for VisionCamera V5 plus fast-tflite running a model on every frame, on any device.
- Could not confirm whether Snapdragon 8 Elite Gen 5 phones (OriginOS 6) ship an NNAPI driver that delegates to Hexagon, or just fall back to the CPU reference.
- Could not confirm whether VisionCamera V5 is officially tested with RN 0.86 / Expo SDK 57. The worklets support RN 0.86, but VC's own peer range was not checked.
- Found no fast-tflite issue or PR adding a QNN delegate.

---

## 2. Option (b): native Android. CameraX ImageAnalysis, LiteRT 2.x (CompiledModel/NPU), QNN delegate, ONNX Runtime QNN EP, OpenCV, Compose overlay

### Takeaway
Native Kotlin has the most direct and best-documented path to the NPU:
- **Official Maven AARs** for LiteRT, the QNN LiteRT delegate and QNN runtime, ONNX Runtime QNN, OpenCV and LiteRT-LM.
- **Explicit SM8850 support** in LiteRT's Qualcomm backend.
- **CameraX ImageAnalysis** delivers RGBA_8888 and drops stale frames for you.

A pure-Kotlin app that only consumes prebuilt AARs compiles no C++, so the NDK-on-Windows path problem disappears. The main extra cost is rebuilding the UI and porting the TypeScript scoring core to Kotlin. The golden vectors already exist.

### Cited Findings
- **CameraX ImageAnalysis.**
  - `STRATEGY_KEEP_ONLY_LATEST` is the default and non-blocking; newer frames overwrite older ones. `STRATEGY_BLOCK_PRODUCER` blocks all bound use cases when its queue is full.
  - `setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)` makes CameraX convert YUV to RGBA internally into plane 0.
  - Frame budget: under 33 ms at 30 fps and under 16 ms at 60 fps.
  - You must call `ImageProxy.close()`.
  - Source: [developer.android.com – Image analysis](https://developer.android.com/media/camera/camerax/analyze)
- **LiteRT Qualcomm backend.**
  - It supports QNN through the `CompiledModel` API "for both AOT and on-device compilation".
  - Listed SoCs: SM8850 (8 Elite Gen 5), SM8750, SM8650, SM8550, SM8475 and SM8450.
  - Building LiteRT itself from source needs Ubuntu 22.04 and Bazel. This matters only if you compile LiteRT, not if you use the AAR.
  - Source: [LiteRT Qualcomm NPU](https://developers.google.com/edge/litert/next/qualcomm)
- **LiteRT NPU guide.**
  - Dependency: `com.google.ai.edge.litert:litert`.
  - Kotlin: `CompiledModel.create(context.assets, "model.tflite", CompiledModel.Options(Accelerator.NPU, Accelerator.GPU))`. It falls back automatically ("if NPU is unavailable, inference will fallback to GPU").
  - Requires minimum API 31 and arm64-v8a.
  - JIT suits small models; AOT "significantly reduces initialization costs". JIT output can be cached via `CompilerCacheDir`.
  - Zero-copy AHardwareBuffer input is available through `TensorBuffer::CreateFromAhwb` (C++).
  - Production delivery is through Google Play for On-device AI and Play Feature Delivery. Runtime libraries come as `litert_npu_runtime_libraries(_jit).zip` from GitHub releases.
  - The guide does not describe sideloading.
  - Source: [LiteRT NPU guide](https://developers.google.com/edge/litert/next/npu)
- **Google NPU benchmarks (Nov 24, 2025).**
  - Up to 100x faster than CPU and 10x faster than GPU.
  - "56 models run in under 5ms with the NPU" on Snapdragon 8 Elite Gen 5. 64 of 72 canonical models were fully delegated.
  - FastVLM-0.5B: TTFT 0.12 s, more than 11,000 tok/s prefill, more than 100 tok/s decode.
  - PODAI and AI Packs are the distribution path.
  - Source: [Google Developers Blog](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)
- **LiteRT releases.**
  - v2.2.0 "Aligned SoC coverage with QAIRT SDK 2.47" and lets you specify the Qualcomm SoC by name or number.
  - v2.1.0rc1 "Added support for NPU JIT mode on Qualcomm and MediaTek".
  - GitHub shows v2.2.0 as "Aug 13" with the year omitted, which implies the current year. Treat as Aug 2026 (unverified).
  - Source: [LiteRT releases](https://github.com/google-ai-edge/LiteRT/releases)
- **Classic LiteRT Interpreter + QNN delegate.**
  - Dependencies: `com.qualcomm.qti:qnn-runtime:2.34.0` and `com.qualcomm.qti:qnn-litert-delegate:2.34.0` ("should be updated accordingly").
  - The HTP backend loads its skel libs from `nativeLibraryDir`, which means the libraries ship inside the APK.
  - Source: [Google – Qualcomm NPUs with LiteRT](https://developers.google.com/edge/litert/android/npu/qualcomm)
  - The latest `qnn-litert-delegate` reported is 2.46.0. — [mvnrepository (search snippet)](https://mvnrepository.com/artifact/com.qualcomm.qti/qnn-litert-delegate)
- **QNN delegate practice (Edge Impulse).**
  - Put `libQnnTFLiteDelegate.so`, `libQnnHtp.so`, `libQnnHtpV**.so`, `libQnnHtpV**Skel.so`, `libQnnSystem.so` and `libQnnIr.so` in `jniLibs/arm64-v8a`.
  - Set `android:extractNativeLibs="true"` and `ADSP_LIBRARY_PATH=<nativeLibDir>:/dsp`.
  - "INT8 quantized is required for HTP acceleration".
  - On QRB6490 inference went from 5,748 µs to 527 µs (10.9x).
  - Source: [Edge Impulse QNN on Android](https://docs.edgeimpulse.com/tutorials/topics/android/qnn-acceleration)
- **Open NPU issues.**
  - LiteRT issue #6059 (SM8750): "No compiler plugin found" and "Failed to initialize Dispatch API". Unresolved, awaiting the user. — [LiteRT #6059](https://github.com/google-ai-edge/LiteRT/issues/6059)
  - The FastVLM-on-NPU guidance request (#5499) is stale with no visible maintainer answer. `litert-community/FastVLM-0.5B` exists on Hugging Face. — [LiteRT #5499](https://github.com/google-ai-edge/LiteRT/issues/5499)
- **ONNX Runtime.**
  - `com.microsoft.onnxruntime:onnxruntime-android-qnn` is an AAR with the QNN EP. A search snippet said 1.27.0 is the latest. — [Maven Central](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android-qnn)
  - The QNN EP runs on Android and Windows on Snapdragon. The HTP backend "only supports quantized models" (uint8/uint16) with fixed shapes. Options include `backend_path` and `htp_performance_mode`. — [ORT QNN EP docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
  - Qualcomm ships the matching QNN runtime separately on Maven as `com.qualcomm.qti:qnn-runtime`. — [OpenMed accelerators page (search snippet)](https://openmed.life/docs/runtimes/android-accelerators/)
  - Non-quantised models do not run on HTP. — [ORT issue #18351 (title)](https://github.com/microsoft/onnxruntime/issues/18351)
- **OpenCV.** Since 4.9.0, OpenCV for Android is on Maven Central (`implementation 'org.opencv:opencv:4.9.0'`). — [OpenCV docs](https://docs.opencv.org/4.x/d5/df8/tutorial_dev_with_OCV_on_Android.html). A search snippet reported 5.0.0.1 as the latest. — [Maven Central](https://central.sonatype.com/artifact/org.opencv/opencv)
- **Target device (iQOO 15).** SM8850-AC Snapdragon 8 Elite Gen 5, Adreno 840, 12 or 16 GB LPDDR5X, UFS 4.1. — [GSMArena](https://www.gsmarena.com/vivo_iqoo_15_5g-14198.php)

### Inferences
- **Two NPU routes in native Kotlin, in order of demo safety:**
  1. The Interpreter API with `qnn-litert-delegate` and `qnn-runtime` from Maven, using the latest 2.4x release to cover SM8850. The .so files arrive through Gradle, so a sideloaded APK works with no Play dependency and no manual QAIRT download.
  2. `CompiledModel` with `Accelerator.NPU` in JIT mode, falling back to GPU. It is cleaner, but the runtime and dispatch libraries have to be packaged and issue #6059 shows setup failures.
  - In both cases, keep a one-line GPU fallback so the demo never depends on the NPU.
- **Quantisation.** HTP generally wants int8 (Edge Impulse; ORT docs). The team already measured that int8 keeps AUROC. So an int8 `backbone_r18_320` is the right NPU artefact, and the float model can stay as the GPU/CPU fallback.
- **Backbone vs scorer cost.** A ResNet18-class backbone should run in a few ms on the SM8850 NPU (56 models under 5 ms) or a few tens of ms on GPU. The ~3.3×10^8-MAC kNN then dominates. Options:
  - Multi-threaded Kotlin loops, roughly tens to low hundreds of ms.
  - OpenCV's NEON-optimised `Core.gemm` / `BFMatcher` (prebuilt).
  - A second TFLite graph (q·bᵀ matmul + min) on GPU/NPU.
  - Throttle the live overlay to about 5-10 Hz while the camera preview stays at 30-60 fps.
- **YUV→RGB cost.** Using CameraX RGBA_8888 plus a crop/resize to 320×320 avoids hand-written YUV conversion. VisionCamera's docs put RGB at about 2.6x the bandwidth of YUV, which is fine at 640×480.
- **ONNX Runtime + QNN adds no value here.** The model is already TFLite, and ORT would mean re-exporting to ONNX QDQ.

### Gaps
- No published latency for CameraX's internal YUV→RGBA conversion, and no fps numbers for a CameraX + LiteRT QNN pipeline on SM8850.
- Not verified:
  - which QNN version first supports SM8850 (Hexagon generation and skel name, e.g. V81);
  - whether `qnn-litert-delegate` 2.46.0 is validated on SM8850;
  - the APK size added by the `qnn-runtime` AAR.
- Did not verify whether the LiteRT CompiledModel NPU JIT path accepts fp32/fp16 models on HTP, or needs int8.
- **Compose overlay patterns** (e.g. `camera-compose` `CameraXViewfinder` vs `AndroidView(PreviewView)` + `Canvas`) were not researched with sources. `AndroidView(PreviewView)` with a custom overlay View/Canvas is the conservative choice (unsourced).

---

## 3. Option (c): Flutter. Camera image stream performance and NPU access

### Takeaway
Flutter would be a full rewrite in a language the codebase does not use.
- **Camera stream.** The Android camera plugin (CameraX-based) streams only YUV_420, with open memory-leak and fps-control issues.
- **Inference.** `tflite_flutter` offers only NNAPI and GPU delegates on Android.
- **NPU.** Reaching it needs custom Kotlin platform-channel code, which is effectively option (b) plus a Dart layer.

This is the highest-risk option for a 10-hour build.

### Cited Findings
- **`tflite_flutter`.** Latest version 0.12.1, publisher tensorflow.org. "Acceleration using NNAPI, GPU delegates on Android". Minimum API 26. No camera-stream example. pub.dev showed "published 11 months ago" at fetch time (Sept 2026). — [pub.dev tflite_flutter](https://pub.dev/packages/tflite_flutter)
- **`camera_android_camerax` limits.**
  - It supports only the YUV_420 image format group; `nv21` still reports as yuv420. — [flutter #145961](https://github.com/flutter/flutter/issues/145961)
  - There is an open request for RGBA_8888. — [flutter #151193](https://github.com/flutter/flutter/issues/151193)
  - "StartImageStream causes memory leak, frequently crashes". — [flutter #145893](https://github.com/flutter/flutter/issues/145893)
  - There is no way to set fps in `startImageStream`. — [flutter #77235](https://github.com/flutter/flutter/issues/77235)
- **Blog claim.** A Feb 2026 blog claims near-60-fps YOLO detection in Flutter with no dropped frames. This is a single-author blog, not a benchmark. — [Medium, Otto Lin](https://medium.com/@cia1099/approached-60-fps-object-detection-without-any-frame-dropout-on-mobile-devices-with-flutter-6ab3c9dc5c4b)
- **Ultralytics.** The official Ultralytics YOLO Flutter plugin exists for Android and iOS. — [ultralytics/yolo-flutter-app](https://github.com/ultralytics/yolo-flutter-app). It is AGPL-3.0 or requires an Enterprise licence. — [Ultralytics licence](https://www.ultralytics.com/license)
- **QNN via a third-party package.** One third-party package mentions a QNN delegate NPU mode that "will fall back to CPU" without the right files. — [flutter_pose_detection docs (search snippet)](https://pub.dev/documentation/flutter_pose_detection/latest/). Low reliability and not a general solution.

### Inferences
- Flutter has no advantage for this project. NPU, a native scorer and the LLM all need Kotlin through platform channels anyway, and the existing TypeScript UI and core would be thrown away.

### Gaps
- No authoritative Flutter + NPU (QNN) benchmark or maintained official plugin was found.
- The `flutter_gemma` or other Flutter LLM plugins were not evaluated.

---

## 4. Option (d): hybrid. A native camera + inference module inside the Expo/RN app

### Takeaway
Expo's Modules API lets you write a Kotlin native view: a `PreviewView` with CameraX, LiteRT/QNN and the scorer, which sends events (score, box, heat-map summary) to JS. This keeps the existing enrolment, settings and profile UI, and avoids VisionCamera/worklets entirely. But the RN C++ build (and its Windows path issues) stays. The native camera, inference and scorer code equals option (b)'s core anyway. There is also a real risk of LiteRT version or native-library clashes with fast-tflite (LiteRT 1.4.0).

### Cited Findings
- **Expo Modules API.** Native Android views are written in Kotlin by extending `ExpoView`, exposed with `requireNativeViewManager('...')`, and send events with `EventDispatcher` / `Events("onLoad")`. They work with `npx expo prebuild --clean`. — [Expo native view tutorial](https://docs.expo.dev/modules/native-view-tutorial/)
- **Current build.** The app is built from `expo prebuild`, and `android/` is generated and not committed. — `D:\Projects\Kaizen_Eye\README.md`
- **RN C++ builds.** CMake runs during RN Android builds (autolinking CMake), so C++ build issues apply to every RN app. — [RN issue #47377](https://github.com/facebook/react-native/issues/47377)

### Inferences
- **Fastest hybrid path:**
  - Create a local Expo module `kaizen-live` in Kotlin with a `LiveInspectView`.
  - Inside it: CameraX, LiteRT with the QNN delegate, a Kotlin scorer and a native overlay drawn in the view.
  - It emits an event only about 5-10 times a second.
  - It loads the memory bank from a file the existing TypeScript enrolment writes, e.g. Float32 binary plus tau. Enrolment stays in TypeScript.
  - This avoids per-frame JS work.
- **Dependency risk.** Two TFLite runtimes in one APK (fast-tflite's LiteRT 1.4.0 and the module's LiteRT 2.x plus QNN delegate) may clash on duplicate native library names or classes. A likely fix is moving the photo path to the native module too and removing fast-tflite. This is inference, not verified.
- **Speed vs risk.** The hybrid is faster to a demo than a native rewrite only if the RN build stays green. Debugging spans Metro, Gradle, Kotlin and C++, which raises the risk of an unrecoverable stall in a 10-hour window.

### Gaps
- Did not verify whether fast-tflite and `com.google.ai.edge.litert` 2.x plus `qnn-litert-delegate` actually conflict in one APK.
- Did not verify how Expo SDK 57's Modules API behaves with a heavy CameraX view under the New Architecture (Fabric).

---

## 5. Open-source starting points (live camera + NPU + overlay; on-device LLM/VLM) and their licences

### Takeaway
The best ready-made starting points are all native Android:
- **Qualcomm AI Hub Apps `object_detection_android`**: live camera, TFLite with QNN/GPU/CPU delegates, OpenCV, overlay. BSD-3.
- **Google `litert-samples`**: Kotlin, CompiledModel, NPU samples, live-camera object detection. Apache-2.0.
- **Google AI Edge Gallery**: LiteRT-LM image + text chat, local `.litertlm` import, per-SoC NPU APKs. Apache-2.0.

MNN Chat and Qualcomm GenieX are strong LLM/VLM references. The Ultralytics app is AGPL.

### Cited Findings
- **Qualcomm AI Hub Apps.**
  - BSD-3-Clause.
  - Android apps: ChatApp (Genie), image classification, object detection, semantic segmentation, super resolution (TFLite) and GenieX Chat.
  - NPU-capable SoCs include Snapdragon 8 Elite Gen 5. Android 11+ (API 30+).
  - Source: [qualcomm/ai-hub-apps](https://github.com/qualcomm/ai-hub-apps)
- **`object_detection_android`.**
  - Detects objects "in real time using live camera input", integrating "streaming camera, TFLite, and OpenCV".
  - Written in Java. Android 14+. YOLOv3 through v11 variants. — [AI Hub app page](https://aihub.qualcomm.com/apps/object_detection_android)
  - Delegates: "Qualcomm Hexagon NPU -- via QNN", "GPU -- via GPUv2", "CPU -- via XNNPack".
  - The model comes from `qai-hub-apps fetch` or by copying to `src/main/assets/detector.tflite`.
  - Needs Android Studio 2023.1.1+ and Git-LFS. A Docker option with QAIRT is provided.
  - "The QNN SDK dependency is also released under a separate license".
  - Source: [README](https://github.com/qualcomm/ai-hub-apps/blob/release/object_detection_android/README.md)
- **`geniex_chat_android`.**
  - Runs LLMs and VLMs on NPU (Snapdragon 8 Elite and 8 Elite Gen 5), Adreno GPU or CPU.
  - "the model weights are not bundled into the APK"; the app downloads them from an in-app catalog at runtime.
  - Uses Maven `com.qualcomm.qti:geniex-android` and Android Studio 2024.3.1+. BSD-3, with the GenieX SDK under its own licence.
  - Source: [README](https://github.com/qualcomm/ai-hub-apps/blob/release/geniex_chat_android/README.md)
- **Google `litert-samples`** (Apache-2.0).
  - CompiledModel samples: image segmentation (including a `kotlin_npu` variant), image classification, and object detection with live camera.
  - Qualcomm NPU examples: Gemma, MobileNet, FastVLM. Also shared Kotlin camera helpers.
  - Source: [google-ai-edge/litert-samples](https://github.com/google-ai-edge/litert-samples)
  - The Kotlin NPU segmentation sample was linked from Google's blog as `v2/image_segmentation/kotlin_npu/android`, but that path returned 404 in Sept 2026. — [blog link](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)
- **Google AI Edge Gallery.**
  - Apache-2.0. "Ask Image" (camera or gallery). Android 12+. "No internet is required" after download. — [google-ai-edge/gallery](https://github.com/google-ai-edge/gallery)
  - Release 1.0.12 added "Gemma3 1B NPU support for Qualcomm Technologies SoCs" with device-specific APK variants (sm8550, sm8650, sm8750, sm8850).
  - 1.0.11 added Gemma 4. 1.0.16 and 1.0.18 added LiteRT-LM model import and multimodal images/audio in AI Chat.
  - The latest is 1.0.19 ("Sep 2", year not shown).
  - Source: [Gallery releases](https://github.com/google-ai-edge/gallery/releases)
  - An aggregator dates 1.0.12 to April 24, 2026. — [Codersera (secondary)](https://codersera.com/blog/install-google-ai-edge-gallery-to-run-ai-models-on-your-phone/)
- **MNN Chat (Android).**
  - Models include Qwen3-VL (4B, 8B, 30B-A3B), Qwen2.5-VL and Qwen Omni. Image-to-text and multiple image input.
  - CPU and OpenCL GPU; QNN NPU is implied but not documented.
  - Local models via `adb push`. Latest 0.8.3 (adds Gemma 4). Build uses NDK 27.2.12479018.
  - Source: [MnnLlmChat README](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md)
  - Qwen3-VL support was added Oct 16, 2025. — [alibaba/MNN](https://github.com/alibaba/mnn)
- **ONNX Runtime Android samples.** Object detection (`ObjectDetector.kt`) and image classification exist under `mobile/examples/...`. — [onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile/examples/object_detection/android)
- **Ultralytics.** The Flutter plugin is AGPL-3.0 or Enterprise. — [Ultralytics licence](https://www.ultralytics.com/license)

### Inferences
- **Best camera + NPU skeleton:** ai-hub-apps `object_detection_android`. It already does the camera, TFLite QNN/GPU/CPU delegate selection and overlay on Snapdragon. Replace YOLO post-processing with backbone → kNN → heat map. It is Java, but an AI agent can extend Java or add Kotlin files.
  - Caveat: check whether its Gradle setup pulls QNN from Maven or needs a manual QAIRT SDK download. A manual download would cost time.
- **Best Kotlin/modern-API skeleton:** `litert-samples` (CompiledModel, Apache-2.0).
- **Best LLM/VLM reference:** AI Edge Gallery's LiteRT-LM integration, since it shares the runtime we would use.
- Licensing is fine for a demo: BSD-3 and Apache-2.0. The QNN SDK and GenieX SDK have their own licences, and Ultralytics AGPL should be avoided.

### Gaps
- Licence of `onnxruntime-inference-examples` was not verified (believed MIT).
- MediaPipe camera samples (`mediapipe-samples`) were not re-checked for 2026 maintenance.
- Not checked:
  - whether `object_detection_android` uses CameraX or Camera2;
  - its QNN library source (Maven vs QAIRT);
  - its Windows build status.

---

## 6. On-device LLM/VLM SDKs and how each plugs into each platform

### Takeaway
The most mature, lowest-risk choice is LiteRT-LM (Kotlin AAR, Apache-2.0):
- It has image input.
- It loads a `.litertlm` file from a filesystem path.
- It supports CPU, GPU and NPU.
- Published numbers for Gemma-4-E2B (2.58 GB) show 0.3 s TTFT on GPU on a Galaxy S26 Ultra.

LLM NPU support is SoC-specific and mostly text-only today (e.g. Gemma3-1B NPU builds). A VLM should therefore run on GPU. Qualcomm GenieX gives LLM/VLM on the SM8850 NPU, but its models come from an in-app catalog download. MNN, llama.cpp (Hexagon) and MLC are more build-heavy or less proven for an offline VLM demo on Windows.

### Cited Findings
- **LiteRT-LM project.** v0.16.0, Apache-2.0. Android Kotlin API is stable. CPU, GPU and NPU. Models: Gemma 3n, Gemma 4, Llama, Phi-4, Qwen. "Support for vision and audio inputs". — [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- **LiteRT-LM on Android.**
  - Dependency: `com.google.ai.edge.litertlm:litertlm-android:latest.release`. Config: `EngineConfig(modelPath="/path/model.litertlm", backend=Backend.GPU()|CPU()|NPU(nativeLibraryDir=...))`.
  - "engine.initialize() ... can take ... up to 10 seconds", so run it off the main thread.
  - GPU needs `libvndksupport.so` and `libOpenCL.so` declared in the manifest.
  - Images go in as `Content.ImageFile(path)` or `Content.ImageBytes(bytes)`.
  - Source: [LiteRT-LM Android guide](https://developers.google.com/edge/litert-lm/android)
- **LiteRT-LM benchmarks.**
  - Gemma-4-E2B (2.58 GB) on Samsung S26 Ultra: CPU prefill 557 tok/s, decode 47 tok/s, TTFT 1.8 s. GPU prefill 3808 tok/s, decode 52 tok/s, TTFT 0.3 s.
  - Other listed sizes: Gemma4-E4B 3654 MB, Gemma-3n-E2B 2965 MB, Qwen2.5-1.5B 1598 MB, Qwen3-0.6B 586 MB, phi-4-mini 3906 MB.
  - Source: [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- **LiteRT-LM on the Qualcomm NPU.**
  - The page lists Gemma3-1B (4-bit, 1280 context, about 658 MB) for SM8750, SM8650 and SM8550.
  - Files are SoC-specific (`..._${ro.soc.model}.litertlm`) and gated on Hugging Face ("login and acknowledge the form").
  - QAIRT libraries (`libQnnHtp*Stub.so`, `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpPrepare.so`, `libQnnHtp*Skel.so`) are pushed via adb for CLI use.
  - Source: [LiteRT-LM NPU guide](https://developers.google.com/edge/litert/next/litert_lm_npu)
  - Gallery 1.0.12 shipped an sm8850 NPU variant for Gemma3 1B. — [Gallery releases](https://github.com/google-ai-edge/gallery/releases)
- **FastVLM on SM8850 NPU (Google demo).** 0.12 s TTFT, "live scene understanding demo". — [Google blog](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/). The public developer path is unclear: the guidance issue is stale. — [LiteRT #5499](https://github.com/google-ai-edge/LiteRT/issues/5499)
- **MediaPipe LLM Inference API.** It is maintenance-only, and Google recommends migrating Android projects to LiteRT-LM. — [MediaPipe LLM Inference Android (search snippet)](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android)
- **Qualcomm Genie / GenieX.**
  - The ChatApp uses the Genie SDK. — [ai-hub-apps](https://github.com/qualcomm/ai-hub-apps)
  - Genie bundles are copied to the device via adb. — [llm_on_genie tutorial (search snippet)](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie)
  - GenieX runs LLM/VLM on the 8 Elite Gen 5 NPU, with weights downloaded at runtime from a catalog. — [GenieX README](https://github.com/qualcomm/ai-hub-apps/blob/release/geniex_chat_android/README.md)
- **Nexa SDK.**
  - The Nexa SDK Android bindings URL now redirects to `qualcomm/GenieX`, with an NPU backend "using QNN library internally" and a `LlmWrapper.builder()` API. — [NexaAI/nexa-sdk bindings → qualcomm/GenieX](https://github.com/NexaAI/nexa-sdk/blob/main/bindings/android/README.md)
  - Qualcomm described Nexa for Android as picking NPU/GPU/CPU "with just three lines of code" for LLM, VLM, ASR and more. — [Qualcomm dev blog, Nov 2025 (search snippet)](https://www.qualcomm.com/developer/blog/2025/11/nexa-ai-for-android-simple-way-to-bring-on-device-ai-to-smartphones-with-snapdragon)
- **MNN.** Covered in section 5: Qwen3-VL / Qwen2.5-VL, `adb push` local models, NDK build. — [MnnLlmChat README](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md)
- **llama.cpp.**
  - It has a Snapdragon backend doc covering CPU, Adreno OpenCL and Hexagon HTP. It is built with a Docker toolchain and needs 8 Gen 2 or newer. — [llama.cpp Snapdragon backend (search snippet)](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
  - Some alternative Hexagon forks are experimental. — [kan-linux/ggml-hexagon](https://github.com/kan-linux/ggml-hexagon)
- **MLC LLM.**
  - It targets the Adreno GPU via OpenCL (TVM). — [MLC blog](https://blog.mlc.ai/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices)
  - A 2026 roundup claims MLC Chat uses the Hexagon NPU (~40 tok/s on Qwen3 1.7B). — [PromptQuorum (low reliability)](https://www.promptquorum.com/power-local-llm/best-local-llm-apps-android-2026). This contradicts MLC's documented OpenCL backend, so treat it as unverified.

### Inferences
- **Per-platform integration:**
  - **Native (b):** add the LiteRT-LM AAR and call `Engine` directly. GenieX is also a plain Maven AAR.
  - **RN (a/d):** wrap LiteRT-LM in a Kotlin Expo module (`AsyncFunction("ask") { imagePath, prompt -> ... }`) and stream tokens through events.
  - **Flutter (c):** a MethodChannel to the same Kotlin code.
  - **MNN and llama.cpp:** need NDK/CMake source builds, or copying large JNI projects. That is high risk on Windows in 10 hours.
- **Recommended demo VLM:** Gemma 3n E2B (2.97 GB) or Gemma 4 E2B (2.58 GB) `.litertlm` on the GPU backend with image input.
  - Latency is about 0.3 s TTFT for text on SM8850-class GPU, plus image encoding.
  - Initialise it once at app start behind a splash or progress indicator (about 10 s).
  - Keep an NPU text-only model (Gemma3-1B sm8850) as an optional "NPU LLM" badge, not a dependency.
- **GenieX is attractive for "VLM on NPU"** but the offline import path is undocumented. The demo would have to download from the catalog before going into airplane mode, and this was not verified to work fully offline afterwards.

### Gaps
- Not confirmed that a Gemma3-1B `.litertlm` for SM8850 is on Hugging Face (Gallery ships an sm8850 APK variant, but the LiteRT-LM NPU page listed only SM8550 to SM8750).
- No image-input (vision encoder) latency figures for Gemma 3n or Gemma 4 on SM8850 GPU.
- GenieX model list, offline sideload support and licence terms for NPU/commercial use were not found.
- No RN or Flutter LiteRT-LM wrapper packages were verified.

---

## 7. Model bundling and offline delivery: APK limits, 1-4 GB weights, first-run load

### Takeaway
Keep LLM weights out of the APK:
- **Transfer.** `adb push` the `.litertlm` (2.6-3 GB) to the phone once.
- **Loading.** Load it by file path. Google's own Gallery documents `/sdcard/Download/` plus an in-app file picker import.

Play size limits (asset packs, PODAI) do not apply to a sideloaded demo APK. APKs over about 2 GB are unreliable to sideload (anecdotal). Vision models and QNN libraries stay in the APK. Allow about 10 s for LLM engine initialisation at app start.

### Cited Findings
- **Gallery offline import.** `adb push /path/to/model.litertlm /sdcard/Download/`, then "+" → file picker → Import. The dialog has "Support image" and "Support audio" toggles and a CPU/GPU preference. — [Gallery wiki: Importing Local Models](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))
- **Loading by path.** LiteRT-LM loads models from a filesystem path via `EngineConfig(modelPath=...)`. Initialisation can take up to about 10 s. — [LiteRT-LM Android guide](https://developers.google.com/edge/litert-lm/android)
- **Gated downloads.** The SoC-specific NPU LLM files are gated on Hugging Face (login + form), so download them before the demo. — [LiteRT-LM NPU guide](https://developers.google.com/edge/litert/next/litert_lm_npu)
- **MNN** also supports local models via `adb push`. — [MnnLlmChat README](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md)
- **Google Play limits** (Play distribution only), as reported by the help page at fetch time:
  - base module 500 MB; legacy APK 100 MB; asset packs 1.5 GB each;
  - install-time total within 4 GB; on-demand/fast-follow cumulative 30 GB; overall maximum 34 GB.
  - The page says nothing about AI-pack-specific limits.
  - Source: [Play Console Help 9859372](https://support.google.com/googleplay/android-developer/answer/9859372?hl=en)
  - The 500 MB base-module figure differs from the older, widely quoted 150-200 MB, so re-check it if Play matters.
- **Sideload size.** Forum lore says Google "officially" supports APKs up to about 2 GB and that most phones fail to install larger ones. The ZIP format allows 4 GB, more with zip64. — [XDA thread (old, low reliability)](https://xdaforums.com/t/q-max-apk-size-for-sideloading.3000575/)
- **LiteRT NPU runtime libraries** for production come through PODAI / Play Feature Delivery, with zips on GitHub releases. — [LiteRT NPU guide](https://developers.google.com/edge/litert/next/npu). Google's own Gallery distributes per-SoC NPU APK variants, including sm8850. — [Gallery releases](https://github.com/google-ai-edge/gallery/releases)
- **Current APK** is about 87 MB and "fully offline; the model is inside the APK". — `D:\Projects\Kaizen_Eye\README.md`

### Inferences
- **Fool-proof offline plan:**
  1. Download the `.litertlm` on the laptop.
  2. Run `adb push` to `/sdcard/Download/`, or to the app-specific `/sdcard/Android/data/<pkg>/files/`, which the app reads without storage permissions.
  3. On first launch, either load it directly from the app-specific directory or copy it from the picker URI into `filesDir`. Copying 3 GB on UFS 4.1 should take seconds to tens of seconds.
  4. Show "model ready" status on the home screen.
- **Rehearse offline.** Test the whole flow in airplane mode the night before.
- **APK contents.** The vision backbone (a few MB) and QNN runtime AARs belong in the APK. Push the LLM weights separately. This keeps the build fast and avoids the 2-4 GB APK question entirely.

### Gaps
- Did not verify, on Android 16 / OriginOS 6:
  - whether `adb push` into `/sdcard/Android/data/<pkg>/files/` works on this OEM build;
  - whether apps can still read from `/data/local/tmp`. Older MediaPipe docs used that path.
- No measured first-load (mmap/initialise) time for a 3 GB `.litertlm` on SM8850 beyond Google's "up to 10 seconds" statement.
- APK size added by `qnn-runtime`, `qnn-litert-delegate` and `litertlm-android` AARs is unknown.

---

## 8. Windows build pitfalls (spaces, long paths, NDK/CMake) and how to avoid them

### Takeaway
The team's bug is known. A user name with a space (`C:\Users\Vijay Chikkala`) makes the NDK clang be called through its 8.3 short name, which drops it into C mode and produces `std::__ndk1` link errors. RN also has documented space-in-path and 260-character path failures with CMake/Ninja.

The most robust fix is to avoid compiling C++ at all: a pure Kotlin app with prebuilt AARs. Failing that:
- keep the project in a short path with no spaces (e.g. `D:\Projects\...`);
- keep the SDK junction `D:\AndroidSdk`;
- use Ninja 1.12 or newer;
- enable Windows long paths.

### Cited Findings
- **Space in the user name.** It causes "hundreds of `ld.lld: error: undefined symbol: std::__ndk1::...`" because clang runs via its 8.3 short name (`CLANG_~1.EXE`) and enters C mode. The fix is `mklink /J D:\AndroidSdk "%LOCALAPPDATA%\Android\Sdk"`, and deleting `node_modules\**\android\.cxx` after a failure. — `D:\Projects\Kaizen_Eye\README.md`
- **RN 0.76.1 with spaces.** A project path with spaces fails with a CMake error ("Cannot specify link libraries for target ... which is not built by this project") because autolinking CMake was not quoted. The issue is marked fixed. — [RN #47377](https://github.com/facebook/react-native/issues/47377)
- **Software Mansion's Windows guide:**
  - Paths over 240 characters trigger CMake warnings. Move the project or `subst` a drive letter.
  - Spaces in paths cause build failures.
  - Enable long paths in the registry.
  - Use Ninja 1.12.0 or newer and CMake 3.22.1 or newer.
  - Unsetting `_JAVA_OPTIONS` sometimes fixes builds. Clear `android\build`, `android\.gradle`, the module build directories and the Gradle caches.
  - Source: [Reanimated: Building on Windows](https://docs.swmansion.com/react-native-reanimated/docs/guides/building-on-windows/)
- **260-character limit.** CMake/Ninja hit it inside RN libraries on Windows. — [react-native-screens #3471](https://github.com/software-mansion/react-native-screens/issues/3471); [Nutrient guide](https://www.nutrient.io/guides/react-native/troubleshooting/windows-path-length-cmake-error/)
- **MNN** source builds pin NDK 27.2.12479018. — [MnnLlmChat README](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md)
- **llama.cpp Hexagon** builds use a Docker toolchain image. — [llama.cpp Snapdragon backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)

### Inferences
- A new native Kotlin project that depends only on Maven AARs has no `externalNativeBuild`, so no CMake, Ninja or clang runs on the Windows machine. This removes the whole bug class the team hit.
  - AARs: LiteRT, qnn-runtime, qnn-litert-delegate, litertlm-android, CameraX and OpenCV.
  - Put the project under `D:\Projects\Kaizen_Eye\native\` (short, no spaces).
  - Point `sdk.dir` at `D:/AndroidSdk`.
- Options (a) and (d) keep RN's CMake build and add more C++ modules: Nitro, worklets, VisionCamera. Each is another chance to hit the path problems above.
- Options that compile llama.cpp (Hexagon) or MNN from source on Windows are the riskiest for a 10-hour window.

### Gaps
- Did not verify that AGP/Gradle itself has no remaining space-in-path issues for pure-Java/Kotlin builds when `GRADLE_USER_HOME` is under `C:\Users\Vijay Chikkala\.gradle`. It is generally fine, but unverified here.

---

## 9. Risk-ranked comparison, recommended stack, starting samples and 10-hour feasibility check

### Takeaway
**Recommended: option (b), a new native Kotlin app.**
- **Camera:** CameraX Preview plus ImageAnalysis (RGBA_8888, KEEP_ONLY_LATEST).
- **Vision inference:** LiteRT with the Qualcomm QNN LiteRT delegate (int8 backbone on the Hexagon NPU) and an automatic GPU fallback.
- **Scoring:** a Kotlin port of the PatchCore scorer, verified with JVM unit tests against the existing golden vectors.
- **LLM/VLM:** LiteRT-LM (Gemma 3n/4 E2B `.litertlm` with image input on GPU), with weights pushed via adb and loaded from local storage.

Start from Qualcomm's `object_detection_android` or Google's `litert-samples`. Borrow the LLM wrapper from AI Edge Gallery. The plan fits in about 10 hours if the NPU is treated as a time-boxed upgrade over a working GPU pipeline.

### Cited Findings
- **Evidence behind the recommendation:**
  - SM8850 is explicitly supported by LiteRT's Qualcomm backend. — [LiteRT Qualcomm](https://developers.google.com/edge/litert/next/qualcomm)
  - The QNN delegate and runtime are on Maven. — [Google QNN delegate page](https://developers.google.com/edge/litert/android/npu/qualcomm)
  - GPU fallback is automatic in CompiledModel. — [LiteRT NPU guide](https://developers.google.com/edge/litert/next/npu)
  - CameraX gives RGBA and handles backpressure. — [CameraX analyze](https://developer.android.com/media/camera/camerax/analyze)
  - LiteRT-LM has a stable Kotlin API with image input. — [LiteRT-LM Android](https://developers.google.com/edge/litert-lm/android)
  - Gallery shows offline `.litertlm` import. — [Gallery wiki](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))
  - AI Hub's live-camera TFLite/QNN/OpenCV sample (BSD-3). — [ai-hub-apps object detection](https://github.com/qualcomm/ai-hub-apps/blob/release/object_detection_android/README.md)
- **Evidence against (a):**
  - fast-tflite Android offers only `android-gpu` and `nnapi`. — [fast-tflite README](https://raw.githubusercontent.com/mrousavy/react-native-fast-tflite/main/README.md)
  - The V5 resizer has a HardwareBuffer crash report. — [VC #3824](https://github.com/mrousavy/react-native-vision-camera/issues/3824)
  - V5 is new (April 2026). — [Margelo](https://margelo.com/blog/whats-new-in-visioncamera-v5)
- **Evidence against (c):** the camera plugin is YUV-only and has leak issues. — [flutter #145961](https://github.com/flutter/flutter/issues/145961), [#145893](https://github.com/flutter/flutter/issues/145893). `tflite_flutter` offers NNAPI and GPU only. — [pub.dev](https://pub.dev/packages/tflite_flutter)
- **int8 fits the NPU.** The team already measured int8 as AUROC-neutral (0.898 vs 0.905), and HTP wants int8. — `D:\Projects\Kaizen_Eye\README.md`; [Edge Impulse QNN](https://docs.edgeimpulse.com/tutorials/topics/android/qnn-acceleration)

### Inferences

#### Risk-ranked comparison (1 = lowest risk for a fool-proof demo in about 10 hours)

| Rank | Option | Time to working live demo | Camera throughput / zero-copy | NPU access on SM8850 | Samples to start from | Windows build risk | LLM/VLM integration | Main failure modes |
|---|---|---|---|---|---|---|---|---|
| 1 | **(b) Native Kotlin** (CameraX + LiteRT/QNN + Kotlin scorer + LiteRT-LM) | ~6-8 h incl. UI rebuild and scorer port (golden vectors de-risk the port) | CameraX RGBA_8888, KEEP_ONLY_LATEST; AHardwareBuffer zero-copy available (C++) | Direct: QNN LiteRT delegate AAR or CompiledModel NPU; SM8850 listed | ai-hub-apps object detection (BSD-3), litert-samples (Apache-2.0), Gallery (Apache-2.0) | Lowest: no C++ compile if only AARs are used | Direct Kotlin AAR (LiteRT-LM, GenieX) | QNN lib/version mismatch on SM8850 (mitigate with the GPU fallback); UI rebuild time |
| 2 | **(d) Hybrid** (Expo local module: native CameraX view + LiteRT/QNN + scorer) | ~6-9 h; keeps the existing UI, but adds a bridge | Same as (b) inside the native view; events to JS throttled | Same as (b) | Same as (b), plus Expo native-view tutorial | Medium: RN CMake build stays (mitigated today by the junction) | Kotlin Expo module wrapping LiteRT-LM | LiteRT 1.4 (fast-tflite) vs 2.x/QNN clashes; two-world debugging; prebuild regenerating `android/` |
| 3 | **(a) RN + VisionCamera V5 + fast-tflite** | ~5-8 h for GPU-only live preview; NPU unlikely in time | Worklet on the camera thread; `yuv`/`rgb` formats; GPU resizer; JS kNN too slow → native Nitro scorer needed | Only NNAPI (deprecated, driver unverified); no QNN delegate | fast-tflite and VisionCamera examples; no inspection-style sample | Medium-high: 6+ new C++ native packages via CMake | Still needs a Kotlin module for LiteRT-LM | V5 immaturity (HardwareBuffer crash), worklets/RN 0.86 pairing, NNAPI not reaching Hexagon |
| 4 | **(c) Flutter** | 10+ h (full rewrite in Dart) | YUV-only stream, leak issues, no fps control | Only via custom Kotlin platform code | Ultralytics plugin (AGPL), community plugins | Medium (Gradle + Flutter toolchain), but a new toolchain for the team | MethodChannel to Kotlin | Rewrite scope; no NPU plugin |

#### Recommended stack (concrete)
- **Project.**
  - A new Android Studio project at `D:\Projects\Kaizen_Eye\native\`, Kotlin, minSdk 31 (LiteRT NPU minimum API 31), `abiFilters "arm64-v8a"`, `sdk.dir=D:/AndroidSdk`.
  - Keep the Expo app untouched as a fallback demo (photo mode).
- **UI.** Views or Compose, whichever the chosen sample uses. For Compose, use `AndroidView(PreviewView)` plus a `Canvas` overlay: heat-map bitmap, PASS/REJECT banner and a latency HUD (CPU/GPU/NPU ms).
- **Camera.** CameraX `Preview` + `ImageAnalysis` (RGBA_8888, `STRATEGY_KEEP_ONLY_LATEST`, about 640×480). Center-crop to 320×320 and feed the backbone.
- **Vision inference.**
  - Dependencies: `com.google.ai.edge.litert:litert` plus `com.qualcomm.qti:qnn-litert-delegate` and `qnn-runtime` at the latest 2.4x.
  - NPU runs the int8 `backbone_r18_320`. The GPU delegate (float model) is the automatic fallback.
  - `CompiledModel(Accelerator.NPU, Accelerator.GPU)` is an alternative to the classic delegate.
- **Scorer.**
  - A pure-Kotlin/JVM `:core` module: coreset, LOO tau, NN distance, 3×3 smoothing, 10% border and the whole-image check.
  - JUnit tests against `testdata/golden_core.json` and `golden_app.json`. These run on the laptop with no device needed.
  - Multi-threaded NN search over the 1,600-entry bank, or OpenCV `Core.gemm`/`BFMatcher` via `org.opencv:opencv`.
  - Update the overlay at 5-10 Hz.
- **LLM/VLM.**
  - `com.google.ai.edge.litertlm:litertlm-android` with `Backend.GPU()`. Model: Gemma 3n E2B or Gemma 4 E2B `.litertlm` with image input.
  - An "Explain" button sends the current frame crop, the heat-map summary and a prompt, and streams the answer.
  - Initialise the engine at app start on a background thread (about 10 s).
  - The weights are pushed via adb and read from app-specific storage.
- **Starting samples:**
  - (1) Qualcomm `ai-hub-apps/object_detection_android`: camera + TFLite QNN/GPU/CPU + overlay (BSD-3).
  - or (2) Google `litert-samples` object detection / CompiledModel Kotlin (Apache-2.0).
  - plus (3) the Google AI Edge Gallery LiteRT-LM chat/"Ask Image" code (Apache-2.0) for the LLM wrapper.

#### Feasibility check: about 10 hours, AI agent implementing, one person on hardware

| Hours | Milestone (exit criterion) | Fallback if blocked |
|---|---|---|
| 0-1 | Clone the sample, build a debug APK on Windows (no NDK), camera preview + analyzer running on the iQOO 15 | Switch between sample (1) and (2) |
| 1-3 | Kotlin `:core` scorer passes golden-vector JUnit; float backbone runs on the GPU delegate from assets; per-frame backbone latency logged | CPU/XNNPack delegate |
| 3-5 | Enrol (20 frames from the live stream or photos) → bank + tau saved; live Inspect with heat-map overlay and PASS/REJECT at ≥5 Hz | Tap-to-inspect still frame (still "live preview") |
| 5-6.5 | int8 backbone on the NPU via the QNN delegate (time-boxed to 90 min); on-screen NPU/GPU latency toggle | Ship on GPU; show NPU as "experimental" |
| 6.5-8.5 | LiteRT-LM VLM "Explain defect" (image + heat-map text) from a pushed `.litertlm`, streamed tokens | Text-only LLM over the score JSON; or the Gallery app as a side demo |
| 8.5-10 | Release APK, airplane-mode full rehearsal, reinstall-from-scratch test, buffer | Keep the existing Expo APK as a photo-mode backup |

- **Why it fits.**
  - The heavy parts are prebuilt AARs.
  - The scoring logic already has a Python reference and golden vectors, so an AI port is mechanically checkable.
  - The phone has 12-16 GB RAM, so a 2.6-3 GB VLM and the camera pipeline fit together.
  - The riskiest part (NPU) is isolated behind an automatic GPU fallback.
- **Biggest residual risks:**
  1. The QNN delegate or runtime version vs SM8850 (Hexagon generation) → use the newest Maven release and time-box it.
  2. VLM image-encode latency on GPU is unknown → prompt with a single small crop.
  3. The HF gated model download must happen before the demo day.
  4. The UI rebuild consumes more time than planned → keep the UI minimal (three screens: Enrol, Inspect, Settings).

### Gaps
- No end-to-end reference (live camera + NPU backbone + kNN + VLM) exists to benchmark against. The hour estimates above are engineering judgement, not measured.
- Not confirmed on SM8850 specifically:
  - QNN delegate compatibility and version;
  - LiteRT-LM image-input latency;
  - whether the `object_detection_android` build needs a manual QAIRT download.
