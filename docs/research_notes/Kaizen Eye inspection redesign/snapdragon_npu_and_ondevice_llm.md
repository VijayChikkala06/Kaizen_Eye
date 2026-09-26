# Snapdragon 8 Elite Gen 5 (iQOO 15) Hexagon NPU: vision runtimes, on-device LLM/VLM, thermals and live telemetry (research as of 2026-09-26)

Labels used on every number: **[V]** = vendor / first-party claim or vendor-published benchmark (Qualcomm, Google, SDK or model authors); **[I]** = independent measurement (academic paper, third-party developer); **[U]** = unverified / low-reliability (aggregator, search-result snippet, forum title only). Chip codes: SM8850 = Snapdragon 8 Elite Gen 5 (Hexagon HTP v81); SM8750 = Snapdragon 8 Elite (Hexagon v79). "8 Elite Gen 5 for Galaxy" = the Samsung Galaxy S26-series bin of SM8850. Maven versions were read directly from repository metadata on 2026-09-26.

## Q1. iQOO 15 specs and Snapdragon 8 Elite Gen 5 NPU specs/claims

### Takeaway
The iQOO 15 is an SM8850 (Snapdragon 8 Elite Gen 5, Hexagon HTP v81) phone with 12 or 16 GB LPDDR5X running OriginOS 6 on Android 16. It launched in China on 20 Oct 2025 and in India on 26 Nov 2025, from Rs 72,999. Qualcomm claims the NPU is "37% faster" and supports INT2 to INT16, FP8 and FP16. Qualcomm does not disclose TOPS, and the gain in performance per watt it has reported is only about 16%. On 22 Sep 2026 Qualcomm announced the Snapdragon 8 Elite Gen 6, so the iQOO 15 is now a previous-generation flagship.

### Cited Findings
- iQOO 15 announced **20 Oct 2025 (China)**. It uses the Snapdragon 8 Elite Gen 5 with a separate "Q3" gaming chip. China configurations: 12/256, 12/512, 16/256, 16/512 and 16 GB/1 TB, with "up to 16GB of LPDDR5X Ultra RAM" and UFS 4.1. OS is "Android 16-based OriginOS 6". Battery is 7,000 mAh (single-cell silicon-carbon) with 100 W wired and 40 W wireless charging. This article says the vapour chamber is **"14,000mm²"**. — [GSMArena China launch](https://m.gsmarena.com/iqoo_15_debuts_with_snapdragon_8_elite_gen_5_soc_7000mah_battery-news-69980.php)
- India launch (article dated 26 Nov 2025): Snapdragon 8 Elite Gen 5 with the Q3 chip, **12 GB or 16 GB LPDDR5X**, 256/512 GB UFS 4.1, OriginOS 6 on Android 16 (the end of FunTouchOS in India), 6.85-inch 2K 144 Hz display, 7,000 mAh, 100 W / 40 W charging, **from Rs 72,999**. — [Business Standard](https://www.business-standard.com/technology/gadgets/iqoo-15-launch-snapdragon-8-elite-gen5-originos-india-125112600468_1.html)
- GSMArena review (26 Nov 2025, 16/512 GB unit): OriginOS 6 on Android 16, with a promise of 5 years of OS updates and 7 years of security patches. Scores: Geekbench 6 3,643 single / 10,466 multi; AnTuTu v11 3,785,250; 3DMark Wild Life Extreme 7,229. This review says the vapour chamber is **"8000 mm2 single-layer"**, which **contradicts** the 14,000 mm² in the China launch article. It may be a regional difference or an error in one of the two. — [GSMArena review, performance page](https://www.gsmarena.com/iqoo_15-review-2905p4.php)
- SM8850 NPU claims [V, as reported by analysts]:
  - The NPU is "37% faster". Futurum quotes "local processing can hit up to 220 tokens per second" but does not say which model that figure is for. It also notes "support for INT2 and FP8 precision". — [Futurum, 1 Oct 2025](https://futurumgroup.com/insights/is-snapdragon-8-elite-gen-5-the-benchmark-for-next-gen-flagship-phones/)
  - Qualcomm added "a tensor (matrix-math) unit and more scalar and vector cores". The NPU supports integer types from 2 to 16 bits plus FP8 and FP16, but **not BF16 or FP4**. A 3B LLM runs "37% faster" and the context window grows eightfold to 32K tokens. Performance per watt rises by **"only 16%"**. **Qualcomm does not disclose Gen 5 NPU TOPS.** — [XPU.pub analysis, 6 Oct 2025](https://xpu.pub/2025/10/06/qualcomm-snapdragon-8-elite-gen-5/)
- An "80 TOPS" figure and a "12 scalar + 8 vector + 1 accelerator" layout circulate on third-party sites [U]. Qualcomm has not confirmed either, and XPU.pub says TOPS is undisclosed. — [multicoreperformance.com](https://multicoreperformance.com/snapdragon-8-elite-gen-5-dedicated-ai-spec-sheet-80-tops-architecture-deep-thermal-truths/)
- SM8850 is **Hexagon arch v81** (htp_arch 81), confirmed by a developer running ONNX Runtime QNN on a Galaxy S26 Ultra (SM-S948U, 8 Elite Gen 5) [I]. — [itsallgoody/onnxruntime-qnn-snapdragon-8-elite-gen5](https://github.com/itsallgoody/onnxruntime-qnn-snapdragon-8-elite-gen5)
- llama.cpp ships HTP libraries for v73, v75, v79 and **v81**. — [llama.cpp Snapdragon backend docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
- Snapdragon 8 Elite (SM8750) uses Hexagon v79. — [MNN releases](https://github.com/alibaba/MNN/releases); [dev.to, May 2026](https://dev.to/jdshah/why-my-llm-runs-4x-faster-on-hardware-i-had-never-heard-of-9b9)
- Snapdragon 8 Elite Gen 6 and 8 Elite Extreme Gen 6 were announced **22 Sep 2026**. The Extreme version "can run a 30-billion-parameter mixture-of-experts (MoE) model locally". — [TechCrunch](https://techcrunch.com/2026/09/22/qualcomm-launches-two-new-smartphone-chips-with-emphasis-on-ai/)

### Inferences
- With 12 or 16 GB of RAM, there is room for one ~1–3 GB VLM plus several small vision models running at the same time (see the memory figures in Q4).
- AI Hub numbers labelled "8 Elite Gen 5 for Galaxy" (the S26-series bin) are the closest published proxy for the iQOO 15. The Galaxy bin may be marginally faster than the standard SM8850 in the iQOO 15. That is not confirmed, so treat those numbers as best case.
- Performance per watt improved far less (+16%) than peak performance (+37%). Running the NPU at peak will therefore draw more power than on the 8 Elite, which is one more reason to duty-cycle (see Q5).

### Gaps
- The official Qualcomm product brief PDF could not be text-extracted. NPU clock speed, TOPS and memory bandwidth are not confirmed from primary sources.
- The vapour-chamber size conflict (8,000 vs 14,000 mm²) is unresolved.
- I found no official statement of iQOO 15 India availability beyond the launch-day details.

## Q2. Runtimes and toolchains that reach the Hexagon NPU from an Android app in 2026: maturity and risk for a 10-hour build

### Takeaway
The app's current path, react-native-fast-tflite with the NNAPI delegate, almost certainly **does not reach the Hexagon NPU**. On recent Qualcomm chips NNAPI exposes only the `nnapi-reference` CPU device, and the library has no QNN or LiteRT-NPU option.

Two practical routes to the NPU exist today. Both need a small native Kotlin module (for example an Expo module):
1. **LiteRT 2.x `CompiledModel` with `Accelerator.NPU`**. It officially lists SM8850, supports on-device (JIT) compilation of the existing `.tflite`, and falls back to GPU.
2. **ONNX Runtime 1.29 + QNN EP + `com.qualcomm.qti:qnn-runtime`**. It is proven on SM8850 from a sideloaded APK after four documented fixes, and it can **guarantee** NPU-only execution.

The classic Interpreter API with Qualcomm's `qnn-litert-delegate` is a third option that keeps the existing `.tflite` files. ExecuTorch QNN is too heavy for 10 hours.

### Cited Findings
**LiteRT (Google) with the Qualcomm AI Engine Direct (QNN) accelerator**
- Supported SoCs: "Snapdragon 8 Elite Gen 5 Mobile Platform (SM8850)", SM8750, SM8650, SM8550, SM8475 and SM8450. "LiteRT supports Qualcomm AI Engine Direct (QNN) through the `CompiledModel` API for both AOT and on-device compilation." — [LiteRT Qualcomm doc](https://developers.google.com/edge/litert/next/qualcomm)
- Announced 24 Nov 2025 as a replacement for the older TFLite QNN delegate [V]. Benchmarked across 72 canonical models on **SM8850**:
  - "up to a 100x speedup over CPU and a 10x speedup over GPU".
  - "over 56 models run in under 5ms with the NPU, while only 13 models achieve that on the CPU".
  - NPU latency is roughly 1–20% of CPU; GPU is roughly 5–70% of CPU.
  - 90 LiteRT ops are lowered, enough for 64 of the 72 models to delegate fully to the NPU.
  - Runtime libraries are delivered through Google Play for On-device AI / feature modules.
  — [Google Developers Blog](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)
- NPU guide:
  - AOT is "best suited for large, complex models where the target SoC is known". JIT is "ideal for ... small models" but has a "higher first-run cost".
  - Dependency is `com.google.ai.edge.litert:litert`, and "API level 31+ is required for NPU support".
  - Sample code: `CompiledModel.create(assets, "model.tflite", CompiledModel.Options(Accelerator.NPU, Accelerator.GPU))`, where "If NPU is unavailable, inference will fallback to GPU".
  - Vendor libraries come via Play Feature Delivery / Play AI Packs, or from `litert_npu_runtime_libraries(_jit).zip`.
  - JIT compile cache example (device unspecified): resnet152 initialises in 7,465 ms the first time and 198 ms when cached.
  — [LiteRT NPU doc](https://developers.google.com/edge/litert/next/npu)
- Kotlin API: `implementation 'com.google.ai.edge.litert:litert:2.1.0'`, with the calls `createInputBuffers()`, `run()` and `readFloat()`. The page does not document how to query which accelerator was actually used. — [LiteRT Kotlin doc](https://developers.google.com/edge/litert/next/android_kotlin)
- Latest version on Google Maven is **litert 2.2.0** (published 13 Aug 2026). — [Google Maven metadata](https://dl.google.com/android/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml)
- v2.2.0 "Aligned SoC coverage with QAIRT SDK 2.47". v2.1.3 added bundling of MediaTek NPU libraries in the app binary. The release notes do not explicitly mention SM8850. — [LiteRT releases](https://github.com/google-ai-edge/LiteRT/releases)
- A third-party wrapper (flutter_litert 3.9.x) confirms that "Android API 31+ arm64 apps can now use an app-provided LiteRT JIT NPU runtime" (validated on SM8550/v73, SM8650/v75 and SM8750/v79, using QAIRT 2.47). It exposes `accelerators`, `isFullyAccelerated` and `didFallback` diagnostics on top of LiteRT Next 2.2.0 [I]. — [flutter_litert changelog](https://pub.dev/packages/flutter_litert/changelog)
- An open issue from 28 Jan 2026 reports a developer unable to get FastVLM running on the NPU via LiteRT / LiteRT-LM. It got no maintainer answer and was marked stale, which is a maturity warning for multimodal models. — [LiteRT issue #5499](https://github.com/google-ai-edge/LiteRT/issues/5499)

**Qualcomm QNN LiteRT delegate (classic Interpreter API)**
- Maven dependencies are `com.qualcomm.qti:qnn-runtime` and `com.qualcomm.qti:qnn-litert-delegate`; the doc's example uses 2.34.0. Code: `QnnDelegate.Options().setBackendType(HTP_BACKEND); setSkelLibraryDir(nativeLibraryDir)`. Listed SoCs are 8 Gen 1 to 8 Elite "and more"; SM8850 is not listed explicitly. On a Galaxy S25 (8 Elite) [V]:
  - MobileNetV2: NPU 0.3 ms, GPU 1.8 ms, CPU 2.8 ms.
  - FFNet-40s: NPU 24.9 ms, GPU 43 ms, CPU 481.7 ms.
  — [LiteRT Qualcomm delegate doc](https://developers.google.com/edge/litert/android/npu/qualcomm)
- The latest `qnn-runtime` and `qnn-litert-delegate` are both **2.50.0** (Maven Central, updated 7 Sep 2026). — [qnn-runtime metadata](https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/maven-metadata.xml); [qnn-litert-delegate metadata](https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-litert-delegate/maven-metadata.xml)

**ONNX Runtime + QNN Execution Provider**
- Provider options:
  - `backend_type` = `htp` (NPU), `gpu`, `cpu` or `saver`.
  - `htp_performance_mode` = `burst`, `balanced`, `default`, `high_performance`, `high_power_saver`, `low_balanced`, `low_power_saver`, `power_saver` or `sustained_high_performance`.
  - `enable_htp_fp16_precision` is on by default, so fp32 models run in fp16 on HTP.
  - `profiling_level` = `basic`, `detailed` or `optrace` (optrace needs QAIRT ≥ 2.39).
- Session options: `"session.disable_cpu_ep_fallback","1"` makes the session throw "an exception if the model can't be run entirely on the QNN HTP backend", and `"ep.context_enable","1"` caches compiled context binaries to cut startup time.
- Quantised HTP models use uint8/uint16 QDQ.
— [ORT QNN EP docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- Latest `com.microsoft.onnxruntime:onnxruntime-android-qnn` is **1.29.0** (updated 12 Aug 2026). — [Maven metadata](https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android-qnn/maven-metadata.xml)
- **Proven on SM8850 from a sideloaded app** (Galaxy S26 Ultra, onnxruntime-android-qnn 1.29.0 + qnn-runtime 2.45.0, htp_arch 81) [I]. Four fixes were needed:
  1. Declare `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>` in the manifest.
  2. Set `ADSP_LIBRARY_PATH` to include `nativeLibraryDir`.
  3. Set `packaging.jniLibs.useLegacyPackaging = true`.
  4. Pin the exact `qnn-runtime` version.
  
  Measured: Whisper-tiny encoder 22.0–33.7 ms and decoder 1.73 ms/token on QNNExecutionProvider. — [itsallgoody repo](https://github.com/itsallgoody/onnxruntime-qnn-snapdragon-8-elite-gen5)

**NNAPI and react-native-fast-tflite (the current stack)**
- The NNAPI NDK API is deprecated starting in Android 15, and Google's migration guide points to TFLite in Play services with the GPU delegate. — [Android NNAPI guide](https://developer.android.com/ndk/guides/neuralnetworks); [NNAPI migration guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide) (from search-result summaries of these pages [U])
- A developer report dated 4 Sep 2026 says "Qualcomm ships no NNAPI driver". On SM8650 and SM8550 phones "NNAPI exposes only `nnapi-reference` (CPU)", and the suggested route to Hexagon is `qnn-litert-delegate` + `qnn-runtime` [I]. — [DroidRunner issue #82](https://github.com/m96-chan/DroidRunner/issues/82)
- An older example: on a Galaxy Tab S9 (8 Gen 2), the TFLite NNAPI delegate fell back to the XNNPACK CPU delegate at `AllocateTensors()` [I]. — [TensorFlow issue #61854](https://github.com/tensorflow/tensorflow/issues/61854)
- react-native-fast-tflite offers only the `android-gpu` and `nnapi` delegates (plus the default CPU). Its README says "NNAPI is deprecated on Android 15. GPU delegate is preferred." It does not mention QNN, NPU or CompiledModel. — [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite)

**ExecuTorch QNN backend**
- The docs verify SM8550 and SM8450 but do not list SM8850, and they recommend QNN 2.37.0 "for stability". Using it requires rebuilding the ExecuTorch AAR yourself (`build_android_library.sh`) plus `com.qualcomm.qti:qnn-runtime`. — [ExecuTorch Qualcomm backend docs](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)

**MediaPipe**
- The MediaPipe LLM Inference API is in maintenance-only mode, and Google recommends migrating to LiteRT-LM. — [MediaPipe LLM Inference Android guide](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android) (from a search-result summary [U])

**Qualcomm AI Hub (build-time only)**
- AI Hub model cards list per-device latency and "compute unit: NPU" for TFLITE, QNN_DLC and ONNX exports (AI Hub v0.63.0; see Q3).
- For licence-restricted models such as Llama, "pre-exported model assets are unavailable" and you must compile them yourself through AI Hub, which needs an AI Hub account and API token for cloud compilation.
- This is a build-time requirement only. The compiled output runs offline.
— [qualcomm/Llama-v3.2-3B-Instruct](https://huggingface.co/qualcomm/Llama-v3.2-3B-Instruct)

### Inferences
- **Lowest-risk NPU path for the vision models in about 10 hours:** a Kotlin Expo native module wrapping LiteRT 2.2.0 `CompiledModel` with `Options(Accelerator.NPU, Accelerator.GPU)`, loading the existing int8 (2.9 MB) or fp32 ResNet18 `.tflite` with JIT compilation and a compiler cache directory.
  - SM8850 is on Google's official support list, and Google's own 72-model benchmark ran on SM8850.
  - The main risk is library delivery for a sideloaded APK. The documented path is Play Feature Delivery, so the agent should bundle the QNN / dispatch `.so` files in `jniLibs` and apply the same manifest, `useLegacyPackaging` and `ADSP_LIBRARY_PATH` fixes as the ORT repo.
- **Fallback, or first choice if "prove 100% NPU" matters most:** ORT 1.29.0 + QNN EP.
  - It is the only path with a documented switch (`session.disable_cpu_ep_fallback`) that fails loudly if any node would run on CPU, and with built-in profiling CSVs.
  - It has a working SM8850 sideload recipe.
  - The cost is exporting the backbone to an ONNX QDQ model.
- **Do not demo NNAPI as "the NPU".** On current Qualcomm phones it most likely runs on the CPU reference path.
- Keep react-native-fast-tflite as the CPU/GPU baseline for the A/B "NPU vs CPU" comparison in the demo.
- Qualcomm context binaries and `.litertlm` NPU builds are SoC-specific (files are named per SoC, and AOT targets a "known" SoC). A model precompiled for SM8750 (v79) should not be assumed to run on SM8850 (v81). Prefer JIT, or AOT artefacts explicitly built for SM8850.

### Gaps
- No Google document explicitly describes bundling the Qualcomm NPU dispatch/QNN libraries inside a sideloaded APK for LiteRT 2.x on SM8850. The inference above relies on third-party evidence (flutter_litert, the ORT repo).
- The official Kotlin API for "which accelerator actually ran" (the equivalent of `isFullyAccelerated`) is not documented on Google's page. It is visible only through the flutter_litert wrapper.
- Not researched: MediaPipe Tasks (vision) NPU support, and whether a PatchCore-style kNN or memory-bank step could run on the NPU. Assume it runs on the CPU.
- I did not verify whether dynamic-range-quantised (int8 weights, float activations) `.tflite` models are accepted by HTP. The AI Hub numbers are for fully quantised w8a8 or float models.

## Q3. Published NPU latencies for candidate vision models (backbone, detector, segmenter, CLIP) on 8 Elite Gen 5 / 8 Elite

### Takeaway
On AI Hub's own measurements [V], a ResNet18 classifier takes **0.23 ms** (TFLite w8a8) and YOLOv11n-class detection **0.73 ms** (TFLite w8a8, 640×640) on the 8 Elite Gen 5 NPU. CLIP ViT image+text takes about **11–13 ms**. The backbone and detector are effectively free, and the camera frame rate will limit throughput, not the NPU. MobileSAM's image encoder is the exception at about **0.5 s** per 720p image even on the NPU, so run segmentation on demand only.

### Cited Findings
All rows below are **[V] Qualcomm AI Hub v0.63.0**, measured on "Snapdragon 8 Elite Gen 5 For Galaxy Mobile" unless noted.
- **ResNet18** (224×224, 11.7M params; 11.3 MB w8a8 / 44.6 MB float):
  - 8 Elite Gen 5: TFLITE w8a8 **0.233 ms**, TFLITE float 0.591 ms, QNN_DLC w8a8 0.271 ms, ONNX w8a8 0.347 ms. Peak memory about 0–30 MB. All on NPU.
  - 8 Elite: TFLITE w8a8 0.254 ms, TFLITE float 0.71 ms.
  — [qualcomm/ResNet18](https://huggingface.co/qualcomm/ResNet18)
- **YOLOv11-Detection** (640×640, 2.64M params; 2.83 MB w8a8 / 10.1 MB float; **licence AGPL-3.0**):
  - 8 Elite Gen 5: TFLITE w8a8 **0.732 ms**, TFLITE float 1.614 ms, QNN_DLC w8a16 1.634 ms, ONNX w8a16 2.204 ms. Peak memory 44–195 MB.
  - 8 Elite: TFLITE w8a8 0.805 ms, TFLITE float 1.977 ms.
  — [qualcomm/YOLOv11-Detection](https://huggingface.co/qualcomm/YOLOv11-Detection)
- **MobileSAM** (720×1280; encoder 6.95M params / 26.6 MB, decoder 6.16M / 23.7 MB, float; Apache-2.0):
  - 8 Elite Gen 5, encoder: TFLITE float **499 ms**, QNN_DLC 640 ms, ONNX 749 ms. Peak memory up to about 1.4 GB.
  - 8 Elite Gen 5, decoder: TFLITE 2.23 ms.
  - 8 Elite, encoder: TFLITE 538 ms and QNN_DLC 678 ms. An ONNX row reads 74.7 ms, which is inconsistent with the other rows and is probably a data error on the card.
  — [qualcomm/MobileSam](https://huggingface.co/qualcomm/MobileSam)
- **OpenAI-CLIP** (224×224, 150M params, 571 MB float; MIT):
  - 8 Elite Gen 5: TFLITE float **12.8 ms**, QNN_DLC w8a16 11.5 ms. Peak memory about 0.5–0.66 GB.
  - 8 Elite: TFLITE float 17.0 ms.
  — [qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)
- Standalone vision encoders on SM8750 (8 Elite) with QNN INT8 on the NPU vs LiteRT FP16 on the GPU, 336×336 input [I]:
  - ViT-B/16: NPU 14.9 ms, GPU 98.9 ms, CPU 663 ms.
  - Phi-3.5-V CLIP: NPU 103.5 ms, GPU 436 ms.
  - NanoVLM encoder: NPU 14.3 ms.
  - Gemma 3n MobileNetV5: NPU 80.1 ms, GPU 134.6 ms, CPU 1,629.5 ms.
  — [Phase Matters, arXiv 2606.27906 (22 Jun 2026)](https://arxiv.org/html/2606.27906v1)
- Google's LiteRT survey on SM8850 [V]: NPU latency is about 1–20% of CPU latency across 72 models, and 56 or more models run in under 5 ms. — [Google Developers Blog](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)

### Inferences
- The app's truncated ResNet18 patch-feature backbone (11 MB fp32 / 2.9 MB int8) is smaller than the full 44.6 MB ResNet18. On the NPU it should run well under 1 ms per 224-pixel crop [V-based estimate]. The kNN / memory-bank scoring on the CPU is likely to dominate.
- For a live "show off the chipset" pipeline, detector (≈0.7 ms) + backbone (≈0.2–0.6 ms) + optional CLIP (≈12 ms) per frame is well within a 30 fps budget. The NPU could run many models per frame and still sit idle most of the time, so thermals will come from the camera, ISP and display rather than from these models.
- MobileSAM costs about 0.5 s per image. Trigger it only on a flagged defect or on user tap.
- YOLOv11 is AGPL-3.0. That is fine for a hackathon demo but relevant for any later commercial use.

### Gaps
- No AI Hub SM8850 numbers were found for **DINOv2** or **EfficientSAM**. Web search was exhausted before these could be checked.
- The AI Hub numbers are for the Galaxy bin of SM8850. No iQOO 15-specific vision-model latency was found.

## Q4. On-device LLM/VLM options with NPU support: SDKs, published numbers, and which small VLM to use

### Takeaway
- **Only NPU-precompiled VLM for SM8850 found:** **FastVLM-0.5B**, built for LiteRT-LM as `.sm8850.litertlm`. Google [V] reports **TTFT 0.12 s including one 1024×1024 image, 106 tok/s decode, 925 MB RAM**. It is small and weak at reasoning, and it uses Apple's research licence.
- **Stronger option on the GPU, not the NPU:** **Gemma 4 E2B** via LiteRT-LM, at 52 tok/s decode and 0.3 s TTFT on the 8 Elite Gen 5 (S26 Ultra) [V]. Its NPU builds exist only for SM8750 and QCS8275.
- **Qualcomm's own LLM stack (Genie to GenieX):** reaches 30 tok/s for Llama 3.2 3B on 8 Elite Gen 5 [V], but independent users see about 3× less. GenieX (the Nexa SDK successor after Qualcomm acquired Nexa) is a fast-moving developer preview.
- **Industrial anomaly detection:** even GPT-4o reaches only 74.9% on MMAD. Use the VLM for explanations and rules text, not as the defect detector.

### Cited Findings
**Google LiteRT-LM (successor to MediaPipe LLM Inference)**
- `com.google.ai.edge.litertlm:litertlm-android` latest is **0.17.1** (updated 16 Sep 2026). — [Google Maven metadata](https://dl.google.com/android/maven2/com/google/ai/edge/litertlm/litertlm-android/maven-metadata.xml)
- **FastVLM-0.5B** [V] (the device name is garbled on the card as "Xiaomi 14 Pro Max"; the NPU file is SM8850, so it is most likely a Xiaomi 17 Pro Max):
  - NPU `.sm8850.litertlm`, 899 MB, dynamic_int8, context 1280: prefill **11,272 tok/s**, decode **106 tok/s**, **TTFT 0.12 s**, memory **925 MB**.
  - GPU file (1,103 MB): prefill 2,220 tok/s, decode 64 tok/s, TTFT 0.55 s, memory 1,766 MB.
  - "TTFT includes encoding time for 1 image and corresponding text prompt".
  - Licence: "Apple Machine Learning Research Model License Agreement".
  — [litert-community/FastVLM-0.5B](https://huggingface.co/litert-community/FastVLM-0.5B)
- Google's own account of FastVLM-0.5B on the SM8850 NPU [V]: int8 weights with int16 activations, "TTFT in just 0.12 second on high-resolution images (1024x1024)", "over 11,000 tokens/sec for prefill and over 100 tokens/sec for decode". — [Google Developers Blog](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)
- **Gemma 4 E2B** LiteRT-LM [V] (Apache 2.0; 1024 prefill / 256 decode, context 2048):
  - **S26 Ultra (8 Elite Gen 5 for Galaxy) GPU:** prefill 3,808 tok/s, **decode 52.1 tok/s, TTFT 0.3 s**, memory 676 MB, model 2,583 MB.
  - S26 Ultra CPU: 557 / 46.9 tok/s, TTFT 1.8 s, 1,733 MB.
  - The only NPU row is Qualcomm Dragonwing IQ8: 3,747 / 31.7 tok/s, TTFT 0.3 s, 1,869 MB.
  — [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
- NPU files in that repo: `gemma-4-E2B-it_qualcomm_sm8750.litertlm` (3.02 GB) and `..._qualcomm_qcs8275.litertlm` (3.29 GB). **There is no sm8850 file.** — [repo file list](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main)
- Gemma 4 E2B on a Galaxy S25 Ultra (SM8750) [I, 18 May 2026]:
  - NPU: **41.7 tok/s decode, TTFT 92 ms**, peak RSS 1,934 MB.
  - GPU: 24.5 tok/s, TTFT 366 ms, 1,375 MB.
  - "Constrained decoding (topK, topP, temperature) is unsupported on NPU".
  — [dev.to](https://dev.to/jdshah/why-my-llm-runs-4x-faster-on-hardware-i-had-never-heard-of-9b9)
- As of 15 May 2026, Gemma 4 **E4B** had no Qualcomm NPU artefact, and "the Gemma 3 1B Qualcomm build is locked to 1280 tokens". — [mamai issue #58](https://github.com/nmrenyi/mamai/issues/58)
- Google's official precompiled Qualcomm NPU LLM is only Gemma3-1B, for SM8750, SM8650 and SM8550 (4-bit, 1280 context, about 658 MB). SM8850 is not listed. — [LiteRT-LM NPU doc](https://developers.google.com/edge/litert/next/litert_lm_npu)

**Qualcomm Genie, GenieX and Nexa**
- **Llama 3.2 3B Instruct, Genie, w4a16, context 4096** [V]:
  - 8 Elite Gen 5: **30.10 tok/s, TTFT 0.072–2.31 s**.
  - 8 Elite: 28.03 tok/s, TTFT 0.077–2.45 s.
  - No pre-exported assets are available because of the Llama licence; you must compile via AI Hub with an account and token.
  - The card says "Genie support will be deprecated" in favour of GenieX.
  — [qualcomm/Llama-v3.2-3B-Instruct](https://huggingface.co/qualcomm/Llama-v3.2-3B-Instruct)
- **Llama 3.1 8B on 8 Elite (Genie)** [V]: 13.05 tok/s, TTFT 0.155–4.94 s. — [qualcomm/Llama-v3.1-8B-Instruct](https://huggingface.co/qualcomm/Llama-v3.1-8B-Instruct) (from a search-result summary)
- A Qualcomm support-forum question is titled "Llama 3.1 8B on Snapdragon 8 Elite: getting 5.1 tok/s vs published 13.1 tok/s" [U; title only, page not retrievable]. — [Qualcomm forum](https://mysupport.qualcomm.com/supportforums/s/question/0D5dK00000GkGOtSAN/llama-31-8b-on-snapdragon-8-elite-getting-51-toks-vs-published-131-toks-what-configuration-is-needed-to-match-the-benchmark)
- Independent Genie measurement on 8 Elite [I, 23 Mar 2026]:
  - Llama 3.2 3B w4a16 "approximately 10 tokens per second"; a 50–80-token answer takes under 8 s.
  - Llama 3.1 8B about 5 tok/s.
  - The config offers "burst" and "sustained_high_performance" profiles.
  — [Grape Up blog](https://grapeup.com/blog/running-llms-on-device-with-qualcomm-snapdragon-8-elite)
- **GenieX:**
  - Described as "the community version of Qualcomm GENIE", with two runtimes: llama.cpp (GGUF on NPU, GPU or CPU) and QAIRT bundles (NPU only).
  - Platforms include Android on 8 Elite and 8 Elite Gen 5. APIs: CLI, Python, Kotlin/Java and an OpenAI-compatible server. VLMs are supported (examples mention Qwen2.5-VL-7B and Qwen3-VL). Licence BSD-3.
  — [qualcomm/GenieX](https://github.com/qualcomm/GenieX)
  - The Android sample app "downloads the model you select at runtime from an in-app catalog". — [geniex_chat_android](https://github.com/qualcomm/ai-hub-apps/tree/main/apps/geniex_chat_android)
  - `com.qualcomm.qti:geniex-android` has moved quickly through 0.3.17 … 0.4.0, 0.6.1 and **0.7.0** (updated 18 Sep 2026). — [Maven metadata](https://repo1.maven.org/maven2/com/qualcomm/qti/geniex-android/maven-metadata.xml)
  - Qualcomm published a "GenieX developer preview" blog in June 2026. The page body did not render; the heading references NexaSDK. — [Qualcomm blog](https://www.qualcomm.com/developer/blog/2026/06/geniex-developer-preview)
  - An open issue reports a "fixed ~1.5s latency per graphExecute on X Elite (decode 0.64 tok/s vs 17)" [U; laptop chip, from a search snippet]. — [GenieX issue #1266](https://github.com/qualcomm/GenieX/issues/1266)
- **Nexa AI acquired by Qualcomm**:
  - Discussion dated 21 Mar 2026. — [qualcomm/nexa-sdk discussion #1058](https://github.com/qualcomm/nexa-sdk/discussions/1058)
  - AI Hub page: "Nexa AI Is Now Part of Qualcomm AI Hub." — [aihub.qualcomm.com/genai](https://aihub.qualcomm.com/genai)
  - Nexa's Android artifact `ai.nexa:core` was last updated on 26 Feb 2026 (0.0.24). — [Maven metadata](https://repo1.maven.org/maven2/ai/nexa/core/maven-metadata.xml)
- **OmniNeural-4B** (Nexa):
  - "currently runs only on Qualcomm NPUs". Claims "3.5× faster image encoding (vs SigLIP encoder)" [V]. No ms or tok/s numbers are given.
  - Weights are **CC BY-NC 4.0**. NPU use requires a Model Hub token (`nexa config set license`).
  — [NexaAI/OmniNeural-4B](https://huggingface.co/NexaAI/OmniNeural-4B)
- NexaSDK Android lists minSdk 27 and day-0 Qwen3-VL-4B/8B support. — [NexaAI/nexa-sdk](https://github.com/NexaAI/nexa-sdk)
- Qualcomm's Nov 2025 blog on NexaSDK for Android reports Granite 4.0-h-350M at 92 tok/s on the NPU vs 40 tok/s on the CPU on an S25 Ultra (8 Elite) [V/U, from a search-result summary]. — [Qualcomm blog](https://www.qualcomm.com/developer/blog/2025/11/nexa-ai-for-android-simple-way-to-bring-on-device-ai-to-smartphones-with-snapdragon)

**Other engines**
- **MNN** 3.6.1 (Jul 2026):
  - The QNN backend supports the Qwen3 series and VLMs, including "Full inference support for Qwen3-VL".
  - "Qwen3-0.6B (W4 ... block64) reaches 2667 tok/s prefill on Snapdragon 8 Elite (Hexagon v79)" vs 336 tok/s on CPU (7.9×) [V].
  - Supported models include Qwen3.5, Qwen3-VL, Gemma 4, FastVLM, SmolVLM, MiniCPM-4 and LFM. MNN Chat 0.8.3 supports offline inference and multi-image input.
  — [MNN releases](https://github.com/alibaba/MNN/releases)
- **llama.cpp Hexagon (HTP) backend**, official in ggml-org [V/I]:
  - Llama-3.2-1B Q4_0 on HTP: pp128 **169.4 tok/s**, tg64 **51.5 tok/s**. The device is not named in the docs.
  - HTP quant types: Q4_0, Q8_0, MXFP4.
  - Builds via Docker toolchain images. There is a 3.5 GB virtual address space per NPU session.
  — [llama.cpp Snapdragon docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
- **MLC-LLM (GPU/OpenCL, not NPU)** on an S24 Ultra (8 Gen 3), Qwen2.5-1.5B 4-bit [I]:
  - 12.21 tok/s peak falling to 10.38 tok/s sustained.
  - 1.486 W (BatteryManager, display off) and 146.4 mJ/token.
  — [arXiv 2603.23640 (v2, Jun 2026)](https://arxiv.org/html/2603.23640)
- **FastVLM-0.5B via a research QNN port on SM8750** [I]:
  - NPU prefill 197.4 tok/s vs CPU 120.6; decode 113.1 vs 95.5.
  - Executor init: NPU 536 ms vs CPU 83.6 ms.
  - **End-to-end cold 26,820 ms vs warm 2,080 ms** on NPU only. The hybrid split (NPU prefill, CPU decode) takes 6,240 ms cold and 2,640 ms warm.
  - Time to first token ranges from 0.36 to 0.85 s across 24–224 visual tokens.
  
  The prefill figure is about 57× lower than Google's LiteRT-LM number on SM8850, reflecting a different runtime, chip and method. — [Phase Matters, arXiv 2606.27906](https://arxiv.org/html/2606.27906v1)

**VLM quality (<4B) and fitness for defect inspection**
- MiniCPM-V 4.0 (4.1B, Apache-2.0) [V, self-reported OpenCompass]:
  - MiniCPM-V 4.0: OpenCompass 69.0, OCRBench 894, MMMU 51.2.
  - Qwen2.5-VL-3B: 64.5 / 828 / 51.2.
  - InternVL2.5-4B: 65.1 / 820 / 51.8.
  - On-device claim: "less than 2s first token delay and more than 17 token/s decoding on iPhone 16 Pro Max".
  — [openbmb/MiniCPM-V-4](https://huggingface.co/openbmb/MiniCPM-V-4)
- MiniCPM-V 4.6 (**1B**, SigLIP2-400M + Qwen3.5-0.8B, Apache-2.0, 17 May 2026) claims "Qwen3.5 2B-level capability" on OpenCompass, OCRBench and others [V]. Its Android demo ran on a Redmi K70. No chip-specific latency is given. — [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6)
- **Qwen3.5-4B** (Feb 2026, Apache 2.0, natively multimodal with image and video input) [V, self-reported]: MMMU 77.6, OCRBench 85.0, RealWorldQA 79.5. — [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- **MMAD** (ICLR 2025; 39,672 questions over 8,366 industrial images; seven subtasks including defect classification and localisation) [I]: "the average accuracy of GPT-4o models reaching 74.9%", which "falls far short of industrial requirements". — [arXiv 2410.09453](https://arxiv.org/abs/2410.09453)

### Inferences
- **Recommended demo VLM for "maximum NPU" on the iQOO 15:** FastVLM-0.5B `.sm8850.litertlm` via LiteRT-LM Android 0.17.1, called only on flagged frames to produce a one-sentence explanation. Expected latency [V-based]: about 0.12 s TTFT plus 60 tokens at about 106 tok/s, so under 1 s per explanation and under 1 GB RAM.
  - Caveats: 0.5B quality is limited, the Apple licence is research-only, and multimodal NPU use via LiteRT has had unresolved community issues (#5499). Budget time to test early.
- **Recommended for better reasoning or natural-language rules:** Gemma 4 E2B on the **GPU** (≈52 tok/s decode, 0.3 s text TTFT on the same chip family [V]). A 100-token answer takes about 2 s. Present it honestly as GPU in the telemetry panel. If an SM8850 NPU build of Gemma 4 E2B appears, switch to it.
- **Qualcomm-native 3B LLM (Genie/GenieX):**
  - Plan on 10–30 tok/s: 30 is Qualcomm's number and 10 is what independents measured on the 8 Elite. A 100-token rationale would take 3–10 s.
  - Llama needs AI Hub compilation with an account (at build time only).
  - GenieX's Android catalog downloads models at runtime, so models must be pre-fetched or pushed via adb before switching to airplane mode.
  - The SDK's version churn (0.3 to 0.7 in months) and "developer preview" status make it higher risk in a 10-hour window.
- **Avoid NexaSDK and OmniNeural for this build:** the SDK has not been updated since Feb 2026 (post-acquisition), the NPU needs a licence token, and the weights are CC BY-NC.
- **Always pre-warm** (load and run once at app start, or keep caches). The cold vs warm gap (26.8 s vs 2.1 s in the SM8750 study, and JIT 7.5 s vs 0.2 s) would ruin a live demo if hit on stage.
- Given MMAD, the defect decision should come from the vision pipeline (anomaly score or detector). The LLM/VLM should turn those scores, boxes and user-written inspection rules into explanations, and can optionally be asked yes/no questions about a crop.
- Rule-checking in natural language could run text-only, with the anomaly-map summary sent as structured text. This avoids image-token prefill and works with any of the LLMs above.

### Gaps
- No published SM8850 NPU latency was found for Qwen3-VL-2B/4B, Qwen2.5-VL-3B, InternVL3-1B/2B, SmolVLM2, MiniCPM-V 4.x or Gemma 3n vision. MNN and GenieX support some of these, but no phone numbers were found.
- PowerServe was not researched because the web-search budget was exhausted.
- Whether GenieX or AI Hub offers a precompiled VLM bundle for SM8850 is unknown.
- The Nexa NPU licence-token activation flow (whether it needs internet) is unknown.

## Q5. Thermal and power behaviour under sustained inference, and strategies to avoid overheating

### Takeaway
The iQOO 15 throttles hard under sustained CPU and GPU stress: it falls below 50% of peak after holding clocks for about the first 10 minutes. Independent SM8750 data shows NPU inference runs about **10 °C cooler with 2.5× less energy per request** than CPU inference. The safest way to show off the chipset is to put all per-frame models on the NPU, cap frame rate, run LLM/VLM calls only on demand, and adapt to Android's thermal headroom.

### Cited Findings
- iQOO 15 (GSMArena, Nov 2025) [I]:
  - In the stress test the CPU "dipped below 50% of its theoretical performance", though "it maintained solid clock speeds in the first 10 minutes". The GPU also "dipped below 50%".
  - The reviewer says the phone "seems to struggle with maintaining high performance in the CPU and GPU stress tests". Heat concentrates in the side frame; the back panel stays cool.
  — [GSMArena review](https://www.gsmarena.com/iqoo_15-review-2905p4.php)
- A search-result snippet [U] reports that the iQOO 15 retains 51.7% of peak in a "Burnout CPU Throttle test". The originating page was not verified; it is possibly the vendor-run iQOO community forum. — [iQOO community thread](https://community.iqoo.com/in/thread/128910)
- SM8750, 100-run stability test with FastVLM, NPU (QNN INT8/W4A8) vs CPU (LiteRT XNNPACK FP16) [I]:
  - NPU 40.8 ± 3.03 °C vs CPU 51.3 ± 3.62 °C, 10.47 °C cooler.
  - The NPU delivers "2.52× lower energy per request".
  - CPU paths triggered governor scaling at 82 °C, while the NPU stayed stable at 66 °C.
  — [Phase Matters, arXiv 2606.27906](https://arxiv.org/html/2606.27906v1)
- Sustained LLM on a phone GPU, S24 Ultra (8 Gen 3), MLC-LLM OpenCL, Qwen2.5-1.5B 4-bit [I]:
  - Throughput fell from 12.21 to 10.38 tok/s (15% drop, plateau by iteration 8), with peak GPU temperature 68.5 °C, 1.486 W, and 146.4 mJ/token.
  - Reaching 20 sustained iterations required "context window set to 2,048 tokens, prefill chunk to 128 tokens, display held off".
  - For comparison, an iPhone 16 Pro dropped 41.5%.
  - The authors conclude that "thermal management supersedes peak compute as the primary constraint".
  — [arXiv 2603.23640](https://arxiv.org/html/2603.23640)
- Snapdragon X Elite, a laptop chip [U, search snippet]: Adreno OpenCL offload used "1.7× more system energy than the NPU", and 6.5× more on a RAG workload. — [arXiv 2606.11257](https://arxiv.org/html/2606.11257v1)
- The SM8850 NPU's performance per watt is up "only 16%" vs the previous generation, while peak is up 37% [V, as reported]. — [XPU.pub](https://xpu.pub/2025/10/06/qualcomm-snapdragon-8-elite-gen-5/)
- Power-mode controls:
  - The ORT QNN EP `htp_performance_mode` offers `burst`, `sustained_high_performance`, `balanced`, `power_saver`, `low_power_saver` and others. — [ORT QNN EP docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
  - Genie configs offer "burst" vs "sustained_high_performance" profiles. — [Grape Up](https://grapeup.com/blog/running-llms-on-device-with-qualcomm-snapdragon-8-elite)
- Android guidance: when approaching unsafe thermal states, apps should "reduce the frame rate, lower fidelity", monitor thermal state, and "proactively adjust" the workload. — [Android ADPF thermal guide](https://developer.android.com/games/optimize/adpf/thermal)

### Inferences
- **Suggested thermal policy for the demo:**
  1. Run camera analysis at a capped rate, for example 10–15 fps while inspecting and 1–2 fps (or paused) when idle or when the scene is unchanged.
  2. Put detector and backbone on the NPU with the `sustained_high_performance` or `balanced` HTP mode. Use `burst` only for one-shot "inspect this part" button presses and the VLM call.
  3. Call the LLM/VLM only per flagged defect or per user request, never per frame.
  4. Poll thermal headroom every 10 s or more. Above about 0.7 headroom, or at thermal status MODERATE or higher, step down fps and HTP mode. At SEVERE, pause the VLM.
  5. Keep screen brightness moderate. The display is a large share of power, and the one phone-power study held the display off.
- Because this workload's NPU inference is sub-millisecond, most heat will come from camera, ISP and display, plus any CPU-side preprocessing (resize, normalise, kNN). Do preprocessing on the GPU or NPU where possible, or at low resolution.
- The "10 minutes before throttling" observation suggests a 3–5 minute stage demo is unlikely to throttle if duty-cycled. Pre-cool the phone and remove the case before presenting.

### Gaps
- No independent measurement of **sustained NPU** inference on SM8850 or the iQOO 15 was found (all NPU thermal data is SM8750).
- No per-rail NPU power figure in watts was found for any Snapdragon 8-series phone. Qualcomm does not publish one, and the available studies report energy ratios or whole-phone power.

## Q6. Proving NPU usage and efficiency live: in-app telemetry and profiling hooks

### Takeaway
Show three kinds of evidence together:
1. **Which accelerator ran.** ORT's `disable_cpu_ep_fallback` gives a hard guarantee, and LiteRT reports fallback status (the diagnostics are exposed by the flutter_litert wrapper).
2. **Per-stage wall-clock latency with a live CPU-vs-NPU A/B toggle.** Expect roughly 5–100× differences.
3. **Android system telemetry.** PowerManager gives thermal status and headroom. BatteryManager gives battery temperature, current and voltage, from which power ≈ V × I can be computed. Remember that headroom can be polled at most once every 10 s and that battery current is whole-phone.

### Cited Findings
- `PowerManager.getThermalHeadroom(forecastSeconds)`:
  - Returns "0.0f (no throttling, THERMAL_STATUS_NONE) to 1.0f (heavy throttling, THERMAL_STATUS_SEVERE)".
  - "You shouldn't call it more than once every 10 seconds", otherwise it returns NaN. "If the initial value ... is NaN, the API is not available on the device". Avoid calling it from multiple threads.
  - Available from Android 11 (API 30); NDK version from API 31.
  - `getThermalHeadroomThresholds()` was added in Android 15.
  - `addThermalStatusListener` / `OnThermalStatusChangedListener` deliver status changes.
  — [Android ADPF thermal guide](https://developer.android.com/games/optimize/adpf/thermal)
- AOSP `PowerManager`: the `THERMAL_STATUS_*` constants run NONE → LIGHT → MODERATE → SEVERE → CRITICAL → EMERGENCY → SHUTDOWN. There is also a flagged `addThermalHeadroomListener(...)` (`@FlaggedApi(FLAG_ALLOW_THERMAL_THRESHOLDS_CALLBACK)`). — [PowerManager.java (AOSP main)](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/core/java/android/os/PowerManager.java)
- `BatteryManager`:
  - `BATTERY_PROPERTY_CURRENT_NOW` is "Instantaneous battery current in microamperes ... Positive values indicate net current entering the battery ... negative values indicate net current discharging". `CURRENT_AVERAGE` is the same but averaged.
  - `ENERGY_COUNTER` gives "remaining energy in nanowatt-hours". `CHARGE_COUNTER` is in µAh.
  - If a property is unsupported, `getIntProperty` returns `Integer.MIN_VALUE` for targetSdk ≥ P, and `getLongProperty` returns `Long.MIN_VALUE`.
  - `EXTRA_TEMPERATURE` and `EXTRA_VOLTAGE` are extras on the sticky `ACTION_BATTERY_CHANGED` intent.
  — [BatteryManager.java (AOSP)](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/core/java/android/os/BatteryManager.java)
- Units behind those extras (Health HAL): "Instantaneous battery temperature in tenths of degrees Celsius" and voltage "in millivolts (mV)". — [HealthInfo.aidl](https://android.googlesource.com/platform/hardware/interfaces/+/refs/heads/main/health/aidl/android/hardware/health/HealthInfo.aidl)
- An academic study measured phone power through BatteryManager with the display off: S24 Ultra, 1.486 W during LLM decode [I]. — [arXiv 2603.23640](https://arxiv.org/html/2603.23640)
- ORT QNN EP hooks:
  - `session.disable_cpu_ep_fallback=1` throws if any part cannot run on HTP.
  - `profiling_level` = `basic`, `detailed` or `optrace`, with `profiling_file_path` for a CSV.
  - `ep.context_enable` caches the compiled graph.
  — [ORT QNN EP docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- The Whisper-on-SM8850 example logs the provider as "QNNExecutionProvider" per stage (encoder 22.0–33.7 ms) [I]. — [itsallgoody repo](https://github.com/itsallgoody/onnxruntime-qnn-snapdragon-8-elite-gen5)
- LiteRT:
  - `CompiledModel.Options(Accelerator.NPU, Accelerator.GPU)` falls back to GPU if the NPU is unavailable. — [LiteRT NPU doc](https://developers.google.com/edge/litert/next/npu)
  - The flutter_litert wrapper (over LiteRT Next 2.2.0) exposes `accelerators`, `isFullyAccelerated`, `requestedAccelerators` and `didFallback` [I]. — [flutter_litert changelog](https://pub.dev/packages/flutter_litert/changelog)
- AI Hub model cards report the "compute unit" (NPU) per model, runtime and precision. This is a pre-demo evidence source that uses AI Hub cloud profiling and requires an account. — [qualcomm/ResNet18](https://huggingface.co/qualcomm/ResNet18)

### Inferences
- **Suggested telemetry panel:**
  - Per-frame stage timings (preprocess, detector, backbone, scoring, VLM TTFT and tok/s) measured with `SystemClock.elapsedRealtimeNanos()` around each call.
  - An "Accelerator" badge per stage. Show NPU only when the session was created with NPU requested and (ORT) CPU fallback disabled, or (LiteRT) the model is reported fully accelerated. Otherwise show "GPU fallback" or "CPU".
  - A live A/B toggle that re-runs the same frame on the CPU (XNNPACK, via the existing fast-tflite path) to show the speedup.
  - Thermal status (listener) and headroom (polled at most every 10 s).
  - Battery temperature in °C (the extra ÷ 10).
  - Estimated power in W, computed as |CURRENT_NOW µA| × voltage mV ÷ 10⁹. It is valid only when unplugged, is whole-phone, and should be smoothed over several seconds.
  - Energy per inspection, from the ENERGY_COUNTER or CHARGE_COUNTER delta where supported.
- To attribute power to the NPU, show the delta vs an idle baseline (camera on, inference off), and the CPU-vs-NPU power delta during the A/B toggle. Do not claim a per-NPU wattage: Android exposes only battery-level current.
- Check the sign and units of CURRENT_NOW on the actual iQOO 15 before showing numbers. Some OEM kernels deviate from the µA convention; this is a known ecosystem quirk that I could not source here.

### Gaps
- There is no public Android API for per-IP (NPU, GPU, CPU) power or utilisation. Qualcomm's on-device profilers were not researched: Snapdragon Profiler is a desktop tool, and whether QNN's profiling API can be read at runtime from an app is unverified.
- The official LiteRT Kotlin API for accelerator/fallback introspection is not documented on Google's pages; it is confirmed only through the third-party wrapper.
- Whether `getThermalHeadroom` is implemented on OriginOS 6 / iQOO 15 (as opposed to returning NaN) is unverified. Test it on the device.
