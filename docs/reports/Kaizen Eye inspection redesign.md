# Rebuild Kaizen Eye around a Visual Twin

Kaizen Eye should stop competing on its algorithm, because learning "good" from about 20 photos is what Siemens Inspekto, Keyence, Zebra, LandingLens and Edge Impulse already sell. It should instead win on execution that nobody ships and that judges can verify. The proposal is a native Kotlin app that learns a part from a **10–15 second handheld video**, locks onto that exact object, prints its own **statistically bounded false-alarm certificate**, judges every piece on a live hand-fed line with a beep, and explains each reject using a vision-language model running offline on the iQOO 15's Hexagon NPU. The round-1 failures (different objects passing, twenty carefully aligned photos) come from scoring background patches with mid-level ResNet18 texture features inside a fixed centre crop. The project's own benchmark shows that changing clutter **roughly triples the threshold and drops detection to a few percent**, so masking, crop normalisation and a semantic identity gate matter more than a bigger model. Moving from ImageNet-CNN PatchCore features to training-free DINOv2 ViT-S patch matching lifts published one-shot MVTec AD accuracy from **83.4 to 96.5 image AUROC**. Its encoder class runs in about **12–18 ms** on this NPU, so once scoring moves off the frame loop, the camera, not the compute, sets the speed limit. 3D enrolment (NeRF, Gaussian splatting, CAD) cannot be built on Android in ten hours and falls to near-chance with few views. A multi-view memory bank built from video captures most of the benefit. Native Kotlin is the only stack with a documented route to the NPU; React Native's TFLite binding offers only GPU and deprecated NNAPI. The business case should target Tier-2/3 auto-component MSMEs, where a **₹72,999 phone** replaces **₹1.65–11 lakh** inspection stations and the money is in customer PPM scores, not inspector wages. Every "Y" in the pitch must be measured on the phone. The report ends with the architecture, an hour-by-hour plan with cut lines, the claim sheet, a demo script and the risk register.

## Twenty good photos is the industry norm, so novelty must come from phone-native execution

The round-1 judges were right: Kaizen Eye's algorithm is the industry default. Siemens Inspekto learns from **"20 good samples"** and "no defective samples" ([Siemens](https://www.siemens.com/en-us/company/artificial-intelligence/industrial-ai/inspekto-ai-inspection/)). Keyence's VS smart camera learns "good" from an average of 20 images ([SDC Automation](https://sdcautomation.com/blog/keyence-ai-vision-systems-an-sdc-supplier-spotlight/)), and Zebra's anomaly tools need "just 20-30 image samples" ([Zebra](https://www.zebra.com/gb/en/about-zebra/newsroom/press-releases/2024/zebra-technologies-adds-new-deep-learning-tools-to-aurora-machine-vision-software.html)). LandingLens recommends 50 normal images plus at least one abnormal image to set its threshold ([LandingLens](https://landinglens.docs.landing.ai/anomaly-detection)). Edge Impulse has shipped **PatchCore scoring, the exact family Kaizen Eye uses, since October 2024** ([Edge Impulse](https://www.edgeimpulse.com/blog/patchcore-boosts-visual-anomaly-detection-in-edge-impulse/)).

The 2025–26 frontier has moved on to four things:

| Frontier feature | Example |
|---|---|
| Continual learning | HALCON 25.11 adds classes "without the need for complete retraining" on a standard CPU ([MVTec](https://www.mvtec.com/company/press-room/press-releases/detail/mvtec-introduces-new-deep-learning-feature-continual-learning-in-halcon-2511)) |
| Learning by watching the line | Elementary builds models "in under 60 seconds" ([Elementary](https://www.elementaryml.com/)) |
| Synthetic defects | UnitX GenX, generated from three real samples ([A3](https://www.automate.org/news/unitx-unveils-genx-a-generative-ai-breakthrough-for-industrial-inspection-unitx-labs)) |
| VLM agents | Omron with NVIDIA's 7B Cosmos Reason, on GPU stacks ([ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2026/07/16/omron-advances-inspection-technology-with-nvidia-omniverse-and-metropolis/27914/)) |

None of these runs on a phone NPU. The competitive scan, which was limited, found **no phone-native, offline inspection app that learns from good parts only**. The nearest open live-camera prototype runs at 20 ms per frame on a Jetson ([anomalib #3220](https://github.com/open-edge-platform/anomalib/discussions/3220)). The offline phone apps that do exist run supervised YOLO detectors ([Ultralytics HUB](https://github.com/ultralytics/ultralytics/blob/main/docs/en/hub/app/index.md)).

The incumbents' strengths come mostly from imaging hardware and fixed stations, not from smarter models. Keyence's IV4 tests **"over 10,000 conditions"** to choose image settings, and its "AI Target Extraction" removes background "sources of false detection" ([Keyence](https://www.keyence.com/products/vision/vision-sensor/iv4/)). UnitX uses 32 independently controlled lights and captures up to 100 MP ([UnitX](https://www.unitxlabs.com/industry/bearing-machining-inspection-application/)). Inspekto's original setup had the user trace the item outline at a fixed station ([automation.com](https://www.automation.com/article/inspekto-releases-s70-autonomous-machine-vision-sy)). A handheld phone has none of this, so it must recreate three guarantees in software: the object is isolated from its surroundings, lighting is held constant, and the pose seen at inspection was already seen at enrolment.

| System | How it works | Why it is strong | Weakness and root cause |
|---|---|---|---|
| Siemens Inspekto | Good-only anomaly model on a bundled industrial PC, camera and light; 20 good parts | Installed in 30–60 min by shop-floor staff; no defect images ([automation.com](https://www.automation.com/article/inspekto-releases-s70-autonomous-machine-vision-sy)) | Fixed station and traced region; about €10,000 in 2019 ([ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/)). The model knows only one pose and one background |
| Keyence IV4 / VS | Optics search plus target extraction; one registered image (IV4), about 20 good images (VS) | Hardware removes clutter before the model runs | ₹1.65–5 lakh per sensor in India before lighting and integration ([IndiaMART IV3](https://www.indiamart.com/proddetail/keyence-vision-sensor-with-built-in-ai-iv3-series-2854380953255.html), [IndiaMART IV4](https://www.indiamart.com/proddetail/keyence-iv4-vision-sensor-camera-with-built-in-ai-2857001721891.html)) |
| Cognex In-Sight and OneVision | Edge learning from 5–10 labelled images ([Cognex](https://www.cognex.com/en-se/products/machine-vision/2d-machine-vision-systems/in-sight-3800)); OneVision trains in the cloud and inspects on the camera | In-Sight 3900 reportedly reaches 5,000 parts/min, on Qualcomm silicon ([Control.com](https://control.com/news/cognex-launches-a-high-speed-ai-powered-machine-vision-system/)) | Needs labels; training and governance live in the cloud ([PR Newswire](https://www.prnewswire.com/news-releases/cognex-onevision-adoption-ramps-as-manufacturers-scale-ai-vision-globally-302768368.html)) |
| LandingLens anomaly detection | 160M-parameter reverse distillation; 50 normal images plus at least 1 abnormal | No-code, one-button training | Warns that different backgrounds may be flagged; needs a real defect to set the threshold; 1 credit per inference ([plans](https://landinglens.docs.landing.ai/plans)) |
| MVTec HALCON | Good-only anomaly detection, plus Global Context AD for missing or misplaced parts ([MVTec](https://www.mvtec.com/knowledge-base/technologies/deep-learning/methods/anomaly-detection)) | Handles logical anomalies | PC software licence and integrator skills |
| Elementary | Watches the line and builds a model in under 60 s; claims "up to 99.9%" accuracy at 1,000 parts/min | Learns from the live stream | The accuracy claim has no public protocol; data is stored in AWS ([Elementary](https://www.elementaryml.com/)) |
| UnitX | 32-light imaging, supervised deep learning, synthetic defects | Solves defect contrast and scarce defect data | Needs real defect samples and turnkey hardware |
| Omron + NVIDIA VSS | 7B VLM plus LLM agents that reason about root causes | Explanations | GPU or cloud stack. VLMs are weak detectors: GPT-4o scores 74.9% on MMAD ([MMAD](https://arxiv.org/abs/2410.09453)) |
| Hyperscaler services | Cloud anomaly detection | — | AWS Lookout for Vision reached end of life on 31 Oct 2025 ([AWS](https://docs.aws.amazon.com/lookout-for-vision/latest/developer-guide/su-awscli-sdk.html)). Azure Percept was retired on 30 Mar 2023 ([Microsoft](https://learn.microsoft.com/en-us/previous-versions/azure/azure-percept/retirement-of-azure-percept-dk)). Custom Vision retires on 25 Sep 2028 ([Microsoft](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/migration-options)) |

Nearly every weakness traces to two technical facts. Patch-level one-class models treat the image as a **bag of local patches** compared against an enrolment set captured under one pose, one light and one background, and their scores are **uncalibrated distances** that carry no meaning and no stated operating point. The next table maps each weakness to the phone-side fix that the rest of this report specifies. Because the fixes target root causes rather than symptoms, each one also answers a specific round-1 criticism.

| Competitor weakness | Root cause | Phone-side fix in Kaizen Eye |
|---|---|---|
| Needs a fixed mount, background and pose | The memory bank holds only patches from the enrolment pose and background. Pose-naive PatchCore scores 78.5 image AUROC against 93.9 for a pose-aware method ([PAD](https://github.com/EricLee0224/PAD), [SplatPose](https://arxiv.org/html/2404.06832)) | Object mask, crop normalised to the object's box, multi-view video memory bank with nearest-view retrieval |
| A different object passes | kNN over patches has no global identity or shape check | Semantic identity gate, mask-geometry check and patch-coverage test, giving three verdicts |
| Missing or misplaced components pass | Each local patch still looks normal. AnomalyDINO scores 50.2 AUROC on "cable swap" ([AnomalyDINO](https://arxiv.org/html/2405.14529)) | Deterministic count gate; VLM checklist as a stretch goal |
| Small defects are missed | 224–320 px inputs dilute defects. On MVTec AD 2, AU-PRO₀.₀₅ is below 31% at 256 px ([MVTec AD 2](https://arxiv.org/abs/2503.21622)) | Object crop from the high-resolution frame at 448 px, with optional tiling |
| False alarms when lighting changes | Features are not illumination-invariant, and auto-exposure drifts | Torch on, auto-exposure/focus/white-balance locked, teach under the same light, drift alarm |
| Real defects needed to set thresholds; unverifiable "99%" claims | Distances are uncalibrated | Threshold from good parts only (an order statistic), printed with its false-alarm bound |
| No explanation | Scores carry no meaning. Ungrounded VLM reports hallucinate 65% of the time ([arXiv 2605.26533](https://arxiv.org/abs/2605.26533)) | On-device VLM explanation grounded in detector outputs (4% hallucination when grounded) |
| Retraining for each product or nuisance alarm | Model weights must be re-optimised | Re-teach by video in seconds; one-tap memory-bank correction, which closed a median 66% of the performance gap in 2026 research ([arXiv 2608.17775](https://arxiv.org/abs/2608.17775)) |
| Cloud dependence and vendor sunsets | Training and analytics are sold as SaaS | 100% on-device; works in airplane mode |

### The headline feature is a ten-second Visual Twin that certifies itself

The operator films a good part for ten seconds with the phone in hand, and Kaizen Eye turns the clip into a **Visual Twin**: 16–32 sharp, diverse keyframes with the background masked out, their patch features, a semantic fingerprint of the part and a calibrated threshold. The phone then prints a **certificate** for that part stating how many independent good presentations back the threshold, the false-alarm bound this implies at 95% confidence, the identity margin separating this part from other objects, and a self-test result on synthetic scratches. From then on the same phone watches a live, hand-fed line. It judges each piece once as it crosses a virtual photo-eye, beeps and keeps a red box on a defective piece as it moves, rejects anything that is not the enrolled part, and explains each reject in words using a vision-language model on the NPU, in airplane mode. "Visual Twin" describes a multi-view 2D reference that behaves like a lightweight 3D model without any 3D reconstruction. That framing is defensible given the enrolment evidence in the next section.

Each ingredient exists somewhere (Elementary learns by watching the line, Neurala and HALCON learn instantly, anomalib builds synthetic validation sets, Omron runs VLM agents), but only on fixed industrial hardware, GPUs or the cloud. The claim to judges should therefore be exactly this: **the first fully offline, handheld implementation of the combination on a phone NPU**. The certificate turns criterion 3 ("give us exact numbers") into a product feature, and sets it against competitors' unprotocolled "up to 99.9%" ([Elementary](https://www.elementaryml.com/)) and "99%+" ([Averroes](https://averroes.ai/)) claims. Live conveyor mode and NPU telemetry are table stakes next to Keyence's AI trigger and Elementary's 1,000 parts/min, so they should be demonstrated but not pitched as novel.

## Locking onto the object, not a bigger model, fixes what the judges saw

### Background patches let different objects pass

The current app has two checks, and background dominates both ([Kaizen Eye README](file:///D:/Projects/Kaizen_Eye/README.md)). The patch check takes a fixed centre crop and extracts 1,600 ResNet18 patch vectors at 320 px. It then takes the 3×3-smoothed maximum patch distance and ignores the outer 10% border. The whole-image check compares the crop's mean patch vector against the enrolment frames.

Patch kNN has no notion of the whole object. It asks whether each local patch resembles *some* normal patch, not whether the ensemble looks like the enrolled part. So a different object placed on the same table finds close neighbours for most of its patches, and the table patches match perfectly. The whole-image check averages the same background-heavy, mid-level texture features, so its global vector barely moves either.

The project's own handheld benchmark measures the damage: with changing clutter at the frame edges, **the threshold roughly triples and detection falls to a few percent, "whatever the model"**, and framing jitter of ±10% costs about 8 AUROC points ([README](file:///D:/Projects/Kaizen_Eye/README.md)). That is the "does not lock onto the object" critique, already quantified.

The fix changes the pipeline in three steps. First, **segment the part**. On the live line a threshold against a plain contrasting sheet gives the mask for free (see the live-line section). On arbitrary backgrounds, AnomalyDINO's zero-shot mask comes from the same forward pass by thresholding the first PCA component of the DINOv2 patch features, then dilating and closing; its safeguard of running a "masking test" and skipping the mask when it fails should be copied too ([AnomalyDINO](https://arxiv.org/html/2405.14529)). Second, **crop to the object**: take the mask's bounding box plus about 10% margin and resize to a fixed side, which removes translation, scale and framing variation. Third, **score object patches only**: build the memory bank from object patches and score with the mean of the top 1% of patch distances, the default in both AnomalyDINO and SubspaceAD ([SubspaceAD](https://arxiv.org/html/2602.23013v3)).

On clean benchmark objects, masking plus rotation adds only about 2 AUROC points at no significant runtime cost ([AnomalyDINO](https://arxiv.org/html/2405.14529)). On cluttered handheld captures the gain should be much larger, because the baseline failure is much worse. That expectation still has to be measured.

For arbitrary backgrounds, three segmenters are available on this chip. **FastSAM-S** runs in 3.1 ms on the Snapdragon 8 Elite Gen 5 NPU, model only ([qualcomm/FastSam-S](https://huggingface.co/qualcomm/FastSam-S)). **MediaPipe's Interactive Segmenter** takes a tap or centre-point prompt and runs in 208 ms on a Pixel 10 CPU, which is fine for teach keyframes but not for every frame ([MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/interactive_segmenter)). **MobileSAM's encoder** takes about 0.5 s even on the NPU and should be avoided ([qualcomm/MobileSam](https://huggingface.co/qualcomm/MobileSam)).

Rejecting a wrong object is semantic novelty detection, which pretrained features handle well. kNN on lightly adapted ImageNet-pretrained features reached **96.2% one-class AUROC on CIFAR-10** ([PANDA](https://arxiv.org/pdf/2010.05903)). Kaizen Eye should embed the masked, normalised crop semantically. The first choice is the **DINOv2 [CLS] token** if the DINOv2 backbone lands, since DuoAD describes [CLS] as capturing "holistic object semantics" while staying "largely insensitive to local anomalous regions" ([DuoAD](https://arxiv.org/html/2607.23924v1)). The fallbacks are **MediaPipe's MobileNetV3 embedder** at 3.9 ms on a Pixel 6 CPU ([MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/image_embedder)) and **CLIP ViT-B/16** at about 13 ms on this NPU ([qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)).

The identity score is the maximum cosine similarity to the Twin's keyframes. Its threshold is set at teach time between two measured distributions: leave-segment-out similarities among the part's own keyframes (positives), and the similarities of 50–100 bundled embeddings of hands, pens, phones, tabletops and other parts to the Twin (negatives). The gap between them is the **identity margin** printed on the certificate. Two cheap checks back up the gate: mask area, aspect ratio and Hu moments must fall within ±3σ of the Twin's values, and a patch-coverage test separates a wrong object, which lights up most of its patches, from a defect, which lights up a cluster. The app then returns four outcomes: **PASS, DEFECT, NOT THE ENROLLED PART, REFRAME**. That is the direct answer to the round-1 demo.

One limit must be stated honestly. PANDA measures novelty between object classes. Two look-alike parts, such as two black mice, are an instance-level problem, and no published margins exist for it. The demo should show the measured margin rather than promise zero look-alike errors.

### A 10–15 second video replaces twenty photos; 3D capture does not fit ten hours

Video enrolment needs no new research, only standard pieces in the right order. Frames come straight from the CameraX analysis stream at 5–10 fps for 10–15 s, giving 60–150 candidates with no MP4 encode or decode. The blurriest 30–50% are dropped by variance of the Laplacian, ranked within each clip because absolute thresholds must be tuned per dataset ([PyImageSearch](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)). The survivors are embedded, and greedy farthest-first (k-center) selection picks 16–32 keyframes, the same greedy logic PatchCore uses to build its coreset ([PatchCore](https://arxiv.org/html/2106.08265v2)). The shot curve supports this range. AnomalyDINO scores **96.6, 96.9, 97.7, 98.2 and 98.4** image AUROC at 1, 2, 4, 8 and 16 shots on MVTec AD, so gains flatten after about eight aligned views ([AnomalyDINO](https://arxiv.org/html/2405.14529)). Handheld jitter justifies going to 16–32. A coverage meter in the UI fills as diverse keyframes accumulate.

The operator should film a slow ±20–30° arc, turn the part once in-plane, and slide it across the sheet under the torch. Real views then supply rotation and shading variety. The project already measured synthetic enrolment augmentation as unhelpful for its ResNet18 pipeline ([README](file:///D:/Projects/Kaizen_Eye/README.md)).

A varied bank can become too permissive. To prevent that, test-time scoring should first retrieve the three keyframes nearest the query by global embedding, and then run patch kNN only against those. This mirrors the "estimate the pose, then compare with a pose-matched reference" logic of OmniposeAD and SplatPose, without any 3D, and it also shrinks the kNN work. The pose problem is real: on the MAD benchmark, pose-naive PatchCore scores **78.5 image AUROC against 90.9 for OmniposeAD and 93.9 for SplatPose** ([PAD](https://github.com/EricLee0224/PAD), [SplatPose](https://arxiv.org/html/2404.06832)).

3D enrolment fails on data, compute and platform. On data, SplatPose used 210 posed training views per category, and in PADFormer's 4-shot table it drops to **52.2 image AUROC, near chance** ([PADFormer](https://arxiv.org/html/2608.04210)). On compute, SplatPose trains in about 5 minutes and needs about 5 s per image on a GPU; PocketGS trains a splat on an iPhone 15 in 255–389 s from 50 images with 2.21 GB peak memory ([PocketGS](https://arxiv.org/html/2601.17354v3)); and Scaniverse's roughly one-minute on-device training is iOS-only ([80.lv](https://80.lv/articles/capture-photorealistic-3d-scenes-on-your-phone-with-gaussian-splatting)). On platform, no Android on-device Gaussian-splatting pipeline was found, cloud capture apps break the offline rule, and no CAD model will exist for whatever a judge pulls out of a bag.

| Enrolment route | Views needed | Compute | Fits a 10 h offline Android build? | Verdict |
|---|---|---|---|---|
| 20 aligned photos (current) | 20 | Trivial | Yes | Slow, and breaks with pose and clutter |
| **Handheld video → keyframes** | 16–32 picked from 10–15 s | Seconds on the NPU | **Yes** | **Build** |
| NeRF / 3D Gaussian splatting (OmniposeAD, SplatPose) | About 200 posed views | Minutes to hours of GPU training; seconds per query | No | Skip |
| Sparse-view reconstruction (PADFormer) | 2–10 | 0.74–1.75 s per image on an A10 GPU, plus a trained ViT | No | Skip |
| CAD rendering | A CAD file | Offline rendering | No CAD for ad-hoc parts | Skip |

### Thresholds must come from independent good parts, not neighbouring frames

Video creates a trap for the current threshold method. Adjacent frames are near-duplicates, so plain leave-one-out leaves a near-identical neighbour in the bank. Normal scores then come out too low, the threshold ends up too tight, and good parts get rejected. The Twin must instead use **leave-segment-out**: exclude a ±1–2 s window, or split the clip into 4–5 blocks and score each against the others.

The cleanest operating threshold is distribution-free. Set τ as an order statistic of m normal scores from independent units, which gives Pr[FPR ≤ α] ≥ 1−δ. At 95% confidence the required counts are ([arXiv 2608.15090](https://arxiv.org/html/2608.15090)):

| False-alarm bound (α) | Independent good units needed |
|---|---|
| 10% | 29 |
| 5% | 59 |
| 2% | 149 |
| 1% | 299 |

Correlated images count only once per independent unit. By the same rule, today's leave-one-out maximum over about 20 photos bounds false alarms at only about **13.9%** at 95% confidence, because 0.861²⁰ ≈ 0.05.

In practice the teach clip builds the Twin, a separate "calibrate" pass of 29 presentations of *distinct* good parts sets τ (a pack of identical M8 nuts or washers makes this easy), and the certificate prints m and the resulting bound. With fewer parts, SuperADD's recipe is a sane default: the 95th percentile of held-out normals times a 1.3–1.5 gain ([SuperADD](https://arxiv.org/html/2605.14808v1)).

Synthetic defects need a caveat. CutPaste reaches 96.6 image AUC as a *training* signal ([CutPaste](https://arxiv.org/abs/2104.04015)). As a proxy for choosing models on industrial data, however, synthetic anomalies picked the best model only **2 of 15 times on MVTec AD and 0 of 12 on VisA** ([SWSA](https://arxiv.org/html/2310.10461)). The synthetic scratches in the certificate therefore prove only that the pipeline sees defects inside the mask; they are not an accuracy claim.

On the live line, voting across two or three crops of the same tracked part (k-of-n) cuts per-part false alarms without loosening τ.

## The live line runs at camera speed once scoring leaves the frame loop

### Detection is nearly free on this chip; per-part scoring is the bottleneck

Nano detectors no longer limit a phone pipeline. Ultralytics measured a full YOLO26n call on a Snapdragon 8 Elite Gen 5 phone, including pre- and post-processing, at **10.7 ms on the NPU, 15.8 ms on the GPU and 52.2 ms on the CPU** ([Ultralytics QNN](https://docs.ultralytics.com/integrations/qnn), [Ultralytics LiteRT](https://docs.ultralytics.com/integrations/litert)). Qualcomm AI Hub lists the same model at about 1.9 ms, model only ([qualcomm/YOLO26-Detection](https://huggingface.co/qualcomm/YOLO26-Detection)). Every latency budget should therefore assume a **3–5× gap between AI Hub figures and in-app reality**.

The camera path sets two further limits. CameraX asks analyzers to finish in under about 32 ms at 30 fps and drops frames when they run late ([CameraX](https://developer.android.com/media/camera/camerax/analyze)). The camera-to-display path alone took about 80 ± 25 ms on a Pixel 2 ([inovex](https://www.inovex.de/de/blog/the-glass-to-glass-latency-on-android/)).

Scoring is where the time goes. By the team's own figures, today's app spends 40–90 ms on the embedding and 100–350 ms on the JavaScript kNN for each photo. Run serially, that caps throughput at **136–430 parts per minute** (60,000 ms ÷ scoring ms), and 180–480 ms pass between a part crossing the line and its verdict. At 50–100 ms per part, the ceiling rises to **600–1,200 parts per minute**, and event latency falls to roughly 90–150 ms plus audio, which feels instant to an operator.

The design therefore runs two loops at two rates. A **fast loop** runs on every frame in under 10 ms: downscale, find parts, track them, test the trigger and draw the overlay. A **slow path** runs once per part, asynchronously: crop the part from the high-resolution frame, run the full Twin judgement, and attach the verdict to the part's track.

The slow path needs its own first-in-first-out queue, because frame-processor runners drop work when busy. VisionCamera's AsyncRunner "will immediately return false and not schedule the work" in that case ([VisionCamera](https://visioncamera.margelo.com/docs/guides/frame-processors-tips)), and a dropped job would mean a part that is never judged. When the queue grows, the app shows a "line too fast" warning, the phone's equivalent of an industrial overrun alarm.

### Short exposure and the torch make moving parts inspectable

Motion blur in pixels equals part speed × exposure time ÷ pixel size ([Vision-Doctor](https://www.vision-doctor.com/en/camera/exposure-time-area-scan-camera.html)). The table below assumes the phone is about 30 cm above the parts, so a 300 mm field spans 640 analysis pixels at about 0.47 mm per pixel. That field size is an assumption; measure it with a ruler in frame. The figures show why default auto-exposure (1/60 s) smears a hand-slid part by about 11 pixels.

| Part speed | Exposure for 1 px blur | Blur at 1/60 s | Frames in view at 30 fps |
|---|---|---|---|
| 0.1 m/s (≈20 ft/min) | 4.7 ms | 1.7 mm (≈3.5 px) | ≈90 |
| 0.3 m/s (hand slide) | 1.6 ms | 5 mm (≈11 px) | ≈30 |
| 0.5 m/s (≈100 ft/min) | 0.94 ms | 8.3 mm (≈18 px) | ≈18 |
| 1.0 m/s (≈200 ft/min) | 0.47 ms | 16.7 mm (≈35 px) | ≈9 |

Android makes the fix available. With Camera2's AE_MODE_OFF, the app sets exposure time and sensitivity directly, and the AE and AWB locks freeze exposure and white balance ([AOSP 3A modes](https://source.android.com/docs/core/camera/camera3_3Amodes)). The capture preset is:

| Setting | Value |
|---|---|
| Frame rate | Fixed 30 fps through the AE target-fps range; 60 fps if the sensor offers it |
| Exposure | 0.5–1 ms, with ISO raised as needed |
| Torch | On |
| Focus and white balance | Locked after teaching |
| Stabilisation, HDR, low-light boost | Off. Stabilisation adds capture latency, and low-light boost lengthens exposure ([VisionCamera performance](https://visioncamera.margelo.com/docs/performance)) |

Teach the Twin in motion with the same preset. Blur, sensor noise and rolling-shutter shear then become part of "normal" instead of triggering false rejects.

The honest speed envelope is **up to about 0.5 m/s for small parts**. That covers the slower half of typical packaging belts, which run at 20–200 ft/min ([PackagingConveyor](https://packagingconveyor.com/resources/conveyor-design-guide/conveyor-engineering/)). It is nowhere near a Cognex In-Sight 3900, reported at up to 5,000 parts per minute. The phone is a tool for final-inspection tables and low-volume lines.

### A virtual photo-eye judges each part exactly once

**Detector.** Place the parts on a plain sheet that contrasts with them. The simplest robust detector is then a colour or brightness threshold plus contour finding at 320×180, restricted to a band around the trigger line. FastOpenCV runs this exact pipeline in real time: resize to 320×180, HSV range check, contours, area filter, bounding boxes ([FastOpenCV](https://lukaszkurantdev.github.io/react-native-fast-opencv/examples/realtimedetection)). It needs no training. It survives a moving phone, because it thresholds against a background colour captured with one "empty sheet" tap. It also hands the judge worker a clean silhouette for free.

Ignoring blobs that touch the frame edge takes care of hands. A hand placing a part forms a single blob with the arm, and that blob touches the border. The part is therefore judged only after the hand leaves.

Two common alternatives fail here. **Background subtraction** is designed for static cameras and breaks when the phone moves ([OpenCV tutorial](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html)), and **ML Kit's object tracker** "might need to process 30 or more frames" before its first detection, is capped at five objects and is an unbundled download ([ML Kit](https://developers.google.com/ml-kit/vision/object-detection/android)). A **COCO-pretrained YOLO nano** with its class filter off is the fallback for cluttered scenes, at about 11–16 ms in-app, with an AGPL licence caveat ([qualcomm/YOLOv11-Detection](https://huggingface.co/qualcomm/YOLOv11-Detection)).

**Tracker.** Tracking borrows SORT's constant-velocity prediction with greedy matching; SORT ran at 260 Hz on a 2016 desktop CPU ([SORT](https://arxiv.org/abs/1602.00763)). Matching is gated by distance between centroids rather than box overlap. At 1 m/s and 30 fps a part moves about 71 px per frame, so consecutive boxes may not overlap at all.

**Trigger.** A track is judged once, when its anchor has been seen downstream of a virtual line for 2–3 consecutive frames. These are the counting rules of supervision's LineZone ([supervision](https://supervision.roboflow.com/latest/detection/tools/line_zone/)). Only tracks first seen upstream are judged, so an ID switch after the line cannot double-count. While a part is in the zone, the tracker keeps its best crop: the sharpest, most central and fully in frame.

A second trigger, **"steady hold"**, judges a part that sits still and sharp inside a guide box for about half a second. This lets judges place objects by hand.

The industrial mapping belongs in the pitch. In factories, hardware photo-eyes and encoders are the standard triggers ([Elementary](https://www.elementaryml.com/blog/the-complete-guide-to-machine-vision-triggering-for-high-speed-image-acquisition)). Kaizen Eye replaces them with an image-based photo-eye and a tracker that estimates speed.

**Screen, sound and logging.** The screen follows high-performance HMI practice: a grey interface, with colour only for abnormal states and red reserved for critical ones ([Control.com](https://control.com/technical-articles/going-gray/)).

| Part state | Box |
|---|---|
| Tracking | Thin grey outline |
| Being judged | Amber outline |
| Pass | Brief green outline |
| Fail | Thick red box, translucent heat map and a label such as "REJECT #042", kept until the part leaves the frame |

The overlay lags the part by the pipeline latency, and 50 ms at 0.5 m/s is 25 mm. The box position is therefore pushed forward by track velocity × measured latency.

Sound and vibration fire only on rejects: a pass is silent to avoid alarm fatigue, a fail plays Android's `TONE_CDMA_ALERT_CALL_GUARD` (three 1,319 Hz bursts) ([ToneGenerator](https://developer.android.com/reference/android/media/ToneGenerator)) with a 120/80/120 ms vibration, and three consecutive rejects escalate to a stop-the-line tone. The counters show total, pass and reject counts, parts per minute over the last 60 s, the reject rate over the last 50 parts, the trigger-to-verdict time, and a p-chart of the reject fraction in subgroups of 20–50 parts that turns amber above its upper control limit ([NIST](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc332.htm)). Every part is written to an append-only JSONL log with its crop and heat map, and the log can be exported through the share sheet.

### Faster analysis comes from shrinking the search, not from a new model

Brute-force kNN is today's slowest stage: with 20 enrolment frames the bank holds 1,600 patches, so each query compares 1,600 patches against 1,600 entries at 128 dimensions, about 3.3×10⁸ multiply-adds. The biggest win is shrinking the search, because nearest-view retrieval and object-only patches mean each query searches three keyframes instead of the whole bank. The next is moving kNN out of JavaScript and expressing cosine similarity as one matrix multiply, using the NEON-optimised GEMM in OpenCV's prebuilt Maven package ([OpenCV Android](https://docs.opencv.org/4.x/d5/df8/tutorial_dev_with_OCV_on_Android.html)) or multi-threaded Kotlin. SubspaceAD's PCA-residual head scores each patch with one projection, stores under 1 MB per part and took 36 ms per image with DINOv2-S at 448 px on a GPU; it deserves an A/B test but not default status, because its ViT-S accuracy is unpublished ([SubspaceAD](https://arxiv.org/html/2602.23013v3)). Two tempting shortcuts should be refused. Compressing the bank backfires: product-quantised PatchCore-Lite fell from 0.95 to 0.86 image AUROC and ran about 4× slower on a CPU ([arXiv 2603.20288](https://arxiv.org/html/2603.20288)). EfficientAD is the fastest accurate model, at 2.2 ms per image and 98.8 image AUROC on a desktop GPU, but it needs 70,000 training iterations (reported at 5 h 26 min) and lost 11.6 AU-PRO₀.₀₅ points under new lighting on MVTec AD 2, so it cannot support ten-second teaching ([EfficientAD](https://arxiv.org/html/2303.14535v3), [arXiv 2403.15463](https://arxiv.org/html/2403.15463), [MVTec AD 2](https://arxiv.org/html/2503.21622)). The target is **at most 100 ms per part**, spent once per part and not once per frame:

| Stage | Budget |
|---|---|
| Backbone on the NPU | ≤20 ms |
| kNN | ≤30 ms |
| Crop, mask and gates | The remainder |

## DINOv2 for reflexes and a small VLM for reasons, both on the NPU through native Kotlin

### DINOv2 ViT-S is the upgrade with the best accuracy per FLOP

Training-free matching on frozen DINOv2 patch features now leads few-shot industrial anomaly detection, and the gap to Kaizen Eye's CNN family is large. On MVTec AD, AnomalyDINO with DINOv2 ViT-S/14 (21M parameters) at 448 px reaches **96.5 image AUROC from one good image and 98.3 from sixteen**, while PatchCore scores 83.4 at one shot and 88.8 at four under the same protocol ([AnomalyDINO](https://arxiv.org/html/2405.14529)). On the harder VisA set the same method climbs from 85.6 to 93.8 between one and sixteen shots. A bigger CNN is not the answer: with full data, Wide ResNet-50 beats ResNet-18 by less than one point ([anomalib](https://github.com/openvinotoolkit/anomalib/blob/v0.7.0/src/anomalib/models/patchcore/README.md)). The 2025–26 leaders (UniVAD 97.8, DuoAD 97.7, VisionAD 97.4 and SubspaceAD 97.1 at one shot) sit within about a point of AnomalyDINO but use larger backbones or multi-model stacks, and UniVAD's authors concede latency limits real-time use ([DuoAD](https://arxiv.org/html/2607.23924v1), [UniVAD](https://arxiv.org/html/2412.03342v3)). The rest of the field fails the brief. CLIP zero-shot methods plateau at 89–92 on MVTec AD ([PA-CLIP](https://arxiv.org/html/2503.01292v1)), and batched zero-shot MuSc needs 200 unlabeled images and 955 ms per image on an RTX 3090 ([MuSc](https://github.com/xrli-U/MuSc)). Real multi-view data stays hard for everyone, with one-shot Real-IAD around 80–85 ([DuoAD](https://arxiv.org/html/2607.23924v1)), which is why the Twin's object lock and view retrieval matter as much as the backbone.

Nobody has published DINOv2 latency on this chip, but its architecture already runs there. Depth-Anything-V2-Small, whose encoder is the DINOv2 ViT-S configuration, runs in **11.9–18.0 ms** at 518 px on the Snapdragon 8 Elite Gen 5 NPU ([qualcomm/Depth-Anything-V2](https://huggingface.co/qualcomm/Depth-Anything-V2)). A 448 px encoder should therefore fit inside a 20 ms per-part budget, against 0.23–0.59 ms for ResNet18 at 224 px ([qualcomm/ResNet18](https://huggingface.co/qualcomm/ResNet18)). Three export rules keep the DINOv2 path safe. First, bake the positional-embedding interpolation into a fixed-resolution graph, because DINOv2-L failed on a Qualcomm RB5 when the QNN GPU backend rejected a Resize operator ([EdgeZSAD](https://arxiv.org/html/2606.16119)). Second, prefer FP16 or w8a16 precision: no study measures INT8 effects on DINOv2 anomaly detection, general ViT evidence says W8A8 is near-lossless only with ViT-aware quantisation ([PTQ4ViT](https://arxiv.org/pdf/2111.12293)), and an FP16 deployment of a ViT detector drifted less than 0.2 AUROC points ([EdgeZSAD](https://arxiv.org/html/2606.16119)). Third, spend spare NPU time on resolution. MVTec AD 2 keeps state-of-the-art methods below 60% AU-PRO largely because of tiny defects ([MVTec AD 2](https://arxiv.org/abs/2503.21622)), every top VAND 3.0 entry used inputs of at least 448 px ([VAND 3.0](https://arxiv.org/html/2509.17615)), and the 2026 winner SuperADD tiles at 640 px ([SuperADD](https://arxiv.org/html/2605.14808v1)). A 2×2 tiling of the object crop at 448 px is the natural stretch goal for small defects. ResNet18 remains the fallback backbone: a 2.4 MB `feature_extractor.tflite` already sits in the native project's assets, and the team measured int8 quantisation as accuracy-neutral (0.898 vs 0.905 AUROC) at 3.2× the CPU speed ([README](file:///D:/Projects/Kaizen_Eye/README.md)).

### A small VLM should explain rejects, never decide them

VLMs are weak defect detectors. Even GPT-4o reaches only **74.9%** on the MMAD industrial benchmark, which its authors call far short of industrial requirements ([MMAD](https://arxiv.org/abs/2410.09453)). Small models are worse at the core task: a base Qwen2.5-VL-3B scores 58.8% accuracy on MVTec LOCO ([LAD-Reasoner](https://arxiv.org/html/2504.12749)), and RobustMAD finds that models of about 4B parameters and below "fall short of safety-critical requirements" ([RobustMAD](https://arxiv.org/html/2607.16243)). What VLMs do well is describe a defect that has already been located. Giving MMAD models masks improved defect localisation by 21.29 points ([MMAD](https://arxiv.org/html/2410.09453)), and a pipeline that feeds detector outputs to the language model as spatial tokens cut report hallucination from 65% to 4% ([arXiv 2605.26533](https://arxiv.org/abs/2605.26533)).

Kaizen Eye's VLM therefore receives the part crop with the defect region drawn on it, plus a one-line list of facts: the defect's grid location, its area as a percent of the part, its score as a multiple of the threshold, and the part name typed at teach time. It must return one sentence. It can never overturn a FAIL, and a deterministic guard replaces any sentence whose location words contradict the heat map with a template sentence. Natural-language rule checklists in the style of LogicQA reach 87.6 AUROC on MVTec LOCO, but only with GPT-4o ([LogicQA](https://arxiv.org/abs/2503.20252)). They are a stretch goal, and a deterministic check that counts holes and components inside the mask against the Twin should come first.

| VLM option | Where it runs on this chip | Published speed | Size and licence | Verdict |
|---|---|---|---|---|
| FastVLM-0.5B `.sm8850.litertlm` via LiteRT-LM | NPU | Time to first token 0.12 s including one 1024×1024 image; 106 tokens/s; 925 MB RAM ([vendor](https://huggingface.co/litert-community/FastVLM-0.5B)) | 899 MB; Apple research licence | **Primary**: the only NPU-precompiled VLM found for this SoC |
| Gemma 4 E2B `.litertlm` | GPU; NPU builds exist only for SM8750 and QCS8275 | 52 tokens/s and 0.3 s time to first token on the S26 Ultra GPU ([vendor](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)); 41.7 tokens/s on the SM8750 NPU ([independent](https://dev.to/jdshah/why-my-llm-runs-4x-faster-on-hardware-i-had-never-heard-of-9b9)) | 2.58 GB; Apache 2.0 | **Fallback** with stronger reasoning |
| Llama 3.2 3B via Genie or GenieX | NPU | 30 tokens/s claimed ([Qualcomm](https://huggingface.co/qualcomm/Llama-v3.2-3B-Instruct)) vs about 10 measured on the 8 Elite ([Grape Up](https://grapeup.com/blog/running-llms-on-device-with-qualcomm-snapdragon-8-elite)) | Needs an AI Hub compile; GenieX downloads models from an in-app catalog ([GenieX](https://github.com/qualcomm/GenieX)) | Avoid within 10 h |
| MNN or llama.cpp Hexagon | NPU or CPU | Qwen3-VL-class models supported | Source builds with the NDK or Docker | Avoid on Windows within 10 h |

Warm-up is the practical trap. LiteRT-LM engine initialisation "can take ... up to 10 seconds" ([LiteRT-LM Android](https://developers.google.com/edge/litert-lm/android)), and a research port of FastVLM on the previous-generation chip took 26.8 s cold against 2.1 s warm, end to end ([Phase Matters](https://arxiv.org/html/2606.27906v1)). The app must load the model and run it once behind the splash screen. One public report of trouble running FastVLM on the NPU through LiteRT sits unanswered ([LiteRT #5499](https://github.com/google-ai-edge/LiteRT/issues/5499)). However, Google's AI Edge Gallery ships device-specific SM8850 NPU builds ([Gallery releases](https://github.com/google-ai-edge/gallery/releases)) and documents offline import of `.litertlm` files ([Gallery wiki](https://github.com/google-ai-edge/gallery/wiki/6.-Importing-Local-Models-(optional))), so the team can confirm the path on the actual phone in five minutes before writing any code. Model weights stay outside the APK: push them over adb and load them by file path.

### Using the NPU to the full means spending spare capacity on quality, not running models more often

The chip's thermal profile favours the NPU. Qualcomm says the 8 Elite Gen 5 NPU is 37% faster than its predecessor, but performance per watt rose only about 16%, and the chip's TOPS figure is undisclosed ([XPU.pub](https://xpu.pub/2025/10/06/qualcomm-snapdragon-8-elite-gen-5/)). In GSMArena's stress test, the iQOO 15 held clocks for about the first 10 minutes, then its CPU and GPU fell below 50% of peak ([GSMArena](https://www.gsmarena.com/iqoo_15-review-2905p4.php)). On the previous-generation chip, NPU inference ran **10.47 °C cooler with 2.52× less energy per request** than CPU inference, and the CPU path triggered governor throttling at 82 °C while the NPU held at 66 °C ([Phase Matters](https://arxiv.org/html/2606.27906v1)).

The vision stages take from under a millisecond to about 15 ms per part on the NPU, so it will sit idle most of the time. The sensible way to "max it out" is to move every neural stage onto it, then spend the spare capacity on quality rather than on running models every frame:

| Where the spare NPU capacity goes | How |
|---|---|
| Resolution | 448 px input and tiling |
| Redundancy | Score two or three crops per part and vote |
| Explanation | Run the VLM on rejects |

Most heat will come from the camera, ISP, display and CPU, so a thermal governor caps the fast loop at 30 fps (1–2 fps when nothing is in view), keeps the NPU in a sustained power mode with burst mode reserved for the VLM call, polls thermal headroom no more than once every 10 s as Android requires ([ADPF](https://developer.android.com/games/optimize/adpf/thermal)), and lowers the frame rate and pauses the VLM when thermal status reaches moderate or severe.

The telemetry panel proves NPU use in three ways. A **per-stage accelerator badge** shows "NPU" only when the model really ran there: ONNX Runtime's QNN provider can be told to throw an error if any node would fall back to the CPU (`session.disable_cpu_ep_fallback`) ([ORT QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)), and LiteRT reports fallback status, which third-party wrappers already surface ([flutter_litert](https://pub.dev/packages/flutter_litert/changelog)). A **live CPU/NPU toggle** re-runs the same frame on the CPU to show the speed-up; on the previous chip, vision encoders ran 20–45× faster on the NPU than on the CPU ([Phase Matters](https://arxiv.org/html/2606.27906v1)). **Android telemetry** shows thermal status and headroom, battery temperature (reported in tenths of a degree) and whole-phone power from battery current and voltage ([BatteryManager](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/core/java/android/os/BatteryManager.java)). No public Android API reports power for the NPU alone. The pitch should therefore show the difference between CPU and NPU modes, never an NPU wattage.

It should also say plainly that the current app never used the NPU. react-native-fast-tflite offers only GPU and NNAPI delegates ([fast-tflite](https://github.com/mrousavy/react-native-fast-tflite)). On recent Qualcomm phones, NNAPI exposes only a CPU reference device, because "Qualcomm ships no NNAPI driver" ([DroidRunner #82](https://github.com/m96-chan/DroidRunner/issues/82)).

### Native Kotlin is the only stack with a documented path to the NPU

The platform decision follows from where the NPU runtimes live. All four options below are plain Kotlin dependencies:

| Runtime | What it offers | Source |
|---|---|---|
| LiteRT `CompiledModel` | Lists the Snapdragon 8 Elite Gen 5 (SM8850) as supported, compiles `.tflite` models for the NPU on the device, and falls back to the GPU automatically | [LiteRT Qualcomm](https://developers.google.com/edge/litert/next/qualcomm), [LiteRT NPU](https://developers.google.com/edge/litert/next/npu) |
| Qualcomm QNN delegate and runtime | Maven artifacts, currently at version 2.50.0 | [Maven](https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-litert-delegate/maven-metadata.xml) |
| ONNX Runtime QNN provider | Shown working on an SM8850 phone from a sideloaded APK after four documented fixes | [itsallgoody](https://github.com/itsallgoody/onnxruntime-qnn-snapdragon-8-elite-gen5) |
| LiteRT-LM | Kotlin library that loads `.litertlm` model files from a path | [LiteRT-LM Android](https://developers.google.com/edge/litert-lm/android) |

A native app built on these compiles no C++. That avoids the Windows clang failure caused by the space in the user name, which already cost the team time ([README](file:///D:/Projects/Kaizen_Eye/README.md)).

| Option | NPU access on SM8850 | Camera pipeline | Windows build risk | Verdict |
|---|---|---|---|---|
| **Native Kotlin** (CameraX + LiteRT/QNN + LiteRT-LM) | Direct, through CompiledModel, the QNN delegate or ONNX Runtime QNN | CameraX RGBA frames; stale frames dropped automatically ([CameraX](https://developer.android.com/media/camera/camerax/analyze)) | Lowest: no C++ compiled | **Build** |
| Hybrid (Kotlin camera module inside the Expo app) | Same as native | Same as native | Medium: the React Native C++ build stays, and two TFLite runtimes may clash | Only if the React Native UI must stay |
| React Native + VisionCamera V5 + fast-tflite | GPU and NNAPI only | Worklets. V5 was released on 16 Apr 2026 ([Margelo](https://margelo.com/blog/whats-new-in-visioncamera-v5)) with an open HardwareBuffer crash report ([VC #3824](https://github.com/mrousavy/react-native-vision-camera/issues/3824)) | Medium-high: six or more C++ packages | Not for this round |
| Flutter | Only through custom Kotlin code | YUV-only stream ([flutter #145961](https://github.com/flutter/flutter/issues/145961)) with an open memory-leak issue ([flutter #145893](https://github.com/flutter/flutter/issues/145893)) | Medium, plus a new toolchain | Full rewrite; no |

The native route also has a head start on this machine. The Android Studio project in the working directory already builds a debug APK, with CameraX 1.4.2 and LiteRT 1.4.0 declared ([build file](file:///C:/Users/Vijay%20Chikkala/AndroidStudioProjects/KaizenEye/app/build.gradle.kts)). The agent's first steps are therefore to raise minSdk from 26 to 31, the minimum for LiteRT's NPU path, and to move to LiteRT 2.2.0.

The starter kit already holds the Python scoring reference and golden test vectors. A Kotlin `:core` port must reproduce them, with "coreset indices exactly, floats to ~1e-4 relative" ([starter README](file:///C:/Users/Vijay%20Chikkala/AndroidStudioProjects/KaizenEye/kaizen-eye-starter/README_STARTER.md)). That turns the scorer port into a mechanical, unit-tested task for an AI agent.

Two samples provide reference code to borrow: Qualcomm's `object_detection_android`, with camera, TFLite on QNN/GPU/CPU delegates and an overlay under a BSD-3 licence ([ai-hub-apps](https://github.com/qualcomm/ai-hub-apps/blob/release/object_detection_android/README.md)), and Google's AI Edge Gallery, with LiteRT-LM image chat under Apache-2.0 ([Gallery](https://github.com/google-ai-edge/gallery)). The existing Expo app stays installed as the photo-mode fallback.

## Tier-2 auto-component suppliers feel the pain in PPM, not payroll

### Human inspectors miss one defect in four and get worse within half an hour

The baseline for reliability comes from decades of inspection research. Sandia's review of 212 documents found that "error rates of 20% to 30% are frequently quoted" across inspection tasks ([See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)). Even 82 expert inspectors of precision parts caught only 85% of defective items, while wrongly rejecting **35% of good ones** ([See 2015](https://www.sandia.gov/research/publications/details/visual-inspection-reliability-for-precision-manufactured-parts-2015-12-01/)). Attention also decays quickly, per the same review ([See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)). Detection can fall up to 40% within 30 minutes. Experienced inspectors of automotive rubber seals scored 27% fewer hits in their second 15 minutes than in their first. And one inspector reversed 23% of decisions when shown the same piston rings again.

These studies come from Western aerospace, nuclear and automotive settings, not Indian MSMEs, and the pitch should say so. Still, they explain where the app's value lies: consistency and immunity to fatigue at human pace. The same review found that two inspectors checking the same item beat one ([See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)). That points to the natural positioning: Kaizen Eye as an independent **second inspector**.

### Auto-component MSMEs are the beachhead

India's auto-component industry turned over **₹7.60 lakh crore in FY26**, up 12.7% ([Autocar Professional](https://www.autocarpro.in/news/acma-indian-auto-component-industry-grows-127-percent-to-inr-76-lakh-crore-in-fy26-133454)). MSMEs make up about 80% of its manufacturers, with ₹2.4–2.9 lakh crore in turnover ([Autocar Professional](https://www.autocarpro.in/news/auto-component-msmes-face-capability-gap-rs-39000cr-tied-up-134505)). Field research in the Delhi–Haryana cluster names four barriers to automation: capital cost, setup time, small batches of 5,000–10,000 units, and a shortage of technicians ([CSEP](https://csep.org/discussion-note/wheels-of-change-automation-in-indias-automotive-sector/)).

Customers enforce quality with money. AVTEC's supplier quality manual scores suppliers in defective-parts-per-million (PPM) bands where **more than 500 PPM earns zero points**, rejects a whole lot when a single sampled part fails (C=0 sampling), and charges 100% sorting to the supplier ([AVTEC SQM](https://www.avtec.in/pdf/AVTEC%20Supplier%20Quality%20Manual.pdf)). The OEM pull is visible too. In January 2026, Maruti Suzuki onboarded startups for AI visual inspection of complex components and zero-defect supplier parts ([Maruti Suzuki](https://www.marutisuzuki.com/corporate/media/press-releases/2026/january/maruti-suzuki-onboards-5-more-startups-to-scale-new-age-technologies-across-business-areas)).

Final inspection at these firms is usually a self-paced table of small, rigid, repeated parts: bolts, nuts, washers, castings, stampings and seals. That suits anomaly detection learned from good parts, and it suits a phone camera. Frequent changeovers also favour ten-second re-teaching over integrator projects.

Two sectors are backups. **Electronics and PCB assembly** comes first: production reached an estimated ₹13.11 lakh crore in FY26 ([PIB](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2284808&reg=48&lang=1)), solder-defect detection ranges from 43% to 100% across inspectors ([See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)), and VisA's four PCB subsets are CC BY 4.0 and usable for benchmarking ([VisA](https://github.com/amazon-science/spot-diff)), although tiny solder defects stretch a phone's optics. **Pharma packaging** comes second: about 8,500 of roughly 10,500 units are MSMEs facing revised Schedule M compliance from 2026 ([Business Standard](https://www.business-standard.com/health/small-pharma-companies-get-1-year-breather-to-implement-schedule-m-125021201111_1.html)), but inline blister lines run at 300–800 packs per minute, too fast for a phone, so the fit is at-line sampling only.

### The money is in escapes and PPM bands; a phone costs 3.9 months of one inspector's wage

| Inspection station | Price | Phone is cheaper by | Months of a skilled inspector's minimum wage |
|---|---|---|---|
| iQOO 15 running Kaizen Eye | ₹72,999 ([91mobiles](https://www.91mobiles.com/hub/iqoo-15-launched-in-india-price-availability/)) | — | 3.9 |
| Keyence IV3 AI sensor | ₹1.65 lakh, dealer price ([IndiaMART](https://www.indiamart.com/proddetail/keyence-vision-sensor-with-built-in-ai-iv3-series-2854380953255.html)) | 2.3× | 8.9 |
| Cognex In-Sight SnAPP | ₹1.80 lakh, dealer price ([IndiaMART](https://www.indiamart.com/proddetail/cognex-insight-snap-vision-sensor-2852699267873.html)) | 2.5× | 9.7 |
| Keyence IV4 | ₹5.0 lakh, dealer price ([IndiaMART](https://www.indiamart.com/proddetail/keyence-iv4-vision-sensor-camera-with-built-in-ai-2857001721891.html)) | 6.8× | 27 |
| Siemens Inspekto S70 | €10,000 in 2019, ≈ ₹11 lakh ([ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/)) | 15.1× | 59.5 |
| Standard vision system (UK integrator price guide) | £5,000–15,000, ≈ ₹6.4–19 lakh ([Clearview](https://clearview-imaging.com/pages/how-much-does-machine-vision-actually-cost)) | 8.7–26× | 34–103 |

Months of wage use Haryana's skilled minimum wage of ₹18,500.81 a month from 1 April 2026 ([Zimyo](https://www.zimyo.com/guides/minimum-wages/haryana/)). The incumbent prices mostly exclude lighting and integration. In fairness, the phone lacks industrial I/O, lighting and ruggedisation.

Labour savings are a weak story. At ₹18,500–20,600 a month, an inspector costs ₹89–99 an hour, which works out to only ₹0.15–1.05 of labour per part at 6–38 s per part ([Zimyo](https://www.zimyo.com/guides/minimum-wages/haryana/), [Indeed](https://in.indeed.com/career/quality-control-inspector/salaries)).

Escapes are a strong story. Suppose 2% of parts arrive defective (the literature's typical range is 1–10%) and humans miss 20–30% of them. The supplier then ships **4,000–6,000 PPM**, deep in AVTEC's zero-point band. An independent second screen with measured recall r cuts escapes to 2% × human miss rate × (1 − r):

| App recall (r) | Escaped defects | AVTEC band |
|---|---|---|
| 90% | 400–600 PPM | Straddles the 500 PPM zero-point line |
| 95% | 200–300 PPM | Inside the 201–500 band |

The team must replace r with its own measured value.

Four further points belong in the pitch. Offline inference has **zero marginal cost**, whereas LandingLens charges a credit per inference ([LandingLens plans](https://landinglens.docs.landing.ai/plans)), and it carries **no vendor-sunset risk** of the kind that ended when AWS retired Lookout for Vision in October 2025 ([AWS](https://docs.aws.amazon.com/lookout-for-vision/latest/developer-guide/su-awscli-sdk.html)). Government schemes subsidise process, not devices: the MSME Competitive (LEAN) scheme pays 90% of consultancy costs for tools that include poka-yoke ([Drishti IAS](https://www.drishtiias.com/daily-updates/daily-news-analysis/msme-competitive-lean-scheme)), and ZED certification is subsidised at 80%, 60% and 50% for micro, small and medium firms ([ZED](https://zed.msme.gov.in/subsidy-on-cost-of-certification)), so Kaizen Eye should be pitched as a **poka-yoke aid that helps a supplier climb ZED levels**, not as a subsidised device. Market figures need care: India's machine-vision systems market is put at US$626.55 million for 2025 ([IMARC](https://www.imarcgroup.com/india-machine-vision-systems-market)) within a global market of US$15.83 billion growing 8.3% a year ([MarketsandMarkets](https://www.marketsandmarkets.com/Market-Reports/industrial-machine-vision-market-234246734.html)), but these are estimates, and a bottom-up count of stations per plant is more persuasive. Phone inspection for Indian factories is already circulating as an unbuilt concept, priced at ₹999–2,999 a month ([StartupBasket](https://startupbasket.ai/ideas/netraqc-phone-ai-factory-inspection/)). That is one more reason the novelty claim must rest on measured execution.

The honest positioning is final-inspection tables, spot checks, audits and low-volume lines up to about 0.5 m/s. It is not a 5,000-parts-per-minute inline camera wired to a PLC.

## The recommended architecture: one pipeline, two triggers, four verdicts

The system is one native Kotlin app with three screens (Teach, Inspect, Telemetry). Every mode shares a single judging pipeline, so behaviour cannot drift between segments of the demo. Every stage has a named fallback that keeps the flow alive, which is what "fool-proof" has to mean in a ten-hour build.

```
KAIZEN EYE 2 — native Kotlin, minSdk 31, arm64, sideloaded APK, runs in airplane mode

CameraX Preview + ImageAnalysis (RGBA, 1280x720, KEEP_ONLY_LATEST)
Camera2Interop: fixed 30 fps, exposure 0.5-1 ms (or AE lock), AF/AWB locked, torch on, no stabilisation/HDR
   |
   +--> FAST LOOP (every frame, CPU, target < 10 ms)
   |      320x180 downscale -> mat-contrast threshold + morphology + contours (ROI band)
   |      -> centroid tracker with velocity gating
   |      -> trigger A: virtual photo-eye (2-3 frames downstream) | trigger B: steady hold (~0.5 s, sharp)
   |      -> best-crop buffer (sharpness x centrality) -> overlay (box extrapolated by velocity x latency)
   |
   +--> JUDGE WORKER (once per part, FIFO queue, async)
          high-res crop + mask -> sanity (area 5-80%, one component, off-border) --fail--> REFRAME
          -> crop to mask box + 10% -> 448 px
          -> BACKBONE on NPU: DINOv2 ViT-S/14 fp16/w8a16   [fallback: ResNet18 int8 -> GPU -> CPU]
          -> Gate 1 identity: CLS cosine vs Twin keyframes >= tau_id --fail--> NOT THE ENROLLED PART
          -> Gate 2 geometry: area/aspect/Hu within +-3 sigma      --fail--> NOT THE ENROLLED PART
          -> nearest-view retrieval: top-3 Twin keyframes by CLS
          -> masked patch kNN (cosine, OpenCV GEMM) -> score = mean of top-1% patch distances
          -> score <= tau: PASS | score > tau: high patch coverage ? NOT THE ENROLLED PART : DEFECT
          -> alerts: red box + heat map ride the track, andon flash, tone + vibration,
             counters, p-chart, JSONL log with crops
               |
               +--> REASONER (rejects or on tap; low priority; never overrides a verdict)
                      LiteRT-LM FastVLM-0.5B .sm8850 on NPU  [fallback: Gemma 4 E2B on GPU -> template]
                      input: crop with box drawn + facts line (grid cell, area %, score/tau, part name)
                      guard: location words must match the heat map, else template sentence

TEACH (Visual Twin, 10-15 s): frames @ 8 fps -> masks -> Laplacian sharpness (drop worst 30-50%)
   -> CLS embeddings -> greedy k-center, 16-32 keyframes (coverage meter) -> object patch features (fp16)
   -> leave-segment-out scores -> tau_id from positives vs 50-100 bundled negatives -> Twin saved
CALIBRATE (29+ distinct good parts): order-statistic tau -> CERTIFICATE
   (m, false-alarm bound at 95%, identity margin, synthetic-scratch self-test, measured ms on this phone)

TELEMETRY + THERMAL GOVERNOR: per-stage ms, accelerator badge (NPU/GPU/CPU), CPU<->NPU A/B toggle,
   thermal listener + headroom poll <= once per 10 s, battery temperature, whole-phone watts;
   steps down fps, NPU power mode and VLM use as heat rises
```

### Three NPU paths share one set of Qualcomm libraries

Vision models have three NPU paths, tried in order; paths 2 and 3 are fallbacks, not guarantees.

| Order | Runtime | Why |
|---|---|---|
| 1 | LiteRT 2.2.0 `CompiledModel` with `Accelerator.NPU`, and `Accelerator.GPU` as automatic fallback ([LiteRT NPU](https://developers.google.com/edge/litert/next/npu)) | Default path |
| 2 | Classic interpreter with Qualcomm's `qnn-litert-delegate` 2.50.0 ([Maven](https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-litert-delegate/maven-metadata.xml)) | Used if path 1's NPU libraries will not start in a sideloaded build within the time box |
| 3 | ONNX Runtime 1.29 with the QNN provider ([itsallgoody](https://github.com/itsallgoody/onnxruntime-qnn-snapdragon-8-elite-gen5)) | The only route with a published SM8850 sideload recipe and a hard switch that forbids falling back to the CPU |

Keeping both vision and the VLM on the LiteRT family puts only one set of Qualcomm libraries in the APK.

Model files are split by size. The vision models, a few MB to a few tens of MB, ship inside the APK. The VLM weights (899 MB for FastVLM, 2.58 GB for Gemma) are pushed by adb into app-specific storage and loaded by file path.

### A Twin is a folder of about 9 MB

Each Twin is a folder holding the keyframe JPEGs, 16-bit patch features (about 9 MB for 24 keyframes × 500 object patches × 384 dimensions), the [CLS] vectors, both thresholds and the certificate as JSON. Several Twins make a multi-part library, and as a stretch goal the identity gate can pick the right Twin automatically.

### Every failure falls back to a weaker mode that still works

Each failure has a pre-built fallback:

| If this fails | The app does this instead |
|---|---|
| A mask fails | Says REFRAME rather than giving a wrong verdict |
| The NPU | Uses the GPU |
| DINOv2 | Uses ResNet18 |
| The VLM | Uses a template sentence |
| Line tracking gets flaky | Uses the steady-hold trigger |
| The live camera misbehaves | Uses REPLAY mode, which feeds an MP4 through the identical pipeline under a visible "REPLAY" badge |
| Everything | The old Expo APK still does photo-mode inspection |

## The ten-hour build plan front-loads the fixes for the critiques and time-boxes the NPU and VLM

The plan assumes the AI coding agent writes all the code while one person holds the phone, builds props and runs measurements, and the laptop, not the phone, runs the unit tests. Three rules govern it: **archive a shippable APK at every cut line**; **never let the NPU or the VLM block the pipeline**, which makes the GPU and template paths first-class code; and **measure the "X" baselines on the current Expo app in the first hour**, because every X→Y claim needs its X measured on the same parts by the same hands. A pre-flight track needs no coding and should finish before the clock starts, or else run in parallel during hours 0–5. It downloads the FastVLM-0.5B `.sm8850.litertlm` (899 MB) and Gemma 4 E2B `.litertlm` (2.58 GB) weights after accepting the gated-model forms on Hugging Face, plus the AI Edge Gallery SM8850 build. It exports DINOv2 ViT-S/14 at a fixed 448 px to TFLite and ONNX and checks the outputs against PyTorch on Colab or WSL, because the LiteRT converter has no Windows wheels ([README](file:///D:/Projects/Kaizen_Eye/README.md)). It gathers matte black and white sheets, 50 or more identical M8 nuts or washers, ten "wrong" objects including two look-alikes, and five to ten parts deliberately damaged with scratches, chips or marker dots. And it photographs 50–100 random desk objects as the identity gate's negatives.

| Clock | AI agent builds | Human does in parallel | Exit test on the iQOO 15 | If blocked |
|---|---|---|---|---|
| 0:00–0:45 | minSdk 31, LiteRT 2.2.0, LiteRT-LM 0.17.1, OpenCV (Maven), arm64 only. CameraX preview and analysis at 1280×720. Camera2Interop fixed fps, manual exposure or AE lock, AF/AWB lock, torch. FPS display. Log camera capabilities | Install the Gallery SM8850 build and run FastVLM on the NPU in airplane mode. On the old Expo app, measure: 20-photo enrolment time (×3), wrong-object accepts (10 objects × 5), per-photo latency | 30 fps preview with 3A locked and the torch on | Manual exposure unsupported: AE lock plus torch, and demo speed limited to ≤0.2 m/s |
| 0:45–2:00 | Kotlin `:core` (coreset, kNN, leave-one-out, scoring) passing JUnit against `golden_core.json`. Existing `feature_extractor.tflite` run through CompiledModel (NPU, then GPU). Stage timers. OpenCV GEMM cosine kNN. Accelerator badge | Prepare props, photograph negatives, seed defects | Golden tests pass. Backbone + kNN ≤60 ms on the phone. Badge honestly shows NPU or GPU | NPU fails after 30 min: QNN delegate for 20 min, then ship on GPU. **Cut line 2:00**: ship whichever accelerator works |
| 2:00–3:45 | Empty-sheet calibration and threshold mask. Mask sanity check. Crop to mask. Memory bank of object patches only. Video teach: 8 fps × 12 s, sharpness filter, k-center 16–32 keyframes, coverage meter. Leave-segment-out scoring. Identity gate (ResNet18 pooled features or MediaPipe embedder until DINOv2 lands) with negatives. Geometry and coverage checks. Four verdicts. Twin save and load. Calibration pass and certificate | Test continuously: same part at 10 placements, 10 wrong objects, 3 marker defects | Teach ≤30 s. Same part passes 10/10 across positions and rotations. At least 9/10 wrong objects flagged "not the enrolled part". Marker dot flagged DEFECT | Video teach flaky: burst teach (auto-capture 20 frames while the part is turned). Mask fails: centre crop plus a REFRAME warning. **Cut line 4:00: this must pass before anything else, because it answers critiques 2 and 5** |
| 3:45–5:15 | Tracker. Line trigger and steady-hold trigger. Best-crop selection. FIFO judge worker. Overlay extrapolation. Tone and vibration. Counters, parts/min, p-chart. JSONL log and export. REPLAY mode (MP4 through the same pipeline) | Record a 60 s REPLAY clip of the hand-fed line early, then run 50-part tests | 50 parts at about 0.3 m/s give 50 verdicts with no double counts. Every seeded defect beeps. 95th-percentile trigger-to-verdict time logged | Tracking flaky: steady-hold trigger only; still a live camera with a beep per part. **Cut line 5:30: freeze the live loop** |
| 5:15–6:30 | DINOv2-S behind a feature flag. [CLS] feeds the identity gate. Top-1% mean scoring. Nearest-view retrieval. Parity test against laptop features | A/B test ResNet18 vs DINOv2 on a mini-set: 20 good, 10 defective, 10 wrong | Parity cosine ≥0.99. ≤30 ms on NPU (≤60 ms on GPU). DINOv2 at least matches ResNet18 on the mini-set | Stay on ResNet18; all object-lock gains are kept. **Hard stop at 6:30** |
| 6:30–7:45 | LiteRT-LM engine warmed at app start. FastVLM on the NPU. Reject card: boxed crop plus facts line gives one streamed sentence. Location guard. Time-to-first-token and tokens/s shown | Airplane-mode tests; log guard outcomes on 20 rejects | In airplane mode, tapping a reject gives a sentence in ≤2 s with an NPU badge | FastVLM on NPU fails within 45 min: Gemma 4 E2B on GPU for 20 min, then template sentence. **Cut line 7:45** |
| 7:45–8:30 | Telemetry screen: per-stage ms, badges, CPU/NPU A/B toggle, thermal status and headroom, battery °C, power estimate. Thermal governor. Certificate card | Start a 30-min REPLAY soak with telemetry logging | A/B toggle shows the speed-up on the same frame; soak running | Drop the power estimate; keep ms and badges |
| 8:30–9:15 | Stretch goals only if every exit test is green: one-tap "Mark OK" correction with undo, then the hole/component count check. Otherwise, fix bugs | Run the proof protocol (see next section) and screen-record it | CSV files and recordings saved | Skip the stretch goals |
| 9:15–10:00 | Release-signed APK and a log-to-CSV summary script | Clean reinstall, push weights, two timed airplane-mode rehearsals, full backup video. Old Expo APK kept installed | Two clean rehearsals | Revert to the last archived green APK |

When time runs short, features are cut in this order: natural-language rules first, then one-tap correction, the count check, DINOv2 (ResNet18 stays), the VLM on the NPU (the GPU or template path stays) and finally the power estimate. Five things are never cut: object lock with the identity gate, video teach with its burst fallback, a live per-part verdict with a beep, proof of airplane-mode operation, and honest accelerator badges.

## Sixteen X→Y claims, each with a proof the team can produce before the round

Every "X" below is either sourced or measured on the current app during hour one. Every "Y" is a **target that goes on a slide only after it is measured on the iQOO 15**; a missed target is shown at its measured value, because a real 88% beats an unproven 99%. The protocols fit the plan's human track and need only the props listed above.

| # | Claim | X (today or incumbent) | Y (target) | Proof produced during the hackathon |
|---|---|---|---|---|
| 1 | Teach time | Current app: 20 aligned photos plus build, stopwatched in hour one. Inspekto: 30–60 min install with 20–30 parts ([automation.com](https://www.automation.com/article/inspekto-releases-s70-autonomous-machine-vision-sy)) | **≤30 s** from "Teach" to "Armed" (10–15 s clip + ≤15 s build) | 3 operators × 3 runs, screen-recorded with an on-screen timer; report median and max |
| 2 | Images needed | About 20 photos (current); 50 normal plus at least 1 abnormal (LandingLens ([docs](https://landinglens.docs.landing.ai/anomaly-detection))) | **One clip**, 16–32 automatically chosen keyframes, **zero defect images** | Twin folder listing and certificate screenshot |
| 3 | Wrong-object acceptance | Old app on 10 objects × 5 presentations, measured in hour one | **0 of 50** dissimilar objects accepted; look-alikes reported separately with the measured identity margin | 50-trial video and CSV |
| 4 | Decision latency | 140–440 ms of compute per photo plus a manual capture (team figures) | **≤100 ms** 95th-percentile trigger-to-verdict on the NPU | Stage timings from the JSONL log over 200 parts; histogram |
| 5 | Throughput | Old app's parts per minute in single-photo mode, stopwatched | **≥120 parts/min** hand-fed at about 0.3 m/s, with 200 of 200 parts judged exactly once; scoring ceiling 60,000 ÷ p95 ms (≈600/min at 100 ms) | Video of a 200-part run plus log counts |
| 6 | Public-benchmark accuracy | PatchCore: 83.4 (1-shot) and 88.8 (4-shot) image AUROC on MVTec AD ([AnomalyDINO](https://arxiv.org/html/2405.14529)) | Reproduce **≈96.5 / 97.6** with the exported DINOv2-S on 3 MVTec AD categories; on-device feature cosine ≥0.99 | Laptop notebook table plus on-device parity log. MVTec AD is non-commercial ([MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad)) but fine for benchmarking |
| 7 | Defect recall on own parts at a fixed false-reject rate | Old app on the same physical set. Its synthetic handheld benchmark: 80% detected at 0% false rejects ([README](file:///D:/Projects/Kaizen_Eye/README.md)) | Recall on ≥20 seeded-defect parts at **≤5% false rejects** on ≥60 good presentations, beating the old app on the same set | Old vs new confusion matrices: same parts, same operator |
| 8 | False-alarm guarantee | None stated. A 20-photo leave-one-out maximum bounds false alarms only at ≈13.9% at 95% confidence (calculated with the rule in [arXiv 2608.15090](https://arxiv.org/html/2608.15090)) | **≤10% at 95% confidence** from 29 distinct good parts (≤5% with 59) | Certificate screen plus a held-out test on good parts |
| 9 | Repeatability | Humans reversed 23% of decisions on re-inspection ([See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)) | **≤5% flip rate** across 20 parts × 10 presentations | 200-decision CSV |
| 10 | Sustained operation | Human detection falls up to 40% within 30 min ([See 2012](https://digital.library.unt.edu/ark:/67531/metadc835891/m2/1/high_res_d/1055636.pdf)); iQOO 15 CPU and GPU drop below 50% of peak under stress ([GSMArena](https://www.gsmarena.com/iqoo_15-review-2905p4.php)) | 30-min REPLAY soak: **zero changed decisions** across loops, **≤10% fps drop** | Telemetry CSV plotted over time |
| 11 | NPU speed-up | Same backbone on the CPU of the same phone (A/B toggle) | **≥10×** backbone speed-up. The literature shows 20–45× for vision encoders on SM8750 ([Phase Matters](https://arxiv.org/html/2606.27906v1)) | Toggle screen recording plus stage CSV |
| 12 | Energy and heat | 10-min CPU-mode run: battery temperature rise and whole-phone watts | **Lower** temperature rise and power in NPU mode. The literature shows 2.52× less energy and 10.47 °C cooler on SM8750 ([Phase Matters](https://arxiv.org/html/2606.27906v1)) | Two unplugged 10-min runs on the same scene; BatteryManager log |
| 13 | Explanation | Score and heat map only. VLM agents need a GPU or the cloud ([ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2026/07/16/omron-advances-inspection-technology-with-nvidia-omniverse-and-metropolis/27914/)) | **≤2 s** offline explanation per reject (vendor time-to-first-token 0.12 s ([FastVLM](https://huggingface.co/litert-community/FastVLM-0.5B))); guard contradiction rate reported | Airplane-mode video plus guard log over 20 rejects |
| 14 | Station cost | ₹1.65 lakh (Keyence IV3) to ≈₹11 lakh (Inspekto) ([IndiaMART](https://www.indiamart.com/proddetail/keyence-vision-sensor-with-built-in-ai-iv3-series-2854380953255.html), [ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/)) | **₹72,999**: 2.3–15.1× cheaper; 3.9 vs 8.9–59.5 months of a skilled inspector's wage | Price links and bill of materials |
| 15 | Marginal cost and connectivity | 1 credit per inference (LandingLens ([plans](https://landinglens.docs.landing.ai/plans))); cloud services being retired | **₹0 per inference**; 100% airplane mode | Entire demo run in airplane mode with an inference counter |
| 16 | Supplier PPM outcome | 2% defect rate × 20–30% human miss = 4,000–6,000 PPM escapes, which is AVTEC's zero-point band ([AVTEC SQM](https://www.avtec.in/pdf/AVTEC%20Supplier%20Quality%20Manual.pdf)) | 2% × human miss × (1 − measured recall from claim 7): for example **200–300 PPM** at 95% recall | Calculated on the slide from claim 7's measured recall |

Four honesty rules protect these numbers. Label literature baselines as literature, since the human miss rates come from Western aerospace, nuclear and automotive studies rather than Indian MSMEs. Describe volunteer panels as volunteers, not trained inspectors. State the operating threshold behind every recall figure. And show one failure case on purpose, such as a reflective part under a changed light: MVTec AD 2 shows the whole field still struggles there ([MVTec AD 2](https://arxiv.org/abs/2503.21622)), and admitting it makes the other fifteen numbers more believable.

## A six-minute phone-only demo that invites the judges to break it

**What you need.** The iQOO 15, with its case off, pre-cooled, and airplane mode on and visible. One plain sheet that contrasts with the parts; this is the only prop, and a plain tabletop also works because the empty-sheet tap learns any uniform background. A handful of identical small parts, three to five of them with seeded defects. A bag of wrong objects.

**Who does what.** One presenter holds the phone 25–35 cm above the sheet with elbows braced, and the other feeds parts. A solo presenter can instead sweep the phone steadily over a pre-laid row of parts. The relative motion is the same, and the threshold detector does not care which side moves.

| Clock | Presenter does | Judges see | Claims proven |
|---|---|---|---|
| 0:00–0:30 | Shows airplane mode and opens telemetry. States the problem with sourced numbers: humans miss 20–30% of defects, inspection stations cost ₹1.65–11 lakh, and cloud inspection services like AWS Lookout for Vision have shut down | Accelerator badges reading NPU; milliseconds per stage | 11, 15 |
| 0:30–1:30 | Taps **Teach** and films a good part for 12 s, turning it slowly and arcing the phone. Runs the 29-part calibration pass live if time allows; otherwise shows the certificate from rehearsal and explains it | Coverage meter fills; Twin armed in under 30 s; certificate shows keyframes, false-alarm bound, identity margin and self-test scratches caught | 1, 2, 8 |
| 1:30–2:15 | Hands judges the wrong objects: a coin, a pen cap, a different nut size. They place them on the sheet | "NOT THE ENROLLED PART" in amber within a second, with the identity margin | 3; round-1 critique 5 fixed |
| 2:15–3:15 | Slides 15 parts across the trigger line at hand speed; three are defective | Each part gets an ID and one verdict. Defective parts get a red box that rides the part, a beep and a vibration. Counters, parts per minute and trigger-to-verdict milliseconds update | 4, 5, 7 |
| 3:15–3:45 | Taps a reject | One-sentence explanation in under 2 s, with the NPU badge, time to first token and tokens per second | 13 |
| 3:45–4:15 | Toggles CPU vs NPU on the same frame; shows thermal status | Milliseconds drop sharply; temperature stays steady | 11, 12 |
| 4:15–5:15 | Presents slides: the measured X→Y table, the PPM escape arithmetic and the cost table | Numbers backed by videos and CSV files | 9, 10, 14, 16 |
| 5:15–6:00 | Invites a judge to try one more object or defect. Shows a deliberate failure case (a shiny part under changed light) and explains why | An honest limit, stated up front | Credibility |

**If something breaks on stage**, each failure has a rehearsed answer. If the live camera misbehaves through glare, flicker or a stuck tracker, switch to REPLAY mode, which carries a visible badge and runs the same pipeline, then return to live for the wrong-object test, which needs only the steady-hold trigger. If the VLM fails to load, the template explanation keeps the reject card working. If everything else fails, the old Expo APK stays installed as a photo-mode last resort.

**Do not film a laptop screen as a fake conveyor.** Moiré patterns can read as texture anomalies. Screen refresh and PWM flicker also push exposure toward 1/30–1/60 s ([Wikipedia](https://en.wikipedia.org/wiki/Flicker_(screen))), which directly conflicts with the sub-millisecond exposure that freezes motion.

## Thirteen risks, the earliest warning of each, and the fallback already built

Most risks fall into three groups. The first is integration risk on a sideloaded NPU stack, which is time-boxed and has GPU fallbacks. The second is physics risk from light, blur and heat, which the capture preset and thermal governor contain. The third is credibility risk with judges, which measured numbers and deliberate honesty defuse. The table orders risks by how much they could damage the round.

| Risk | Likelihood | Impact | Early warning | De-risking and fallback |
|---|---|---|---|---|
| Agent time overrun, or an integration stall | High | High | A missed exit test at a cut line | Strict time boxes, an APK archived at every cut line, three screens only, stretch goals last |
| Glare, reflections or exposure drift cause false rejects | High | High | Good parts rejected; p-chart drifts | Torch and locked 3A; teach under the demo preset; matte parts on a matte sheet; sensitivity slider. MVTec AD 2 shows lighting shift remains unsolved ([MVTec AD 2](https://arxiv.org/abs/2503.21622)) |
| NPU runtime will not start in a sideloaded build on SM8850 | Medium | High | GPU fallback engages; dispatch errors like those reported on SM8750 ([LiteRT #6059](https://github.com/google-ai-edge/LiteRT/issues/6059)) | Test the Gallery SM8850 build in hour 0. Automatic GPU fallback. QNN delegate 2.50.0 next. Then the ONNX Runtime QNN recipe with its four fixes: declare `libcdsprpc.so`, set `ADSP_LIBRARY_PATH`, use legacy JNI packaging, pin the QNN runtime version ([itsallgoody](https://github.com/itsallgoody/onnxruntime-qnn-snapdragon-8-elite-gen5)) |
| Threshold too tight (correlated frames, too few calibration parts) | Medium | High | Good parts fail the held-out test | Leave-segment-out scoring; 29 distinct calibration parts; SuperADD's percentile × gain as a fallback ([SuperADD](https://arxiv.org/html/2605.14808v1)) |
| DINOv2 export or quantisation hurts accuracy or speed | Medium | Medium | Parity cosine below 0.99, or the NPU rejects an operator | Fixed-resolution FP16 or w8a16 export done before the clock; ResNet18 fallback keeps all object-lock gains |
| VLM fails on the NPU, or its cold start stalls the demo | Medium | Medium | Initialisation over 10 s; errors like [LiteRT #5499](https://github.com/google-ai-edge/LiteRT/issues/5499) | Warm up behind the splash screen; Gemma 4 E2B on the GPU; template sentence; weights downloaded and pushed in advance |
| Thermal throttling mid-demo | Medium | Medium | Thermal status reaches moderate; fps falls | Neural stages on the NPU; 30 fps cap and idle down-clocking; sustained power mode; case off, pre-cooled phone; 30-min soak in rehearsal. The iQOO 15 held clocks for about 10 minutes under stress ([GSMArena](https://www.gsmarena.com/iqoo_15-review-2905p4.php)) |
| A look-alike object passes the identity gate | Medium | Medium | Small identity margin on the certificate | DINOv2 [CLS] embedding; geometry and coverage checks; a demo part with a distinctive shape; margin shown openly |
| No manual exposure, or blur and rolling shutter at speed | Medium | Medium | Low crop sharpness; blurred heat maps | Read camera capabilities in hour 0; fall back to AE lock plus torch; demo speed limited to 0.2–0.3 m/s; teach in motion |
| Tracker double-counts or misses parts because of hand shake | Medium | Medium | Log count differs from parts fed | Require 2–3 frames past the line; count by direction; judge only tracks first seen upstream; steady-hold trigger as fallback |
| Two copies of the Qualcomm libraries clash (vision and LLM) | Low–medium | Medium | Native-library load errors | Keep both on the LiteRT family; if they still clash, move the VLM to the GPU |
| Airplane-mode surprise from a library that downloads models at runtime | Low–medium | High | Anything that fails only offline | Bundle every model; avoid ML Kit object tracking and the GenieX catalog; rehearse twice in airplane mode |
| Judges dismiss the novelty ("Inspekto and Elementary do this"), or mistake the self-test for an accuracy claim | High | Medium | Questions in the pitch | Show the competitor table; concede that each component exists; claim the offline, handheld phone-NPU combination plus the self-issued certificate; label the synthetic self-test a sanity check, citing the SWSA results ([SWSA](https://arxiv.org/html/2310.10461)) |

## Conclusion

The judges' five criticisms reduce to two engineering omissions. The system never knew what the object was, and it never stated its own operating point. Both are cheaper to fix than a model upgrade. Geometry (mask, object crop, identity gate) fixes the first, and statistics (an order-statistic threshold published as a certificate) fixes the second. That changes the goal for round two. The winning move is not a *better* anomaly detector but a *self-describing* one: the phone publishes, per part, the evidence that competitors' "99.9%" brochures leave out. The same logic applies to the chip. On the 8 Elite Gen 5 the vision models are so cheap that the showcase is not raw NPU utilisation. It is what the spare capacity buys while the phone stays cool: higher resolution, votes across several crops, and explanations in plain words.

The plan's ordering matters as much as its contents. Because object lock, video teaching, live per-part verdicts and the certificate come first, and all run on the ResNet18 backbone and GPU that already work, the round-one critiques are answered even if both time-boxed upgrades (DINOv2 and the NPU VLM) fail. The phone's real limits remain: no strobe, no encoder, no PLC I/O. So its defensible market is where no station exists today, the manual final-inspection table at Tier-2/3 suppliers. There, its durable asset is not the model. It is the Twin, the certificate and the per-part log, which together double as quality evidence for customer audits and for climbing ZED levels.
