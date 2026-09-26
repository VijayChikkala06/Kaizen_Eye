# Video / 3D enrolment and object-locked inspection for Kaizen Eye (on-device, Android, offline)

Scope: how a phone-only, offline inspection app can (A) enrol a "good" part from a 10-20 s video (or 3D capture) instead of ~20 photos, and (B) lock onto the object so that background, clutter, pose and lighting stop dominating the anomaly score, including rejecting a completely different object. Current Kaizen Eye pipeline for reference: centre crop -> ImageNet ResNet18 patch features at 320 px -> PatchCore-style memory bank from ~20 good photos -> kNN patch distance -> leave-one-out threshold. Target device: iQOO 15 (Snapdragon 8 Elite Gen 5). Build window: ~10 h by an AI coding agent.

Device-labelling note: all Qualcomm AI Hub numbers below were measured by Qualcomm on "Snapdragon 8 Elite Gen 5 For Galaxy Mobile" (and older chips). The iQOO 15 uses the standard Snapdragon 8 Elite Gen 5, so expect similar but not identical numbers. MediaPipe numbers were measured on Google Pixel phones (Tensor chips), not Snapdragon.

---

## Q1. Video enrolment: how to turn a 10-20 s video into a good enrolment set (keyframe selection, frame count, coverage)

### Takeaway
No paper was found that does "video enrolment" for few-shot industrial anomaly detection specifically. The right approach is to put together standard parts: a per-video relative sharpness filter (variance of Laplacian), then diversity sampling with greedy k-center (farthest-first) on global embeddings, then PatchCore's own greedy coreset on patches. Few-shot evidence shows about 8-16 well-chosen aligned views capture most of the gain. Walk-around (pose-varying) enrolment needs far more views, or a pose-aware memory bank.

### Cited Findings
**Sharpness / blur scoring**
- The variance-of-Laplacian focus measure comes from Pech-Pacheco et al. (ICPR 2000, diatom autofocusing). It is computed as `cv2.Laplacian(image, cv2.CV_64F).var()`. Blurry images have fewer edges and so lower variance. — [PyImageSearch](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
- The PyImageSearch example uses a fixed threshold of 100.0, but the author warns you "often need to tune it on a per-dataset basis". Too low a threshold marks sharp images blurry, and too high a threshold lets blurry ones through. — [PyImageSearch](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)

**Motion-based frame selection (as used for SfM / Gaussian splatting capture)**
- The open-source `adaptive-frame-extractor` picks keyframes "dynamically based on camera motion". It extracts fewer frames when the camera is stationary or slow, and more during fast movement or rotation. It has Low/Medium/High density presets. Its README says "Blur/sharpness-aware automatic selection is not yet implemented." — [morishuz/adaptive-frame-extractor](https://github.com/morishuz/adaptive-frame-extractor)

**Diversity / dedup sampling (k-center / farthest-point)**
- PatchCore builds its memory bank with greedy coreset subsampling. It iteratively picks the feature that is farthest from the already-selected set (a greedy k-center approximation). This removes redundancy and cuts memory and inference time. — [PatchCore paper (arXiv 2106.08265)](https://arxiv.org/html/2106.08265v2); [Anomalib PatchCore docs](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/patchcore.html)
- Anomalib's PatchCore default `coreset_sampling_ratio` is 0.1, i.e. it keeps about 10% of patch embeddings. — [Anomalib PatchCore docs](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/patchcore.html)

**How many frames are enough (fixed/aligned view)**
- AnomalyDINO (DINOv2 ViT-S, 672 px, patch kNN), MVTec-AD image AUROC by number of normal shots: 1-shot 96.6, 2-shot 96.9, 4-shot 97.7, 8-shot 98.2, 16-shot 98.4, full-shot 99.5. — [AnomalyDINO (arXiv 2405.14529)](https://arxiv.org/html/2405.14529)
- The same method on VisA: 1-shot 87.4, 2-shot 89.7, 4-shot 92.6, 8-shot 93.8, 16-shot 94.8, full-shot 97.6. — [AnomalyDINO](https://arxiv.org/html/2405.14529)
- AnomalyDINO raised one-shot MVTec-AD from 93.1% to 96.6% AUROC, is training-free, and needs no extra data. — [AnomalyDINO abstract](https://arxiv.org/abs/2405.14529)

**How many views for full pose coverage (walk-around / 3D)**
- The PAD / MAD-Sim benchmark (20 LEGO toys) has 4,000+ multi-pose RGB images, about 200 training views per object, and training images come with full camera poses (transforms.json). — [PAD GitHub](https://github.com/EricLee0224/PAD)
- SplatPose used 210 training views per category on MAD. — [SplatPose (arXiv 2404.06832)](https://arxiv.org/html/2404.06832)
- PADFormer (Aug 2026) runs pose-agnostic detection from sparse reference views: N in {2, 4, 10, all}, with no camera poses needed. — [PADFormer (arXiv 2608.04210)](https://arxiv.org/html/2608.04210)

**Capture guidance from phone 3D-capture apps (aggregator source, lower reliability)**
- KIRI Engine recommends "3 complete orbits at different heights" for small objects. Polycam auto-captures at about 1 frame/s, typically 120-240 frames in 2 minutes. General advice: move at "walking pace or slower" to avoid motion blur. — [Polyvia3D mobile capture guide](https://www.polyvia3d.com/guides/gaussian-splatting-mobile-capture)
- PocketGS (on-device 3DGS research) uses 50 images per scene captured on the phone. — [PocketGS (arXiv 2601.17354)](https://arxiv.org/html/2601.17354v3)

**Video-based enrolment literature**
- The searches found few-shot industrial AD methods that use a handful of normal reference images (e.g. FastRecon, VisionAD). None of them enrol a static industrial part from a handheld video. — [FastRecon ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf); [VisionAD (arXiv 2504.11895)](https://arxiv.org/html/2504.11895v2)

### Inferences
- **Recommended frame-selection recipe** (all cheap, all on-device):
  1. Pull frames straight from the CameraX analysis stream at about 5-10 fps for 10-20 s, giving roughly 60-200 candidates. This avoids encoding and then decoding an MP4.
  2. Compute the Laplacian variance on a downscaled grey image, ideally only inside the object mask or centre region. Drop the bottom 30-50% within that video. Use a relative, per-video percentile rather than PyImageSearch's absolute 100. The metric measures edge content, so a sharp but texture-poor part (e.g. a matte black mouse) scores low even when in focus. Ranking frames within one video of the same object avoids that problem.
  3. Embed the survivors with a global descriptor: global-average-pooled ResNet18 features (already in the pipeline and sub-millisecond on the NPU, see Q3), or MediaPipe's MobileNetV3 embedder.
  4. Run greedy k-center (farthest-first) to pick K = 16-32 keyframes. This one step does both diversity sampling and dedup; stop early when the next farthest distance falls below a small epsilon, which means the remaining frames are near-duplicates.
  5. Build the patch memory bank from the K frames and apply PatchCore greedy coreset (about 10%) to keep kNN fast.
- **Frame count:** the AnomalyDINO curve suggests 8-16 diverse, aligned views get within about 1 point of the 16-shot result on MVTec. For handheld video with pose, scale and light jitter, 16-32 keyframes is a reasonable target. Full 360-degree coverage needs far more (PAD and SplatPose use about 200 posed views). Scope the demo to "inspection side up, moderate viewpoint change" rather than arbitrary pose.
- **Fixed-view vs walk-around:** for a hackathon demo, guide the user to a slow arc of about ±20-30 degrees around the inspection face, with small distance changes, and optionally a second pass at another height. Full walk-around enrolment is only worth it if the pose-aware memory bank from Q4 is built too; otherwise views from the far side just make the bank more permissive.
- **Threshold pitfall specific to video:** adjacent video frames are near-duplicates. Plain leave-one-out on video frames therefore leaves a near-identical neighbour in the bank, so normal scores come out too low and the threshold ends up too tight (false rejects at test time). Use leave-one-*segment*-out: exclude a ±1-2 s window, or split the video into 4-5 temporal blocks and score each block against the others.
- **Ratings (accuracy gain / phone runtime / effort for ~10 h build):**
  - Relative Laplacian sharpness filter — gain Medium (stops blurred frames making the bank permissive and inflating the threshold) / runtime trivial, estimated a few ms per frame on CPU at about 480 px / effort about 0.5 h.
  - Greedy k-center on global embeddings (diversity + dedup) — gain Medium-High / runtime negligible, about 1 ms per frame on NPU plus an O(N·K) CPU loop / effort about 1 h.
  - Optical-flow motion-adaptive sampling — gain Low (redundant once k-center is in place) / runtime moderate / effort 1-2 h. Skip, or only use gyro or frame-difference to reject shaky frames.
  - Patch-level greedy coreset (10%) — gain neutral on accuracy but keeps kNN latency flat as frames are added / effort about 1 h.
  - Live UX coverage meter (show how many diverse keyframes have been captured so far) — gain indirect (better enrolments) / effort about 1 h.

### Gaps
- No published study was found that measures how many video keyframes a PatchCore/AnomalyDINO-style detector needs under handheld pose jitter. The 16-32 target is an inference.
- No source was found that gives a validated absolute Laplacian threshold for phone-captured industrial parts; it has to be calibrated per video.
- No paper was found on "video enrolment" for industrial anomaly detection as such.

---

## Q2. 3D enrolment (NeRF / 3D Gaussian Splatting / CAD rendering) and what multi-view / pose-agnostic AD research shows

### Takeaway
3D enrolment can deliver pose-agnostic detection, but only with dense posed capture (about 200 views) and GPU-scale compute. It collapses to near-chance with 4 views. On-device 3DGS training exists only on iOS (Scaniverse, PocketGS research), taking about 1-5 minutes. No open Android implementation was found. For a ~10 h Android build it is not realistic. A multi-view 2D memory bank from video captures most of the practical benefit.

### Cited Findings
**Phone 3D capture apps**
- Niantic's Scaniverse (announced March 2024): "The entire process of splat capture and training happens locally on a user's device". Training "takes around a minute". The article mentions iOS devices only. — [80.lv (Mar 22, 2024)](https://80.lv/articles/capture-photorealistic-3d-scenes-on-your-phone-with-gaussian-splatting)
- **Conflict:** an aggregator guide says Scaniverse, Polycam, Luma and KIRI all process splats "in the cloud", with Scaniverse at 3-5 min and Android support. It also says Polycam and KIRI support Android and Luma is iOS-only. This contradicts the Niantic-sourced on-device claim above; weight the 80.lv/Niantic statement higher for Scaniverse. — [Polyvia3D](https://www.polyvia3d.com/guides/gaussian-splatting-mobile-capture)
- The same aggregator says LiDAR-enhanced Polycam "produced 2.1M Gaussians with no holes". — [Polyvia3D](https://www.polyvia3d.com/guides/gaussian-splatting-mobile-capture)

**On-device 3DGS training (research)**
- PocketGS (arXiv 2601.17354) trains 3DGS fully on an iPhone 15 (A16) in Swift + Metal:
  - Per-scene time: 255-389 s (about 4-5 min) from 50 images.
  - Peak memory: 2.21 GB average (1.82-2.65 GB) during training.
  - Quality: LPIPS 0.225 vs 0.398 (workstation 3DGS-SfM) and 0.281 (3DGS-MVS); PSNR 23.67 vs 21.16 dB against the sparse baseline.
  - The paper does not mention Android.
  — [PocketGS](https://arxiv.org/html/2601.17354v3)

**Pose-agnostic anomaly detection (PAD / MAD dataset)**
- The MAD dataset has 20 LEGO toys:
  - MAD-Sim: 4,000+ multi-pose views.
  - MAD-Real: 10 classes, 7,000+ simulated and real images.
  - Anomaly types: Stains, Burrs, Missing.
  - Training needs full camera poses.
  — [PAD GitHub](https://github.com/EricLee0224/PAD); [PAD paper (NeurIPS 2023 D&B)](https://arxiv.org/abs/2310.07716)
- OmniposeAD pipeline: anomaly-free NeRF, then coarse-to-fine pose estimation of the query, then comparison of the rendered reference with the query. — [PAD GitHub](https://github.com/EricLee0224/PAD)
- PAD README results in Pixel-AUROC / Image-AUROC format:
  - OmniposeAD 97.8 / 90.9
  - PatchCore 74.7 / 78.5
  - DRAEM 89.4 / 58.0
  - FastFlow 90.8 / 71.3
  - UniAD 88.5 / – 
  
  In other words, PatchCore loses heavily when test poses differ from training poses. — [PAD GitHub](https://github.com/EricLee0224/PAD)
- SplatPose (3DGS instead of NeRF; coarse pose via LoFTR, then differentiable pose refinement):
  - Image AUROC 93.9 vs OmniposeAD 90.9 vs PatchCore 78.5.
  - Pixel AUROC 99.5 vs 98.4 (OmniposeAD).
  - AUPRO 95.8 vs 86.6.
  - Training about 4 min 54 s vs about 4 h 34 min (55x faster).
  - Inference about 5 s vs 66 s per image (13x faster), on GPU.
  - Uses 210 training views per category.
  
  Its OmniposeAD pixel figure (98.4) differs from the PAD README (97.8), perhaps a re-run or different averaging. — [SplatPose (arXiv 2404.06832)](https://arxiv.org/html/2404.06832)
- PADFormer (arXiv 2608.04210, 2026): a ViT with MAE-pretrained encoder that reconstructs an anomaly-free version of the query in image space from sparse reference views, with no camera poses and no NeRF/3DGS.
  - 4-shot on MAD-Sim, image / pixel AUROC: PADFormer 81.3 / 92.9, "OmniAD" 50.6 / 84.6, SplatPose 52.2 / 86.7.
  - 4-shot on PIAD, image AUROC: 80.7 vs 47.5 vs 48.8.
  - Inference: 0.74-1.75 s per image on an NVIDIA A10.
  — [PADFormer](https://arxiv.org/html/2608.04210)

**Multi-view AD datasets**
- Real-IAD (CVPR 2024):
  - 150K high-resolution images of 30 objects.
  - 5 cameras: one top view plus four at about 45 degrees.
  - Images 2,000-5,000 px.
  - 8 defect types: pit, deformation, abrasion, scratch, damage, missing parts, foreign objects, contamination.
  - PatchCore I-AUROC is reported as 90.4 in the single-view setting vs 89.4 multi-view, with sample-level S-AUROC 93.7. The exact setting behind each figure should be checked in the paper tables.
  — [Real-IAD (arXiv 2403.12580)](https://arxiv.org/html/2403.12580)
- MANTA: 137.3K images, 38 categories in 5 domains, tiny objects, each specimen captured from 5 viewpoints, 8.6K anomalous images with pixel masks, plus a text subset. — [MANTA (arXiv 2412.04867)](https://arxiv.org/abs/2412.04867)
- MVTec 3D-AD: 10 categories scanned with a high-resolution industrial 3D sensor. Defects: scratches, dents, holes, contaminations, deformations. The authors say 3D AD has "considerable room for improvement". — [MVTec 3D-AD (arXiv 2112.09045)](https://arxiv.org/abs/2112.09045)
- Eyecandies: synthetic, photo-realistic procedurally generated candies, 10 classes, rendered "under multiple lighting conditions", with RGB + depth + normal maps. The authors report extra modalities can raise detection performance. — [Eyecandies (arXiv 2210.04570)](https://arxiv.org/abs/2210.04570)
- MVTec AD 2 (2025): 8 scenarios, 8,000+ images, including lighting-condition changes, transparent/overlapping objects and very small defects. State-of-the-art methods stay "below 60% average AU-PRO". Lighting shift is still an open problem even for SOTA. — [MVTec AD 2 (arXiv 2503.21622)](https://arxiv.org/abs/2503.21622)

### Inferences
- The PAD numbers show why Kaizen Eye's PatchCore breaks under pose change: a pose-naive PatchCore gets 78.5 image AUROC on MAD vs about 91-94 for pose-aware 3D methods. The pose-aware methods need about 200 posed views plus GPU minutes-to-hours, and PADFormer's table shows them near chance (about 50%) with only 4 views.
- A 10-20 s phone video does not yield reliable camera poses without SfM (COLMAP-class). No Android on-device SfM + 3DGS training pipeline was found, and PocketGS/Scaniverse are iOS. **3D enrolment is not realistic in a 10 h Android build.** Cloud apps (Polycam, Luma) break the offline requirement.
- CAD rendering needs CAD models of whatever the judges bring, which won't exist for ad-hoc objects such as a mouse. Skip it for the demo.
- **Ratings:**
  - On-device NeRF/3DGS enrolment — gain High in theory for arbitrary pose / runtime minutes of training plus seconds per query even on GPU / effort far beyond 10 h (no Android code path). **Do not attempt.**
  - Novel-view augmentation from 3DGS — same blocker.
  - "Pseudo-3D" pose-aware multi-view 2D memory bank from video keyframes (see Q4) — gain Medium-High for moderate viewpoint change / runtime within budget / effort 1-2 h. **Recommended substitute.**
  - PADFormer-style image-space reconstruction — promising for sparse views, but GPU-scale (0.7-1.75 s per image on A10) and needs a trained ViT. Not feasible in 10 h.
- **Pitch framing:** "Video enrolment builds a multi-view reference set, like a lightweight 3D model of the good part, without needing 3D reconstruction." This is defensible given the evidence above.

### Gaps
- No published Android on-device 3DGS training times were found.
- Scaniverse's current (2026) processing mode and Android support could not be confirmed from a primary source (the Niantic Medium post returned 403).
- No sources were gathered on CAD-to-image rendering for synthetic normals/defects.
- PADFormer's accuracy with 10 or "all" views and its full baseline tables were not extracted.

---

## Q3. Object localisation / segmentation on mobile before anomaly scoring

### Takeaway
Masking the background is the single most direct fix for "doesn't stick to the object". AnomalyDINO shows zero-shot foreground masking plus rotation augmentation adds about 2 AUROC points on object categories. On the Snapdragon 8 Elite Gen 5 NPU, class-agnostic segmenters are cheap: FastSAM-S about 3 ms, EdgeTAM 3-8 ms, MobileSAM decoder about 2-3 ms. MobileSAM's encoder numbers on AI Hub are inconsistent. For a 10 h build, the lowest-effort robust path is tap-or-centre-point prompting (MediaPipe Interactive Segmenter, or FastSAM-S "everything" plus picking the mask under the point).

### Cited Findings
**Why masking helps (evidence from AD literature)**
- AnomalyDINO's zero-shot masking:
  - Method: threshold the first PCA component of DINOv2 patch features, then dilate and apply morphological closing.
  - Safeguard: a "masking test" on the first reference sample; masking is only used if the test passes.
  - Scope: used for objects (e.g. Capsule, Hazelnut, Screw), not for textures (Wood, Tile, Leather), and not where the test failed (Cable, Transistor, Bottle, Metal nut, Zipper).
  — [AnomalyDINO](https://arxiv.org/html/2405.14529)
- AnomalyDINO preprocessing (masking + rotation) improves detection AUROC by "approximately 2%" without significant runtime increase. — [AnomalyDINO](https://arxiv.org/html/2405.14529)
- AnomalyDINO-S takes about 60 ms per image at 448 px on an NVIDIA A40 GPU. It scores images by cosine distance and the mean of the 1% most anomalous patches. — [AnomalyDINO](https://arxiv.org/html/2405.14529)

**Qualcomm AI Hub on-device numbers (NPU)**
- ResNet18 (Kaizen Eye's current backbone) at 224x224 on 8 Elite Gen 5 For Galaxy:
  - TFLite: 0.591 ms float, 0.233 ms w8a8.
  - QNN: 0.586 ms float, 0.271 ms w8a8.
  - On 8 Gen 3: 0.92 ms TFLite float.
  — [qualcomm/ResNet18](https://huggingface.co/qualcomm/ResNet18)
- FastSAM-S (640x640, 45.1 MB float) on 8 Elite Gen 5 For Galaxy:
  - TFLite 3.088 ms, QNN 3.287 ms, ONNX 3.714 ms (float, NPU).
  - 8 Elite: 3.917 ms TFLite. 8 Gen 3: 5.155 ms TFLite.
  — [qualcomm/FastSam-S](https://huggingface.co/qualcomm/FastSam-S)
- MobileSAM on AI Hub (input 720x1280; encoder 26.6 MB, decoder 23.7 MB float):
  - Decoder on 8 Elite Gen 5: 2.233 ms TFLite, 2.385 ms QNN, 2.65 ms ONNX.
  - **Encoder numbers are internally inconsistent.** 8 Elite Gen 5: 499 ms TFLite / 640 ms QNN / 749 ms ONNX. 8 Elite: 74.7 ms ONNX but 678 ms QNN. 8 Gen 3: 81.0 ms QNN but 888 ms ONNX.
  — [qualcomm/MobileSam (Hugging Face)](https://huggingface.co/qualcomm/MobileSam); [AI Hub MobileSAM page](https://aihub.qualcomm.com/mobile/models/mobilesam)
- EdgeTAM (a lightweight SAM 2 for on-device video segmentation and tracking; 1024x1024 encoder; components: encoder 8.3M params, memory encoder 1.2M, video decoder 6.22M): the card lists 7.88 ms float and 3.039 ms w8a8 (ONNX, NPU) on 8 Elite Gen 5, and 9.405 / 3.581 ms on 8 Elite. The fetched summary did not make clear which component these rows cover. — [qualcomm/EdgeTAM](https://huggingface.co/qualcomm/EdgeTAM)
- Depth-Anything-V2-Small (24.7M params, DINOv2 ViT-S-based, 518x518) runs in 18.027 ms (ONNX float, NPU) on 8 Elite Gen 5 For Galaxy. It is a proxy upper bound for running a DINOv2-S backbone on this NPU. — [qualcomm/Depth-Anything-V2](https://huggingface.co/qualcomm/Depth-Anything-V2)
- The AI Hub catalogue includes MobileSam, FastSam-S/X, SAM2, SAM3, EdgeTAM, YOLOv8/v11/YOLO26-Segmentation, YOLOE-Segmentation, OpenAI-Clip, SigLIP2, ResNet18/50, MobileNet-v3, and Depth-Anything V1/V2/V3. It does **not** list SuperPoint, LightGlue, XFeat, DINOv2 (standalone), U^2-Net or EfficientSAM. — [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)

**SAM-family papers / repos**
- MobileSAM: image encoder about 5M params (SAM: 611M); total 9.66M params; 12 ms per image on a single GPU (8 ms encoder + 4 ms decoder). SAM takes 456 ms; FastSAM has 68M params and takes 64 ms. Apache-2.0, ONNX export available. — [MobileSAM GitHub](https://github.com/ChaoningZhang/MobileSAM)
- EdgeSAM: 38.7 FPS on iPhone 14, "40-fold" faster than SAM and "14 times" faster than MobileSAM on edge devices. 9.6M params, 22.1 GFLOPs at 1024x1024. +2.3 mIoU (COCO) and +3.2 (LVIS) over MobileSAM. Point (positive/negative) and box prompts; CoreML and ONNX exports; NTU S-Lab License 1.0. — [EdgeSAM GitHub](https://github.com/chongzhou96/EdgeSAM)

**Tap-to-select on Android (MediaPipe)**
- The MediaPipe Interactive Segmenter splits an image into "a selected object and everything else" from user points or strokes. It uses the MagicTouch model (MobileNetV3-like CNN with a custom decoder, 768x768 input, int8). Latency on Pixel 10: CPU 208.2 ms, GPU 580.6 ms. It outputs a float32 confidence mask, supports Android/Python/Web, and has a new stateful API that encodes image features once for repeated interactions. — [MediaPipe Interactive Segmenter](https://developers.google.com/edge/mediapipe/solutions/vision/interactive_segmenter)

### Inferences
- **Root cause of the judges' failure:** with a centre crop and a background-dominated memory bank, most test patches (table, mat, hand) find close neighbours, so a different object only moves the score a little. Restricting scoring to object patches, and normalising the crop to the object's bounding box, turns a wrong object into a mostly-anomalous patch set.
- **Background subtraction** (MOG2-style) assumes a static camera, which does not hold for handheld phone-only capture. Skip it.
- **Recommended path (lowest risk):** segment once per enrolment keyframe and once per test shot, using a centre-point prompt by default and a tap override in the UI.
  - Option A: MediaPipe Interactive Segmenter. It is a ready Android Tasks API with no model conversion, and about 200 ms CPU per image (Pixel 10) is fine for about 30 keyframes plus single test shots.
  - Option B: FastSAM-S "everything" via AI Hub TFLite (about 3 ms NPU for the network, plus Kotlin post-processing for mask assembly and NMS). Pick the mask containing the tap or centre point with sensible area bounds.
- MobileSAM/EdgeSAM give better box/point-prompted masks. However, the AI Hub MobileSAM encoder timings are erratic (75-888 ms), and EdgeSAM has no published Snapdragon numbers and a non-Apache licence. Treat these as stretch options, benchmarked on the iQOO before committing.
- EdgeTAM is the natural "tap once on the first video frame, track the mask through the enrolment video" model. It is attractive but has three components plus memory handling, which is high integration risk in 10 h.
- DINOv2-PCA masking à la AnomalyDINO would need a custom DINOv2 export (not in the AI Hub catalogue). The Depth-Anything-V2-Small proxy (18 ms at 518 px) suggests runtime is fine, but export and integration risk is high. As a cheap fallback, try PCA over ResNet18 patch features on enrolment frames to separate object from border patches. This is untested; validate visually.
- Copy AnomalyDINO's safeguard: run a mask sanity test (area ratio within e.g. 5-80% of the frame, one dominant connected component, not touching all borders). Fall back to the unmasked centre crop and warn the user if it fails.
- **Ratings:**
  - Mask-restricted scoring plus bbox-normalised crop — gain **High** (directly fixes background dominance and helps wrong-object rejection) / runtime: MediaPipe about 200 ms CPU or FastSAM-S about 3 ms NPU plus post-processing / effort 2-3 h.
  - MobileSAM/EdgeSAM prompted — gain High / runtime uncertain on this chip / effort 3-4 h, risk medium-high.
  - EdgeTAM video propagation — gain High for enrolment consistency / runtime 3-8 ms listed / effort 4+ h, risk high.
  - DINOv2 PCA mask — gain High (proven in AnomalyDINO) / runtime about 18 ms or less (proxy) / effort 3-4 h plus export risk.
  - YOLO-seg (COCO classes) — gain Low for arbitrary industrial parts (class-specific).

### Gaps
- No Qualcomm AI Hub latency was fetched for YOLOE-Segmentation, SAM2 or SAM3 on 8 Elite Gen 5.
- No Snapdragon numbers were found for EdgeSAM, EfficientSAM or U^2-Net.
- The MobileSAM encoder rows on AI Hub are contradictory, and the reason was not determined.
- MediaPipe Interactive Segmenter latency on a Snapdragon 8 Elite Gen 5 was not found; only Pixel 10 figures are published.

---

## Q4. Alignment to a canonical pose vs pose-aware multi-view memory banks

### Takeaway
For a 3D handheld object, full geometric alignment is only reliable for near-planar parts. Classic homography (ORB/AKAZE + RANSAC) and ECC work there; ECC is photometrically invariant. The best return for effort is:
1. Normalise translation and scale from the object mask (crop to the mask bounding box and resize).
2. Handle in-plane rotation with rotation augmentation of the memory bank, which AnomalyDINO found at least as good as explicit alignment.
3. Handle viewpoint change with a pose-aware memory: retrieve the nearest enrolment views by global embedding, then do patch kNN against those views only.

### Cited Findings
- AnomalyDINO's default "agnostic" setting rotates all reference samples to enrich the memory bank, and "agnostic preprocessing slightly outperforms the informed counterpart" (where rotations are known). — [AnomalyDINO](https://arxiv.org/html/2405.14529)
- RegAD uses registration (feature alignment with spatial transformer networks) as a proxy task. It compares registered test features with registered support features, and reports 3-8% AUC gains over prior few-shot methods on MVTec and MPDD. It needs pre-training. — [RegAD (arXiv 2207.07361)](https://arxiv.org/abs/2207.07361)
- OpenCV's standard recipe for locating a known object: detect and match features (ORB descriptors with a Hamming-distance BFMatcher), then `findHomography` with RANSAC. It needs at least 4 correct correspondences and returns an inlier mask; `perspectiveTransform` then maps the object outline. — [OpenCV feature homography tutorial](https://docs.opencv.org/3.4.20/d1/de0/tutorial_py_feature_homography.html); [OpenCV AKAZE and ORB planar tracking](https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html)
- ECC alignment (`findTransformECC`, Evangelidis & Psarakis 2008) estimates a parametric motion model by maximising the Enhanced Correlation Coefficient. It is "invariant to photometric distortions in contrast and brightness", and its iterative scheme is linear, so it is computationally efficient. — [LearnOpenCV ECC article](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/); [ECC author page](https://sites.google.com/site/georgeevangelidis/ecc)
- `cv.minAreaRect()` returns the centre (x, y), (width, height) and rotation angle of the minimum-area rotated rectangle around a contour; `cv.boxPoints()` gives its corners. — [OpenCV contour features tutorial](https://docs.opencv.org/4.13.0/dd/d49/tutorial_py_contour_features.html)
- In the older convention, minAreaRect's angle lies in [-90, 0), and the "next edge" is used once rotation passes 90 degrees. — [TheAILearner](https://theailearner.com/tag/angle-of-rotation-by-cv2-minarearect/)
- XFeat is a lightweight learned local feature, "up to 5x faster" than deep local features. It runs in real time for sparse VGA matching on a laptop i5 CPU (vanilla PyTorch), and at about 1,400 FPS batched on an RTX 4090. With LighterGlue it scores AUC 0.564/0.710/0.819 vs SuperPoint+LightGlue 0.591/0.738/0.841. The repo has **no** ONNX or mobile export yet. Apache-2.0. — [XFeat GitHub](https://github.com/verlab/accelerated_features); [XFeat paper](https://arxiv.org/abs/2404.19174)
- SuperPoint, LightGlue and XFeat are not in the Qualcomm AI Hub catalogue. — [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)
- Pose-aware 3D methods estimate the query pose first (SplatPose: coarse pose via LoFTR, then gradient refinement; rotation error 0.040 rad vs iNeRF 0.056), then compare against a pose-matched reference. — [SplatPose](https://arxiv.org/html/2404.06832)
- VisionAD applies identical augmentations to query and reference images to get multi-view anomaly scores from a patch-level memory bank. — [VisionAD (arXiv 2504.11895)](https://arxiv.org/html/2504.11895v2)

### Inferences
- **Mask-based normalisation is the cheapest big win:** crop to the mask's bounding box plus about 10% margin, resize to a fixed side, and zero or ignore background patches. This removes translation and scale variation and framing differences. Effort about 1 h; runtime negligible.
- **In-plane rotation:**
  - Use `minAreaRect` on the mask, or second-order image moments, to rotate to a canonical angle. Rectangle symmetry leaves a 90/180-degree ambiguity, and the angle convention differs between OpenCV versions.
  - Resolve the ambiguity by scoring against the memory bank at 0/90/180/270 degrees and keeping the minimum. Alternatively, follow AnomalyDINO and add rotated copies of the enrolment frames to the bank (for example 8 x 45 degrees). This multiplies bank size, so apply coreset afterwards.
  - Gain Medium; effort about 1 h.
- **Viewpoint (out-of-plane) change — pose-aware multi-view memory bank:**
  1. Store each enrolment keyframe's global embedding alongside its patch features.
  2. At test time, retrieve the top-k (e.g. 3-5) nearest keyframes by cosine similarity.
  3. Run patch kNN only against those keyframes' patches, optionally with neighbouring cells only, as a lightweight positional prior.
  
  This mirrors the "retrieve pose, then compare" logic of OmniposeAD/SplatPose without 3D. It also speeds up kNN because the bank per query is smaller. Gain Medium-High for walk-around enrolment; effort 1-2 h.
- **Homography (ORB/AKAZE + RANSAC) against the retrieved nearest keyframe:** useful for near-planar parts such as PCBs, labels and flat panels. It fails on textureless or strongly 3D objects (few inliers). Gate it on inlier count and inlier ratio, and fall back to mask normalisation otherwise. Runtime is likely tens of ms on CPU at 320-640 px (no phone benchmark found). Effort 2 h. Gain Medium for planar parts, Low otherwise.
- **ECC refinement** after coarse (mask/homography) alignment adds sub-pixel accuracy and robustness to brightness/contrast. It needs a good initial warp; use a pyramid and an inputMask. Effort 1 h. Gain Low-Medium; mostly helps pixel-level heatmaps.
- XFeat, SuperPoint and LightGlue: no mobile export path within 10 h. Skip. RegAD needs training. Skip.

### Gaps
- No published phone-CPU timings were found for ORB+RANSAC or ECC on Snapdragon 8 Elite Gen 5.
- No paper was found that directly evaluates "retrieve nearest enrolment view, then patch-kNN" on PAD/Real-IAD. The pose-aware bank is an engineering inference from the OmniposeAD/SplatPose design.
- The OpenCV findTransformECC reference page returned 403, so exact parameter semantics were not re-verified. The four supported motion types (translation/Euclidean/affine/homography) come from general knowledge and the LearnOpenCV article title/summary.

---

## Q5. Wrong-object / out-of-distribution rejection

### Takeaway
Rejecting "a completely different object" is semantic novelty detection, which is much easier for pretrained features than subtle defect detection. kNN scoring on (lightly adapted) ImageNet-pretrained features reached 96.2% one-class AUROC on CIFAR-10 in 2020 (PANDA). The recommended design is a separate gate before defect scoring: a global-embedding cosine check against the enrolled keyframes, with a threshold self-calibrated from enrolment, plus mask shape/area and object-count sanity checks. No published "typical cosine margins" for this setting were found; calibrate on-device.

### Cited Findings
- PANDA (ImageNet-pretrained features adapted to the normal class, scored with kNN) reached 96.2% ROC-AUC on one-class CIFAR-10 (each class in turn is normal, the other classes are anomalies), vs a previous self-supervised SOTA of 90.1%. — [PANDA (arXiv 2010.05903)](https://arxiv.org/pdf/2010.05903)
- A 2021 study argues that "learning representations from in-domain data may be unnecessary for outlier detection". A single ImageNet-pretrained feature extractor gives "competitive or better performance" on outlier-detection benchmarks. — [arXiv 2105.09270](https://arxiv.org/abs/2105.09270)
- On-device embedders:
  - OpenAI CLIP ViT-B/16 image encoder (224x224): 13.245 ms on 8 Elite Gen 5 For Galaxy, 16.662 ms on 8 Elite, 21.83 ms on 8 Gen 3 (ONNX float, NPU). — [qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)
  - SigLIP2 is also in the AI Hub catalogue. — [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)
  - ResNet18 at 224 runs at 0.23-0.59 ms on 8 Elite Gen 5 NPU. — [qualcomm/ResNet18](https://huggingface.co/qualcomm/ResNet18)
- MediaPipe Image Embedder offers MobileNet-V3 small and large (224x224). Latency on Pixel 6: small 3.94 ms CPU / 7.83 ms GPU; large 9.75 ms CPU / 9.08 ms GPU. It has a built-in cosine-similarity utility, L2 normalisation and scalar-quantisation options on Android. — [MediaPipe Image Embedder](https://developers.google.com/edge/mediapipe/solutions/vision/image_embedder)
- AnomalyDINO checks whether its PCA foreground mask captures the object on the first reference sample before using masking. This is a precedent for automatic "is the object segmentable/present" sanity checks. — [AnomalyDINO](https://arxiv.org/html/2405.14529)

### Inferences
- **Gate 1: global embedding similarity (primary).**
  - Compute an L2-normalised embedding of the masked, bbox-normalised crop. The cheapest option is GAP over ResNet18 layer3/layer4 features (already computed). The best semantic separation is CLIP ViT-B/16 (13 ms NPU) or SigLIP2. The simplest API is MediaPipe MobileNetV3.
  - Score = max cosine similarity to the K enrolment keyframes, plus optionally the mean of the top 3.
  - **Self-calibrate:** compute leave-segment-out similarities among enrolment frames (the "same object" distribution) and the similarities of a small bundled "negative" set (e.g. 50-100 generic desk objects, hands, tabletops, embedded as vectors at build time) to the enrolment set. Place the threshold between them, e.g. midway between the 5th percentile of positives and the 99th percentile of negatives, and warn if they overlap.
  - Gain **High** for the judges' "mouse vs other objects" test / runtime 1-15 ms / effort 1-1.5 h.
- **Gate 2: mask geometry.** Compare mask area fraction, minAreaRect aspect ratio and Hu moments against the enrolment distribution (mean ± 3σ). This catches objects of very different shape or size. Gain Medium / runtime negligible / effort about 1 h.
- **Gate 3: object count / clutter.**
  - Count large connected components or masks (e.g. FastSAM "everything" masks above an area threshold in the ROI).
  - Return "multiple objects / clutter: reframe" instead of a verdict.
  - Gain Medium for robustness / effort about 1 h if FastSAM is already integrated.
- **Gate 4: patch-coverage.** Report the fraction of object patches above the patch threshold. A wrong object typically lights up most of its patches, whereas a defect lights up a small cluster. This gives a three-way outcome, "WRONG OBJECT" vs "DEFECT" vs "OK", using data already computed. Gain Medium / effort about 0.5 h.
- **UX:** show three distinct outcomes: PASS, DEFECT (with heatmap), and NOT THE ENROLLED PART. This directly answers the judges' critique.

### Gaps
- No published cosine-similarity margins were found for "same instance under viewpoint change vs different object" for DINOv2 CLS, CLIP or SigLIP embeddings on phone captures. They must be measured on-device.
- The PANDA figure is for CIFAR-10 class-level novelty. Instance-level discrimination of similar-looking parts (e.g. two different black mice) will be harder, and no source quantifies it.

---

## Q6. Self-calibrating the threshold with synthetic defects (no real defect samples)

### Takeaway
Cheap synthetic defects (CutPaste, Perlin-mask texture/colour blends as in DRAEM, Poisson-blended patches as in NSA) are proven as training signals, with 96.6-98.0 image AUROC on MVTec. The evidence that synthetic anomalies reliably predict real-defect performance or select models/thresholds is mixed. SWSA shows it works well for semantic anomalies on natural images, but "is less effective" on MVTec/VisA, where CutPaste-style anomalies were the better proxy. Use synthetic defects as a sensitivity sanity check and to set a floor, not as a guarantee.

### Cited Findings
- CutPaste cuts an image patch and pastes it at a random location, then learns to tell altered images from normal ones. On MVTec AD it gains 3.1 AUC from scratch over prior methods and reaches 96.6 image AUC with ImageNet transfer. — [CutPaste (arXiv 2104.04015)](https://arxiv.org/abs/2104.04015)
- DRAEM uses a Perlin-noise generator for random anomaly shapes filled with augmented DTD texture images. It reports 98.0 image AUROC on MVTec, 2.5 points above the prior best unsupervised method, and (I-AUROC, P-AUROC, PRO) = (98.0, 97.3, 93.0). — [DRAEM overview (alphaXiv)](https://www.alphaxiv.org/overview/2108.07610v2); [DRAEM paper](https://arxiv.org/abs/2108.07610)
- NSA (Natural Synthetic Anomalies) uses Poisson image editing to seamlessly blend scaled patches from separate images, reaching an overall MVTec detection AUROC of 97.2. — [NSA (arXiv 2109.15222)](https://arxiv.org/abs/2109.15222)
- SWSA (Selection With Synthetic Anomalies, arXiv 2310.10461):
  - **How the synthetic validation sets are built:** from a small normal support set (20 seed images, giving 100 synthetic anomalies for CUB/VisA/MVTec; "as few as 10 samples" possible), using CutPaste and training-free diffusion style interpolation (DiffStyle, γ=0.7).
  - **Natural images:** CUB picked the best model 109/200 times (selected-model AUROC 0.988 vs oracle 0.991). Flowers picked 62/102 (0.994 vs 0.997), with Kendall's τ 0.866 for ranking.
  - **Industrial:** MVTec-AD 2/15 picks (0.706 vs oracle 0.757) and VisA 0/12 (0.764 vs 0.824) with diffusion anomalies.
  - **Authors' conclusion:** "For MVTec-AD and VisA ... SWSA is less effective". Subtle local defects sit close to the normal data, and CutPaste anomalies were the better industrial proxy (0.674 vs 0.643 on VisA one-vs-closest).
  - **Theory:** total-variation bounds exceeded 0.5 in all settings, i.e. there is no tight guarantee.
  — [SWSA](https://arxiv.org/html/2310.10461)

### Inferences
- **On-device calibration recipe:**
  1. For each held-out enrolment segment (leave-segment-out, see Q1), generate 5-10 synthetic defects *inside the object mask*:
     - (a) CutPaste / scar: a small rectangle or thin rotated strip copied from elsewhere on the object or another frame.
     - (b) DRAEM-style: a Perlin-noise mask blended with a colour/brightness shift, blur or a patch from a different region. Procedural textures avoid bundling DTD.
     - (c) Occlusion or "missing" blobs: fill with local background colour.
     - (d) Optionally NSA-style Poisson blending.
  2. Score the synthetic defects with the exact same pipeline.
  3. **Threshold:** set τ from the normal leave-segment-out scores (e.g. max, or the 99th percentile plus a margin). Report the synthetic-defect detection rate at τ as "estimated sensitivity". If it is below e.g. 80%, warn "enrolment too variable: record again with steadier framing / more even light".
  4. **Model/hyper-parameter selection:** if choosing between two feature layers or resolutions, SWSA suggests CutPaste-style anomalies are the better industrial proxy, but expect weak agreement with real defects. Keep choices fixed rather than auto-tuned in a 10 h build.
- **Why threshold on normals first:** false alarms on good parts are what judges will see most. SWSA's industrial results warn that synthetic-defect-optimised thresholds can be over-optimistic about real subtle defects.
- **Ratings:**
  - CutPaste/scar synthetic check — gain Medium (credible "self-test" in the demo, catches bad enrolments) / runtime about 5-10 extra scorings per held-out frame, seconds in total at enrolment / effort about 1 h.
  - Perlin + colour/texture blend — gain Medium / effort 1-1.5 h.
  - Poisson/NSA blend — gain Low-Medium over the above / effort 1-2 h.
  - Diffusion-generated anomalies (AnomalyDiffusion, DiffStyle) — not feasible on-device in 10 h.
- The synthetic self-test also doubles as a demo feature: show judges the "self-check heatmaps" on synthetic scratches right after enrolment.

### Gaps
- No study was found that validates threshold (rather than model) selection from synthetic anomalies for patch-kNN detectors against real industrial defects.
- DRAEM's numbers here come from a secondary overview page (alphaXiv) and search summaries, not from the paper PDF itself.

---

## Q7. Which combination should Kaizen Eye build in ~10 hours? (synthesis of Q1-Q6)

### Takeaway
Build order by value/effort:
1. Object mask plus bbox-normalised crop and mask-restricted scoring.
2. A global-embedding wrong-object gate with self-calibrated threshold.
3. Video enrolment with sharpness filter and k-center keyframe selection.
4. A pose-aware (nearest-view) memory bank plus rotation augmentation.
5. Leave-segment-out threshold plus a synthetic-defect self-test.

Skip on-device 3D (NeRF/3DGS/CAD), learned matchers and learned registration. Every component above has published evidence or fits the latency budget on Snapdragon 8 Elite Gen 5 class NPUs.

### Cited Findings
- Background masking plus rotation augmentation gives about +2% AUROC in AnomalyDINO. — [AnomalyDINO](https://arxiv.org/html/2405.14529)
- Pose-naive PatchCore scores 78.5 image AUROC vs 90.9-93.9 for pose-aware methods on MAD. — [PAD GitHub](https://github.com/EricLee0224/PAD); [SplatPose](https://arxiv.org/html/2404.06832)
- 3D pose-aware methods drop to about 50% image AUROC with 4 views. — [PADFormer](https://arxiv.org/html/2608.04210)
- Latency building blocks on 8 Elite Gen 5 For Galaxy (NPU): ResNet18 at 224 in 0.23-0.59 ms; FastSAM-S at 640 in 3.1 ms; CLIP ViT-B/16 in 13.2 ms; Depth-Anything-V2-Small (DINOv2-S proxy) at 518 in 18.0 ms. — [ResNet18](https://huggingface.co/qualcomm/ResNet18); [FastSam-S](https://huggingface.co/qualcomm/FastSam-S); [OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip); [Depth-Anything-V2](https://huggingface.co/qualcomm/Depth-Anything-V2)
- MediaPipe Interactive Segmenter runs at 208 ms CPU on Pixel 10. — [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/interactive_segmenter)
- Few-shot saturation: MVTec 8-shot 98.2 vs 16-shot 98.4 (AnomalyDINO). — [AnomalyDINO](https://arxiv.org/html/2405.14529)
- On-device 3DGS training is only published for iOS (PocketGS 4-5 min, 50 images, iPhone 15; Scaniverse about 1 min). — [PocketGS](https://arxiv.org/html/2601.17354v3); [80.lv](https://80.lv/articles/capture-photorealistic-3d-scenes-on-your-phone-with-gaussian-splatting)

### Inferences
- **Suggested 10 h plan (AI coding agent):**
  1. **(2-3 h) Object focus.**
     - Integrate MediaPipe Interactive Segmenter; FastSAM-S via AI Hub TFLite is the alternative.
     - Use a centre-point default prompt and a tap override.
     - Add the mask sanity test.
     - Crop to mask bbox plus margin, resize to 320, and zero out or exclude background patches from both the memory bank and scoring.
     - Score = mean of the top 1% object-patch distances (AnomalyDINO-style).
  2. **(1-1.5 h) Wrong-object gate.** Global embedding from ResNet18 GAP (free) or CLIP/MobileNetV3, max-cosine to enrolled keyframes, threshold from enrolment positives vs bundled negatives, and a mask-geometry check. Outputs: PASS / DEFECT / WRONG PART / REFRAME.
  3. **(2 h) Video enrolment.**
     - A 10-20 s capture from the CameraX analysis stream at about 5-10 fps, with an on-screen coverage meter.
     - Relative Laplacian filter, then k-center selection of 16-32 keyframes (dedup built in).
     - Segment each keyframe; PatchCore greedy coreset at 10%.
  4. **(1-1.5 h) Pose robustness.** Rotation-augmented bank (4-8 angles; AnomalyDINO "agnostic") plus nearest-view retrieval (top-3 keyframes by global embedding) before patch kNN.
  5. **(1-1.5 h) Calibration and self-test.** Leave-segment-out normal scores set τ. CutPaste/Perlin synthetic defects inside the mask estimate sensitivity, and are shown as a post-enrolment self-check.
  6. **(1 h buffer)** On-device benchmarking (confirm each stage's ms on the iQOO 15), lighting advice in the UI (MVTec AD 2 shows lighting shift remains hard even for SOTA), and demo scripting.
- **Expected effect on the judges' tests:**
  - "Other object shown": caught by Gate 1/2 and by patch coverage. High confidence for dissimilar objects; lower for look-alikes.
  - "Background/framing changes": largely neutralised by masking and bbox normalisation.
  - "Pose changes": moderate viewpoint change handled by keyframe diversity plus nearest-view retrieval. Large out-of-plane changes remain out of scope; say so in the pitch and guide the user to present the enrolled face.
  - "Lighting": partially handled by video diversity (different angles give different highlights) and ECC-style photometric invariance if alignment is added. Otherwise instruct users to keep lighting consistent.

### Gaps
- None of the combined-pipeline accuracy numbers above have been measured. The individual components have published evidence, but the gains are not additive by construction. Validate on a small held-out set of real good and deliberately damaged parts on the device before the demo.
