# Competitive landscape: AI visual defect / anomaly inspection (commercial and open-source), as of Sept 2026, for Kaizen Eye

Research date: 26 Sep 2026. Flags used: **[OLD]** means the source is older than 2024 and may be out of date. **[VENDOR CLAIM]** means marketing numbers with no independent test protocol. **[COMPETITOR-AUTHORED]** means the source is written by a rival vendor. **[SNIPPET]** means the figure came from a search-result summary and the full page was not read.
Kaizen Eye baseline (from the user brief, no external source): Expo/React Native Android app. It extracts ImageNet ResNet18 patch features at 320 px with TFLite on the CPU, builds a PatchCore-style greedy-coreset memory bank from about 20 good photos, sets the threshold by leave-one-out, and scores by kNN. It works on single photos only.

---

## Q1. Commercial products: approach, images needed, training and labels, hardware and price, speed, strengths and weaknesses

### Takeaway
Commercial tools have converged on one recipe. They learn from good parts only, need about 20 to 50 images (as few as 1 to 10 for some tools), and run inference on a fixed smart camera or edge PC. Most add optional cloud training and analytics. Kaizen Eye's current algorithm (PatchCore-style kNN on about 20 good photos) is therefore the same kind of system as Siemens Inspekto, Zebra Aurora, Keyence VS, LandingLens AD and Edge Impulse FOMO-AD, and is not ahead of them. The 2025-26 frontier features are:
- continual or instant learning (HALCON 25.11, Neurala L-DNN)
- generative synthetic defects (UnitX GenX)
- VLM "agents" for reasoning (Omron with NVIDIA Cosmos Reason)
- cloud-trained, edge-run fleets (Cognex OneVision, Elementary)

Meanwhile the hyperscalers are pulling out of turnkey visual inspection: AWS Lookout for Vision reached end of life on 31 Oct 2025, Azure Percept was retired on 30 Mar 2023, Azure Custom Vision retires on 25 Sep 2028, and Google Visual Inspection AI's status is unclear.

### Cited Findings

#### Compact comparison table (details and sources in the bullets below)
| Product | Algorithm family | Images needed | Defect images or labels? | Where it runs | Hardware / price class | Speed claim | Status |
|---|---|---|---|---|---|---|---|
| Siemens Inspekto (S70 lineage) | Unsupervised anomaly (good-only) | 20 good (20-30 in S70-era sources) | No | Bundled Siemens IPC, camera and light (edge) | ~EUR 10k (2019) [OLD] | n/a | Active under Siemens |
| Keyence IV4 (AI vision sensor) | AI Identify (master matching) plus AI imaging/background removal | 1 registered image | No for Identify | On-sensor | Entry USD 2-4k [COMPETITOR-AUTHORED] | n/a | Active |
| Keyence VS (smart camera) | Edge-learning defect detection | ~20 good (integrator claim) | No | On-camera | Mid USD 8-15k [COMPETITOR-AUTHORED] | n/a | Active |
| Cognex In-Sight 3800 / 2800 edge learning | Pre-trained edge-learning classifiers | 5-10 examples | Yes, class labels | On-camera | Entry USD 3-6k [COMPETITOR-AUTHORED]; used 2800 ~USD 2.1k | "line speeds" | Active |
| Cognex In-Sight D900 (ViDi Detect/Red) | Deep-learning anomaly from good parts | n/a | No for Detect | On-camera, no PC | n/a | n/a | Active [OLD launch] |
| Cognex OneVision | Cloud training and governance, edge inference; includes "Anomaly Detect" | n/a | n/a | Cloud train, In-Sight 3900/6900 run | n/a | "solution in less than a day" | GA 13 May 2026 |
| Landing AI LandingLens AD | Reverse distillation, 160M params | 50 normal recommended | At least 1 abnormal (ideally 10+) to set the threshold | Cloud train; cloud/Docker/LandingEdge run | SaaS | n/a | Active |
| MVTec HALCON / MERLIC | Deep-learning AD plus Global Context AD (logical anomalies) | "few" good | No | PC/embedded (software licence) | Licence (n/a) | "seconds or minutes" to train | Active; GCAD since 22.05 [OLD] |
| Zebra Aurora Deep Learning | Reconstruction AD plus one-class per-region AD | 20-30 | No (unsupervised) | Zebra cameras/PC via Aurora DL Editor | n/a | n/a | Active (2024 tools) |
| Teledyne DALSA Astrocyte | Anomaly-detection classifier | "few tens" | No, or unbalanced | PC training, runtime on Teledyne SW | n/a | <10 min training | Active |
| Elementary (VisionStream) | Learns "good" by watching the line | "under 60 seconds" | No labels | Edge controller plus AWS cloud (QualityOS) | Turnkey cameras, lights, controllers | up to 1,000 ppm | Active |
| Instrumental (Discover/Detect) | Unsupervised CNN anomaly | "5 units" (Discover); 30 images (older Detect) | No | Managed cloud or own environment; drop-in stations | Enterprise SaaS | n/a | Active |
| Neurala VIA | Lifelong-DNN (incremental learning) | n/a | Annotation, trains instantly | On-prem, cloud or offline | n/a | n/a | Active; OEM licensing 2026 |
| UnitX (OptiX + CorteX + GenX) | Supervised DL plus programmable 32-light imaging plus synthetic defects | "few images" per defect; GenX from 3 real defects | Yes | CorteX Edge inference | Turnkey | 1 m/s capture; up to 100 MP | Active |
| Averroes.ai | Supervised no-code DL on existing AOI images | 20-40 per defect class | Yes | Software on existing systems | n/a | "99%+" [VENDOR CLAIM] | Active |
| Roboflow | Supervised RF-DETR/YOLO plus embedding "Identify Outliers" | Labeled sets | Yes (supervised path) | Cloud or edge inference | SaaS tiers | n/a | Active |
| Omron + NVIDIA Metropolis VSS | VLM (Cosmos Reason 7B) plus LLM agents | n/a | n/a | GPU edge/cloud | NVIDIA GPU/Jetson | n/a | Announced Jul 2026 |
| Edge Impulse FOMO-AD | PatchCore scoring | n/a | No | MCU to NPU to GPU | Enterprise tier only | n/a | Since Oct 2024 |
| AWS Lookout for Vision | Cloud anomaly/segmentation | n/a | Normal/anomalous labels | Cloud plus edge app | Pay per use | n/a | **DISCONTINUED**: new customers closed 10 Oct 2024; end of life 31 Oct 2025 |
| Google Visual Inspection AI | Anomaly/defect/assembly models | "as few as 10 labelled images" (2021) | Labels | Cloud train, edge inference | GCP | n/a | **Unclear in 2026** (see below) |
| Kaizen Eye (current) | PatchCore-style kNN, ResNet18 at 320 px | ~20 good photos | No | Phone CPU (TFLite) | Phone | Single photo | Prototype |

#### Siemens Inspekto
- Only "20 good samples" are needed and "no defective samples"; setup is "Unbox, Mount, Ready to use"; it integrates with PLC, MES/ERP and TIA Portal; it does anomaly and presence detection; it is described as "self-adaptive" to surface variation. — [Siemens Inspekto page](https://www.siemens.com/en-us/company/artificial-intelligence/industrial-ai/inspekto-ai-inspection/)
- S70-era setup [OLD, 2018-2019]: the user traces the outline of the item (the field of view) with a mouse, presents 20-30 good items, and starts inspecting. Installation takes 30-60 minutes by shop-floor staff with no integrator. — [automation.com](https://www.automation.com/article/inspekto-releases-s70-autonomous-machine-vision-sy); [New Equipment Digest](https://www.newequipment.com/product-directory/controls-instrumentation/inspection-equipment/product/55119028/72507-inspekto-s70-autonomous-machine-vision-ai-inspection-system)
- It ships with a Siemens Industrial PC, camera, light and all cables. — [SE Automation, Aug 2025](https://seawi.com/2025/08/11/siemens-inspekto-ai-based-visual-inspection/)
- Siemens showed Inspekto as "newly acquired" at Global Industrie in March 2025. — [DirectIndustry e-Magazine, 12 Mar 2025](https://emag.directindustry.com/2025/03/12/global-industrie-siemens-showcases-inspekto-its-newly-acquired-and-democratized-ai-vision-tool-for-quality-inspection/)
- Price of about EUR 10,000 for the S70 [OLD, 2019] [SNIPPET]. — [ManufacturingTomorrow, Apr 2019](https://www.manufacturingtomorrow.com/news/2019/04/10/inspekto-%E2%80%94-the-gold-mine-of-productivity/13232/)

#### Keyence (IV4 and VS)
- IV4 features:
  - "AI Imaging II" tests "over 10,000 conditions" to pick the best image settings.
  - "AI Target Extraction" removes background "sources of false detection".
  - "AI Identify" gives stable detection "by registering just a single image".
  - It outputs X, Y and θ positional deviation.
  - An "AI trigger" removes the need for a separate trigger sensor.
  - No cycle time is published on the page.
  — [Keyence IV4](https://www.keyence.com/products/vision/vision-sensor/iv4/); [Control.com on IV4-H](https://control.com/news/ai-powered-imaging-keyence-introduces-iv4-h-series-vision-sensor)
- VS series: "With an average of 20 images, it quickly learns to identify what a 'good' product looks like and flags any deviations" (integrator blog). — [SDC Automation](https://sdcautomation.com/blog/keyence-ai-vision-systems-an-sdc-supplier-spotlight/)

#### Cognex
- In-Sight 3800 edge learning trains with "as few as 5 to 10 example images". Its tasks are classification and OCR, "from presence/absence detection to multi-class sortation". Processing is on the device. ViDi Red is the Cognex anomaly-detection tool. — [Cognex In-Sight 3800](https://www.cognex.com/en-se/products/machine-vision/2d-machine-vision-systems/in-sight-3800)
- In-Sight D900: the ViDi Detect tool "learns from images of good parts" to find defects "that do not need to be pre-defined". It runs on the smart camera without a PC. — [Cognex D900 tools](https://www.cognex.com/products/machine-vision/2d-machine-vision-systems/in-sight-d900/tools); [Gibson Engineering](https://www.gibsonengineering.com/pages/insight_d900) [OLD product generation]
- OneVision:
  - Announced 9 Jun 2025, with beta on In-Sight 3800/8900. GA was announced 13 May 2026, with ">100 customers".
  - Architecture: cloud for training, management and governance; inspection runs on In-Sight 3900/6900 with "no cloud connectivity required during production".
  - Customer claim (Essity): "viable solution in less than a day".
  - Scaling costs "reduced by up to 50%".
  — [PR Newswire, 13 May 2026](https://www.prnewswire.com/news-releases/cognex-onevision-adoption-ramps-as-manufacturers-scale-ai-vision-globally-302768368.html); [Cognex PR, Jun 2025](https://www.cognex.com/company/press-releases/2025/onevision-software-press-release)
- OneVision "Anomaly Detect automatically flags anomalies and out-of-spec conditions". — [Cognex OneVision page](https://www.cognex.com/en/products/machine-vision-software/one-vision)
- A used In-Sight 2800 (IS2801C) was listed at USD 2,099.99. — [eBay listing](https://www.ebay.com/itm/276486796739)
- [COMPETITOR-AUTHORED, Averroes, 4 May 2026] price bands:
  - Smart cameras: Cognex USD 3-6k entry, Keyence USD 2-4k entry; both USD 8-15k mid-range; advanced AOI USD 15-50k+ (Cognex) and USD 20-40k (Keyence).
  - Integration/NRE: USD 20k-150k.
  - Claim that both "typically need hundreds to thousands of labeled images per defect class". **This contradicts** Cognex's own "5 to 10 images" and the ~20-image VS claim above.
  — [Averroes blog](https://averroes.ai/blog/cognex-vs-keyence-vision-systems)

#### Landing AI LandingLens
- Anomaly Detection guidance:
  - "Start with 50 Normal images": at least 10 ideal plus at least 10 with natural variation.
  - Abnormal images: "at least 1, but ideally 10+", in the Dev set to set the threshold.
  - Maximum 1,000 images.
  - Architecture: "reverse distillation (RD) paradigm with 160 million parameters".
  - Warning: "use a consistent background... Otherwise, the model might flag different backgrounds as anomalies."
  — [LandingLens docs](https://landinglens.docs.landing.ai/anomaly-detection)
- The platform is no-code with one-button training. It deploys to the cloud, Docker or LandingEdge (factory edge). Visual Prompting speeds up labelling. — [Labellerr 2026 overview](https://www.labellerr.com/blog/top-computer-vision-model-training-service-providers/); [Landing AI release notes](https://docs.landing.ai/landinglens/landinglens-release-notes)

#### MVTec HALCON / MERLIC
- Anomaly detection: "No labeling required – training with 'good' images only"; "only a few defect-free images"; training "in seconds or minutes". Global Context AD finds "missing components, deformed parts, or incorrectly positioned objects". — [MVTec AD method page](https://www.mvtec.com/knowledge-base/technologies/deep-learning/methods/anomaly-detection)
- Global Context Anomaly Detection was introduced in HALCON 22.05 as a "world's first". It "understands" the logical content of the whole image and catches missing or incorrectly arranged components [OLD, 2022 but still shipping]. — [MVTec PR 22.05](https://www.mvtec.com/company/press-room/press-releases/detail/mvtec-sets-new-standards); [MVTec GCAD article](https://www.mvtec.com/knowledge-base/news/article/global-context-anomaly-detection-in-mvtec-halcon)
- HALCON 25.11 (12 Nov 2025) adds "Continual Learning – Classification":
  - "only five to ten images" per class
  - needs "only a standard CPU"
  - resistant to catastrophic forgetting
  - classes can be added or updated "without the need for complete retraining"
  — [MVTec PR](https://www.mvtec.com/company/press-room/press-releases/detail/mvtec-introduces-new-deep-learning-feature-continual-learning-in-halcon-2511)

#### Zebra Aurora Deep Learning
- Anomaly tools train unsupervised, "only needing normal references". "Just 20-30 image samples" are needed. — [Zebra PR 2024](https://www.zebra.com/gb/en/about-zebra/newsroom/press-releases/2024/zebra-technologies-adds-new-deep-learning-tools-to-aurora-machine-vision-software.html)
- There are two methods: image reconstruction, and "Anomaly Detection 2" one-class classification of each part of the image. Running it needs Aurora Deep Learning Editor. A threshold-analysis UI is provided. — [Zebra Aurora Focus docs](https://docs.zebra.com/us/en/machine-vision/zebra-aurora-focus-user-guide/c-aurora-using-machine-vision-tools/c-aurora-presence-absense-tools/t-aurora-mint-using-anomaly-detection.html); [Zebra developer portal](https://developer.zebra.com/products/machine-vision/aurora-deep-learning)

#### Teledyne DALSA Astrocyte
- Anomaly detection is for when "only good samples are available" and can use unbalanced data. "A few tens of samples" are needed and training takes "under 10 minutes with good data". — [Teledyne Astrocyte](https://www.teledynedalsa.com/en/products/imaging/vision-software/astrocyte/); [1stVision](https://www.1stvision.com/cameras/Teledyne-DALSA-Astrocyte)

#### Elementary
- VisionStream is "AI that watches your line, learns what 'good' looks like, and catches up to 99.9% of defects—in under 60 seconds". "Edge controllers build AI models in under 60 seconds". No labelling is needed. It claims "up to 99.9% accuracy at 1,000 ppm" [VENDOR CLAIM]. QualityOS stores data "in the AWS cloud". — [Elementary home](https://www.elementaryml.com/); [Elementary platform](https://www.elementaryml.com/product/platform)
- VisionLink adds this AI to existing (legacy) vision cameras. — [Elementary VisionLink](https://www.elementaryml.com/product/visionlink)

#### Instrumental
- Discover AI surfaces unexpected variation with "5 units and an anomaly-detecting AI". Processing is "Managed cloud or within your own environment". It uses drop-in stations and ingests data from existing AOI and X-ray systems. Its focus is electronics (AI servers, consumer electronics). — [Instrumental product](https://instrumental.com/product/)
- Older Detect blog: Detect needs "no training, and no golden units" and "starts working after seeing only 30 images" [likely OLD]. — [Instrumental Detect](https://instrumental.com/resources/company/instrumental-detect-automated-defect-detection-using-machine-learning/)

#### Neurala VIA
- VIA uses Lifelong-DNN. "Images being trained immediately as they're annotated" means there is no separate model-building step. It deploys on-prem, in the cloud, on a local server or with "no connectivity". — [Neurala L-DNN blog](https://www.neurala.com/how-lifelong-dnn-helps-neurala-via-be-competitive-for-visual-inspections/); [GlobalSpec](https://insights.globalspec.com/article/16848/neurala-via-flexible-vision-ai-implementation)
- On 1 Jul 2026 Neurala expanded its technology-licensing programme to bring edge vision AI to OEM hardware and software. — [Neurala press releases](https://www.neurala.com/news/press-releases/)

#### UnitX
- OptiX has "32 independently controllable lighting sources" and "fly capture speeds of 1m/s". CorteX Edge handles "up to 100 MP". Models are "sample efficient", needing "only a few images to train on new defect types". — [UnitX bearing application](https://www.unitxlabs.com/industry/bearing-machining-inspection-application/)
- GenX generates synthetic defect images "from as few as three real samples". It claims accuracy gains "up to 9×" and says one EV-battery supplier cut training time by 80% [VENDOR CLAIM]. — [A3 news](https://www.automate.org/news/unitx-unveils-genx-a-generative-ai-breakthrough-for-industrial-inspection-unitx-labs)

#### Averroes.ai
- Claims "20–40 images per defect class" and "99%+ accuracy and near zero false positives" [VENDOR CLAIM]. It is a no-code platform that works with existing inspection systems and needs "no extra hardware". — [Averroes](https://averroes.ai/); [Averroes features](https://averroes.ai/features/ai-defect-detection)

#### Roboflow
- Offers two paths:
  - Supervised: RF-DETR/YOLO trained on labelled defects.
  - Unsupervised: Workflows "embedding blocks and the Identify Outliers block".
- Borderline cases go to a human review queue, and "Vision Events" logs every inspection.
- Roboflow positions itself as a migration target for Lookout for Vision.
— [Roboflow visual anomaly detection](https://blog.roboflow.com/visual-anomaly-detection/); [Roboflow YOLO anomaly](https://blog.roboflow.com/yolo-for-anomaly-detection/); [Roboflow L4V migration](https://blog.roboflow.com/aws-lookout-for-vision-migration/)

#### NVIDIA Metropolis / Omron
- Omron combines the NVIDIA Metropolis VSS blueprint with Cosmos. VSS "uses Cosmos Reason, an open vision language model (VLM)", together with an LLM, to reason and propose next steps. The agents "identify defects, reason potential root causes". Cosmos Reason is a 7B-parameter VLM. — [ManufacturingTomorrow, 16 Jul 2026](https://www.manufacturingtomorrow.com/news/2026/07/16/omron-advances-inspection-technology-with-nvidia-omniverse-and-metropolis/27914/); [NVIDIA blog](https://blogs.nvidia.com/blog/physical-ai-partners-metropolis-updates-siggraph/)

#### Edge Impulse
- Visual Anomaly Detection (FOMO-AD) added PatchCore scoring, "available for Enterprise users". It targets devices "from MCU to NPU and GPU". Post dated 4 Oct 2024. — [Edge Impulse blog](https://www.edgeimpulse.com/blog/patchcore-boosts-visual-anomaly-detection-in-edge-impulse/)

#### Indian vendors
- Lincode LIVIS is a no-code platform for engineers, operators and QA to "train and deploy AI models for new parts, defects, and production lines". — [Lincode](https://lincode.ai/product); [Metrology News](https://metrology.news/lincodes-livis-platform-sets-new-standard-for-ai-powered-visual-quality-inspection/)
- Qualitas Technologies (Bangalore, founded 2008) offers the EagleEye DLOps platform (launched 2021) plus custom mechanical rigs for complex parts. — [Qualitas](https://qualitastech.com/products/); [Business Standard 2021](https://www.business-standard.com/content/press-releases-ani/qualitas-technologies-launches-qualitas-eagleeye-platform-for-ai-powered-visual-inspection-121091600535_1.html) [OLD launch]

#### Discontinued or uncertain hyperscaler offerings
- **AWS Lookout for Vision is discontinued.** New customers were blocked from 10 Oct 2024. Support ended 31 Oct 2025, after which the console and resources became inaccessible. — [AWS docs](https://docs.aws.amazon.com/lookout-for-vision/latest/developer-guide/su-awscli-sdk.html)
- AWS's recommended alternatives are AWS Partner solutions, SageMaker (JumpStart defect models) and Bedrock foundation models. — [AWS ML blog, 10 Oct 2024](https://aws.amazon.com/blogs/machine-learning/exploring-alternatives-and-seamlessly-migrating-data-from-amazon-lookout-for-vision/)
- Lookout had an edge "Defect Detection App" for offline analysis on edge devices. — [AWS DDA guide](https://docs.aws.amazon.com/lookout-for-vision/latest/dda-user-guide/dda-understanding-usage.html)
- **Microsoft:**
  - Azure Percept DK and its services were retired on 30 Mar 2023. — [Microsoft Learn](https://learn.microsoft.com/en-us/previous-versions/azure/azure-percept/retirement-of-azure-percept-dk)
  - Azure Custom Vision will be retired on 25 Sep 2028, with a transition plan advised by 25 Sep 2026. — [Microsoft Learn migration options](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/migration-options); [Azure Aggregator, 14 Oct 2025](https://azureaggregator.wordpress.com/2025/10/14/retirement-azure-custom-vision-will-be-retired-on-september-25-2028/)
- **Google Visual Inspection AI:**
  - Launched 22 Jun 2021, claiming "as few as 10 labelled images", "up to 300 times fewer human-labelled images" and accuracy "up to 10X". Models were trained in the cloud and could be downloaded for edge operation. — [PR Newswire 2021](https://www.prnewswire.com/news-releases/google-clouds-visual-inspection-ai-reinvents-manufacturing-quality-control-301317486.html) [OLD]
  - 2026 status is **unclear**. IAM role docs are still maintained ([Google IAM docs](https://docs.cloud.google.com/iam/docs/roles-permissions/visualinspection)), and an Oct 2025 forum thread reports trouble accessing the API ([Google Dev forum](https://discuss.google.dev/t/problems-with-visual-inspection-ai-on-the-skills-platform/277898)).
  - On 26 Sep 2026 my fetch of `cloud.google.com/solutions/visual-inspection-ai` **redirected to the Vertex AI Model Garden docs page**. This suggests the standalone product page has been retired, but I found no official deprecation notice. Treat this as an observation, not a confirmed fact.
- Tulip's no-code defect-detection integration was built on Lookout for Vision, so integrators were affected by the shutdown. — [Tulip KB](https://support.tulip.co/docs/defect-detection-with-lookout)

### Inferences
- The "no novelty" critique is technically valid:
  - Good-only, ~20-image, on-device anomaly detection is standard (Inspekto 20, Keyence VS ~20, Zebra 20-30, LandingLens 50).
  - PatchCore specifically already ships in Edge Impulse (since Oct 2024) and anomalib.
  - Kaizen Eye cannot differentiate on the algorithm family. It must differentiate on form factor (a handheld phone with no fixture), fully offline reasoning and explanation, self-calibration, and operator-in-the-loop learning.
- Several pros come from the imaging hardware, not the AI:
  - Keyence (AI Imaging, 10,000 conditions; background extraction) and UnitX (32 programmable lights) raise defect contrast and remove clutter before any model runs. That is why they get away with 1-20 images.
  - Inspekto bundles fixed optics and light, so the model only has to learn "this station".
  - A phone has none of this, so it must compensate in software: object lock, exposure lock, pose handling.
- Offline is a real differentiator in 2026:
  - Hyperscalers are pulling out (AWS Lookout end of life 2025, Azure Percept 2023, Custom Vision 2028, Google VIAI unclear).
  - Specialists keep training or analytics in the cloud (Cognex OneVision training, Elementary QualityOS on AWS, Instrumental managed cloud).
  - A fully offline phone system avoids vendor-sunset and data-egress risk. This is a credible narrative for judges.
- The frontier features vendors shipped in 2025-26 are instant or continual learning (HALCON 25.11, Neurala), synthetic defects (UnitX GenX) and VLM agents (Omron with NVIDIA). They all run on PCs, GPUs or the cloud. None was found running on a phone NPU.

### Gaps
- No published cycle times were found for Keyence IV4/VS or Cognex In-Sight 3800/D900 anomaly tools. The vendor pages checked give no numbers.
- There is no official pricing for Keyence, Cognex, Landing AI, Elementary, Instrumental, UnitX or Averroes. The only bands found are from a rival's blog (Averroes) and a 2019 Inspekto figure.
- The number of good images needed for Cognex ViDi Red/Detect and OneVision Anomaly Detect is not published in the sources found.
- Not researched in depth: Sick, Basler, Axis, Mech-Mind, Eigen Innovations, Kitov, Datategy. Robro Systems (India) was not found in searches.
- Neurala headcount and funding (34 employees, USD 37.5M raised) appear only on aggregator sites (Tracxn/PitchBook) and were not verified.

---

## Q2. Open-source, research and hackathon-level prototypes: what exists, which run on a phone, which are offline

### Takeaway
Open-source tooling (anomalib 2.x, Edge Impulse) and 2024-26 research already cover several things:
- training-free few-shot detection (AnomalyDINO, 1-shot 96.6% image AUROC on MVTec AD)
- VLM-based explanation and logical checks (AnomalyGPT, LogicQA, LogSAD)
- on-device continual PatchCore
- human-in-the-loop memory-bank correction (Aug 2026)

I found **no phone-native, offline, good-only anomaly-inspection app**, commercial or open-source. Phone apps that run offline (Ultralytics HUB, up to 30 fps) are supervised detectors. Measured phone-NPU VLM numbers exist (Snapdragon 8 Elite), but they show VLMs are too slow to run on every frame on a conveyor.

### Cited Findings

#### Libraries and platforms
- Anomalib versions:
  - 2.0.0 released 19 Mar 2025; the latest is 2.6.2 (11 Sep 2026).
  - 2.5.0 (29 May 2026) added INP-Former, GLASS, AnomalyVFM and CFM.
  - 2.5.1 (17 Jul 2026) migrated DINOv2 models to `timm` "with DINOv3 backbone support".
  - 2.6.0 (25 Jul 2026) added the SuperADD VAND 4.0 winning method and the AutoVI dataset.
  — [PyPI history](https://pypi.org/project/anomalib/#history); [GitHub releases](https://github.com/open-edge-platform/anomalib/releases)
- Most anomalib models export to OpenVINO IR for Intel hardware. — [anomalib GitHub](https://github.com/open-edge-platform/anomalib)
- Dinomaly (CVPR 2025) reports 99.6% image AUROC on MVTec AD and is in anomalib from v2.2.0 (DINOv2 ViT). — [Datature guide 2026](https://datature.io/blog/visual-anomaly-detection-with-anomalib-a-hands-on-guide-2026)
- Anomalib VLM-AD compares test images with reference normal images through natural-language prompting. Backends include GPT-4o-mini, Ollama (Llama), Vicuna and Mistral. — [anomalib VLM-AD docs](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/vlm_ad.html)
- Anomalib can build a **synthetic validation set** (`ValSplitMode.SYNTHETIC`) by applying Perlin-noise perturbations to normal images. — [anomalib synthetic utils](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/data/utils/synthetic.html)
- A lightweight, anomaly-focused "geti-inspect" is being developed as a feature branch in the anomalib repo [SNIPPET]. — [anomalib GitHub](https://github.com/open-edge-platform/anomalib)
- Intel publishes an edge "PCB anomaly detection" reference suite. — [Open Edge Platform docs](https://docs.openedgeplatform.intel.com/2025.1/edge-ai-suites/pcb-anomaly-detection/index.html)
- Community "live camera anomaly detector" built on anomalib (12 Dec 2025):
  - Django web UI with multi-camera support.
  - Uses synthetic anomalies during training so normalisation and threshold can be refined in production.
  - "20ms per frame" on an NVIDIA Orin NX.
  - Closed source.
  — [anomalib Discussion #3220](https://github.com/open-edge-platform/anomalib/discussions/3220)
- Edge Impulse FOMO-AD with PatchCore scoring runs from MCU to NPU (Enterprise tier). — [Edge Impulse](https://www.edgeimpulse.com/blog/patchcore-boosts-visual-anomaly-detection-in-edge-impulse/)

#### Phone apps
- The Ultralytics HUB app (iOS/Android) runs YOLO **on-device and offline** at "up to 30 frames per second". It uses the Neural Engine on iOS and TFLite delegates on Android. These are supervised detectors, not anomaly models. — [Ultralytics HUB app docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/hub/app/index.md)
- A Nov 2025 listicle of "smartphone-based visual inspection apps" (InfiView, VisualScope Pro, etc.) gives no offline or anomaly-method details. It estimates custom app development at USD 60k-500k+. This is a low-quality source. — [Indie Hackers](https://www.indiehackers.com/post/top-5-smartphone-based-visual-inspection-apps-for-2026-features-cost-development-guide-3af96dd8be)
- "DeepInspect" inspects smartphones as the product being checked. It is not a phone-based inspector. — [SwitchOn](https://switchon.io/smartphone-defect-detection/)

#### Research: few-shot and training-free anomaly detection
- AnomalyDINO (WACV 2025):
  - Method: training-free DINOv2 patch kNN.
  - MVTec AD 1/4/16-shot image AUROC: 96.6 / 97.7 / 98.4% (1-shot up from 93.1% for the previous state of the art). VisA 1/4/16-shot: 87.4 / 92.6 / 94.8%.
  - Speed: ~60 ms per image (ViT-S, 448 px, A40 GPU). ViT-S has 21M parameters.
  - **Zero-shot PCA foreground masking** (skipped when a "masking test" fails) and rotation augmentation.
  - It **fails on semantic/logical anomalies**: "Cable Swap" scores only 50.2% AUROC because every patch matches the reference.
  — [AnomalyDINO arXiv](https://arxiv.org/html/2405.14529v2)
- On-device continual PatchCore (15 Dec 2025): lightweight extractor plus incremental k-center coreset; +12% AUROC and −80% memory versus baseline; "eliminates costly cloud retraining". — [arXiv 2512.13497](https://arxiv.org/abs/2512.13497)
- PatchCore-Lite / PaDiM-Lite (18 Mar 2026): −79% memory for PatchCore-Lite; −77% memory and −31% inference time for PaDiM-Lite. — [arXiv 2603.20288](https://arxiv.org/abs/2603.20288)
- Training-free human-in-the-loop memory-bank correction (18 Aug 2026):
  - Operators edit a PatchCore memory bank with a "self-calibrating novelty gate" instead of retraining.
  - With 10 golden samples plus corrections it "closes a median 66% performance gap".
  - It improves 12 of 15 MVTec categories and costs 43% of exhaustive review.
  — [arXiv 2608.17775](https://arxiv.org/abs/2608.17775)
- Related paper: "Sequential PatchCore: Anomaly Detection for Surface Inspection using Synthetic Impurities" (2025; title only, not read). — [arXiv 2501.09579](https://arxiv.org/pdf/2501.09579)

#### Research: VLM-based inspection
- AnomalyGPT (AAAI 2024): accuracy 86.1%, image AUC 94.1%, pixel AUC 95.3% on MVTec AD; one-normal-shot in-context use; no manual threshold [OLD, submitted Aug 2023]. — [arXiv 2308.15366](https://arxiv.org/abs/2308.15366)
- MMAD benchmark (ICLR 2025): 39,672 questions over 8,366 images. Average scores: GPT-4o 74.9%, which "falls far short of industrial requirements"; GPT-4o-mini 66.3%; Gemini-1.5-flash 68.9%; InternVL2-76B 70.8%. — [MMAD arXiv](https://arxiv.org/abs/2410.09453)
- LogicQA (ACL 2025 Industry): a VLM generates a checklist of questions; training-free and few-shot. MVTec LOCO: AUROC 87.6%, F1-max 87.0%. — [arXiv 2503.20252](https://arxiv.org/abs/2503.20252)
- LogSAD (2025): training-free "match-of-thought" using large multimodal models such as GPT-4V; reported 90.2% AUROC on MVTec LOCO [SNIPPET]. — [arXiv 2503.18325](https://arxiv.org/abs/2503.18325)
- Hybrid VLM report generation (26 May 2026):
  - Pipeline: YOLO26-x-obb, then deterministic spatial tokens, then a 4-bit QLoRA Qwen-2.5-1.5B.
  - Speed: 47 tok/s on a T4-class GPU.
  - Hallucination 4% versus 65% for a zero-shot baseline; BLEU-4 0.41 versus 0.07.
  — [arXiv 2605.26533](https://arxiv.org/abs/2605.26533)
- A glove-manufacturing MLLM with reinforcement fine-tuning handles different products and defects "via natural language prompts". Its mAP of 0.63 is comparable to a specialised YOLO (0.62). — [PLOS One 2026](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0339867)

#### Research: pose and multi-view
- PAD/MAD (NeurIPS 2023) has 11,000+ images of 20 LEGO toys. It notes existing datasets assume the "anomaly-free training dataset is pose-aligned and testing samples have the same pose". — [PAD arXiv](https://arxiv.org/abs/2310.07716) [OLD]
- SplatPose (CVPRW 2024) uses 3D Gaussian splatting for pose-agnostic AD. — [SplatPose](https://arxiv.org/pdf/2404.06832)

#### Phone-NPU VLM feasibility
- "Phase Matters" (MobiSys workshop, 26 Jun 2026) on Snapdragon 8 Elite (SM8750):
  - The NPU gives 1.64× speedup for prefill but only 1.18× for decode (FastVLM-0.5B).
  - Vision encoders run 20-45× faster on the NPU than on the CPU.
  - A heterogeneous CPU/NPU setup uses 2.52× less energy and runs 10.47 °C cooler at steady state.
  — [arXiv 2606.27906](https://arxiv.org/abs/2606.27906)
- Qwen2.5-VL-7B (W4A16, Qualcomm AI Hub) on a Dragonwing IQ-9075 EVK NPU (not a phone): 7.6 tok/s through the GenieX API, about 10 tok/s native, time to first token about 1.1 s (7 Jul 2026). — [Macnica](https://www.macnica.co.jp/en/business/semiconductor/articles/qualcomm/150066/)
- MagicVL-2B reportedly reaches 23.9 tok/s on Snapdragon 8 Elite [SNIPPET]. — [arXiv 2508.01540](https://arxiv.org/pdf/2508.01540)

### Inferences
- Kaizen Eye's direct open-source analogue is anomalib PatchCore or Edge Impulse FOMO-AD. The closest "live camera" prototype runs on a Jetson (Discussion #3220), not a phone.
- A phone-native app that is offline, good-only and runs on a live feed would be a genuine form-factor gap. It is not an algorithmic gap.
- Swapping ResNet18 for DINOv2/v3 ViT-S (21M parameters) with PCA foreground masking follows the published AnomalyDINO recipe. This is the lowest-risk accuracy upgrade:
  - It could cut enrolment from about 20 photos to 1-4 good frames with better AUROC.
  - The mask also partly addresses "not locking onto the object".
  - Its runtime on the iQOO NPU must be measured. The 60 ms figure is from a desktop GPU.
- VLMs should be positioned as a second-stage reasoner and explainer, not the primary detector:
  - Even GPT-4o scores only 74.9% on MMAD.
  - Grounded pipelines cut hallucination from 65% to 4%.
  - At about 1 s time-to-first-token (7B, Qualcomm NPU) or tens of tokens per second (0.5-2B on 8 Elite), the VLM fits "explain the reject" or "check the rule on sampled parts". It does not fit "every frame at conveyor speed".

### Gaps
- No peer-reviewed paper or GitHub repo was found for a **phone** (Android/iOS) PatchCore or AnomalyDINO inspection app with measured latency. Absence of evidence after limited searching; a deeper GitHub search could still find one.
- No measured on-phone latency was found for AnomalyDINO or DINOv2 ViT-S patch-kNN on Snapdragon 8 Elite Gen 5.
- iQOO 15 / Snapdragon 8 Elite Gen 5 NPU specs were not researched here (out of scope).
- The AnomalyGPT backbone and size were not confirmed from the abstract.

---

## Q3. What these systems do badly, the technical root causes, and how each weakness could be fixed

### Takeaway
The recurring weaknesses are:
- they need fixed mounting, a consistent background and consistent pose
- lighting and pose changes cause false alarms
- small defects are missed at low input resolution
- logical or assembly errors are missed by patch methods
- supervised tools need defect labels, and even anomaly tools often need abnormal images to set the threshold
- they depend on the cloud or a vendor (sunset risk)
- they give no human-readable explanation
- new products need retraining
- they cost USD thousands to tens of thousands

Most of these trace to two technical facts. First, patch-level one-class models are "bags of local patches" compared against an enrolment set captured under one pose, light and background. Second, anomaly scores are uncalibrated distances with no semantics. Kaizen Eye's judge complaints (objects not locked, different objects passing, too many photos, no numbers) are exactly these failure modes.

### Cited Findings
- **Background/pose sensitivity:**
  - LandingLens warns that different backgrounds "might" be flagged as anomalies and requires a consistent background. — [LandingLens docs](https://landinglens.docs.landing.ai/anomaly-detection)
  - Keyence built "AI Target Extraction" specifically to remove "sources of false detection from the surroundings". — [Keyence IV4](https://www.keyence.com/products/vision/vision-sensor/iv4/)
  - Inspekto setup requires tracing the item outline (region of interest) at a fixed station. — [automation.com](https://www.automation.com/article/inspekto-releases-s70-autonomous-machine-vision-sy) [OLD]
  - The PAD benchmark notes that datasets assume pose-aligned training and test samples. — [PAD](https://arxiv.org/abs/2310.07716)
- **Lighting, hard conditions and small defects:**
  - MVTec AD 2 (2025; IJCV 2026) includes lighting changes, transparent and overlapping objects, and "extremely small defects".
  - State-of-the-art methods stay "below 60% average AU-PRO", and AU-PRO0.05 is "below 31%" at the default 256×256 input.
  — [MVTec AD 2 arXiv](https://arxiv.org/abs/2503.21622); [MVTec AD 2 page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
  - Vendors fight this with imaging hardware: Keyence tests "over 10,000 conditions" ([Keyence](https://www.keyence.com/products/vision/vision-sensor/iv4/)); UnitX uses 32 lights and up to 100 MP ([UnitX](https://www.unitxlabs.com/industry/bearing-machining-inspection-application/)).
- **Logical anomalies:**
  - Logical anomalies need spatial and contextual reasoning that patch-based approaches "struggle to achieve". — [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
  - AnomalyDINO: Cable Swap 50.2% AUROC because "all patches matched". — [AnomalyDINO](https://arxiv.org/html/2405.14529v2)
  - MVTec's fix is Global Context AD. — [MVTec](https://www.mvtec.com/company/press-room/press-releases/detail/mvtec-sets-new-standards)
- **Defect labels and threshold:**
  - Supervised tools need labelled defects: Averroes 20-40 per class ([Averroes](https://averroes.ai/)); Cognex edge learning 5-10 examples per class ([Cognex](https://www.cognex.com/en-se/products/machine-vision/2d-machine-vision-systems/in-sight-3800)); Google VIAI "as few as 10 labelled images" ([PR 2021](https://www.prnewswire.com/news-releases/google-clouds-visual-inspection-ai-reinvents-manufacturing-quality-control-301317486.html)).
  - LandingLens anomaly detection still needs at least 1 (ideally 10+) abnormal images to set its threshold. — [LandingLens](https://landinglens.docs.landing.ai/anomaly-detection)
  - UnitX GenX exists because real defect data is scarce. — [A3](https://www.automate.org/news/unitx-unveils-genx-a-generative-ai-breakthrough-for-industrial-inspection-unitx-labs)
- **Cloud/vendor dependence:**
  - AWS Lookout end of life 31 Oct 2025 ([AWS](https://docs.aws.amazon.com/lookout-for-vision/latest/developer-guide/su-awscli-sdk.html)); Azure Percept retired 30 Mar 2023 ([Microsoft](https://learn.microsoft.com/en-us/previous-versions/azure/azure-percept/retirement-of-azure-percept-dk)); Azure Custom Vision retires 25 Sep 2028 ([Microsoft](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/migration-options)).
  - Cognex OneVision trains in the cloud ([PR 2026](https://www.prnewswire.com/news-releases/cognex-onevision-adoption-ramps-as-manufacturers-scale-ai-vision-globally-302768368.html)); Elementary stores data in AWS ([Elementary](https://www.elementaryml.com/)).
- **Cost:** smart cameras USD 2k-50k+ plus integration USD 20k-150k [COMPETITOR-AUTHORED]. — [Averroes](https://averroes.ai/blog/cognex-vs-keyence-vision-systems)
- **Explanation:**
  - VLMs are weak detectors (GPT-4o 74.9% on MMAD). — [MMAD](https://arxiv.org/abs/2410.09453)
  - Ungrounded VLM reports hallucinate 65% of the time; grounded ones 4%. — [arXiv 2605.26533](https://arxiv.org/abs/2605.26533)
  - The VLM agents that do exist run on NVIDIA GPU stacks. — [Omron/NVIDIA](https://www.manufacturingtomorrow.com/news/2026/07/16/omron-advances-inspection-technology-with-nvidia-omniverse-and-metropolis/27914/)
- **Retraining for new products:**
  - HALCON 25.11 continual learning exists to avoid "complete retraining". — [MVTec](https://www.mvtec.com/company/press-room/press-releases/detail/mvtec-introduces-new-deep-learning-feature-continual-learning-in-halcon-2511)
  - On-device continual PatchCore targets "frequent variant changes". — [arXiv 2512.13497](https://arxiv.org/abs/2512.13497)
- **Unverifiable accuracy claims:** "99.9%" ([Elementary](https://www.elementaryml.com/)) and "99%+" ([Averroes](https://averroes.ai/)) come with no public protocol. Meanwhile MVTec AD 2 shows state-of-the-art below 60% AU-PRO in hard conditions ([MVTec AD 2](https://arxiv.org/abs/2503.21622)).

### Inferences

#### Weaknesses, root causes and fixes
| Weakness | Who shows it | Technical root cause | Fix in general | Fix on a phone (Kaizen Eye) |
|---|---|---|---|---|
| Needs fixed mount, consistent background and pose; false alarms when the part moves or rotates | LandingLens, Inspekto, most patch-AD tools | The memory bank only contains patch features seen at enrolment pose, scale and background. New background or pose puts patches far from memory, so false alarms follow. | Fixtures, background removal (Keyence), ROI tracing | Foreground mask (PCA on DINO features or small segmenter), crop to the object, rotation/scale augmentation of the bank, pose-binned multi-view bank from video |
| **Different object passes** (Kaizen judge #5) | Any bag-of-patches kNN (PatchCore, AnomalyDINO) | kNN is permutation-invariant over patches with no global shape or identity check. Generic ImageNet texture features let "similar-material" patches match. The threshold only bounds max patch distance. | Global-context models (MVTec GCAD), alignment | Two-gate design: (1) global object-identity gate (cosine of global embedding vs enrolled prototypes plus area/aspect of foreground mask, with its own leave-one-out threshold); (2) position-aware patch kNN (search only among memory patches from the same normalised region) |
| Misses small defects | Most tools at 224-320 px input | Downsampling plus stride-8/16 feature maps: one feature cell covers many pixels, so tiny defects are diluted (MVTec AD 2 AU-PRO0.05 <31% at 256 px) | High-resolution sensors and tiling (UnitX up to 100 MP) | Tile the object crop at native camera resolution (e.g., 2×2 or 3×3 tiles through the NPU); tap-to-zoom ROI |
| False alarms when lighting changes | All, especially phones with auto-exposure | Features are not illumination-invariant; the bank only covers enrolled light. Phone auto-exposure and white balance drift between frames. | Controlled lighting (Keyence AI imaging, UnitX OptiX) | Lock AE/AWB/focus after enrolment, enrol under light variation (video naturally gives this), photometric augmentation, drift alarm on the score distribution |
| Misses logical and assembly errors (missing screw, wrong orientation, wrong count) | Patch-kNN, RD models | Local patches are individually "normal"; the error is only in the arrangement or count | MVTec GCAD; VLM checklists (LogicQA 87.6%, LogSAD 90.2% on LOCO) | Natural-language rule checklist verified by a small on-device VLM on sampled or flagged parts, plus deterministic count/position checks from detections |
| Needs defect images or labels (supervised) or abnormal images for the threshold | Averroes, Cognex edge learning, Google VIAI, UnitX, LandingLens (threshold) | Discriminative models need negatives; anomaly scores are uncalibrated distances | One-class models; synthetic defects (GenX, anomalib Perlin, GLASS) | On-device synthetic-defect self-calibration at enrolment, reported as a detectability spec |
| No explanation (only a score or heatmap) | Nearly all smart cameras | Distance scores carry no semantics; VLMs hallucinate if ungrounded | VLM agents on GPU (Omron/NVIDIA) | Explain by retrieval (show the nearest good patch next to the defect) plus a VLM caption grounded in detector outputs (spatial tokens), generated only for rejects |
| Retraining for each new product or false alarm | Supervised tools; most AD tools | Weights must be re-optimised; catastrophic forgetting | HALCON 25.11 continual learning; Neurala L-DNN | Memory-bank edits (insert with a novelty gate; see the Aug 2026 paper), per-SKU banks with automatic SKU recognition, instant, with undo and audit |
| Cloud dependence and vendor sunset | AWS Lookout (dead), Azure Percept (dead), Custom Vision (2028), OneVision cloud training, Elementary AWS | SaaS architecture centralises training and analytics | On-prem options (Instrumental, Neurala) | 100% on-device enrol, infer, report; export by share sheet |
| Cost (USD 2k-50k hardware plus USD 20-150k integration) | Cognex, Keyence, Inspekto | Industrial optics, lighting, I/O, support | n/a | Phone as commodity camera and NPU. Be honest: no industrial I/O, lighting or IP rating, so position the phone for audits, low-volume lines, SMEs and spot checks |

- Mapping to Kaizen Eye's judge feedback:
  - #5 (no object lock) is the bag-of-patches root cause.
  - #2 (too many photos) is a UX issue, not a model limit. A 1-4-shot DINO recipe exists, and Elementary already "learns by watching" on fixed lines.
  - #3 (exact X→Y numbers) is best answered with a reproducible protocol: MVTec AD/VisA plus own parts, before and after each change, per-stage ms on the NPU. This contrasts with competitors' unverifiable "99.9%".
  - #4 (live feed) is table stakes for smart cameras (Keyence AI trigger; Elementary 1,000 ppm).

### Gaps
- No independent head-to-head benchmark of commercial products (Cognex vs Keyence vs Inspekto vs LandingLens) was found. All accuracy numbers are vendor-reported.
- No quantitative source was found for false-alarm rates of commercial tools under lighting or pose change. The evidence is indirect (MVTec AD 2, PAD, LandingLens guidance).

---

## Q4. Gap analysis: which novel features a phone-only, offline, NPU-accelerated system can credibly add, and who already offers each

### Takeaway
None of the proposed features is novel as a concept in isolation:
- Elementary already learns from watching the line.
- Neurala and HALCON do instant or continual learning.
- UnitX and anomalib synthesise defects.
- Omron with NVIDIA runs VLM agents.

The **combination, running fully offline on a handheld phone NPU**, is not offered by any product found. The strongest novelty-plus-credibility bets are:
1. Grounded on-device VLM rule checks and explanations for rejects.
2. Handheld 10-second video enrolment into a pose-aware bank with an object-lock gate.
3. A zero-defect synthetic self-calibration that prints a detectability "spec sheet" (X→Y numbers).
4. One-tap memory-bank correction with no retraining.

Live conveyor mode, NPU telemetry and offline SPC are expected features and do not count as novelty.

### Cited Findings (who already offers something similar)
- **Video/stream enrolment:** Elementary VisionStream "watches your line, learns what 'good' looks like... in under 60 seconds" on edge controllers (fixed cameras). — [Elementary](https://www.elementaryml.com/). Instrumental ingests "images, videos". — [Instrumental](https://instrumental.com/product/). No handheld or multi-view enrolment product was found. Pose-agnostic AD is still a research topic (PAD, SplatPose). — [PAD](https://arxiv.org/abs/2310.07716); [SplatPose](https://arxiv.org/pdf/2404.06832)
- **Natural-language rules / VLM reasoning:**
  - Omron with NVIDIA VSS uses Cosmos Reason 7B on GPU stacks. — [ManufacturingTomorrow](https://www.manufacturingtomorrow.com/news/2026/07/16/omron-advances-inspection-technology-with-nvidia-omniverse-and-metropolis/27914/)
  - Anomalib VLM-AD uses GPT-4o-mini or Ollama. — [anomalib](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/vlm_ad.html)
  - LogicQA / LogSAD are research only. — [LogicQA](https://arxiv.org/abs/2503.20252); [LogSAD](https://arxiv.org/abs/2503.18325)
  - A glove MLLM switches products "via natural language prompts" (research). — [PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0339867)
  - No phone-offline VLM inspection product was found.
- **Explainable reports:** AnomalyGPT (research) ([arXiv](https://arxiv.org/abs/2308.15366)); hybrid YOLO + Qwen-1.5B report generator on a T4 GPU with 4% hallucination ([arXiv 2605.26533](https://arxiv.org/abs/2605.26533)).
- **Instant feedback with no retraining:**
  - Neurala L-DNN trains "immediately as they're annotated". — [Neurala](https://www.neurala.com/how-lifelong-dnn-helps-neurala-via-be-competitive-for-visual-inspections/)
  - HALCON 25.11 Continual Learning covers classification, with 5-10 images on a CPU. — [MVTec](https://www.mvtec.com/company/press-room/press-releases/detail/mvtec-introduces-new-deep-learning-feature-continual-learning-in-halcon-2511)
  - Elementary says models "continuously improve" with each inspection. — [Elementary platform](https://www.elementaryml.com/product/platform)
  - PatchCore memory-bank correction is research from Aug 2026 (66% of the gap closed). — [arXiv 2608.17775](https://arxiv.org/abs/2608.17775)
  - On-device incremental coreset research is from Dec 2025. — [arXiv 2512.13497](https://arxiv.org/abs/2512.13497)
- **Synthetic-defect calibration:**
  - Anomalib offers a synthetic Perlin validation split (a developer tool). — [anomalib](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/data/utils/synthetic.html)
  - The Jetson live-camera prototype uses synthetic anomalies to refine its threshold. — [Discussion #3220](https://github.com/open-edge-platform/anomalib/discussions/3220)
  - UnitX GenX needs at least 3 real defects and targets supervised training. — [A3](https://www.automate.org/news/unitx-unveils-genx-a-generative-ai-breakthrough-for-industrial-inspection-unitx-labs)
  - LandingLens still asks for real abnormal images to set the threshold. — [LandingLens](https://landinglens.docs.landing.ai/anomaly-detection)
- **Real-time and trigger:** Keyence "AI trigger" needs no separate trigger sensor ([Keyence](https://www.keyence.com/products/vision/vision-sensor/iv4/)); Elementary runs at 1,000 ppm ([Elementary](https://www.elementaryml.com/)). A phone can run offline detection at up to 30 fps ([Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/docs/en/hub/app/index.md)).
- **NPU and thermal:** on Snapdragon 8 Elite, vision encoders run 20-45× faster on the NPU than the CPU, and a heterogeneous CPU/NPU setup uses 2.52× less energy and runs 10.47 °C cooler. — [arXiv 2606.27906](https://arxiv.org/abs/2606.27906)
- **VLM latency:** Qwen2.5-VL-7B W4A16 reaches ~1.1 s time to first token and 7.6-10 tok/s on a Qualcomm NPU (IQ-9075, not a phone). — [Macnica](https://www.macnica.co.jp/en/business/semiconductor/articles/qualcomm/150066/)
- **Analytics:** cloud dashboards are the norm (Elementary QualityOS on AWS; Cognex OneVision cloud governance; Instrumental managed cloud). — [Elementary](https://www.elementaryml.com/); [Cognex](https://www.prnewswire.com/news-releases/cognex-onevision-adoption-ramps-as-manufacturers-scale-ai-vision-globally-302768368.html); [Instrumental](https://instrumental.com/product/)

### Inferences

#### Ranked candidate features (novelty is 20% of marks)
| # | Feature | Competitor weakness it targets | Who partially offers it | Novelty vs market | Credibility risk | Demo moment and numbers to show |
|---|---|---|---|---|---|---|
| 1 | **Grounded on-device VLM "rule inspector" and explainer**: operator types rules ("4 screws present", "label upright", "no cable crossing"); VLM compiles them into yes/no checklist questions (LogicQA-style); runs only on flagged or sampled parts; also writes a one-line reason for each reject, grounded in detector outputs (heatmap peak location, size, nearest-good-patch similarity) | Patch methods miss logical errors (LOCO; AnomalyDINO cable swap 50.2%); smart cameras give no explanation; VLM agents need GPU or cloud (Omron/NVIDIA, VLM-AD with GPT-4o-mini) | Omron+NVIDIA (GPU), anomalib VLM-AD (cloud or PC), LogicQA/LogSAD (research) | **High**: no phone-offline product found | Medium: VLMs are weak raw detectors (GPT-4o 74.9% MMAD) and hallucinate ungrounded (65%). Mitigate by grounding and by never letting the VLM override a detector FAIL. | Airplane mode on; remove a screw (a logical defect the patch model passes), VLM rule catches it; show time to first token and tok/s on NPU; show a hallucination guard |
| 2 | **Handheld 10-s video enrolment with a pose-coverage meter**, auto keyframe selection (sharpness, exposure, diversity), pose-binned multi-view memory bank, plus **object-lock gate** (global identity plus foreground mask) | Fixed mount, consistent pose and background required (LandingLens, Inspekto, PAD); 20-50 photos typical; "different objects pass" | Elementary learns from line video (fixed camera); PAD/SplatPose research (GPU, 3DGS) | **Medium-high**: video enrolment itself is not new (Elementary); handheld multi-view on a phone is | Low-medium: must show rotated good part passes and a different or similar part fails | Enrol in 10 s (vs 20 photos or 50 images for LandingLens); show "coverage 300°"; false-accept rate on a "wrong object" set drops from X% to Y%; 1-4 shot AUROC with the DINO backbone |
| 3 | **Zero-defect self-calibration and detectability spec sheet**: at enrolment, synthesise defects (Perlin blobs, scratches, CutPaste, occlusion/missing region) on held-out good keyframes; pick the threshold for target false-reject rate from leave-one-out normals; print "detects ≥N px (≈M mm at this distance) with R% recall at F% false reject" | LandingLens needs real abnormal images for the threshold; vendors publish unverifiable "99.9%"; judges asked for exact X→Y numbers | Anomalib synthetic validation (developer tool), Jetson live-cam prototype (closed source), UnitX GenX (needs real defects, supervised) | **Medium-high** as an automatic, on-device, per-SKU spec | Medium: synthetic is not real. Validate the synthetic-vs-real correlation on a handful of real defects and state it. | The spec card appears seconds after enrolment; compare predicted versus measured recall on 5-10 real defective parts |
| 4 | **One-tap operator correction with no retraining**: tap "this is OK" on a false alarm, and its normal patches are inserted into the bank through a novelty gate; tap "missed defect" to record a negative anchor; undo plus supervisor PIN plus audit log | Retraining cycles (supervised tools); nuisance false alarms after lighting or supplier changes | Neurala L-DNN (supervised classes), HALCON 25.11 (classification), Elementary "continuously improve", Aug 2026 paper (research) | **Medium**: concept exists, not in a phone AD app | Low: Aug 2026 paper reports a 66% gap closed with 10 golden samples plus corrections | Live: false alarm, one tap, re-inspect passes in under 1 s while a real defect still fails; show the bank size change |
| 5 | **Live conveyor mode**: motion/object-entry "AI trigger", per-piece tracking so each piece gets one verdict (majority over N frames), beep, vibrate and red flash, pass/fail counter, plus **NPU telemetry overlay** (per-stage ms, fps, CPU vs NPU, temperature) | Table stakes vs Keyence AI trigger and Elementary 1,000 ppm; judges explicitly asked for the NPU | Every smart camera (trigger and I/O); nobody on a phone | Low-medium (form factor only) | Low, if the NPU path is real (QNN/LiteRT delegate) and measured | Side-by-side CPU vs NPU ms; sustained 30-min run showing thermals stay stable (cite 2.52× energy / 10.47 °C from the Phase Matters paper) |
| 6 | **Offline shift intelligence**: p-chart / defect-rate SPC, defect-location heatmap across the shift, anomaly-score drift alarm (lighting or process change), PDF/CSV shift report generated on-device (optionally VLM-summarised) | Analytics are cloud-hosted (Elementary AWS, OneVision cloud, Instrumental cloud); hyperscaler sunsets (AWS Lookout 2025, Azure Percept 2023, Custom Vision 2028) | Cloud dashboards everywhere | Low-medium | Low | Airplane-mode shift report; drift alarm fires when a lamp is switched off |

- Recommended pitch framing for judges:
  - Lead with #2 and the object-lock gate. It fixes judge points 2 and 5, the most visible failure.
  - Then #1, the offline VLM on the NPU, which is judge point 6 and the "wow" factor.
  - Back both with #3, the spec sheet that gives exact numbers for judge point 3.
  - #4 is a crowd-pleasing live demo with fresh research backing (Aug 2026).
  - #5 is mandatory and should not be pitched as novel.
  - Say explicitly that each concept exists somewhere (Elementary, Neurala, UnitX, Omron/NVIDIA) but only on fixed industrial hardware, GPUs or the cloud. The novelty claim is "first fully offline, handheld, phone-NPU implementation of the combination", which judges can check against this list.
- Why 3D is not recommended: SplatPose-style 3D Gaussian splatting is research-grade and GPU-trained. A pose-binned 2D memory bank built from video keyframes gets most of the pose robustness at a fraction of the compute. This is an inference; no phone 3DGS AD benchmark was found.
- Credibility guardrails that pre-empt the "multiple prototypes do the same" critique:
  - Publish a before-and-after table on public data (MVTec AD/VisA 1-/4-shot AUROC, ResNet18 vs DINO-S) plus own parts.
  - Include a "wrong object" false-accept test.
  - Report per-stage NPU latency.
  - Show a VLM hallucination check on rejects.
  - Say openly that the phone lacks industrial lighting and I/O and is positioned for spot checks, audits, SMEs and low-volume lines.

### Gaps
- It is not confirmed that any small VLM (0.5-3B) reaches usable accuracy on industrial rule-checking questions. MMAD covers larger models, and small on-device VLM accuracy on MMAD/LOCO-style questions was not found.
- No measured end-to-end latency was found for DINO-S patch-kNN plus a VLM on the iQOO 15 (Snapdragon 8 Elite Gen 5). The Phase Matters data is for the 8 Elite (SM8750). The Qwen2.5-VL-7B data is for the QCS9075 (non-phone).
- No source was found on how well synthetic-defect recall predicts real-defect recall in production. This is the key credibility risk for feature #3.
- Whether Cognex OneVision or Keyence ship any natural-language rule interface as of Sep 2026 could not be confirmed. None was found in the sources checked.
