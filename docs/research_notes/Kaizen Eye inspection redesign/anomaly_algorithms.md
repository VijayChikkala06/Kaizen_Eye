# Training-free / few-shot / zero-shot visual anomaly detection for real-time on-device (smartphone NPU) industrial inspection (state as of Sept 2026)

Notation used throughout: I-AUROC = image-level AUROC, P-AUROC = pixel-level AUROC, PRO = per-region overlap (AU-PRO, FPR limit 0.3 unless stated), k-shot = k normal reference images per category. "Full" = the whole normal training set. Year tags such as [2021] mark older numbers. Every number states dataset, shot count and metric. "Search snippet only" means the full text could not be fetched and the number comes from the search engine's extract, so treat it as less reliable.

## Q1. Which few-shot / zero-shot training-free methods lead in 2026, and what are their benchmark numbers by shot count (MVTec AD, VisA, Real-IAD)?

### Takeaway
Training-free matching on frozen DINOv2/DINOv3 patch features now leads few-shot industrial AD. At 1-shot, these methods reach about 97 to 98 I-AUROC on MVTec AD: UniVAD 97.8, DuoAD 97.7, VisionAD 97.4, SubspaceAD 97.1 and AnomalyDINO 96.6. PatchCore (WRN50) reaches 83.4 in the same setting. On VisA, 1-shot scores range from 87 to 95. The cheapest of the strong methods is AnomalyDINO with DINOv2 ViT-S/14 (21M parameters). CLIP-based zero-shot methods that use no normal images plateau at about 89 to 92 I-AUROC on MVTec AD and 78 to 86 on VisA. "Batched zero-shot" methods do better (MuSc 97.8, PA-CLIP 98.4 on MVTec AD), but they need a batch of unlabeled test images and cost up to about 1 s per image on a desktop GPU. Real-world multi-view data (Real-IAD) remains much harder: the best 1-shot score is about 85 I-AUROC.

### Cited Findings
**Leaderboard status**
- Papers with Code was shut down on July 24, 2025. Its leaderboards are gone and the domain redirects to Hugging Face Trending Papers. Its raw data was archived on GitHub (last updated Sept 8, 2025), and a revival exists at paperswithcode.co. Because of this, the numbers below come from recent papers' comparison tables, not from a leaderboard. — [Coursera](https://www.coursera.org/articles/papers-with-code); [HyperAI](https://hyper.ai/en/news/42900); [HF blog](https://huggingface.co/blog/nielsr/paperswithcode-launch)

**AnomalyDINO [WACV 2025]**
- **Design.** Training-free. It uses DINOv2 ViT-S/14 (21M params) at 448 or 672 px and does patch kNN against a memory bank of reference patches. The image score is the mean of the 1% highest patch distances (tail value-at-risk). Reference images are augmented by rotation. PCA-based foreground masking is applied per category, but not to textures. Runtime is about 60 ms/image at 448 px on an NVIDIA A40. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **MVTec AD, AnomalyDINO-S at 672 px** (I-AUROC / P-AUROC / PRO) — [arXiv 2405.14529, Tables 2 and 5](https://arxiv.org/html/2405.14529)

  | Shots | I-AUROC | P-AUROC | PRO |
  |---|---|---|---|
  | 1 | 96.6±0.4 | 96.8 | 92.7 |
  | 2 | 96.9±0.7 | 97.0 | 93.1 |
  | 4 | 97.7±0.2 | 97.2 | 93.4 |
  | 8 | 98.2±0.2 | 97.4 | 93.8 |
  | 16 | 98.4±0.1 | 97.5 | 94.0 |
  | Full | 99.5 | 98.2 | 95.0 |

- **MVTec AD, AnomalyDINO-S at 448 px** (I-AUROC / P-AUROC / PRO) — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)

  | Shots | I-AUROC | P-AUROC | PRO |
  |---|---|---|---|
  | 1 | 96.5 | 96.3 | 91.7 |
  | 2 | 96.7 | 96.5 | 92.0 |
  | 4 | 97.6 | 96.7 | 92.4 |
  | 8 | 98.0 | 97.0 | 92.7 |
  | 16 | 98.3 | 97.1 | 92.9 |
  | Full | 99.3 | 97.9 | 93.9 |

- **VisA, AnomalyDINO-S at 672 px** (I-AUROC / P-AUROC / PRO) — [arXiv 2405.14529, Tables 3 and 5](https://arxiv.org/html/2405.14529)

  | Shots | I-AUROC | P-AUROC | PRO |
  |---|---|---|---|
  | 1 | 87.4±1.2 | 97.8 | 92.5 |
  | 2 | 89.7 | 98.0 | 93.4 |
  | 4 | 92.6 | 98.2 | 94.1 |
  | 8 | 93.8 | 98.4 | 94.8 |
  | 16 | 94.8 | 98.5 | 95.3 |
  | Full | 97.6 | 98.8 | 96.1 |

- **VisA, AnomalyDINO-S at 448 px** (I-AUROC): 85.6, 88.3, 91.3, 92.6 and 93.8 at 1, 2, 4, 8 and 16 shots; 97.2 at full. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **Batched zero-shot** (no labelled normals; uses other unlabeled test images), I-AUROC:
  - AnomalyDINO-S at 672 px: 94.2 on MVTec AD, 90.7 on VisA.
  - AnomalyDINO-S at 448 px: 93.0 on MVTec AD, 89.7 on VisA.
  - MuSc: 97.8 on MVTec AD, 92.8 on VisA.
  - — [arXiv 2405.14529, Table 4](https://arxiv.org/html/2405.14529)
- **Other backbones.** The authors found "no considerable differences" between ViT-S, ViT-B and ViT-L on MVTec AD. Larger backbones are slightly better on VisA. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **Code.** Apache-2.0. It uses FAISS for the kNN search, with a CPU option. It has "agnostic" mode (masking plus rotations) and "informed" mode, and a separate batched zero-shot script. — [GitHub dammsi/AnomalyDINO](https://github.com/dammsi/AnomalyDINO)

**SubspaceAD [arXiv 2602.23013 v3, May 13, 2026]**
- **Design.** Training-free. It fits a PCA subspace (explained variance τ=0.99) to frozen DINOv2-G features, mean-pooled over layers 22 to 28, at 672 px. References are augmented with 30 random rotations. The patch score is the reconstruction residual ||x − proj(x)||². The image score is the mean of the top 1% of patch scores. Storage is under 1 MB per category. — [arXiv 2602.23013](https://arxiv.org/html/2602.23013v3)
- **MVTec AD** (I-AUROC / P-AUROC / PRO) — [arXiv 2602.23013, Tables 1 and 8](https://arxiv.org/html/2602.23013v3)

  | Shots | I-AUROC | P-AUROC | PRO |
  |---|---|---|---|
  | 1 | 97.1±0.9 | 97.5 | 94.5 |
  | 2 | 97.5 | 97.8 | 94.9 |
  | 4 | 98.0 | 97.9 | 95.1 |
  | Full | 99.2 | 98.2 | 95.6 |

- **VisA** (I-AUROC / P-AUROC / PRO) — [arXiv 2602.23013, Tables 1 and 8](https://arxiv.org/html/2602.23013v3)

  | Shots | I-AUROC | P-AUROC | PRO |
  |---|---|---|---|
  | 1 | 93.2±0.8 | 98.2 | 95.5 |
  | 2 | 93.8 | 98.3 | 95.7 |
  | 4 | 94.7 | 98.4 | 96.0 |
  | Full | 98.2 | 99.1 | 96.9 |

- **Batched 0-shot, I-AUROC:** 96.6 on MVTec AD and 94.1 on VisA. MuSc scores 97.8 and 94.1 in the same table. — [arXiv 2602.23013, Table 2](https://arxiv.org/html/2602.23013v3)
- **Inference time per image:** DINOv2-S at 448 px 36 ms; B at 448 px 56 ms; L at 448 px 112 ms; G at 672 px 127 ms. The GPU type was not captured in my extract. — [arXiv 2602.23013, App. D](https://arxiv.org/html/2602.23013v3)
- **Ablations, 4-shot MVTec AD:**
  - Mean-pooling 7 middle layers gives 98.4 I-AUROC and 95.0 PRO, versus 97.6 and 92.5 with the last layer only.
  - τ from 0.95 to 0.99 is flat (98.4 to 98.5 I-AUROC). τ=1.00 collapses to 40.5.
  - DINOv3-7B underperformed DINOv2-G.
  - Per the authors, performance "improves with increasing model capacity", but ViT-S numbers were not tabulated.
  - — [arXiv 2602.23013, Tables 3, 4, 7](https://arxiv.org/html/2602.23013v3)
- **Few-shot baselines from the same table** (I-AUROC at 1, 2 and 4 shots) — [arXiv 2602.23013, Table 1](https://arxiv.org/html/2602.23013v3)
  - PromptAD: 94.6 / 95.7 / 96.6 on MVTec AD; 86.9 / 88.3 / 89.1 on VisA.
  - IIPAD: 94.2 / 95.7 / 96.1 on MVTec AD; 85.4 / 86.7 / 88.3 on VisA.

**DuoAD [arXiv 2607.23924, July 27, 2026, WACV]**
- **Design.** Training-free. It uses the [CLS] token in two ways. First, [CLS] cosine similarity (which the authors call "semantic stability") selects augmentations automatically. Second, pre-softmax [CLS]-to-patch attention logits reweight each patch's contribution. Backbones are DINOv2 ViT-B/14-reg at 448 px or DINOv3 ViT-B/16 at 512 px. — [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)
- **Results** (I-AUROC / PRO; values given as DINOv2 / DINOv3 backbone) — [arXiv 2607.23924, Tables 1–2](https://arxiv.org/html/2607.23924v1)

  | Dataset | Shots | I-AUROC | PRO |
  |---|---|---|---|
  | MVTec AD | 1 | 97.2 / 97.7 | 94.1 / 94.7 |
  | MVTec AD | 4 | 98.0 / 98.5 | 94.6 / 95.0 |
  | VisA | 1 | 92.5 / 93.3 | 93.3 / 93.2 |
  | VisA | 4 | 94.8 / 95.5 | 94.5 / 94.8 |
  | Real-IAD | 1 | 85.1 / 84.3 | 95.5 / 94.6 |
  | Real-IAD | 4 | 88.7 / 88.1 | 96.3 / 95.7 |

- **Baselines in the same paper, 1-shot I-AUROC:**
  - MVTec AD: UniVAD 97.5±0.3, AnomalyDINO 96.5, FoundAD 96.1, PromptAD 94.6, WinCLIP 93.1, PatchCore 83.4.
  - Real-IAD: AdaptCLIP 81.8, AnomalyDINO 80.6, WinCLIP 74.7.
  - Limitations: the method only works on ViTs, and self-calibrated augmentation needs a small warm-up set of unlabeled test images.
  - — [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)

**VisionAD / "Search is All You Need" [ACM MM 2025]**
- **Design.** Training-free nearest-neighbour search on DINOv2-Reg ViT-L/14. Images are resized to 448 and centre-cropped to 392. References get rotation, flip and affine augmentation, plus a "pseudo multi-view" transform. — [arXiv 2504.11895](https://arxiv.org/html/2504.11895v2)
- **Results** (I-AUROC / P-AUROC / PRO) — [arXiv 2504.11895](https://arxiv.org/html/2504.11895v2)

  | Dataset | Shots | I-AUROC | P-AUROC | PRO |
  |---|---|---|---|---|
  | MVTec AD | 1 | 97.4 | 96.2 | 92.5 |
  | MVTec AD | 2 | 98.1 | 96.6 | 93.2 |
  | MVTec AD | 4 | 98.6 | 96.9 | 93.7 |
  | VisA | 1 | 94.8 | 97.6 | 91.6 |
  | VisA | 2 | 95.0 | 97.7 | 91.8 |
  | VisA | 4 | 95.7 | 98.0 | 92.5 |
  | Real-IAD | 1 | 70.8 | — | — |
  | Real-IAD | 2 | 74.7 | — | — |
  | Real-IAD | 4 | 79.1 | — | — |

- **Throughput** on an RTX 4090 at batch 8: 10.1 img/s with ViT-L, 26.6 img/s with ViT-B. — [arXiv 2504.11895](https://arxiv.org/html/2504.11895v2)
- **Conflict on prior state of the art.** VisionAD cites 95.8 (KAG-prompt) as the previous 1-shot MVTec AD best, but AnomalyDINO already reported 96.6. VisionAD's Real-IAD 1-shot figure (70.8) also differs strongly from DuoAD's AnomalyDINO figure (80.6), so the Real-IAD protocols probably differ. — [arXiv 2504.11895](https://arxiv.org/html/2504.11895v2) vs [arXiv 2405.14529](https://arxiv.org/html/2405.14529) and [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)

**UniVAD [CVPR 2025]**
- **Design.** Training-free, but it stacks several models: CLIP-L/14@336, DINOv2-G/14, RAM (Recognize Anything), Grounded-SAM and SAM. — [arXiv 2412.03342](https://arxiv.org/html/2412.03342v3)
- **1-shot results** (I-AUROC / P-AUROC): 97.8 / 96.5 on MVTec AD, 93.5 / 98.2 on VisA, 71.0 / 75.1 on MVTec LOCO. — [arXiv 2412.03342](https://arxiv.org/html/2412.03342v3)
- **Motivating example.** PatchCore's 1-shot I-AUROC drops from 84.1 on MVTec AD to 62.0 on MVTec LOCO. — [arXiv 2412.03342](https://arxiv.org/abs/2412.03342)
- **Limitation.** The authors state that "computational latency during inference somewhat limits its applicability in real-time scenarios". — [arXiv 2412.03342](https://arxiv.org/html/2412.03342v3)

**Older few-shot baselines** (from AnomalyDINO's tables, WinCLIP protocol) — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **PatchCore [2022 method]** (I-AUROC / P-AUROC / PRO):

  | Dataset | 1-shot | 2-shot | 4-shot |
  |---|---|---|---|
  | MVTec AD | 83.4±3.0 / 92.0 / 79.7 | 86.3 / 93.3 / 82.3 | 88.8 / 94.3 / 84.3 |
  | VisA | 79.9 / 95.4 / 80.5 | 81.6 / 96.1 / 82.6 | 85.3 / 96.8 / 84.9 |

- **WinCLIP+ [CVPR 2023]** (I-AUROC / P-AUROC / PRO):

  | Dataset | 1-shot | 2-shot | 4-shot |
  |---|---|---|---|
  | MVTec AD | 93.1 / 95.2 / 87.1 | 94.4 / 96.0 / 88.4 | 95.2 / 96.2 / 89.0 |
  | VisA | 83.8 / 96.4 / 85.1 | 84.6 / 96.8 / 86.2 | 87.3 / 97.2 / 87.6 |

- **APRIL-GAN [2023]** (I-AUROC / PRO; P-AUROC only where given):

  | Dataset | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
  |---|---|---|---|---|---|
  | MVTec AD | 92.0 / 90.6 | 92.4 / 91.3 | 92.8 / 91.8 | 93.1 / 92.4 | 93.2 / 92.6 |
  | VisA | 91.2 / 90.0 (P-AUROC 96.0) | 92.2 | 92.6 | 93.0 | 93.2 |

  APRIL-GAN has the best 1-shot VisA I-AUROC in that table, beating AnomalyDINO-S's 87.4.
- **SPADE / PaDiM on VisA** (I-AUROC): SPADE 79.5 at 1-shot and 81.7 at 4-shot; PaDiM 62.8 at 1-shot and 72.8 at 4-shot.
- **GraphCore on MVTec AD** (I-AUROC): 89.9 at 1-shot, 95.9 at 8-shot.
- **ADP on MVTec AD** (I-AUROC): 95.4, 96.2, 97.0 and 97.0 at 2, 4, 8 and 16 shots.

**Training requirements of the CLIP-family "few-shot" methods**
- PromptAD [CVPR 2024] learns prompts from normal samples, so it needs per-category training. — [arXiv 2404.05231](https://arxiv.org/abs/2404.05231)
- InCTRL [CVPR 2024] (CLIP ViT-B/16+) is trained on an auxiliary dataset (MVTec AD) and then applied to other domains with 2, 4 or 8 shots. — [GitHub mala-lab/InCTRL](https://github.com/mala-lab/InCTRL)
- APRIL-GAN adds extra linear layers that map image features to the joint embedding space, and uses memory banks of reference features for few-shot. It placed 1st in zero-shot and 4th in few-shot at the VAND 2023 challenge. — [arXiv 2305.17382](https://arxiv.org/abs/2305.17382)

**Zero-shot CLIP family** (no normal images of the target class)
- **PA-CLIP comparison table** (I-AUROC / P-AUROC, PRO where given) — [arXiv 2503.01292](https://arxiv.org/html/2503.01292v1)

  | Method (0-shot) | MVTec AD | VisA |
  |---|---|---|
  | WinCLIP | 91.8 / 85.1, PRO 64.6 | 78.1 / 79.6, PRO 56.8 |
  | AnomalyCLIP | 91.5 / 91.1, PRO 81.4 | 82.1 / 95.5, PRO 87.0 |
  | AdaCLIP | 89.2 / 88.7 | 85.8 / 95.5 |
  | VCP-CLIP | P-AUROC 92.0, PRO 87.3 | P-AUROC 95.7, PRO 90.7 |
  | MuSc | 97.8 / 97.1, PRO 93.5 | 92.6 / 98.7, PRO 92.4 |
  | PA-CLIP (author-reported) | 98.4 / 97.5, PRO 93.7 | 93.4 / 98.8, PRO 92.4 |

  PA-CLIP builds its memory bank from unlabeled test images, so it is a batched zero-shot method. It uses ViT-L-14-336 at 518 px.
- **AA-CLIP [arXiv Mar 2025]** is trained on one auxiliary dataset and tested on the other, using OpenCLIP ViT-L/14 at 518 px. — [arXiv 2503.06661](https://arxiv.org/html/2503.06661)
  - MVTec AD I-AUROC: 85.9, 89.7, 92.0 and 90.5 with 2, 16, 64 and full auxiliary shots. P-AUROC is 91.0 to 91.9.
  - VisA I-AUROC: 78.4 to 84.6.
  - Its baselines: AnomalyCLIP I-AUROC 90.9 on MVTec AD and 82.1 on VisA; AdaCLIP 90.0 and 84.3.
- **MuSc [ICLR 2024; MuSc-V2 accepted to TPAMI 2026]** — [GitHub xrli-U/MuSc](https://github.com/xrli-U/MuSc)
  - Setup: CLIP ViT-L-14-336 at 518 px, scoring each image against a default batch of 200 unlabeled test images.
  - MVTec AD: I-AUROC 97.77, P-AUROC 97.11, PRO 93.45.
  - VisA: I-AUROC 92.57, P-AUROC 98.71, PRO 92.43.
  - With a DINOv2 ViT-B/14 backbone, MVTec AD I-AUROC is 96.31.
  - Time per image on an RTX 3090: 955 ms (ViT-L at 518 px), 270 ms (ViT-L at 336 px), 48 ms (ViT-B/32 at 256 px).
- **EdgeZSAD [arXiv 2606.16119, June 2026]**
  - Design: a TinyViT-21M encoder at 512 px (53.77 GFLOPs) with a global-ranking head and a spatial-correction head. It is trained once on 216,917 Real-IAD-derived images (2,000 steps).
  - Zero-shot results (I-AUROC / P-AUROC / PRO): MVTec AD 91.6 / 79.3 / 67.8; VisA 88.2 / 92.9 / 74.2.
  - Latency: Jetson Orin Nano Super (TensorRT FP16) 32.83 ms, or 30.46 FPS; Qualcomm RB5 Gen2 (QNN GPU FP16) 139.97 ms.
  - FP16 host-vs-device drift is under 0.2 AUROC points. INT8 was not evaluated.
  - — [arXiv 2606.16119](https://arxiv.org/html/2606.16119)

**Other 2025–2026 names**
- AnomalyAny [CVPR 2025] is not a detector. It generates unseen anomalies with Stable Diffusion, conditioned on a single normal image and a text prompt, and the authors report it improves downstream AD on MVTec AD and VisA (search snippet only). — [arXiv 2406.01078](https://arxiv.org/html/2406.01078v3); [GitHub EPFL-IMOS/AnomalyAny](https://github.com/EPFL-IMOS/AnomalyAny)

### Inferences
- **Swap the feature extractor first.** Moving from ImageNet-CNN patch features (PatchCore) to DINOv2 features is the largest single few-shot gain available. On MVTec AD at 1-shot, I-AUROC rises from 83.4 to 96.6, about +13 points, with the same kNN logic. Kaizen Eye's current ResNet18 PatchCore sits in the weaker family.
- **Differences among the 2025–2026 leaders are small.** At 1 to 4 shots on MVTec AD they are within about 1 to 1.5 points of each other. Most of the gap comes from backbone size (ViT-L/G versus ViT-S) and from augmentation. For a phone, AnomalyDINO-S is the best-documented point on the accuracy-per-FLOP curve.
- **Batched zero-shot does not fit a live feed.** MuSc and PA-CLIP need about 200 unlabeled images and roughly 1 s per image with ViT-L on a desktop GPU. The idea of scoring against other frames could still be adapted to video, but only with a small backbone.

### Gaps
- There is no single authoritative 2026 leaderboard since Papers with Code closed. Cross-paper numbers mix protocols; Real-IAD 1-shot numbers in particular disagree between VisionAD and DuoAD.
- I did not collect numbers for FiLo, Myriad, AnomalyMoE, AdaptCLIP (beyond Real-IAD), KAG-prompt or FoundAD beyond the single figures quoted above.
- SubspaceAD's accuracy with ViT-S is not tabulated; it is only shown in a figure without numbers in my extract.
- It is unclear whether EdgeZSAD's weights are public.

## Q2. Full-data trained methods (PatchCore, PaDiM, SPADE, FastFlow, CFlow, SimpleNet, EfficientAD, RD4AD, DRAEM): accuracy, latency and training cost

### Takeaway
EfficientAD is still the reference for speed at high accuracy. EfficientAD-S reaches 98.8 I-AUROC on MVTec AD at 2.2 ms per image on an RTX A6000. However, it needs 70,000 training iterations, reported as 5 h 26 min in one benchmark, so it cannot support "enrol from a short video" on a phone. It was also the least robust to lighting changes on MVTec AD 2. PatchCore needs no gradient training (about 8 min to build), but it is slower per image and weaker at few-shot. With full data, WRN50 beats ResNet18 by less than 1 point I-AUROC.

### Cited Findings
**EfficientAD [WACV 2024]**
- **Per-collection I-AUROC:**
  - EfficientAD-S: 98.8 on MVTec AD, 97.5 on VisA, 90.0 on MVTec LOCO (85.8 logical, 94.1 structural).
  - EfficientAD-M: 99.1 on MVTec AD, 98.1 on VisA, 90.7 on MVTec LOCO (86.8 logical, 94.7 structural).
  - — [arXiv 2303.14535](https://arxiv.org/html/2303.14535v3)
- **Speed:** EfficientAD-S 2.2 ms per image and 614 img/s; EfficientAD-M 4.5 ms and 269 img/s. Measured on an NVIDIA RTX A6000, batch 1 for latency and batch 16 for throughput. — [arXiv 2303.14535](https://arxiv.org/html/2303.14535v3)
- **Training:** 70,000 iterations at batch size 1 with Adam (lr 1e-4, decayed after 66,500 iterations). — [arXiv 2303.14535](https://arxiv.org/html/2303.14535v3)
- **Comparison table, mean over MVTec AD, VisA and LOCO** (I-AUROC / AU-PRO / latency on RTX A6000) — [arXiv 2303.14535, Table 1](https://arxiv.org/html/2303.14535v3)

  | Method | I-AUROC | AU-PRO | Latency |
  |---|---|---|---|
  | EfficientAD-S | 95.4 | 92.5 | 2.2 ms |
  | EfficientAD-M | 96.0 | 93.3 | 4.5 ms |
  | PatchCore | 91.1 | 80.9 | 32 ms |
  | PatchCore ensemble | 92.1 | 80.7 | 148 ms |
  | FastFlow | 90.0 | 86.5 | 17 ms |
  | SimpleNet | 87.9 | 74.4 | 12 ms |
  | AST | 92.4 | 77.2 | 53 ms |
  | GCAD | 85.4 | 88.0 | 11 ms |
  | DSR | 90.8 | 78.6 | 17 ms |
  | Student–Teacher | 88.4 | 89.7 | 75 ms |

**Training times** (MVTec, continual-learning benchmark; GPU not stated): PaDiM 6 min, PatchCore 8 min, STFPM 27 min, CFA about 2 h, FastFlow about 2 h 19 min, EfficientAD 5 h 26 min, DRAEM 11 h 46 min. — [arXiv 2403.15463, Table 1](https://arxiv.org/html/2403.15463)

**PatchCore backbone comparison** (full-shot MVTec AD, anomalib v0.7.0 reproduction) — [anomalib PatchCore README v0.7.0](https://github.com/openvinotoolkit/anomalib/blob/v0.7.0/src/anomalib/models/patchcore/README.md)
- Average I-AUROC: 0.980 with Wide ResNet-50, 0.973 with ResNet-18.
- Average P-AUROC: 0.980 with WRN50, 0.976 with ResNet-18.
- The gap is largest on screw (0.960 vs 0.943) and capsule (0.982 vs 0.965).

**Robustness on MVTec AD 2** (AU-PRO at FPR 0.05; seen lighting / unseen mixed lighting) — [arXiv 2503.21622](https://arxiv.org/html/2503.21622)

| Method | Seen lighting (TEST_priv) | Mixed lighting (TEST_priv,mix) |
|---|---|---|
| EfficientAD | 30.8 | 19.2 |
| PatchCore | 28.8 | 26.0 |
| RD++ | 27.1 | 25.2 |
| RD (RD4AD) | 26.4 | 25.0 |
| MSFlow | 24.3 | 11.9 |
| SimpleNet | 21.1 | 13.7 |
| DSR | 20.3 | 17.4 |

**Search-snippet-only claim:** PatchCore-Lite shows that product-quantizing PatchCore's memory bank cuts image-level performance from 0.95 to 0.86. — [search snippet, arXiv 2603.20288](https://arxiv.org/html/2603.20288)

### Inferences
- **EfficientAD does not fit the brief.** The training time (hours on a desktop GPU) rules it out for a 10-hour hackathon with on-phone enrolment. Its MVTec AD 2 drop under new lighting (−11.6 AU-PRO₀.₀₅ points) is also a concern for handheld use.
- **PatchCore-style kNN is more lighting-robust in MVTec AD 2's own benchmark** (−2.8 points). This supports keeping a kNN or statistical design and upgrading the features.
- **A bigger CNN is not the fix.** With full data, WRN50 over ResNet18 is worth less than 1 point I-AUROC. Kaizen Eye's problems (clutter, the wrong object passing) are unlikely to be solved by WRN50.

### Gaps
- I found no published CFlow numbers in the sources fetched, and no mobile or NPU latency for EfficientAD, PatchCore or FastFlow.
- The GPU used for the 5 h 26 min EfficientAD figure is not stated.

## Q3. What is achievable with 10–30 normal images (a short enrolment video)?

### Takeaway
With 8 to 16 normal images, a training-free DINOv2-S method reaches about 98 I-AUROC and 94 PRO on MVTec AD, and about 93 to 95 I-AUROC on VisA. That is within 1 to 3 points of full-data results. Gains flatten after about 8 shots. The remaining gap is mostly on complex scenes (VisA-type) and real multi-view data (Real-IAD: 85 to 89 I-AUROC at 1 to 4 shots).

### Cited Findings
- **MVTec AD, AnomalyDINO-S (672 px):** 98.2 I-AUROC at 8-shot, 98.4 at 16-shot, 99.5 at full. PRO is 93.8, 94.0 and 95.0 respectively. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **VisA, AnomalyDINO-S (672 px):** 93.8 I-AUROC at 8-shot, 94.8 at 16-shot, 97.6 at full. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **AnomalyDINO-S at 448 px (cheaper):** 16-shot I-AUROC is 98.3 on MVTec AD and 93.8 on VisA. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **APRIL-GAN saturates early:** 93.2 I-AUROC on both MVTec AD and VisA at 16-shot. It barely improves after 4 shots. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **4-shot I-AUROC:**
  - SubspaceAD (DINOv2-G): 98.0 on MVTec AD, 94.7 on VisA. — [arXiv 2602.23013](https://arxiv.org/html/2602.23013v3)
  - DuoAD (DINOv3-B): 98.5 on MVTec AD, 95.5 on VisA. — [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)
  - VisionAD (ViT-L): 98.6 on MVTec AD, 95.7 on VisA. — [arXiv 2504.11895](https://arxiv.org/html/2504.11895v2)
- **Real-IAD, 4-shot I-AUROC:** 88.7 for DuoAD and 79.1 for VisionAD. — [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1); [arXiv 2504.11895](https://arxiv.org/html/2504.11895v2)
- **Pose diversity matters more than count.** In 1 to 2-shot settings, uninformative references with pose variations led to false positives, with high variance on categories such as capsule. Rotation augmentation of references is part of AnomalyDINO's default "agnostic" preprocessing. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)

### Inferences
- **About 20 good captures is enough** for near-saturated accuracy on MVTec-like objects, provided they span the poses and lighting seen at test time. For video enrolment, sample 20 to 40 diverse keyframes (for example, by maximizing feature distance) rather than using hundreds of near-duplicate frames. This limits memory-bank size and kNN time without losing coverage. This reasoning is my own and has not been benchmarked.

### Gaps
- No paper benchmarks enrolment from a handheld video (correlated frames) versus independent photos.
- There are no 16-shot numbers for SubspaceAD, DuoAD or VisionAD in my extracts.

## Q4. Backbone choice and NPU deployment: DINOv2 ViT-S/14 vs ResNet18/WRN50/MobileNet; quantization impact; published mobile latency

### Takeaway
A DINOv2 ViT-S/14 encoder is realistic at video rate on the Snapdragon 8 Elite Gen 5 NPU. Depth-Anything-V2-Small (DINOv2 ViT-S encoder plus a DPT head, 24.7M params, 518×518) runs in 11.9 to 18.0 ms on that chip per Qualcomm AI Hub, and w8a16 quantization is about 1.4× faster than float. ResNet18 at 224 px is about 50× cheaper, at 0.23 to 0.59 ms, but its features are much weaker few-shot. I found no published measurement of how INT8 (w8a8) affects DINOv2 anomaly-detection accuracy. General ViT post-training-quantization evidence says W8A8 is near-lossless only with ViT-aware quantization; FP16 on device shows under 0.2 AUROC drift.

### Cited Findings
- **Depth-Anything-V2-Small on Snapdragon 8 Elite Gen 5 "For Galaxy"** (518×518, 24.7M params, NPU; peak memory about 367 to 420 MB) — [HF qualcomm/Depth-Anything-V2](https://huggingface.co/qualcomm/Depth-Anything-V2)

  | Runtime | Float | w8a16 |
  |---|---|---|
  | ONNX | 18.0 ms | 11.9 ms |
  | QNN_DLC | 16.7 ms | 13.4 ms |
  | TFLite | 16.8 ms | — |

- **Same model on older chips (ONNX):** Snapdragon 8 Elite 19.0 ms float and 14.6 ms w8a16; Snapdragon 8 Gen 3 28.1 ms float and 19.8 ms w8a16. — [HF qualcomm/Depth-Anything-V2](https://huggingface.co/qualcomm/Depth-Anything-V2)
- **Depth-Anything-V2-Small's encoder is the DINOv2 ViT-S config** ("vits"), 24.8M params, with a default input of 518. — [GitHub DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **ResNet18 (224×224, 11.7M params) on Snapdragon 8 Elite Gen 5 "For Galaxy" NPU:** TFLite 0.233 ms w8a8 and 0.591 ms float; QNN 0.271 ms w8a8 and 0.586 ms float. — [HF qualcomm/ResNet18](https://huggingface.co/qualcomm/ResNet18)
- **Few-shot gap between backbone families:** PatchCore (ImageNet CNN) scores 83.4 I-AUROC at 1-shot on MVTec AD, versus 96.6 for AnomalyDINO-S (DINOv2 ViT-S). — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **Among DINOv2 sizes,** ViT-S, ViT-B and ViT-L showed "no considerable differences" on MVTec AD, with larger models slightly better on VisA. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **DINOv3 vs DINOv2:** DINOv3-7B underperformed DINOv2-G in SubspaceAD. DuoAD's DINOv3 ViT-B was about 0.5 points better on MVTec AD and VisA, but worse on Real-IAD (84.3 vs 85.1 at 1-shot). — [arXiv 2602.23013](https://arxiv.org/html/2602.23013v3); [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)
- **General ViT quantization evidence** (ImageNet classification, not AD; search-snippet level): PTQ4ViT reports near-lossless W8A8, with drops under 0.5%, including DeiT-S at 79.47% top-1. Earlier basic post-training quantization lost more than 1% even at 8 bits, because post-softmax and GELU activations are non-Gaussian. — [PTQ4ViT arXiv 2111.12293](https://arxiv.org/pdf/2111.12293)
- **FP16 edge deployment** of a ViT-based zero-shot detector (EdgeZSAD, TinyViT-21M) showed host-vs-device AUROC drift below 0.2 points on Jetson and on RB5 (QNN GPU). — [arXiv 2606.16119](https://arxiv.org/html/2606.16119)
- **Deployment gotcha:** DINOv2-L failed on RB5 "because the QNN GPU backend rejects the embedding-stage Resize operator". — [arXiv 2606.16119](https://arxiv.org/html/2606.16119)
- **MobileNetV2 PatchCore-Edge vs PatchCore-Lite** (product-quantized memory bank, m=8, 8 bits), measured on an i5-9600K CPU, not a phone — [arXiv 2603.20288](https://arxiv.org/html/2603.20288)

  | Variant | MVTec AD I / P-AUROC | VisA I / P-AUROC | Total memory | Latency |
  |---|---|---|---|---|
  | PatchCore-Edge | 0.95 / 0.96 | 0.86 / 0.91 | 7.05 MB | 30 ms |
  | PatchCore-Lite | 0.86 / 0.94 | 0.68 / 0.90 | 1.49 MB | 130 ms |

  PaDiM-Edge scores 0.94 / 0.97 on MVTec AD, and PaDiM-Lite (diagonal covariance) 0.85 / 0.94.

### Inferences
- **Latency budget for DINOv2 ViT-S on the phone.** At 448 px (32×32 = 1,024 tokens), the ViT-S forward pass should take about 12 ms or less on the 8 Elite Gen 5 NPU. That is extrapolated from Depth-Anything-V2-Small at 518 px (1,369 tokens plus a DPT decoder) running at 11.9 to 18 ms. This leaves room for about 30 FPS end-to-end if the scoring head is cheap. The iQOO 15 uses the standard 8 Elite Gen 5, and the "For Galaxy" SKU may be clocked slightly higher, so expect a small penalty.
- **Quantization precision.** Prefer w8a16, or FP16, for the feature extractor. Only try w8a8 after measuring AUROC on your own benchmark. Keep the memory bank or PCA basis in FP16 rather than product-quantizing it: PatchCore-Lite lost 9 to 18 I-AUROC points and became 4× slower.
- **Export at a fixed input resolution.** Bake the positional-embedding interpolation into the graph to avoid Resize-operator rejections on QNN (the EdgeZSAD RB5 failure).

### Gaps
- There is no published Snapdragon latency for DINOv2 ViT-S alone at 392 or 448 px, and none for kNN or PCA scoring on the Hexagon NPU.
- There is no published study of INT8 or w8a16 effects on DINOv2-based AD accuracy (AUROC or PRO).
- There are no published AD results with MobileNet or EfficientNet backbones in the few-shot regime beyond the CPU-simulated edge paper.

## Q5. Logical anomalies (MVTec LOCO): which methods handle them, and how well do VLMs, including small ones, perform (MMAD and related)?

### Takeaway
Logical anomalies (missing, extra or misplaced parts, wrong counts) remain the weak spot of patch-kNN methods. PatchCore scores 62.0 I-AUROC on MVTec LOCO at 1-shot and UniVAD 71.0. The best few-shot results come from component-level pipelines: ObjectCore reaches 80.8 AUROC at 4-shot, and the VAND 3.0 winners (FastLogSAD, UniVAD++) reach about 93 to 94.5 image F1-max at 1 to 8 shots. The best training-free VLM approach, LogicQA with GPT-4o and 3 normal images, reaches 87.6 AUROC. Small (3 to 4B) VLMs are weak binary detectors: base Qwen2.5-VL-3B gets 58.8% accuracy on LOCO, and a GRPO-tuned 3B model gets 53.5% on MMAD anomaly discrimination, where chance is 50%. They are more useful as explainers of an anomaly that has already been localized.

### Cited Findings
**Full-data trained methods on LOCO**
- **EfficientAD** (I-AUROC): logical 85.8 for S and 86.8 for M; overall LOCO 90.0 and 90.7. — [arXiv 2303.14535](https://arxiv.org/html/2303.14535v3)
- **SALAD [ICCV 2025]:** 96.1 AUROC on MVTec LOCO, +3.0 points over the prior best (search snippet only). — [CVF ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Fucka_SALAD_--_Semantics-Aware_Logical_Anomaly_Detection_ICCV_2025_paper.pdf)

**Few-shot and training-free on LOCO**
- **ObjectCore [WACV 2026]** (search snippet only; PDF fetch returned 403) — [CVF WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Fucka_ObjectCore_-_Efficient_Few-shot_Logical_Anomaly_Detection_using_Object_Representations_WACV_2026_paper.pdf); [WACV poster page](https://wacv.thecvf.com/virtual/2026/poster/684)
  - It models each image as a set of object representations and uses bipartite matching against the most similar support image.
  - 4-shot I-AUROC: 80.8 on MVTec LOCO and 96.5 on CAD-SD.
  - It beats AnomalyMoE by +4.8 points at k=2 and +5.7 at k=4.
- **UniVAD** (1-shot): 71.0 I-AUROC and 75.1 P-AUROC on MVTec LOCO. PatchCore scores 62.0 at 1-shot. — [arXiv 2412.03342](https://arxiv.org/html/2412.03342v3)
- **VAND 3.0 [CVPR 2025] Category 2** (few-shot, VLM allowed, MVTec LOCO), image F1-max — [arXiv 2509.17615](https://arxiv.org/html/2509.17615)

  | Method | k=1 | k=2 | k=4 | k=8 |
  |---|---|---|---|---|
  | FastLogSAD (winner) | 92.88 | 93.30 | 93.72 | 94.55 |
  | UniVAD++ (runner-up) | 92.18 | 92.59 | 92.97 | 93.36 |

  - FastLogSAD uses BEiT features, multi-feature projection and zero-shot priors.
  - UniVAD++ uses RAM, Grounded-SAM masks, component-aware patch matching and graph modelling.
  - The organizers note that "data contamination cannot be verified", and no submission reported runtime or memory.

**LogicQA [arXiv 2503.20252, 2025]**
- **Design.** Training-free and annotation-free. A VLM describes 3 normal images, summarizes them, and generates checklist questions; questions scoring below 80% accuracy on normal images are filtered out. At test time, each question is asked as 5 paraphrases with majority vote. — [arXiv 2503.20252](https://arxiv.org/html/2503.20252)
- **MVTec LOCO results (GPT-4o):** AUROC 87.6 and F1-max 87.0. Comparisons in the same table:
  - LogicAD 86.0 / 83.7 (AUROC / F1-max)
  - GCAD 86.0 AUROC
  - AST 79.7 AUROC
  - PatchCore 74.0 AUROC
  - WinCLIP 64.3 / 59.5 (AUROC / F1-max)
  - — [arXiv 2503.20252](https://arxiv.org/html/2503.20252)
- **Other VLMs with LogicQA** (F1-max): Gemini-1.5 Flash 79.7, InternVL-2.5-38B 77.6. The weakest category is screw bag (GPT-4o F1 64.5). — [arXiv 2503.20252](https://arxiv.org/html/2503.20252)
- **Preprocessing:** LogicQA required Back Patch Masking and Lang-SAM on LOCO. — [arXiv 2503.20252](https://arxiv.org/html/2503.20252)

**LAD-Reasoner [arXiv 2504.12749, preprint]**
- Built on Qwen2.5-VL-3B with SFT and GRPO. — [arXiv 2504.12749](https://arxiv.org/html/2504.12749)
- MVTec LOCO results (accuracy / F1):

  | Model | Accuracy | F1 |
  |---|---|---|
  | LAD-Reasoner | 60.4% | 63.5% |
  | Base Qwen2.5-VL-3B | 58.8% | 54.8% |
  | Qwen2.5-VL-72B | 62.1% | 59.6% |
  | APRIL-GAN | 52.4% | 47.1% |
  | AnomalyGPT | 42.6% | 57.3% |

**MMAD [ICLR 2025]**
- **Benchmark:** 8,366 images and 39,672 questions drawn from MVTec AD (1,691), MVTec LOCO (1,566), VisA (2,141) and GoodsAD (2,968). Evaluation is 1-shot, with one normal template image. — [arXiv 2410.09453](https://arxiv.org/html/2410.09453)
- **Average accuracy:**
  - GPT-4o 74.92%, Gemini-1.5-pro 73.09%, Claude-3.5-sonnet 68.36%, AnomalyGPT 36.52%.
  - Humans: expert 86.65%, ordinary 78.69%. Random is 28.57%.
  - — [arXiv 2410.09453](https://arxiv.org/html/2410.09453)
- **GPT-4o per task:** anomaly discrimination 68.63%, defect localization 55.62%, defect description 73.21%, object classification 94.98%. — [arXiv 2410.09453](https://arxiv.org/html/2410.09453)
- **Retrieval and masks help:**
  - RAG raised InternVL2-40B from 69.59% to 76.35%.
  - An "expert agent" given ground-truth masks improved defect localization by +21.29 points.
  - — [arXiv 2410.09453](https://arxiv.org/html/2410.09453)
- **Conflicting per-model numbers between arXiv versions:**
  - The latest HTML gives InternVL2-8B 63.14%, MiniCPM-V2.6 66.25%, LLaVA-OneVision-7B 63.27%. — [arXiv 2410.09453](https://arxiv.org/html/2410.09453)
  - v3 gives InternVL2-8B 59.97%, MiniCPM-V2.6 57.31%, and 7B models between 49.96% and 60.14%. v3 contains no models under 4B, and it reports a 23.37-point spread across the InternVL2 size series. — [arXiv 2410.09453v3](https://arxiv.org/html/2410.09453v3)

**EMIT [arXiv 2507.21619, July 2025]** (MMAD, 1-shot; average accuracy / anomaly discrimination) — [arXiv 2507.21619](https://arxiv.org/html/2507.21619)

| Model | Size | Average | Anomaly discrimination |
|---|---|---|---|
| AnomalyR1 (Qwen2.5-VL-3B tuned) | 3B | 74.64% | 53.47% |
| Qwen2.5-VL | 7B | 75.70% | 64.42% |
| InternVL3 | 8B | 74.18% | 66.71% |
| GLM-4.1V-Thinking | 9B | 78.92% | 72.31% |
| EMIT (InternVL3-8B + GRPO) | 8B | 81.95% | 73.87% |

Its LLaVA-OneVision-7B figure (72.86%) differs from the MMAD paper's, so the evaluation protocols probably differ.

**RobustMAD [arXiv 2607.16243, June 2026]** (small multimodal models of about 4B and under) — [arXiv 2607.16243](https://arxiv.org/html/2607.16243)
- **Best overall:** Qwen3-VL-4B-Instruct at 88.31% multiple-choice accuracy.
- **Across models:** stand-alone anomaly detection ranges from 55% to 86%, and pairwise anomaly detection (with reference) from 47% to 87%. Motion blur or low light costs 0.8 to 2.4 points.
- **Open-ended quality (1–5 scale):** Qwen3-VL-4B 3.12, MiniCPM-V 4.0 2.89, GPT-5 Nano 2.73, Gemini 3 Flash 3.71.
- **Conclusion:** the models "fall short of safety-critical requirements". Failure modes are fragile grounding, answers that are not comprehensive, and hallucination on ill-posed queries.

**On-device LLM speed (weak evidence):** Llama 3.2 3B generates about 10 tokens/s on Snapdragon 8 Elite. This comes from a blog search snippet, and the runtime is unspecified. — [Grape Up blog](https://grapeup.com/blog/running-llms-on-device-with-qualcomm-snapdragon-8-elite)

### Inferences
- **A small on-device VLM should not be the primary detector.** Its binary anomaly discrimination is near chance to mid-60s% even at 7 to 8B. It is reasonable as a second stage that explains an anomaly the reflex detector has already localized. MMAD's mask-guided "expert agent" result (+21.29 points on localization) supports feeding it the heatmap, crop and reference image.
- **Handle countable logical rules without a VLM where possible.** Examples are "N screws", "part X present" and "label on the left". Use segmentation or detection counts compared against enrolled references (ObjectCore-style). This is more reliable and much cheaper; use a VLM for free-form explanation.
- **Reasoning must be asynchronous.** At roughly 10 tokens/s for a 3B model, a 30 to 60 token verdict takes several seconds plus image encoding. Run it only on flagged frames, not per frame.
- **LOCO F1-max numbers are not comparable to AUROC.** The VAND 3.0 results should not be compared directly with the AUROC figures above.

### Gaps
- I found no per-source (MVTec LOCO-only) MMAD accuracy, and no MMAD numbers for models under 3B in the original paper.
- There are no published on-device latency or accuracy results for Qwen3-VL-4B or Qwen2.5-VL-3B on Snapdragon 8 Elite Gen 5.
- ObjectCore and SALAD full-text details (per-k tables, runtime) could not be fetched (HTTP 403); the numbers above come from search-engine extracts.
- I did not find numbers for Myriad.

## Q6. Known failure modes (pose, clutter, lighting, textureless/close-up objects, small defects, "wrong object passes") and published fixes

### Takeaway
The main failure modes are well documented:
- Pose variation in references causes false positives.
- Cluttered or high-variance backgrounds hurt SubspaceAD (VisA macaroni2 80.4 I-AUROC).
- Lighting shift breaks learned methods (EfficientAD −11.6 AU-PRO₀.₀₅ points on MVTec AD 2).
- Semantic or logical changes are invisible to local patch matching: AnomalyDINO scores 50.2 AUROC, chance level, on cable "swap".
- PCA foreground masking fails when the object fills about half the frame or more.

The published fixes are reference augmentation (rotations, flips, affine, ±20% intensity), foreground masking, multi-layer features, top-1% mean scoring instead of max, [CLS]-attention patch reweighting, and tiling for small defects. No benchmark measures Kaizen Eye's specific "a different object passes" failure. It follows from the local, order-free nature of patch kNN, which DuoAD's [CLS] "semantic stability" signal could address.

### Cited Findings
**Documented failure modes**
- **Semantic swaps:** AnomalyDINO scores 50.2 ± 4.9 AUROC, chance level, on the MVTec cable "swap" anomaly. — [arXiv 2405.14529, App. B](https://arxiv.org/html/2405.14529)
- **Pose in references:** at 1 to 2 shots, references with pose variations cause false positives, with high variance on capsule. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **Close-ups break PCA masking:** masking fails on cable and transistor close-ups where objects cover roughly 50% of patches or more. Masking is therefore off for textures (wood, tile, leather, carpet, grid). — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **Clutter and structural anomalies** (SubspaceAD):
  - Degrades on "categories with high normal variance or complex, cluttered backgrounds". VisA macaroni2 scores 80.4 I-AUROC and pcb3 86.5.
  - Handles structural or logical anomalies poorly: transistor scores 68.9 PRO at 1-shot.
  - Cannot tell benign artifacts from defects if the artifacts are absent from the support set.
  - — [arXiv 2602.23013, App. F](https://arxiv.org/html/2602.23013v3)
- **MVTec AD 2 real-world difficulties** — [arXiv 2503.21622](https://arxiv.org/html/2503.21622)
  - The scenes include transparent, reflective and overlapping objects, back-light and dark-field illumination, and high normal variance.
  - Defects are tiny relative to 2.6 to 5 MP images, so resizing to 256 px is destructive.
  - Many defects sit at the image borders, unlike MVTec AD and VisA where they concentrate at the centre.
  - The mean state of the art is 52.6% AU-PRO₀.₃.
- **Lighting robustness on MVTec AD 2** (AU-PRO₀.₀₅, seen → unseen lighting): EfficientAD 30.8 → 19.2, MSFlow 24.3 → 11.9, PatchCore 28.8 → 26.0, RD 26.4 → 25.0. — [arXiv 2503.21622](https://arxiv.org/html/2503.21622)
- **Real multi-view data:** Real-IAD 1-shot I-AUROC is 80.6 for AnomalyDINO and 85.1 for DuoAD. — [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)

**Published fixes**
- **AnomalyDINO:** reference rotations plus PCA-based masking, applied selectively; image score is the mean of the top 1% of patch distances. — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
- **SubspaceAD:** 30 random rotations per normal image; multi-layer mean pooling (98.4 I-AUROC versus 97.6 with the last layer only, 4-shot MVTec AD). — [arXiv 2602.23013](https://arxiv.org/html/2602.23013v3)
- **VisionAD:** rotation, flip and affine support augmentation, plus "pseudo multi-view" transforms (thresholding, channel swap, axis flip) applied to both query and reference. — [arXiv 2504.11895](https://arxiv.org/html/2504.11895v2)
- **DuoAD:** [CLS] cosine selects augmentations without labels, and [CLS]-to-patch attention logits reweight patches. The authors describe [CLS] as capturing "holistic object semantics" and being "largely insensitive to local anomalous regions". — [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)
- **Challenge winners:**
  - SuperADD (VAND 4.0 winner, CVPR 2026) uses ±20% intensity augmentation to simulate illumination changes, overlapping 640×640 tiling, and multi-layer kNN. It still struggles with missing parts, thin scratches and single hairs. — [arXiv 2605.14808](https://arxiv.org/html/2605.14808v1)
  - All top VAND 3.0 Category 1 solutions used inputs of at least 448 px. ISVL used tiling plus morphological post-processing; SuperAD used DINOv2 with a PatchCore-style memory bank, training-free. — [arXiv 2509.17615](https://arxiv.org/html/2509.17615)

### Inferences
- **"Different objects passed" is expected from Kaizen Eye's current design.** Max patch-kNN over a coreset only asks whether each local patch looks like some normal patch. A different item made of similar materials, textures or edges can pass. Add a global identity gate: the cosine similarity of the DINOv2 [CLS] (or pooled) embedding against the enrolled references' [CLS] distribution. Reject as "not the enrolled object" when it falls outside the enrolled range. This is cheap, since it comes from the same forward pass, and is motivated by DuoAD's semantic-stability finding. It has not been benchmarked for this purpose.
- **Background-clutter collapse is expected** when background patches enter the memory bank and the score. Replace the fixed 10% border crop with a foreground mask: PCA first component (AnomalyDINO) or [CLS]-attention. MVTec AD 2 shows real defects often sit at image borders, so a fixed border crop can hide them. For close-ups where PCA masking fails, fall back to a centre ROI or no mask.
- **Use the mean of the top 1% of patches, not the 3×3-smoothed max.** It is the published default in both AnomalyDINO and SubspaceAD and is less sensitive to single-patch noise.

### Gaps
- No published benchmark measures "wrong object accepted" (object-identity confusion) or handheld phone capture conditions such as motion blur and autofocus hunting. RobustMAD measured blur and low light only for VLMs.
- I found no published per-image alignment or registration evaluation for DINOv2 few-shot methods. RegAD-style alignment was not researched.

## Q7. Threshold calibration without defect samples

### Takeaway
The strongest practical evidence favours normal-only thresholds: order statistics or percentiles on held-out normal scores, with an explicit guarantee on the false-alarm rate. Synthetic anomalies (diffusion or CutPaste) did not reliably stand in for real industrial defects. With about 20 normals, the "max of normals" threshold only guarantees about 14% FPR at 95% confidence. Guaranteeing 1% FPR needs about 299 independent normals. Correlated video frames do not count as independent.

### Cited Findings
- **Distribution-free false-alarm calibration [arXiv 2608.15090, Aug 15, 2026]**
  - The threshold is an order statistic of m normal calibration scores, with Pr[FPR ≤ α] ≥ 1−δ. — [arXiv 2608.15090](https://arxiv.org/html/2608.15090)
  - Normals needed at 95% confidence:

    | Target FPR | Normals needed |
    |---|---|
    | 10% | 29 |
    | 5% | 59 |
    | 2% | 149 |
    | 1% | 299 |
    | 0.5% | 598 |

  - Empirically, mean FPR was 4.8% to 6.8% against a 10% target on ISP-AD.
  - Warnings: calibration scores must be effectively independent draws from future normals; the scoring pipeline must stay fixed; recalibrate under distribution shift; and batch- or specimen-correlated images must be calibrated at the independent-unit level.
- **MVTec AD 2 practice**
  - Thresholds were set from validation normals as the mean of per-pixel scores plus 3 standard deviations. — [arXiv 2503.21622](https://arxiv.org/html/2503.21622)
  - PatchCore's dataset-wide normalization gave an "excessively high threshold" from defect-free data, so its F1 was poor despite good AU-PRO. — [arXiv 2503.21622](https://arxiv.org/html/2503.21622)
- **SuperADD (2026 challenge winner):** the threshold is the 95th percentile of anomaly values on a held-out 1/8 of the normal training images, times a gain factor of 1.3 to 1.5. — [arXiv 2605.14808](https://arxiv.org/html/2605.14808v1)
- **Synthetic validation [ICML 2024]** — [arXiv 2310.10461](https://arxiv.org/html/2310.10461v3)
  - Setup: synthetic anomalies from 10 to 20 normal images (diffusion style-mixing, or CutPaste), used for model and prompt selection.
  - Natural images: works (Kendall τ = 0.866 on CUB).
  - MVTec AD: picked the best model 2 of 15 times. VisA: 0 of 12. There was no significant correlation on industrial datasets.
  - The paper does not address threshold selection.
- **AnomalyAny [CVPR 2025]** can generate realistic unseen anomalies from a single normal image with Stable Diffusion (search snippet). Its value for calibration specifically is not reported. — [arXiv 2406.01078](https://arxiv.org/html/2406.01078v3)
- **EfficientAD's map normalization** (computed on normal validation images) added +0.8 points in its ablation, which shows that normalizing scores on normal data matters. — [arXiv 2303.14535](https://arxiv.org/html/2303.14535v3)

### Inferences
- **Kaizen Eye's leave-one-out max threshold over about 20 photos gives a weak guarantee.** For max-of-m, Pr[FPR ≤ α] ≥ 0.95 requires 1−(1−α)^m ≥ 0.95. At m = 20 that gives α ≈ 13.9%. This is computed from the order-statistic rule above.
- **Recommended calibration recipe:**
  - Hold out normal frames from a separate enrolment pass, not neighbours of reference frames.
  - Set τ at a high percentile or order statistic, optionally with a SuperADD-style gain margin.
  - Report the guaranteed FPR in the UI ("≤10% false alarms at 95% confidence with 29 held-out normals").
  - On a conveyor, add temporal k-of-n voting across consecutive frames. This cuts per-object false alarms without lowering the threshold, but assumes frames of the same object are scored several times.
- **Synthetic defects are best used for demos and sanity checks.** For example, paste a Perlin or CutPaste blob and confirm the heatmap fires. The evidence says they are unreliable for choosing thresholds or models on industrial data.

### Gaps
- I found no study quantifying CutPaste, DRAEM-Perlin or NSA synthetic anomalies specifically for threshold calibration in few-shot, training-free detectors.
- There is no study of per-image adaptive thresholds for DINOv2 few-shot AD.

## Q8. Ranked recommendation for Kaizen Eye: (a) per-frame "reflex" detector on the phone NPU, (b) slower "reasoning" stage

### Takeaway
**(a) Reflex:** replace ResNet18 PatchCore with a training-free DINOv2 ViT-S/14 patch-feature detector (AnomalyDINO-style kNN, optionally with a SubspaceAD-style PCA-residual head). Add rotation and intensity-augmented video enrolment, a foreground mask, a [CLS] object-identity gate, top-1% mean scoring and an order-statistic threshold. Published 16-shot accuracy at 448 px is 98.3 I-AUROC on MVTec AD and 93.8 on VisA, and the encoder should run at about 12 to 18 ms on the 8 Elite Gen 5 NPU.

**(b) Reasoning:** run asynchronously on flagged frames only. Combine deterministic component and count checks for logical rules with a small on-device VLM (about 4B; Qwen3-VL-4B led small models in RobustMAD). Prompt it with the reference image, test crop and heatmap in a LogicQA-style checklist. Treat it as an explainer and verifier, not the primary detector.

### Cited Findings
**Evidence for the reflex choice**
- **Few-shot accuracy by shot count** (DINOv2 ViT-S, 21M params) — [arXiv 2405.14529](https://arxiv.org/html/2405.14529)
  - 1-shot: 96.5 I-AUROC on MVTec AD at 448 px, 96.6 at 672 px.
  - 16-shot: 98.3 at 448 px and 98.4 at 672 px on MVTec AD; 93.8 and 94.8 on VisA.
  - Runtime: about 60 ms per image at 448 px on an A40, including kNN.
- **NPU proxy latency:** the DINOv2 ViT-S-based Depth-Anything-V2-Small at 518 px takes 11.9 ms (ONNX w8a16) to 18.0 ms (ONNX float) on the Snapdragon 8 Elite Gen 5 "For Galaxy" NPU. — [HF qualcomm/Depth-Anything-V2](https://huggingface.co/qualcomm/Depth-Anything-V2)
- **SubspaceAD:** constant-cost PCA-residual scoring, under 1 MB per category, 36 ms per image with DINOv2-S at 448 px. Headline accuracy (97.1 I-AUROC on MVTec AD at 1-shot) uses DINOv2-G. — [arXiv 2602.23013](https://arxiv.org/html/2602.23013v3)
- **DuoAD's [CLS]-based add-ons** reach 97.2 I-AUROC on MVTec AD at 1-shot, versus 96.5 for AnomalyDINO, with ViT-B and no training. — [arXiv 2607.23924](https://arxiv.org/html/2607.23924v1)

**Methods ruled out for the reflex stage**
- EfficientAD needs 70k training iterations (5 h 26 min reported) and is lighting-fragile on MVTec AD 2. — [arXiv 2303.14535](https://arxiv.org/html/2303.14535v3); [arXiv 2403.15463](https://arxiv.org/html/2403.15463); [arXiv 2503.21622](https://arxiv.org/html/2503.21622)
- MuSc takes 955 ms per image with ViT-L at 518 px on an RTX 3090, plus a batch of 200 images. — [GitHub MuSc](https://github.com/xrli-U/MuSc)
- UniVAD's own authors cite latency limits. — [arXiv 2412.03342](https://arxiv.org/html/2412.03342v3)
- CLIP zero-shot methods reach only about 89 to 92 I-AUROC on MVTec AD and 78 to 86 on VisA. — [arXiv 2503.01292](https://arxiv.org/html/2503.01292v1)

**Evidence for the reasoning choice**
- LogicQA with GPT-4o and 3 normal images: 87.6 AUROC on LOCO; with InternVL-2.5-38B, 77.6 F1-max. — [arXiv 2503.20252](https://arxiv.org/html/2503.20252)
- Base Qwen2.5-VL-3B: 58.8% accuracy on LOCO. — [arXiv 2504.12749](https://arxiv.org/html/2504.12749)
- Qwen3-VL-4B is the best small model in RobustMAD at 88.31% multiple-choice accuracy, but is "insufficient" for safety-critical use. — [arXiv 2607.16243](https://arxiv.org/html/2607.16243)
- In MMAD, mask guidance improved localization by +21.29 points and RAG added +6.75 points. — [arXiv 2410.09453](https://arxiv.org/html/2410.09453)
- ObjectCore-style object matching: 80.8 I-AUROC on LOCO at 4-shot (search snippet). — [CVF WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Fucka_ObjectCore_-_Efficient_Few-shot_Logical_Anomaly_Detection_using_Object_Representations_WACV_2026_paper.pdf)

### Inferences
All recommendations below are my synthesis of the cited evidence, not published results for this exact configuration.

**(a) Reflex detector, ranked**

1. **"DINO-kNN" (AnomalyDINO-style), best fit for the 10-hour build**
   - **Encoder.** DINOv2 ViT-S/14 at a fixed 448×448 input (32×32 patches), or 392 px if FPS is short. Export with baked positional embeddings, in w8a16 or FP16 on QNN/LiteRT.
   - **Enrolment from a 10 to 20 s video.**
     - Pick about 20 to 40 diverse keyframes.
     - Add rotations and flips, plus ±20% brightness augmentation (SuperADD).
     - Build the memory bank from foreground patches only, then coreset it to bound kNN cost.
   - **Per frame.**
     - Take the foreground mask from PCA component 1, or from [CLS] attention for close-ups.
     - Score as the mean of the top 1% of patch-NN distances.
     - Run a [CLS]-cosine gate against enrolled frames: "not the enrolled object" is reported separately from "defect".
     - Apply k-of-n temporal voting.
   - **Threshold.** Use the order statistic on a separate held-out normal pass (Q7).
   - **Why first.** It has the best-documented few-shot accuracy per FLOP, no training, Apache-2.0 reference code, and it directly addresses Kaizen Eye's clutter and wrong-object failures.
2. **Same encoder with a SubspaceAD-style PCA-residual head, built alongside**
   - Enrolment is PCA on the augmented patch features (τ≈0.99); scoring is a single matrix multiply per frame; storage is under 1 MB.
   - Run it next to the kNN head from the same features, and pick one (or average the rank-normalized scores) on the team's own handheld benchmark.
   - It ranks second only because its accuracy with ViT-S is unpublished.
3. **DuoAD refinements as a stretch goal:** [CLS]-attention patch reweighting and self-calibrated augmentation. They need the attention logits exposed in the exported graph.
4. **Keep the current ResNet18 PatchCore only as a fallback or ensemble member.** It is very cheap (0.23 to 0.59 ms backbone on the NPU at 224 px) but clearly weaker at few-shot (83.4 versus 96.6 at 1-shot on MVTec AD, WRN50-PatchCore versus DINOv2-S).
5. **Not for the reflex path:** EfficientAD (training time), CLIP-L zero-shot methods (compute and accuracy), MuSc or PA-CLIP (batch plus about 1 s per image on a desktop GPU), UniVAD (multi-model latency), and EdgeZSAD (140 ms on a Qualcomm RB5 GPU in FP16; 91.6 I-AUROC zero-shot on MVTec AD).

**(b) Reasoning stage, ranked**

1. **Deterministic logical checks from enrolment**
   - Record component counts and positions from segmentation or clustering of foreground patches on the enrolled frames. Flag missing, extra or misplaced parts by count or position mismatch; this is the ObjectCore idea, simplified.
   - It is fast and explainable, and targets exactly the LOCO-type errors where patch kNN fails (PatchCore 62.0 and UniVAD 71.0 at 1-shot on LOCO).
2. **On-device small VLM, about 4B (Qwen3-VL-4B, else Qwen2.5-VL-3B), called asynchronously on flagged frames**
   - Input: the reference frame, the test crop and a heatmap overlay.
   - Prompt: a short checklist generated once at enrolment, LogicQA-style from 3 normal frames, with majority vote over paraphrases if time allows.
   - Output: a natural-language explanation and a confirm/reject vote.
   - Do not let it overrule a strong reflex alarm. Small VLMs' anomaly-discrimination accuracy is only about 53% to 66% on MMAD.
3. **Heavy few-shot LOCO pipelines** (UniVAD++, FastLogSAD, about 93 to 94.5 F1-max at k=1 to 8 in VAND 3.0) are not realistic on-device in the build window. They are worth citing to judges as the state-of-the-art ceiling.

**Novelty angle for judges:** combine video enrolment with pose and lighting augmentation, a foundation-model (DINOv2) few-shot detector on the NPU, an object-identity gate, a distribution-free false-alarm guarantee, and an on-device VLM explainer. Each piece is backed by 2025–2026 literature, and the combination on a phone NPU appears unpublished.

### Gaps
- None of the recommended components has published results on a smartphone NPU. Accuracy after w8a16 quantization, the latency of kNN versus PCA scoring on Hexagon, and small-VLM latency on the 8 Elite Gen 5 must be measured by the team.
- The [CLS] identity gate and k-of-n temporal voting are engineering proposals without published AD benchmarks.
- SubspaceAD with ViT-S has no published accuracy.
