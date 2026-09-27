# SYNTHETIC - Twin A/B report (r18)

- export: `D:\Projects\Kaizen_Eye\testdata\synthetic_export` (twin `synthetic-bracket`, crops 448 px, analysis factor 4)
- backbone: `backbone_r18_320.tflite` 320 px -> 40x40x128; sensitivity 1.0
- evaluation parts: good 24, defect 15, wrong 12, rotated 10, lookalike 4; calib parts (CALIBRATED tau rule only): 24
- export consistency: cov (phone grid) 0/158 mismatching, geometry max rel diff 0.0e+00, theta max |diff| 0.0e+00
- phone vs laptop (default variant, same backbone, laptop features from the JPEG crops): verdicts agree 63/65; |s| diff median 0.0212 max 0.0882; |sim| diff median 0.000411 max 0.00473
- rotation variants rotate the stored crop about its centre (sheet-colour padding) - an approximation of the phone's full-frame re-crop
- **winner: `SMOOTHED_MAX/CORESET/NONE/LSO`** (AUROC good-vs-defect, then identity AUROC; exact ties -> defaults); default: `SMOOTHED_MAX/CORESET/NONE/LSO`
- note: 24 of 24 variants tie on both AUROCs (1.000 / 1.000), so the winner is decided by the defaults-win-ties rule; compare the operating-point rates below before changing a default

| variant (score/bank/rotation/tau) | tau | tau_id | AUROC def | AUROC id | AUROC id look | PASS good | PASS rotated | NOT_ENR wrong | NOT_ENR look | DEFECT |
|---|---|---|---|---|---|---|---|---|---|---|
| `SMOOTHED_MAX/CORESET/NONE/LSO` **(winner)** | 0.873 | 0.9728 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 8/15 = 53 % [27, 79] |
| `SMOOTHED_MAX/CORESET/NONE/CALIBRATED` | 0.873 | 0.9727 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 8/15 = 53 % [27, 79] |
| `SMOOTHED_MAX/TOPK_VIEWS/NONE/LSO` | 0.873 | 0.9728 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |
| `SMOOTHED_MAX/TOPK_VIEWS/NONE/CALIBRATED` | 0.873 | 0.9727 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |
| `TOP1_MEAN/CORESET/NONE/LSO` | 0.880 | 0.9728 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/CORESET/NONE/CALIBRATED` | 0.880 | 0.9727 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/TOPK_VIEWS/NONE/LSO` | 0.880 | 0.9728 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 11/15 = 73 % [45, 92] |
| `TOP1_MEAN/TOPK_VIEWS/NONE/CALIBRATED` | 0.880 | 0.9727 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 11/15 = 73 % [45, 92] |
| `SMOOTHED_MAX/CORESET/CANONICAL/LSO` | 0.716 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `SMOOTHED_MAX/CORESET/CANONICAL/CALIBRATED` | 0.716 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `SMOOTHED_MAX/TOPK_VIEWS/CANONICAL/LSO` | 0.716 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `SMOOTHED_MAX/TOPK_VIEWS/CANONICAL/CALIBRATED` | 0.716 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `TOP1_MEAN/CORESET/CANONICAL/LSO` | 0.745 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `TOP1_MEAN/CORESET/CANONICAL/CALIBRATED` | 0.745 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `TOP1_MEAN/TOPK_VIEWS/CANONICAL/LSO` | 0.745 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `TOP1_MEAN/TOPK_VIEWS/CANONICAL/CALIBRATED` | 0.745 | 0.9749 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 15/15 = 100 % [78, 100] |
| `SMOOTHED_MAX/CORESET/AUGMENT4/LSO` | 0.864 | 0.9797 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |
| `SMOOTHED_MAX/CORESET/AUGMENT4/CALIBRATED` | 0.864 | 0.9798 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |
| `SMOOTHED_MAX/TOPK_VIEWS/AUGMENT4/LSO` | 0.864 | 0.9797 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |
| `SMOOTHED_MAX/TOPK_VIEWS/AUGMENT4/CALIBRATED` | 0.864 | 0.9798 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |
| `TOP1_MEAN/CORESET/AUGMENT4/LSO` | 0.897 | 0.9797 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/CORESET/AUGMENT4/CALIBRATED` | 0.897 | 0.9798 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/TOPK_VIEWS/AUGMENT4/LSO` | 0.897 | 0.9797 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |
| `TOP1_MEAN/TOPK_VIEWS/AUGMENT4/CALIBRATED` | 0.897 | 0.9798 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 4/4 = 100 % [40, 100] | 9/15 = 60 % [32, 84] |

Rates: k/n = % [two-sided 95 % Clopper-Pearson]; REFRAME counts as a failure. AUROC per spec 14 (ties 0.5). Twin summary per variant in results.json.

Runtime 147.2 s (453 backbone runs, 14.2 s; k-NN cache 1794 hits / 670 misses).
