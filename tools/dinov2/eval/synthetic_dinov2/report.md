# SYNTHETIC - Twin A/B report (dinov2)

- export: `D:\Projects\Kaizen_Eye\testdata\synthetic_export` (twin `synthetic-bracket`, crops 448 px, analysis factor 4)
- backbone: `dinov2_s14_448_fp32.tflite` 448 px -> 32x32x384 (per-patch L2); sensitivity 1.0
- evaluation parts: good 24, defect 15, wrong 12, rotated 10, lookalike 4; calib parts (CALIBRATED tau rule only): 24
- export consistency: cov (phone grid) 0/158 mismatching, geometry max rel diff 0.0e+00, theta max |diff| 0.0e+00
- rotation variants rotate the stored crop about its centre (sheet-colour padding) - an approximation of the phone's full-frame re-crop
- **winner: `SMOOTHED_MAX/CORESET/NONE/LSO`** (AUROC good-vs-defect, then identity AUROC; exact ties -> defaults); default: `SMOOTHED_MAX/CORESET/NONE/LSO`
- note: 24 of 24 variants tie on both AUROCs (1.000 / 1.000), so the winner is decided by the defaults-win-ties rule; compare the operating-point rates below before changing a default

| variant (score/bank/rotation/tau) | tau | tau_id | AUROC def | AUROC id | AUROC id look | PASS good | PASS rotated | NOT_ENR wrong | NOT_ENR look | DEFECT |
|---|---|---|---|---|---|---|---|---|---|---|
| `SMOOTHED_MAX/CORESET/NONE/LSO` **(winner)** | 0.761 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `SMOOTHED_MAX/CORESET/NONE/CALIBRATED` | 0.761 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `SMOOTHED_MAX/TOPK_VIEWS/NONE/LSO` | 0.761 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 8/10 = 80 % [44, 97] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `SMOOTHED_MAX/TOPK_VIEWS/NONE/CALIBRATED` | 0.761 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 8/10 = 80 % [44, 97] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/CORESET/NONE/LSO` | 0.890 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/CORESET/NONE/CALIBRATED` | 0.890 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/TOPK_VIEWS/NONE/LSO` | 0.890 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 9/10 = 90 % [55, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/TOPK_VIEWS/NONE/CALIBRATED` | 0.890 | 0.7967 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 9/10 = 90 % [55, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `SMOOTHED_MAX/CORESET/CANONICAL/LSO` | 0.622 | 0.8265 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 12/15 = 80 % [52, 96] |
| `SMOOTHED_MAX/CORESET/CANONICAL/CALIBRATED` | 0.622 | 0.8255 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 12/15 = 80 % [52, 96] |
| `SMOOTHED_MAX/TOPK_VIEWS/CANONICAL/LSO` | 0.622 | 0.8265 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 11/15 = 73 % [45, 92] |
| `SMOOTHED_MAX/TOPK_VIEWS/CANONICAL/CALIBRATED` | 0.622 | 0.8255 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 11/15 = 73 % [45, 92] |
| `TOP1_MEAN/CORESET/CANONICAL/LSO` | 0.793 | 0.8265 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/CORESET/CANONICAL/CALIBRATED` | 0.793 | 0.8255 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/TOPK_VIEWS/CANONICAL/LSO` | 0.793 | 0.8265 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/TOPK_VIEWS/CANONICAL/CALIBRATED` | 0.793 | 0.8252 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 0/4 = 0 % [0, 60] | 10/15 = 67 % [38, 88] |
| `SMOOTHED_MAX/CORESET/AUGMENT4/LSO` | 0.823 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 9/15 = 60 % [32, 84] |
| `SMOOTHED_MAX/CORESET/AUGMENT4/CALIBRATED` | 0.823 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 9/15 = 60 % [32, 84] |
| `SMOOTHED_MAX/TOPK_VIEWS/AUGMENT4/LSO` | 0.823 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 10/15 = 67 % [38, 88] |
| `SMOOTHED_MAX/TOPK_VIEWS/AUGMENT4/CALIBRATED` | 0.823 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 10/15 = 67 % [38, 88] |
| `TOP1_MEAN/CORESET/AUGMENT4/LSO` | 1.014 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 4/15 = 27 % [8, 55] |
| `TOP1_MEAN/CORESET/AUGMENT4/CALIBRATED` | 1.014 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 4/15 = 27 % [8, 55] |
| `TOP1_MEAN/TOPK_VIEWS/AUGMENT4/LSO` | 1.014 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 5/15 = 33 % [12, 62] |
| `TOP1_MEAN/TOPK_VIEWS/AUGMENT4/CALIBRATED` | 1.014 | 0.8008 | 1.000 | 1.000 | 1.000 | 24/24 = 100 % [86, 100] | 10/10 = 100 % [69, 100] | 12/12 = 100 % [74, 100] | 1/4 = 25 % [1, 81] | 5/15 = 33 % [12, 62] |

Rates: k/n = % [two-sided 95 % Clopper-Pearson]; REFRAME counts as a failure. AUROC per spec 14 (ties 0.5). Twin summary per variant in results.json.

Runtime 340.0 s (501 backbone runs, 213.3 s; k-NN cache 1811 hits / 653 misses).
