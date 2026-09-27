# Kaizen Eye 2 — evaluation export folder (schema 1)

The phone writes one **export folder** per Twin evaluation session; it is pulled with `adb pull` and evaluated on the
laptop by `tools/lab/twin_eval.py` (A/B of scoring / bank / rotation / τ rules and ResNet18 vs DINOv2, features
recomputed on the laptop from the stored crops). `tools/lab/make_synthetic_export.py` writes a synthetic folder in
exactly this format (`testdata/synthetic_export/`) — use it as the reference example and as a parser test.

Spec references (§) are to `docs/verification/twin-spec.md` v1. Everything here is *data the phone already computes*
for its own verdicts; the export only serialises it. Nothing in this folder is ever read back by the app.

## 1. Location and layout

Suggested location: `getExternalFilesDir(null)/exports/<twinId>_<yyyyMMdd-HHmmss>/`
(= `/sdcard/Android/data/com.kaizeneye.v2/files/exports/...`, pullable without root). Write to `<name>.tmp/` and
rename when complete (the laptop ignores `*.tmp` folders).

```
<export>/
  manifest.json
  teach/0000.jpg  0000.json  0001.jpg  0001.json ...      every frame of ONE teach recording, in time order
  parts/<clip>_<trackId>.jpg  <clip>_<trackId>.json ...    one entry per judged presentation (live line / steady hold)
  negatives/0000.jpg  0000.json ...                        the negatives library objects used for τ_id (may be empty)
```
- `NNNN` = zero-padded 4-digit index (`0000`, `0001`, …) in capture order, per folder.
- `<clip>` = the clip/session name chosen in the app, `[A-Za-z0-9-]{1,32}` (no `_`), e.g. `good1`, `defects`, `wrong2`.
- `<trackId>` = the tracker's integer track id (§11.2), zero-padded to 4 digits (`line-a_0017`). Unique within a clip.
  Steady-hold/tap captures use the same scheme (the track that fired).
- The folders may be empty but must exist. A `.json` without a `.jpg` is allowed where stated below.

All JSON files: UTF-8 without BOM, `.` decimal separator, numbers as plain JSON numbers (doubles: shortest
round-trip form, e.g. Kotlin `Double.toString()`; never `NaN`/`Infinity` — use `null`). Enums are SCREAMING_CASE
strings exactly as in `testdata/golden_twin.json`: verdict `PASS | DEFECT | NOT_ENROLLED | REFRAME`,
sanity `OK | NO_OBJECT | TOUCHES_BORDER | TOO_SMALL | TOO_LARGE | MULTIPLE | NO_CORE`,
not-enrolled reason `IDENTITY | SHAPE | COVERAGE`, trigger `LINE | STEADY | TAP`.

## 2. Images (`*.jpg`)

- The §1.4 **crop bytes** of the presentation: `cropResize(fullResFrame, crop.x0, crop.y0, crop.side, cropSize)`,
  RGB, `cropSize × cropSize`, JPEG quality **95** (`Bitmap.compress(JPEG, 95, …)`), no EXIF rotation, same
  orientation as the analysis frames (sensor/landscape — do not rotate to portrait).
- `cropSize` is stated in the manifest. **Recommended: 448** whatever the phone's backbone is, so both backbones can
  be evaluated without up-sampling (the laptop resizes 448 → 320 for ResNet18 with §1.4, `side = cropSize`).
  If the export must be the exact backbone input, use `cropSize = inputSize` (320 for ResNet18).
- JPEG is lossy: laptop features are recomputed from the decoded JPEG, so they differ slightly from the phone's
  (the phone's own numbers are stored in `phone` for the comparison).

## 3. Common per-crop object (`CROPINFO`)

Every `.json` in `teach/`, `parts/` and `negatives/` contains these keys (they describe the chosen component, §2.5,
and its crop square, §3). All pixel coordinates at **analysis resolution** unless stated.

| key | type | meaning |
|---|---|---|
| `tMs` | integer | frame timestamp in ms (the frame's own timestamp, §11 — monotonic, only differences matter) |
| `sanity` | enum | §2.6 result for this crop (`OK` or the REFRAME reason) |
| `sharpness` | number | §1.3 variance of the Laplacian on the grey `inputSize × inputSize` backbone crop, full rectangle (the teach keep step, §7 steps 2–3); `null` if no crop |
| `crop` | `{"x0","y0","side"}` numbers | the §3 crop square in **full-resolution** continuous coordinates (float64 as computed) |
| `mask` | `{"x","y","w","h","rle"}` | the chosen component's pixels over its bounding box: `x = minX`, `y = minY`, `w = maxX−minX+1`, `h = maxY−minY+1` (§2.4, inclusive bbox); `rle` = run lengths over the `w·h` window in row-major order, **alternating 0-runs and 1-runs, starting with a 0-run** (which may be `0`); `sum(rle) = w·h`. A pixel is `1` iff its final label (§2.4, after morphology, small-blob removal and renumbering) is this component's label |
| `cov` | number[gh·gw] | §4 coverage of every patch cell for the **phone's** grid (`pipeline.gh × pipeline.gw`), row-major `p = r·gw + c`, values `k/16` |
| `geometry` | `{"area","fill","aspect","hu1","solidity"}` | §2.7 features of the component (`area` integer pixels) |
| `theta` | number | §2.7 principal angle `0.5·atan2(2·μ11, μ20 − μ02)` in radians |

`mask` + `crop` + `manifest.analysisFactor` let the laptop recompute `cov` for any grid (32×32 for DINOv2,
40×40 for ResNet18) and for rotated crops exactly as §4 does, and re-derive `geometry`/`theta` (§2.7) — the laptop
checks that its `cov`/`geometry` match the phone's for the phone's grid (a free consistency test of the port).

**Frames without a component** (`sanity = NO_OBJECT`) write only `{"tMs", "sanity"}` (other keys `null` or absent)
and no `.jpg`. Every other sanity result has a component and a crop square, so all keys are present and the `.jpg`
is written (the laptop needs it only for `OK`, but REFRAME crops are useful for debugging).

## 4. `manifest.json`

```json
{
  "schema": 1,
  "createdAtMs": 1790000000000,
  "twinId": "20260927-031500-a1b2",
  "twinName": "M8 nut",
  "pipeline": { "...": "the Twin's twin.json \"pipeline\" object, verbatim (§8)" },
  "fingerprint": "<twin.json fingerprint>",
  "analysisFactor": 4,
  "inputSize": 320,
  "cropSize": 448,
  "frameW": 1280,
  "frameH": 720,
  "sheet": {"mu": [191.0, 199.0, 196.0], "sigma": [3.0, 3.0, 3.0]},
  "twin": {"tau": 1.43, "tauTeach": 1.43, "tauId": 0.91, "sensitivity": 1.0, "coverageCut": 0.5, "calibrated": false},
  "counts": {"teach": 96, "parts": 120, "negatives": 30},
  "app": {"versionName": "2.0.0", "device": "iQOO 15", "socModel": "SM8850"}
}
```

| key | required | meaning |
|---|---|---|
| `schema` | yes | `1` |
| `createdAtMs` | yes | wall-clock ms when the export was written (informational) |
| `twinId` | yes | id of the Twin whose teach recording is in `teach/` and which produced `phone` verdicts |
| `twinName` | no | display name |
| `pipeline` | yes | the Twin's `pipeline` object (§8): backbone id/sha/inputSize/gh/gw/dim/l2NormalizePatches, `mask` params (`analysisFactor`, `kSigma`, `sigmaMin`, `minBlobPx`, `borderMargin`, `cropMargin`, `coreThreshold`), `scoreRule`, `bankRule`, `rotation` |
| `fingerprint` | no | the Twin's fingerprint (§8) |
| `analysisFactor` | yes | §1.1 factor `f` (full-res px per analysis px); equals `pipeline.mask.analysisFactor` |
| `inputSize` | yes | the phone backbone's input `N` (320 ResNet18, 448 DINOv2) = `pipeline.inputSize` |
| `cropSize` | yes | side of every stored `.jpg` crop (§2 above) |
| `frameW`, `frameH` | yes | full-resolution frame size (e.g. 1280×720; analysis size = `frameW/f × frameH/f`) |
| `sheet` | yes | the §2.1 sheet model at export time: `mu`, `sigma` per RGB channel. The laptop fills areas outside a stored crop with `mu` when it rotates crops (§6 below) |
| `twin` | no | the phone Twin's thresholds at the time of the parts (`thresholds` of twin.json, §8): `tau`, `tauTeach`, `tauId`, `sensitivity`, `coverageCut`, `calibrated` |
| `counts` | no | number of `.json` files per folder (informational; the laptop lists the folders) |
| `app` | no | free-form app/device info |

## 5. `teach/NNNN.json`

`CROPINFO` plus:

| key | type | meaning |
|---|---|---|
| `index` | integer | = `NNNN` (frame order within the teach recording) |

Write **every** analysed teach frame (also the rejected ones), in time order, so the laptop sees the same
`framesSeen`/`framesAccepted` as the phone and re-runs §7 (accept → sharpness keep → k-center → bank → LSO → τ).
One teach recording per export (the Twin's own).

## 6. `parts/<clip>_<trackId>.json`

`CROPINFO` plus:

| key | type | meaning |
|---|---|---|
| `clip` | string | clip/session name (= the file-name prefix) |
| `trackId` | integer | tracker id (§11.2) |
| `label` | enum string | ground truth chosen by the operator for the clip (or edited per part): `good` (held-out good part, same lot), `defect` (seeded defect), `wrong` (a different object, never in the negatives library), `rotated` (good part presented at a large in-plane rotation), `lookalike` (same outline, different part — reported separately from `wrong`), `calib` (good part used for calibration, §10; never also `good`) |
| `trigger` | enum | `LINE`, `STEADY` or `TAP` (what fired the judgement, §11.3–11.4) |
| `phone` | object | the phone's own judgement of this crop: `{"verdict", "reason", "raw", "s", "sim", "a", "peak", "ms"}` — `verdict` enum; `reason` = the not-enrolled reason (`IDENTITY`/`SHAPE`/`COVERAGE`), the REFRAME sanity reason, or `null`; `raw` §6.3 raw score, `s` normalised score, `sim` identity similarity (§7.3), `a` anomalous fraction (§6.4), `peak` `[row, col]` or `null`, `ms` trigger→verdict latency; unknown/unused values `null` (e.g. all numbers are `null` for REFRAME) |

The crop is the one the phone judged (best crop on `LINE`, the firing frame on `STEADY`, §11.5). If borderline voting
re-scored other crops, export the first (judged) crop and the **final** `phone` numbers.

## 7. `negatives/NNNN.json`

`CROPINFO` plus `index` (integer). These are the negatives-library captures (steady-hold auto-capture, §7.3) whose
globals are stored in the Twin's `negatives.f16`. Wrong objects used for evaluation must **not** also be negatives.

## 8. What the laptop does with it (for reference)

`tools/lab/twin_eval.py --export <export> --backbone r18|dinov2 --out <dir>`:
1. decodes every `OK` crop, resizes it to the backbone input with §1.4 when `cropSize ≠ N`, embeds it with the
   backbone TFLite (`ai_edge_litert`; DINOv2 patch vectors L2-normalised), recomputes `cov` for the backbone grid
   from `mask`/`crop`/`analysisFactor` (§4);
2. re-teaches the Twin from `teach/` (§7) for every variant: score rule × bank rule × rotation × τ rule
   (τ rule `CALIBRATED` only when `calib` parts exist: `τ = max(τ_teach, max raw over valid calib parts)`, §10.1);
   negatives feed `τ_id` (§7.3);
3. judges every part (§9, no borderline voting), and reports per variant AUROC good-vs-defect (score `s`),
   AUROC good-vs-wrong (identity `sim`), PASS rate on held-out `good` (and on `rotated`), NOT_ENROLLED rate on `wrong`
   (and `lookalike` separately), DEFECT rate on `defect`, each with two-sided 95 % Clopper–Pearson intervals.
   **Rotation variants rotate the stored crop about its centre** (bilinear resampling of the decoded JPEG; points
   outside the stored square take the sheet colour `sheet.mu`) instead of re-cropping the full-resolution frame as
   the phone would — an approximation (double resampling, and real sheet texture / neighbouring objects in the
   corners are replaced by a flat colour).
