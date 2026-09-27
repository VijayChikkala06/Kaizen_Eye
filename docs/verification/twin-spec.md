# Kaizen Eye 2 — Visual Twin algorithm spec (v1)

This is the **single source of truth** for the Kaizen Eye 2 inspection maths. Two implementations are written from it
independently and checked against each other:

- `native/core` (Kotlin/JVM, runs on the phone) — unit-tested against `testdata/golden_twin.json`;
- `tools/lab/twin_ref.py` (numpy, laptop) — writes `testdata/golden_twin.json` and is used by `tools/lab/twin_eval.py`.

The legacy PatchCore maths (`mobile/src/core/patchcore.ts`, spec `tools/lab/patchcore_ref.py`, goldens
`testdata/golden_core.json` + `golden_app.json`) is ported unchanged as `com.kaizeneye.core.legacy` and is **not**
redefined here.

If an implementation finds the spec ambiguous, it must not guess silently: pick the reading closest to the text, write
the question and the choice at the top of its source file under `SPEC-QUESTION:`, and report it.

Numbers in `code` are the defaults; every one of them is a named parameter (`Params` objects), never a literal buried
in code.

---------------------------------------------------------------------------------------------------------------------

## 0. Conventions (apply everywhere)

- **Arrays** are row-major. A feature map is `[gh, gw, D]` flattened as `f[(r*gw + c)*D + t]`; patch index
  `p = r*gw + c`. Images are `[H, W, C]`, `x` = column, `y` = row.
- **Continuous image coordinates:** pixel `(x, y)` covers `[x, x+1) × [y, y+1)`; its centre is `(x+0.5, y+0.5)`.
- **Precision:** inputs are float32 (features) or uint8 (pixels). All reductions (sums, distances, moments, variances,
  dot products, means) accumulate in **float64**, in index order. Outputs that are stored (feature rows) are float32 or
  binary16 as stated. Tolerances in the golden tests: `1e-6` relative (+`1e-9` absolute) for float64 results, `1e-4`
  relative for results that pass through float32 feature maps, exact for integers, indices and verdicts.
- **Ties** always go to the lowest index (the first in iteration order), exactly like `numpy.argmax`.
- **Rounding** of a non-negative real `v` to an integer: `floor(v + 0.5)` (half up). Clamp to `[0, 255]` for pixels.
- **Percentile** `P_q(x)` of `n ≥ 1` values, `q ∈ [0, 100]` (numpy's default `linear` method): sort ascending into
  `s[0..n-1]`; `h = (n-1)·q/100`; `lo = floor(h)`; `hi = min(lo+1, n-1)`; `P = s[lo] + (h-lo)·(s[hi]-s[lo])`.
- **Median** = `P_50`. **MAD** (raw) = median of `|x_i − median(x)|`. Robust sigma = `1.4826 · MAD`.
- **Mean / std:** population std (divide by `n`, ddof = 0).
- **Cosine similarity** of two L2-normalised vectors = their dot product (float64).
- **L2 normalise** `v`: `v / ||v||`; if `||v|| == 0` return `v` unchanged (all zeros).
- **Binary16 (f16):** IEEE-754 half precision, round-to-nearest-even (numpy `astype(float16)`), incl. subnormals,
  ±inf on overflow, NaN preserved. Stored little-endian.

## 1. Image operations

### 1.1 Analysis downscale (RGBA → small RGB)
Camera frames arrive as RGBA_8888 `W×H` (default `1280×720`) with a row stride in bytes. The analysis image is the
`f×f` box average (default factor `f = 4` → `320×180`), for `W, H` divisible by `f`:
`out[y][x][c] = floor( (Σ_{i,j<f} in[f·y+i][f·x+j][c] + f²/2) / f² )` for `c ∈ {R,G,B}` (alpha ignored). Integer maths.

### 1.2 Grey
`G = 0.299·R + 0.587·G + 0.114·B` in float64 from uint8 values.

### 1.3 Sharpness (variance of the Laplacian)
On a float64 grey image `G[H][W]` and a rectangle `[x0, x1) × [y0, y1)` (clamped to the image): for every pixel with
`max(x0,1) ≤ x < min(x1, W−1)` and `max(y0,1) ≤ y < min(y1, H−1)`:
`L = G[y−1][x] + G[y+1][x] + G[y][x−1] + G[y][x+1] − 4·G[y][x]`.
Sharpness = population variance of those `L` values; `0.0` if fewer than 9 pixels qualify.

### 1.4 Bilinear sample and square crop-resize
`sample(I, u, v)` at continuous point `(u, v)`: `x = u − 0.5`, `y = v − 0.5`; `x0 = floor(x)`, `y0 = floor(y)`;
`fx = x − x0`, `fy = y − y0`; clamp the four indices `x0, x0+1` to `[0, W−1]` and `y0, y0+1` to `[0, H−1]`;
`(1−fx)(1−fy)·I[y0][x0] + fx(1−fy)·I[y0][x1] + (1−fx)fy·I[y1][x0] + fx·fy·I[y1][x1]` (per channel, float64).

`cropResize(I, x0, y0, side, N)` → `N×N×3`: output pixel `(i, j)` (row `i`, column `j`) =
`sample(I, x0 + (j+0.5)·side/N, y0 + (i+0.5)·side/N)`. The **crop bytes** are `clamp(round(v))` per channel.
The backbone always receives the rounded crop bytes (as float32 values 0..255), never the unrounded floats, so a crop
saved losslessly and re-embedded gives the same features.

`cropResizeRotated(I, cx, cy, side, θ, N)` (rotation challenger, §6.5): output pixel `(i, j)` samples at
`(cx + a·cosθ − b·sinθ, cy + a·sinθ + b·cosθ)` with `a = (j+0.5)·side/N − side/2`, `b = (i+0.5)·side/N − side/2`.

## 2. Empty-sheet model and foreground mask (analysis resolution)

### 2.1 Sheet model
Fitted once on the operator's "empty sheet" tap from `K ≥ 1` analysis frames, over the ROI (default: whole frame).
Per channel `c`: `μ_c = median` of all ROI pixel values of channel `c` in all `K` frames (uint8 values; exact),
`σ_c = max(1.4826 · MAD_c, sigmaMin)` with `sigmaMin = 3.0`.
(Exact integer median/MAD are possible with 256-/512-bin histograms; the result must equal the sort-based definition.)

Slow adaptation (live only, not in golden tests): on a frame with **no** component, `μ_c ← (1−β)·μ_c + β·median_c(frame)`
with `β = 0.02`; `σ` unchanged.

### 2.2 Foreground
`d²(x, y) = Σ_c ((I_c(x, y) − μ_c) / σ_c)²` (float64). Foreground iff `d² > kSigma²`, `kSigma = 4.0`.

### 2.3 Morphology
3×3 square structuring element. **Open** (erode then dilate) then **close** (dilate then erode).
Erosion: a pixel stays 1 iff all its in-image 3×3 neighbours are 1 (neighbours outside the image are ignored).
Dilation: a pixel becomes 1 iff any in-image 3×3 neighbour is 1. (Same as OpenCV's default border handling.)

### 2.4 Connected components
8-connectivity on the final mask. Labels `1, 2, …` are assigned in the order of each component's first pixel in a
row-major scan (label `0` = background). Components with `area < minBlobPx = 30` are removed (their pixels become 0)
and the remaining labels are renumbered `1, 2, …` in the same order. Per component:
`area`, `minX, minY, maxX, maxY` (inclusive pixel indices), centroid `cx = mean(x) + 0.5`, `cy = mean(y) + 0.5`
(continuous coordinates), `touchesBorder` = any pixel with `x < m`, `y < m`, `x ≥ W−m` or `y ≥ H−m`,
`m = borderMargin = 2`.

### 2.5 Main object (teach, steady-hold, calibration)
Candidates = components with `touchesBorder == false`. Main object = largest `area` (ties → lowest label).

### 2.6 Sanity → REFRAME reasons (checked in this order)
For a chosen component `C` in an analysis frame of `W·H` pixels:
1. `NO_OBJECT` — no component (or no candidate in single-object modes).
2. `TOUCHES_BORDER` — `C.touchesBorder`.
3. `TOO_SMALL` — `C.area / (W·H) < minAreaFrac = 0.01`.
4. `TOO_LARGE` — `C.area / (W·H) > maxAreaFrac = 0.80`.
5. `MULTIPLE` — another component `D ≠ C` with `D.area ≥ multipleRatio · C.area` (`multipleRatio = 0.25`) whose
   centroid lies inside `C`'s crop square (§3). (Other parts elsewhere in the frame are fine on a moving line.)
6. `NO_CORE` — the patch core set (§4) is empty.

### 2.7 Geometry features of a component (scale-, rotation- and translation-invariant)
Over the component's pixels with integer coordinates `(x, y)` (the 0.5 offset cancels):
`A = area`; `x̄, ȳ` = means; `μ20 = Σ(x−x̄)²`, `μ02 = Σ(y−ȳ)²`, `μ11 = Σ(x−x̄)(y−ȳ)`.
- `hu1 = (μ20 + μ02) / A²` (first Hu invariant; ≈ 1/(2π) = 0.159 for a disc, larger for elongated shapes).
- Covariance eigenvalues of `[[μ20, μ11], [μ11, μ02]] / A`: `λ1 ≥ λ2 ≥ 0`;
  `aspect = sqrt(λ2 / λ1)` (`1.0` if `λ1 == 0`).
- Principal angle `θ = 0.5 · atan2(2·μ11, μ20 − μ02)`; `u = (cosθ, sinθ)`, `v = (−sinθ, cosθ)`. Oriented extents:
  `wu = max(u·(x,y)) − min(u·(x,y)) + 1`, `wv = max(v·(x,y)) − min(v·(x,y)) + 1`; `fill = A / (wu · wv)`.
- `solidity = A / hullArea`, where `hullArea` is the shoelace area of the convex hull of the corner points
  `(x, y), (x+1, y), (x, y+1), (x+1, y+1)` of all component pixels (Andrew's monotone chain; collinear points dropped).
- The raw `area` (pixels) is also recorded (used only with a wide tolerance, §7.4).

## 3. Crop square (full resolution)
From component `C` (analysis coordinates) and the analysis factor `f`, in full-resolution continuous coordinates:
`bx0 = f·minX`, `by0 = f·minY`, `bx1 = f·(maxX+1)`, `by1 = f·(maxY+1)`; `cx = (bx0+bx1)/2`, `cy = (by0+by1)/2`;
`side = max(bx1−bx0, by1−by0) · (1 + 2·margin)`, `margin = 0.10`; `side = min(side, Wfull, Hfull)`;
`x0 = clamp(cx − side/2, 0, Wfull − side)`, `y0 = clamp(cy − side/2, 0, Hfull − side)`. All float64.
For `MULTIPLE` (§2.6), the same square is used in analysis coordinates (divide by `f`); "inside" = `x0 ≤ cx < x0+side`
and `y0 ≤ cy < y0+side`.
The crop is resized to the backbone input `N` (`320` for ResNet18, `448` for DINOv2) with §1.4.

## 4. Patch-grid object masks
For a crop square (full-res `x0, y0, side`), the component label map at analysis resolution (factor `f`) and a
backbone grid `gh × gw`: `cov[r][c]` = fraction of the `4×4` sample points of cell `(r, c)` that land on the component:
sample `(i, j) ∈ {0..3}²` → full-res point `u = x0 + (c + (j+0.5)/4)·side/gw`, `v = y0 + (r + (i+0.5)/4)·side/gh`
→ analysis pixel `(floor(u/f), floor(v/f))`; a point outside the analysis image counts as 0.
- `core = {p : cov[p] ≥ 0.5}`;
- `S = dilate(core, 1)` (score set: 3×3 dilation on the patch grid, grid border = outside);
- `B = dilate(core, 2)` (bank set: 5×5 dilation).
If `core` is empty the frame is `NO_CORE` (REFRAME).

## 5. Features, descriptors, k-NN
- The backbone maps the `N×N×3` crop bytes to a feature map `[gh, gw, D]` (ResNet18@320: 40×40×128; DINOv2-S/14@448:
  32×32×384). **DINOv2 patch vectors are L2-normalised** (per patch) before anything else; ResNet18 vectors are not.
- **Global object descriptor** `g = L2normalise( Σ_{p ∈ core} f_p )` (all patches if `core` is empty).
- **k-NN** `minSq(p) = min over allowed bank rows b of ||f_p − b||²` (float64 accumulation of float32 inputs). A row is
  excluded by adding `KNN_MASK = 1e30` to its squared distance (the LiteRT graph semantics: `bankNorm` input); if all
  rows are excluded the result is ≥ `1e30`. `d(p) = sqrt(max(0, minSq(p)))`.
  The LiteRT matmul graphs (`knn_p{P}_d{D}_k{K}.tflite`, inputs `feats[P,D]`, `bankT[D,K]`, `bankNorm[K]`, output
  `minD2[P]`) compute the same thing in float32; banks are padded to the smallest bucket `K ≥ rows` with zero columns
  whose `bankNorm = 1e30`, banks larger than the largest bucket are split into chunks and the minima combined, queries
  are padded to `P` with zero rows (their outputs ignored). Graph results must agree with the exact search to
  `max |Δd| ≤ 1e-3 · max(d)`.

## 6. Scoring one crop against a Twin

### 6.1 Distance map
`d(p)` for all `P = gh·gw` patches against the Twin bank (all rows allowed).

### 6.2 Masked smoothing (default)
For `p ∈ S`: `sm(p) = mean{ d(q) : q ∈ N(p) ∩ S }` where `N(p)` = the 3×3 neighbourhood of `p` on the grid (incl. `p`).
Background patches never leak into the score.

### 6.3 Frame score rules
- `SMOOTHED_MAX` (default, legacy-proven): `raw = max_{p ∈ S} sm(p)`.
- `TOP1_MEAN` (challenger): `raw = mean of the n largest d(p), p ∈ S`, `n = max(1, ceil(0.01 · |S|))`.
- Normalised score `s = raw / (τ · sensitivity)`; `s > 1` = "further from normal than calibrated".

### 6.4 Peak, anomalous fraction
- `peak = argmax_{p ∈ S} sm(p)` (ties → lowest `p`) → `(row, col)`.
- `anomalousFraction a = |{p ∈ core : sm(p) > τ·sensitivity}| / |core|`; reported as area % = `100·a`.

### 6.5 Challengers (options, default off; decided offline by `twin_eval.py`)
- **Bank = TOPK_VIEWS** (retrieval): select the `k = 3` keyframes with the highest `g_q · g_k` (ties → lowest index);
  `d(p)` = exact k-NN against all `B`-set patches of those keyframes (float32 rows from `kf_feats`, §8).
- **Rotation = CANONICAL:** crop with `cropResizeRotated` around the crop-square centre, same side, with
  `θ = principal angle` (§2.7) so the principal axis is horizontal; the component is scored at `θ` and `θ + π` and the
  lower `raw` wins. Teach keyframes are canonicalised the same way (at `θ` only).
- **Rotation = AUGMENT4:** keyframes are added to the bank at 0°, 90°, 180°, 270° (rotated crops) before the coreset.

## 7. Teach (build a Twin from a recording)

Input: the recorded frames in time order, each with timestamp `t` (ms), the analysis frame's sanity result for its
main object (§2.5–2.6), and for sane frames: the crop bytes, `cov`, geometry features, and the feature map.
(Golden tests feed per-frame `t, sane, sharpness, cov, geometry, features` directly.)

1. **Accept** frames whose sanity is OK (sane).
2. **Sharpness** of an accepted frame = §1.3 on the grey `N×N` crop, full rectangle.
3. **Keep** accepted frames with `sharpness ≥ P_40(sharpness of accepted frames)` (`keepPercentile = 40`). If fewer than
   `minKept = 8` frames are kept → fail `NOT_ENOUGH_FRAMES`.
4. **Keyframes** by greedy k-center on the kept frames' global descriptors, distance `δ(a, b) = 1 − g_a·g_b`:
   first = the kept frame with the highest sharpness (ties → earliest); then repeatedly add the kept frame with the
   largest `min_{s ∈ sel} δ(·, s)` (ties → earliest). Stop when `|sel| = Kmax = 32`, or when `|sel| ≥ Kmin = 16` and the
   next largest min-distance `< kcEps = 0.002`, or when all kept frames are selected. `sel` keeps selection order;
   keyframe index `k` = position in `sel`.
5. **Segments:** `t0, t1` = first and last **kept** frame times; keyframe `k` with time `t` gets
   `seg = min(B−1, floor(B · (t − t0) / (t1 − t0 + 1e-9)))`, `B = segments = 5`.
6. **Bank:** pool the `B`-set patches (§4) of every keyframe, keyframes in index order, patches in row-major order,
   remembering each row's keyframe. `N` = pooled rows. `k = min(maxRows, max(minK, floor(ratio · N)))` with
   `ratio = 0.10`, `minK = 256`, `maxRows = 2400`; `k = min(k, N)`. Greedy coreset (float64 squared Euclidean, start at
   pooled row 0, add the row with the largest min-distance, ties → lowest index) → bank rows (float32 copies) +
   `bankKf[]` (keyframe index per bank row) + `bankSeg[]`.
   **All stored feature rows are rounded through binary16 before any later step** (bank rows, keyframe globals, and
   the keyframe feature maps used by §6.5), so a Twin reloaded from disk reproduces every number below exactly.
   (Rounding order: select the coreset on the float32 pooled rows, then round the selected rows.)
7. **Leave-segment-out (LSO) scores:** for each keyframe `k` (segment `s`): distance map of its feature map (as
   binary16-rounded values) against bank rows with `bankSeg ≠ s` (the others get `KNN_MASK`); `lso_k` = §6.3 frame score
   `raw` with the keyframe's own `S`. If no row qualifies, `k` is skipped. **Fallback:** if the keyframes occupy fewer
   than 2 distinct segments, leave out the keyframe itself (`bankKf ≠ k`) instead. `τ_teach = tauFactor · max(lso)`,
   `tauFactor = 1.4`. `τ = τ_teach` until calibration (§9).
8. **Identity positives:** `pos_k = max_{j : seg_j ≠ seg_k} g_k·g_j` (fallback as in 7: `j ≠ k`).
   `τ_id` from §7.3 with the negatives library (if any).
9. **Geometry model** (§7.4) from the geometry features of **all kept frames**.
10. `coverageCut = 0.5`.

### 7.3 Identity threshold `τ_id`
`pos` = positive similarities, `neg` = for each negative object its `max_k g_neg · g_k` over the Twin keyframes.
- With `|neg| ≥ 1`: `lo = P_5(pos)`, `hi = P_99(neg)`, `τ_id = (lo + hi)/2`, `margin = lo − hi`, `overlap = margin ≤ 0`.
- Without negatives: `τ_id = P_5(pos) − max(0.02, 3 · 1.4826 · MAD(pos))`, `margin = null`, rule `"no-negatives"`.
Identity passes iff `sim ≥ τ_id`, `sim = max_k g_q · g_k`.

### 7.4 Geometry model and gate
For each feature `φ ∈ {fill, aspect, hu1, solidity}` over the sample set: `m_φ = mean`, `s_φ = max(std, floor_φ)`,
bounds `[m_φ − 3·s_φ, m_φ + 3·s_φ]` (`kGeo = 3.0`). Floors: `fill 0.04`, `aspect 0.04`, `solidity 0.03`,
`hu1 0.04 · m_hu1`. Area: `meanArea = mean(area)`; passes iff `meanArea/2.5 ≤ area ≤ 2.5·meanArea`.
The geometry gate passes iff all five checks pass; the report lists the failing features.

## 8. Twin storage (`filesDir/twins/<id>/`, written to `<id>.tmp/` then renamed)
- `twin.json` — schema below.
- `bank.f16` — `[M, D]`; `globals.f16` — `[K, D]`; `kf_feats.f16` — `[K, P, D]`; `kf_cov.f16` — `[K, P]`;
  `negatives.f16` — `[Nneg, D]` (the negative globals used for `τ_id`; may have 0 rows).
- `keyframes/kf_00.jpg …` — the `N×N` crop bytes of each keyframe, JPEG quality 95 (a Twin rebuilt from these is a
  re-teach from stored crops; its numbers may differ slightly from the original because of JPEG loss).
- **`.f16` format:** 8-byte ASCII magic `KZF16v1\n`, int32 LE `rows`, int32 LE `cols` (`cols` = product of the
  remaining dims, e.g. `P·D`), then `rows·cols` binary16 LE values.

`twin.json` (all keys required unless marked optional):
```json
{
  "schema": 1,
  "id": "20260927-031500-a1b2",          "name": "M8 nut",       "createdAtMs": 1790000000000,
  "pipeline": {
    "backboneId": "r18_320_f32", "backboneSha256": "<hex>", "inputSize": 320, "gh": 40, "gw": 40, "dim": 128,
    "precision": "fp32", "l2NormalizePatches": false, "preprocessVersion": 1,
    "mask": {"analysisFactor": 4, "kSigma": 4.0, "sigmaMin": 3.0, "minBlobPx": 30, "borderMargin": 2,
             "cropMargin": 0.10, "coreThreshold": 0.5},
    "knnGraphId": "knn_p1600_d128", "scoreRule": "SMOOTHED_MAX", "bankRule": "CORESET", "rotation": "NONE"
  },
  "fingerprint": "<sha256 hex of the canonical pipeline JSON>",
  "teach": {"framesSeen": 96, "framesAccepted": 80, "framesKept": 48, "keyframes": 24, "segmentsUsed": 5,
            "durationMs": 12000, "bankRows": 2400, "pooledRows": 21000},
  "keyframes": [{"index": 0, "tMs": 1234, "segment": 0, "sharpness": 812.5, "file": "keyframes/kf_00.jpg"}],
  "bankKf": [0, 0, 3],
  "lso": [0.91, 1.02],
  "positives": [0.97, 0.95],
  "negatives": {"count": 0, "similarities": []},
  "thresholds": {
    "tau": 1.43, "tauTeach": 1.43, "tauFactor": 1.4, "calibrated": false,
    "tauId": 0.91, "tauIdRule": "no-negatives", "identityMargin": null,
    "coverageCut": 0.5, "sensitivity": 1.0,
    "geometry": {"kGeo": 3.0, "meanArea": 1520.0,
                 "fill": [0.61, 0.05], "aspect": [0.93, 0.04], "hu1": [0.161, 0.0064], "solidity": [0.95, 0.03]}
  },
  "certificate": null
}
```
(`geometry.<φ>` = `[mean, sigma-after-floor]`.) The **canonical pipeline JSON** for the fingerprint is the
`pipeline` object serialised with keys sorted lexicographically at every level, no whitespace, numbers in their
shortest round-trip form (integers without `.0`, doubles like Kotlin/JSON `0.1`, `4.0` written as `4.0`).
A Twin whose fingerprint differs from the running pipeline's is refused ("Rebuild from stored crops").

## 9. Verdict for one presentation (order matters)
1. Sanity fails → **REFRAME** (reason from §2.6).
2. Compute `sim`, geometry gate, score. If `sim < τ_id` → **NOT_ENROLLED** (reason `identity`; also report shape
   failures). Else if geometry fails → **NOT_ENROLLED** (reason `shape`).
3. `s ≤ 1` → **PASS**.
4. `a > coverageCut` → **NOT_ENROLLED** (reason `coverage` — a wrong object lights up most of its patches).
5. Else **DEFECT** with `peak (row, col)` and `areaPct = 100·a`.

**Borderline voting** (live, when the governor allows it; excluded from latency claims): if the verdict is DEFECT with
`1 < s ≤ 1.15`, re-score up to 2 more buffered crops of the same track (next best quality), take the **median** of the
available `s` values (1 to 3) as the final `s` and re-apply steps 3–5 with that crop's `a`, `peak` (the crop whose `s`
is the median; for 2 values the higher one).

## 10. Calibration and the certificate

### 10.1 Calibrate
Each distinct good presentation (one per track/trigger; the taught part must not be reused) gives a sample:
`sanityOk, idOk, geoOk, raw, sim, geometry`. Valid = all three gates pass. `m = #valid`.
- `τ_cal = max(τ_teach, max raw over valid samples)` — **never lowered**. `calMax = max raw over valid samples`.
- `τ_id` re-derived (§7.3) with `pos = teach positives ∪ sims of sanity-ok samples`.
- Geometry model re-derived (§7.4) from the sanity-ok samples' geometry (floors apply).
- The certificate is valid while `τ · sensitivity ≥ calMax`; the UI marks it void below that.

### 10.2 Bounds
- Order statistic (score gate only): `α = 1 − (1 − conf)^(1/m)`, `conf = 0.95` (m=20 → 13.9 %, 29 → 9.8 %, 59 → 4.95 %).
- Clopper–Pearson one-sided upper bound on a rejection rate with `r` rejections out of `n`:
  `U(r, n) = BetaInv(conf; r+1, n−r)` for `r < n`, `1.0` for `r = n`. (For `r = 0`: `1 − (1−conf)^(1/n)`.)
- Two-sided `conf` interval for `k` successes out of `n` (G3 reporting):
  `lower = BetaInv((1−conf)/2; k, n−k+1)` (`0` if `k = 0`), `upper = BetaInv((1+conf)/2; k+1, n−k)` (`1` if `k = n`).
- `BetaInv(p; a, b)`: the `x ∈ [0,1]` with `I_x(a,b) = p` (regularised incomplete beta), by bisection to `|Δx| < 1e-12`
  (or any method with the same result to 1e-9). `I_x` via the Lentz continued fraction (Numerical Recipes `betacf`),
  using the symmetry `I_x(a,b) = 1 − I_{1−x}(b,a)` when `x > (a+1)/(a+b+2)`.
- **Within-part bound (before calibration):** `α_within = 1 − (1−conf)^(1/B_used)`, `B_used` = number of segments that
  contain keyframes — labelled *"within-part bound from B_used time blocks — not a false-alarm guarantee"*.

### 10.3 Certificate lines (the data; the UI renders them)
1. Score gate: `m` valid parts, `α` at 95 % → "false alarms ≤ α (score gate only)".
2. Gate pass counts over the `n` calibration presentations: sanity `k/n`, identity `k/n'`, geometry `k/n'`
   (`n'` = sanity-ok count) each with its CP upper bound on the rejection rate.
3. Identity margin (`P_5(pos) − P_99(neg)`) or "no negatives captured".
4. Decision latency p50/p95 (ms) and the accelerator that ran the backbone on this phone.

## 11. Live line: tracker and triggers (analysis resolution; timestamps from the frames, never the wall clock)

### 11.1 Detections
All components of the frame (§2.4, including border-touching ones) with their `area`, bbox, centroid, `touchesBorder`
and `sharpness` (§1.3 on the analysis grey image over the bbox expanded by 1 px).

### 11.2 Association (greedy, one frame at a time)
Tracks hold `id, pos (cx, cy), vel (vx, vy) in px/ms, area, bbox, hits, missed, lastT, confirmed, state`.
For frame time `t`: each live track predicts `p̂ = pos + vel·(t − lastT)`; gate
`G = gateBase + gateArea·sqrt(area) + gateSpeed·|vel|·(t − lastT)` (`12`, `0.75`, `0.5`).
Candidate pairs = (track, detection) with `|c − p̂| ≤ G`, sorted by distance ascending (ties: lower track id, then lower
detection index); assign greedily when both are free.
- Matched: if `hits == 1`: `vel = (c − pos)/(t − lastT)`; else `vel = α·(c − pos)/(t − lastT) + (1−α)·vel`
  (`α = 0.5`); then `pos = c`, `area`, bbox, `hits += 1`, `missed = 0`, `lastT = t`; `confirmed = hits ≥ minHits (2)`.
  (If `t == lastT`, velocity is left unchanged.)
- Unmatched tracks: `missed += 1`; removed when `missed > maxMissed (5)` → event `EXITED` (with `judged` flag).
- Unmatched detections: new track, `id = nextId++` (starting at 1), `hits = 1`, `vel = 0`, `lastT = t`.

### 11.3 Trigger A — virtual photo-eye
A line `axis ∈ {X, Y}` at position `L` (default `X`, `L = 0.5·W`), direction `ANY | POSITIVE | NEGATIVE`.
Side of a track = `sign(coord − L)` (`coord` = `cx` or `cy`; exactly on the line counts as the negative side).
Upstream side: `NEGATIVE` for direction POSITIVE, `POSITIVE` for NEGATIVE, and for ANY the side where the track was
when it became confirmed. A confirmed, not-yet-judged track that has been seen on its upstream side while confirmed
fires `LINE` when it has been on the downstream side for `crossFrames = 2` consecutive matched frames. A track first
confirmed on the downstream side of a POSITIVE/NEGATIVE line never fires (no double count after an ID switch).

### 11.4 Trigger B — steady hold
A confirmed, not-yet-judged track that does not touch the border, with `|vel| < vStill = 0.02 px/ms` and sharpness
`≥ 0.7 ×` its own maximum sharpness so far, continuously (every matched frame) for `≥ holdMs = 500 ms` fires `STEADY`.
The hold clock restarts whenever a matched frame breaks any condition. (`stillSince` = time of the first frame of the
current run; fire when `t − stillSince ≥ holdMs`.)

Each track fires **at most once** (`judged = true` afterwards). If both triggers are enabled, the first one wins.

### 11.5 Best crop
For every matched frame of a confirmed track that does not touch the border: quality
`q = sharpness · centrality`, `centrality = 1 − min(1, |c − frameCentre| / (0.5·hypot(W, H)))`. The track keeps its
best `3` frames by `q` (ties → earlier). On `LINE`, the judge uses the best buffered frame (the others are for voting);
on `STEADY`, the firing frame. A track that fires with no buffered frame is judged REFRAME (`TOUCHES_BORDER`).

### 11.6 Overlay and counters
Displayed box centre = `pos + vel · latencyMs` (`latencyMs` = measured camera-to-screen delay). "Line too fast" when
the judge queue holds more than `3` jobs. Counters: judged, pass, defect, not-enrolled, reframe, exited-unjudged.

## 12. Thermal governor
Inputs every sample: `status` (Android `PowerManager` thermal status: NONE 0, LIGHT 1, MODERATE 2, SEVERE 3, CRITICAL 4,
EMERGENCY 5, SHUTDOWN 6), `headroom` (float, may be NaN = unknown; polled ≤ 1 per 10 s), `idle` (no component for
`idleMs = 3000`). Target level: `L2` if `status ≥ 3` or `headroom > 0.85`; else `L1` if `status == 2` or
`headroom ≥ 0.70`; else `L0` (NaN headroom = ignored). The level rises immediately; it falls only after the lower
target has held continuously for `cooldownMs = 30000` (then it drops straight to that target).
Outputs: `L0` → fps 30, voting on, VLM `AUTO`; `L1` → fps 24, voting off, VLM `ON_TAP`; `L2` → fps 15, voting off,
VLM `PAUSED`, banner. `idle` overrides fps only → 2. Every level change is logged with its inputs.
The governor never changes the backbone or the HTP performance mode.

## 13. Accelerator honesty (runtime rules; pure logic lives in `:core` or `:runtime`)
- Every candidate (NPU, GPU, CPU) runs the same model and input; float candidates pass iff all outputs finite,
  cosine ≥ 0.999 and relative L2 error ≤ 1 % vs the CPU fp32 output; int8 candidates pass iff mean per-patch
  cosine ≥ 0.99.
- A non-CPU candidate is labelled with its accelerator **only** if it also needs ≤ 0.9× the CPU time for the same model
  (best of the timed runs); otherwise it is reported as "<X> requested — no speed-up (ran on CPU?)" and not used.
- The badge text is the chosen accelerator plus the measured speed-up, e.g. `NPU 4.1 ms (7.9× CPU)`.

## 14. Evaluation helpers
- **AUROC** of scores `pos` (should be high) vs `neg`: Mann–Whitney `U / (|pos|·|neg|)` with ties counted `0.5`.
- **Latency stats:** p50/p95 by §0 percentiles over a ring buffer of the last `N = 512` samples.

## 15. Golden file `testdata/golden_twin.json` (written by `twin_ref.py`, read by the Kotlin tests)
UTF-8, no BOM. Every section has `"inputs"` and `"expected"`; images are `{"w":…, "h":…, "c":…, "data":[…]}` flat
row-major integer lists; masks are flat 0/1 lists; feature maps flat float lists with `gh, gw, dim`.
Required sections (each exercising the matching spec section, including tie and edge cases):
`stats` (§0 percentile/median/MAD incl. n = 1, even n, q = 0/100), `half` (§0 f16 bits incl. RNE ties, subnormal,
overflow), `binomial` (§10.2: α for m ∈ {1,20,29,59,299}; CP upper for several (r, n); two-sided intervals incl. k = 0
and k = n), `auroc` (§14 incl. ties), `image` (§1.1 downscale of a random RGBA 16×12 with stride padding, §1.2–1.3
sharpness on a rect, §1.4 crop-resize and rotated crop of a small RGB image — float outputs and rounded bytes),
`mask` (§2.1–2.7: sheet fit from 2 frames, a test frame with ≥ 4 shapes incl. a border-touching one, a small speck and
a ring; foreground, after-open, after-close, labels, components, geometry features, main object, sanity reason),
`crop_patch` (§3–4: crop square for a component, `cov`, core, S, B on a 6×6 grid), `knn` (§5 with a mask vector),
`scoring` (§6.2–6.4: several dmaps/masks, both rules, sensitivity ≠ 1, ties), `teach` (§7 end-to-end on synthetic
per-frame inputs: ≥ 30 frames of 6×6×8 features spanning 12 s, some insane; expected kept, keyframes, segments,
pooled row count, coreset indices, bank values after f16 rounding (first rows), LSO, τ, positives, τ_id with and
without negatives, geometry model), `verdicts` (§9: queries against that Twin covering every verdict and reason,
plus a borderline-vote case), `calibration` (§10: samples → τ_cal, τ_id, geometry, certificate numbers, within-part
bound), `tracker` (§11: ≥ 3 synthetic sequences — two parts crossing at speed with occlusion gaps, a part that
reverses and re-crosses, a steady hold with a blur dip, an ID switch after the line — expected events per frame:
fired track id + trigger + best-crop frame, exits), `governor` (§12 step sequence with hysteresis), `twin_json`
(§8: a pipeline object and its canonical string + SHA-256 fingerprint).
