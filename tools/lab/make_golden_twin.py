#!/usr/bin/env python3
"""Golden vectors for the Kaizen Eye 2 Visual Twin maths (docs/verification/twin-spec.md v1, section 15).

    .venv\\Scripts\\python.exe tools/lab/make_golden_twin.py --out testdata/golden_twin.json

Always use --out (UTF-8, no BOM, json.dump); never shell redirection (PowerShell 5.1 writes UTF-16 + BOM).
The numbers come from tools/lab/twin_ref.py, a numpy reference written independently of the Kotlin port.
Fixed seeds: re-running produces a byte-identical file (checked by tools/lab/test_twin_ref.py).

=====================================================================================================================
JSON LAYOUT (the contract the Kotlin tests are written against)
=====================================================================================================================

Top level
---------
  {"meta": {...}, "stats": SEC, "half": SEC, "binomial": SEC, "auroc": SEC, "image": SEC, "mask": SEC,
   "crop_patch": SEC, "knn": SEC, "scoring": SEC, "teach": SEC, "verdicts": SEC, "calibration": SEC,
   "tracker": SEC, "governor": SEC, "twin_json": SEC}

  meta = {"spec": "docs/verification/twin-spec.md v1", "generator": "tools/lab/make_golden_twin.py", "seed": int,
          "tolerances": {"f64Rel": 1e-6, "f64Abs": 1e-9, "f32Rel": 1e-4, "f32Abs": 1e-6, "betaAbs": 1e-9}}

  Every section SEC = {"cases": [CASE, ...]}  and every  CASE = {"name": str, "inputs": {...}, "expected": {...}}.
  Sections whose cases have different input shapes also put "kind": str in each case (listed below).

Common value encodings
----------------------
  * Floats are plain JSON numbers written with Python repr (shortest round-trip). Float32 inputs (features, bank
    rows, ...) are written as the exact float32 value, so Double.parse(...).toFloat() recovers the bits exactly.
    The file never contains NaN/Infinity; a JSON null stands for "not applicable" (or NaN where stated).
  * Integers are JSON integers; booleans JSON true/false; enums SCREAMING_CASE strings:
      verdict    PASS | DEFECT | NOT_ENROLLED | REFRAME
      sanity     OK | NO_OBJECT | TOUCHES_BORDER | TOO_SMALL | TOO_LARGE | MULTIPLE | NO_CORE
      notEnrolled reason  IDENTITY | SHAPE | COVERAGE           (spec 9 writes them lowercase; golden uses caps)
      scoreRule  SMOOTHED_MAX | TOP1_MEAN        bankRule  CORESET | TOPK_VIEWS
      trigger    LINE | STEADY     lineAxis  X | Y     lineDirection  ANY | POSITIVE | NEGATIVE
      governor level  L0 | L1 | L2               vlm  AUTO | ON_TAP | PAUSED
  * IMAGE  = {"w": int, "h": int, "c": int, "data": [int 0..255]}   flat row-major, index (y*w + x)*c + ch.
    RGBA input images add "rowStride" (bytes per row, >= 4*w); then "data" has h*rowStride bytes INCLUDING the
    padding bytes (random junk, must be ignored), index y*rowStride + x*4 + ch.
  * LABELS = {"w": int, "h": int, "data": [int]}  flat row-major label map (0 = background).
  * MASK   = flat list of 0/1 ints (length w*h of its image, or gh*gw for patch-grid masks, index p = r*gw + c).
  * FMAP   = {"gh": int, "gw": int, "dim": int, "data": [float]}   feature map, f[(r*gw + c)*dim + t].
  * MAT    = {"rows": int, "cols": int, "data": [float]}          row-major matrix, m[i*cols + j].
  * COMPONENT = {"label","area","minX","minY","maxX","maxY": int, "cx","cy": float, "touchesBorder": bool}
  * GEOMETRY  = {"area": int|float, "fill","aspect","hu1","solidity": float}   (outputs also carry informational
                "label", "theta" (principal angle, rad) and "hullArea"; inputs carry only the five features)
  * CROP      = {"x0","y0","side": float}   (full-resolution continuous coordinates)
  * GEOMODEL  = {"kGeo": float, "meanArea": float, "areaFactor": float,
                 "fill": [mean, sigmaAfterFloor], "aspect": [..], "hu1": [..], "solidity": [..]}
  * Every tunable parameter a case uses is written into its inputs (mostly "params"); nothing relies on defaults.

Tolerances (spec 0)
-------------------
  exact : all integers, indices, masks, labels, bytes, f16 bits, booleans, enum strings, list lengths.
  f64   : |got - exp| <= 1e-6*|exp| + 1e-9   for floats derived from floats/uint8 without feature maps.
  f32   : |got - exp| <= 1e-4*|exp| + 1e-6   for floats that pass through float32 feature maps (k-NN distances,
          dmap/sm/raw/s, lso, tau, sims, positives, tauId from teach, globals, bank values).
  beta  : |got - exp| <= 1e-9                for betaInv / Clopper-Pearson / two-sided bounds.
  Inputs are designed so no exact field depends on a floating-point near-tie (>= 1e-6 relative separation between
  competing choices); explicit tie cases use exactly representable values so ties are exact in f32 and f64.

---------------------------------------------------------------------------------------------------------------------
stats   (spec 0: percentile / median / MAD / mean / std)                                               class f64
---------------------------------------------------------------------------------------------------------------------
  inputs   {"x": [float] (unsorted, n >= 1), "qs": [float in 0..100]}
  expected {"percentiles": [float, one per q], "median": float, "mad": float, "robustSigma": float (1.4826*mad),
            "mean": float, "std": float (population, ddof 0)}
  Cases include n = 1, n = 2, even n with duplicates, q = 0 and q = 100.

---------------------------------------------------------------------------------------------------------------------
half    (spec 0: IEEE binary16, round-to-nearest-even)                                                   exact
---------------------------------------------------------------------------------------------------------------------
  kind "encode":  inputs {"f32Bits": [uint32 int], "labels": [str]}   (the float32 values to convert, as raw bits)
                  expected {"f16Bits": [int 0..65535], "decodedF32Bits": [uint32 int]}   (f16 widened back to f32)
  kind "decode":  inputs {"f16Bits": [int]}   expected {"f32Bits": [uint32 int]}
  Covers exact values, RNE ties (to even up and down), just-above/below ties, subnormals (min, max, tie to 0, tie to
  2^-23), float32 values that underflow to +-0, 65504, 65519.99.. -> 65504, 65520 -> +inf (tie rounds to even =
  overflow), large -> +-inf, +-inf, and the canonical quiet NaN 0x7FC00000 <-> 0x7E00 (a test may instead just
  assert "is NaN" for NaN entries; labels start with "nan").

---------------------------------------------------------------------------------------------------------------------
binomial (spec 10.2)                                                         class f64 (alpha) / beta (the rest)
---------------------------------------------------------------------------------------------------------------------
  one case "bounds":
  inputs   {"conf": 0.95, "alphaM": [int], "cpUpper": [{"r": int, "n": int}], "twoSided": [{"k": int, "n": int}],
            "betaInc": [{"x": float, "a": float, "b": float}], "betaInv": [{"p": float, "a": float, "b": float}],
            "withinPartBUsed": [int]}
  expected {"alpha": [float]            alpha = 1 - (1-conf)^(1/m) per alphaM entry (m in 1,20,29,59,299)
            "cpUpper": [float]          U(r, n) per cpUpper entry (r = 0 closed form, r = n -> 1.0)
            "twoSided": [{"lower": float, "upper": float}]   (k = 0 -> lower 0.0, k = n -> upper 1.0)
            "betaInc": [float]          regularised incomplete beta I_x(a, b)
            "betaInv": [float]          x with I_x(a, b) = p
            "withinPart": [float]}      alpha_within = 1 - (1-conf)^(1/B_used) per withinPartBUsed entry

---------------------------------------------------------------------------------------------------------------------
auroc   (spec 14)                                                                                        class f64
---------------------------------------------------------------------------------------------------------------------
  inputs {"pos": [float], "neg": [float]}   expected {"auroc": float}      (ties count 0.5; tie cases included)

---------------------------------------------------------------------------------------------------------------------
image   (spec 1)
---------------------------------------------------------------------------------------------------------------------
  kind "downscale":  inputs {"rgba": IMAGE (c 4, with rowStride), "factor": int}
                     expected {"rgb": IMAGE (c 3)}                                                         exact
  kind "sharpness":  inputs {"rgb": IMAGE (c 3), "rects": [{"x0","y0","x1","y1": int}]}   ([x0,x1) x [y0,y1),
                     may extend outside the image)
                     expected {"grey": [float, w*h], "count": [int per rect: qualifying pixels],
                               "sharpness": [float per rect]  (0.0 when count < 9)}                     class f64
  kind "crop":       inputs {"rgb": IMAGE (c 3), "samples": [{"u","v": float}],
                             "crops": [{"x0","y0","side": float, "n": int}],
                             "rotated": [{"cx","cy","side","theta": float, "n": int}]}
                     expected {"samples": [[r, g, b] floats],
                               "crops":   [{"values": [float, n*n*3], "bytes": [int, n*n*3]}],
                               "rotated": [{"values": [float, n*n*3], "bytes": [int, n*n*3]}]}
                     values class f64; bytes exact (= clamp(floor(v + 0.5), 0, 255)). Output layout (i*n + j)*3 + ch,
                     i = output row, j = output column. Some crops use dyadic geometry so exact .5 ties occur.

---------------------------------------------------------------------------------------------------------------------
mask    (spec 2.1 - 2.7, analysis resolution)                        every case kind "scene"
---------------------------------------------------------------------------------------------------------------------
  inputs {"frame": IMAGE (c 3),
          "sheetFrames": [IMAGE, ...]   (case "sheet_fit_scene": fit the sheet model from these, ROI = whole frame)
          | "sheet": {"mu": [3 floats], "sigma": [3 floats]}   (other cases: sheet model given directly),
          "params": {"sigmaMin", "kSigma": float, "minBlobPx", "borderMargin": int, "minAreaFrac", "maxAreaFrac",
                     "multipleRatio", "cropMargin", "coreThreshold": float, "analysisFactor", "fullW", "fullH",
                     "gh", "gw": int}}          (fullW = analysisFactor*w, fullH = analysisFactor*h; gh x gw is the
                                                 patch grid used for the NO_CORE check)
  expected {"sheet": {"mu": [3], "sigma": [3], "mad": [3] | null},     (null mad when the sheet was given)
            "d2": [float, w*h],                                  class f64
            "foreground": MASK, "afterOpen": MASK, "afterClose": MASK,   (afterClose = final mask)
            "rawAreas": [int]            areas of the 8-connected components of afterClose in first-pixel order,
                                         BEFORE removing those with area < minBlobPx
            "labels": [int, w*h]         final label map after speck removal and renumbering
            "components": [COMPONENT]    in label order (cx, cy class f64)
            "geometry": [GEOMETRY]       one per component, same order (class f64; "area" is the int area)
            "mainObject": int            label of the main object (spec 2.5), 0 = none
            "perComponent": [{"label": int, "crop": CROP, "coreCount": int, "sanity": str}]
                                         sanity (spec 2.6 steps 2-6) if THIS component were the chosen one
            "sanity": str}               sanity of the main object ("NO_OBJECT" when mainObject == 0)
  Cases: the sheet-fit scene (>= 4 shapes: main object, a ring with a hole, a border-touching blob, specks removed
  by the opening and by minBlobPx, a thin line), plus scenes for NO_OBJECT / TOO_LARGE / MULTIPLE / NO_CORE.

---------------------------------------------------------------------------------------------------------------------
crop_patch (spec 3 - 4)
---------------------------------------------------------------------------------------------------------------------
  inputs {"labels": LABELS (analysis resolution), "label": int,
          "params": {"analysisFactor": int, "fullW": int, "fullH": int, "cropMargin": float,
                     "coreThreshold": float, "gh": int, "gw": int},
          "component": {"minX","minY","maxX","maxY": int}    -> crop square derived per spec 3
          | "crop": CROP}                                     -> crop square given (may extend outside the image)
  expected {"crop": CROP (derived or echoed, class f64), "cov": [float, gh*gw] (multiples of 1/16, exact),
            "core": MASK, "S": MASK, "B": MASK}   (grid masks, index r*gw + c)

---------------------------------------------------------------------------------------------------------------------
knn     (spec 5)                                                                                         class f32
---------------------------------------------------------------------------------------------------------------------
  inputs {"feats": FMAP (P = gh*gw query rows), "bank": MAT (M x dim), "excluded": [0/1 per bank row],
          "knnMask": 1e30}
  expected {"minSq": [float, P], "d": [float, P], "allExcluded": bool}
  When allExcluded is true every minSq is >= 1e30: assert minSq >= 1e30 instead of comparing values.

---------------------------------------------------------------------------------------------------------------------
scoring (spec 6.2 - 6.4)                                                                  class f64 (inputs dyadic)
---------------------------------------------------------------------------------------------------------------------
  inputs {"gh","gw": int, "dmap": [float, P], "core": MASK, "S": MASK (= 3x3 dilation of core), "tau": float,
          "sensitivity": float, "scoreRule": str, "params": {"topFraction": 0.01}}
  expected {"sm": [float, P]  (masked 3x3 mean for p in S; 0.0 for p not in S - do not compare those),
            "raw": float, "s": float, "topN": int (n of TOP1_MEAN; 0 for SMOOTHED_MAX),
            "peak": {"row": int, "col": int, "index": int}, "coreCount": int, "anomalousCount": int,
            "anomalousFraction": float, "areaPct": float}
  Includes exact ties (peak -> lowest index; sm == tau*sensitivity is NOT anomalous; s == 1.0 exactly).

---------------------------------------------------------------------------------------------------------------------
teach   (spec 7)                                              kinds "ok" | "fail"; float outputs class f32
---------------------------------------------------------------------------------------------------------------------
  inputs {"params": TEACHPARAMS, "negatives": MAT (Nneg x dim raw float32 negative globals; may have 0 rows),
          "frames": [{"tMs": int, "sane": bool, "sharpness": float | null, "cov": [float, P] | null,
                      "geometry": GEOMETRY (5 features) | null, "features": FMAP | null}]}
          (insane frames carry null for sharpness/cov/geometry/features)
  TEACHPARAMS = {"gh","gw","dim": int, "keepPercentile": float, "minKept": int, "kMin": int, "kMax": int,
                 "kcEps": float, "segments": int, "ratio": float, "minK": int, "maxRows": int, "tauFactor": float,
                 "coreThreshold": float, "scoreRule": str, "topFraction": float, "knnMask": 1e30,
                 "kGeo": float, "fillFloor": float, "aspectFloor": float, "solidityFloor": float,
                 "hu1FloorFrac": float, "areaFactor": float, "coverageCut": float,
                 "posPercentile": 5.0, "negPercentile": 99.0, "idMinGap": 0.02, "idMadK": 3.0}
  expected (kind "ok"):
    {"accepted": [frame idx], "sharpnessCut": float (P_keepPercentile of accepted sharpness, class f64),
     "kept": [frame idx],
     "keyframes": [{"index": int, "frame": int (frame idx), "tMs": int, "segment": int, "sharpness": float}]
                  in selection order (index = position),
     "kcMinDist": [float, K-1]   min-distance (1 - g.g) at which keyframes 1..K-1 were selected,
     "kcStop": "KMAX" | "EPS" | "ALL",  "kcNext": float | null  (next largest min-distance when stopping; null for ALL)
     "t0": int, "t1": int, "segmentsUsed": int,
     "pooledRows": int, "bankRows": int (k),
     "coresetIndices": [int, k]  pooled-row indices in selection order (pooled rows = B-set patches of keyframes
                                 in index order, patches row-major),
     "bank": MAT (k x dim, binary16-rounded values),  "bankKf": [int, k],  "bankSeg": [int, k],
     "globals": MAT (K x dim, keyframe globals after binary16 rounding),
     "lsoMode": "SEGMENT" | "KEYFRAME" (KEYFRAME = fallback, fewer than 2 segments used),
     "lso": [float] (keyframes that were not skipped, index order), "lsoKeyframes": [int] (their indices),
     "tauTeach": float, "tau": float,
     "positives": [float, K],
     "identity": {"noNegatives":   {"tauId": float, "rule": "no-negatives", "margin": null, "overlap": null},
                  "withNegatives": {"tauId": float, "rule": "midpoint", "margin": float, "overlap": bool,
                                    "negSims": [float, Nneg]}  (null when Nneg == 0)},
     "geometry": GEOMODEL (from all kept frames, class f64), "coverageCut": float}
  expected (kind "fail"): {"error": "NOT_ENOUGH_FRAMES", "accepted": [int], "kept": [int]}
  Cases: "main" (>= 30 frames, 12 s, some insane, two segments+), "lso_fallback" (keyframes in one segment ->
  leave-one-keyframe-out), "lso_skip" (1-row bank: keyframes of that row's segment are skipped),
  "not_enough_frames".
  The "rule" string of the with-negatives tau_id is not fixed by the spec (SPEC-QUESTION 4) - do not compare it.
  Readings the expected values depend on (see the SPEC-QUESTION list in twin_ref.py):
    * negSims use binary16-rounded negatives AND binary16-rounded keyframe globals (SPEC-QUESTION 2; not rounding
      the negatives moves negSims[0] of "main" by 1.2e-4 relative);
    * keyframe globals are computed in float64 and rounded float64 -> binary16 once (SPEC-QUESTION 3; the data is
      checked to give identical bits through float32);
    * positives / LSO / negSims use the rounded values; keyframe selection (k-centre) uses the unrounded float64
      globals; the coreset runs on the float32 pooled rows and only the selected rows are rounded.

---------------------------------------------------------------------------------------------------------------------
verdicts (spec 9, 6.5 TOPK_VIEWS)                                     one case "main"; floats class f32
---------------------------------------------------------------------------------------------------------------------
  inputs {"twin": TWIN, "params": {"coreThreshold", "topFraction": float, "topKViews": int, "knnMask": 1e30,
                                    "voteLo": 1.0, "voteHi": 1.15, "voteMaxExtra": 2},
          "queries": [QUERY]}
  TWIN = the teach case "main" result: {"gh","gw","dim": int, "bank": MAT, "bankKf": [int], "bankSeg": [int],
          "globals": MAT (K x dim), "kfFeats": MAT (K x P*dim), "kfCov": MAT (K x P), "tau": float,
          "tauId": float, "coverageCut": float, "geometry": GEOMODEL}   (all feature values already binary16-rounded)
  QUERY = {"name": str, "sanity": str (spec 2.6 steps 1-5 result), "features": FMAP | null, "cov": [float, P] | null,
           "geometry": GEOMETRY | null, "sensitivity": float, "scoreRule": str, "bankRule": str,
           "voting": bool, "extraCrops": [{"features": FMAP, "cov": [float, P]}]  (0..2, next-best quality first)}
  expected {"results": [RESULT, one per query]}
  RESULT = {"verdict": str        verdict of steps 1-5 on the primary crop,
            "reason": str | null  sanity reason for REFRAME (incl. NO_CORE when the core from "cov" is empty),
                                  IDENTITY | SHAPE | COVERAGE for NOT_ENROLLED, null for PASS / DEFECT,
            "sim", "raw", "s": float, "idOk", "geoOk": bool, "geoFailures": [str]  (subset of
            "fill","aspect","hu1","solidity","area" in that order), "peak": {"row","col","index": int},
            "coreCount": int, "anomalousFraction": float, "areaPct": float, "dmap": [float, P],
            "views": [int] | null  (TOPK_VIEWS: chosen keyframes, best first),
            "vote": null | {"s": [float, primary first then extra crops], "chosen": int (crop index),
                            "finalS": float, "verdict": str, "reason": str | null,
                            "peak": {...}, "anomalousFraction": float, "areaPct": float},
            "finalVerdict": str   (vote.verdict when a vote ran, else verdict)}
  REFRAME results carry null in every field except verdict / reason / finalVerdict. Voting runs only when "voting"
  is true and the primary verdict is DEFECT with voteLo < s <= voteHi; the median of 2 values is the higher one;
  the chosen crop is the first crop (primary first) whose s equals the median value.
  The TWIN equals the teach case "main" twin (same bank / globals / thresholds), so verdict tests can either build
  it from this JSON or re-teach it from the teach section.

---------------------------------------------------------------------------------------------------------------------
calibration (spec 10)                                                                                  class f64
---------------------------------------------------------------------------------------------------------------------
  inputs {"params": {"conf", "kGeo", "fillFloor", "aspectFloor", "solidityFloor", "hu1FloorFrac", "areaFactor",
                     "posPercentile", "negPercentile", "idMinGap", "idMadK": float, "latencyWindow": int},
          "tauTeach": float, "positives": [float], "negSims": [float] (may be empty), "segmentsUsed": int,
          "samples": [{"sanityOk", "idOk", "geoOk": bool, "raw": float | null, "sim": float | null,
                       "geometry": GEOMETRY | null}]   (sanity-failed samples: nulls and idOk = geoOk = false),
          "sensitivities": [float], "latencyMs": [float]}
  expected {"n": int, "nSanityOk": int, "m": int (valid), "tauCal": float, "calMax": float, "alpha": float,
            "identity": {"tauId", "rule", "margin", "overlap"} (as in teach),
            "geometry": GEOMODEL (from the sanity-ok samples),
            "certificate": {"sanity":   {"k": int, "n": int, "rejections": int, "upper": float},
                            "identity": {...same, n = nSanityOk...}, "geometry": {...same...},
                            "identityMargin": float | null, "withinPart": float,
                            "latencyP50": float, "latencyP95": float (last latencyWindow samples)},
            "valid": [bool per sensitivity]  (tauCal * sensitivity >= calMax)}
  "upper" values use tolerance beta. tau_id is re-derived with pos = teach positives + the sims of ALL sanity-ok
  samples, including those with idOk = false (SPEC-QUESTION 15; "with_negatives" ends with overlap = true).

---------------------------------------------------------------------------------------------------------------------
tracker (spec 11)                                                                   states class f64, events exact
---------------------------------------------------------------------------------------------------------------------
  inputs {"params": {"w", "h": int (analysis frame), "gateBase", "gateArea", "gateSpeed", "velAlpha": float,
                     "minHits", "maxMissed": int, "lineEnabled": bool, "lineAxis": str, "linePos": float,
                     "lineDirection": str, "crossFrames": int, "steadyEnabled": bool, "vStill": float,
                     "holdMs": float, "holdSharpRatio": float, "bestK": int},
          "frames": [{"tMs": int, "detections": [{"area","minX","minY","maxX","maxY": int, "cx","cy": float,
                                                   "touchesBorder": bool, "sharpness": float}]}]}
  expected {"frames": [{"assign": [int trackId per detection, in detection order],
                        "events": [EVENT]  (FIRED events by ascending trackId, then EXITED by ascending trackId),
                        "tracks": [{"id": int, "cx","cy","vx","vy": float, "hits","missed": int,
                                    "confirmed","judged": bool}]  (live tracks after the frame, ascending id)}],
            "fired": [{"frame": int, "trackId": int, "trigger": str, "judgeFrame": int | null}],   (summary)
            "exited": [{"frame": int, "trackId": int, "judged": bool}]}                          (summary)
  The current frame is added to the best-crop buffer BEFORE the triggers are checked, so a LINE fire can be judged
  on the firing frame itself (SPEC-QUESTION 10; two_parts_crossing track 2 and negative_and_border track 1 do).
  EVENT = {"type": "FIRED", "trackId": int, "trigger": "LINE" | "STEADY",
           "judgeFrame": int | null   (frame index: LINE -> best buffered frame, STEADY -> the firing frame;
                                        null = no buffered frame -> judged REFRAME TOUCHES_BORDER),
           "buffer": [int]            (buffered frame indices, best quality first, ties earlier first),
           "reframe": null | "TOUCHES_BORDER"}
        | {"type": "EXITED", "trackId": int, "judged": bool, "confirmed": bool}
  Frame indices are positions in inputs.frames. Cases: two_parts_crossing (opposite directions, ANY line, occlusion
  gaps, a one-frame spurious blob that exits unconfirmed), reverse_recross (one-frame crossing, a frame exactly on
  the line = negative side, fires once), steady_blur_dip (STEADY; the blur dip restarts the hold clock, fires at
  exactly holdMs; the later line crossing is ignored), id_switch_after_line (POSITIVE; the new track is first
  confirmed downstream and never fires), negative_and_border (NEGATIVE; a wrong-way part never fires; a part that
  always touches the border fires with no buffered crop), axis_y_positive (lineAxis Y).

---------------------------------------------------------------------------------------------------------------------
governor (spec 12)                                                                                         exact
---------------------------------------------------------------------------------------------------------------------
  inputs {"params": {"cooldownMs": int, "statusL2": int, "statusL1": int, "headroomL2": float,
                     "headroomL1": float, "fpsL0", "fpsL1", "fpsL2", "fpsIdle": int},
          "samples": [{"tMs": int, "status": int, "headroom": float | null (null = NaN/unknown), "idle": bool}]}
  expected {"steps": [{"target": str, "level": str, "changed": bool, "fps": int, "voting": bool, "vlm": str,
                       "banner": bool}]}      (initial level before the first sample is L0)

---------------------------------------------------------------------------------------------------------------------
twin_json (spec 8)                                                                                         exact
---------------------------------------------------------------------------------------------------------------------
  kind "pipeline": inputs {"pipeline": {...the twin.json "pipeline" object...}}
                   expected {"canonical": str (sorted keys at every level, no whitespace),
                             "fingerprint": str (lowercase hex SHA-256 of the canonical string's UTF-8 bytes)}
  kind "f16File":  inputs {"rows": int, "cols": int, "values": [float32 values, rows*cols]}
                   expected {"f16Bits": [int], "bytesHex": str (lowercase hex of the whole .f16 file:
                             "KZF16v1\\n", int32 LE rows, int32 LE cols, binary16 LE values)}
"""
import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_ref as tr  # noqa: E402

SEED = 20260927
GAP = 1e-6          # minimum relative separation of every discrete decision in the golden data
TOL = {"f64Rel": 1e-6, "f64Abs": 1e-9, "f32Rel": 1e-4, "f32Abs": 1e-6, "betaAbs": 1e-9}
DIAG = []           # notes printed with --diag


# ================================================================================================= JSON helpers
def floats(a):
    return [float(v) for v in np.asarray(a, np.float64).ravel()]


def f32s(a):
    return [float(v) for v in np.asarray(a, np.float32).ravel()]


def ints(a):
    return [int(v) for v in np.asarray(a).ravel()]


def img(a):
    a = np.asarray(a, np.uint8)
    return dict(w=int(a.shape[1]), h=int(a.shape[0]), c=1 if a.ndim == 2 else int(a.shape[2]), data=ints(a))


def fmap(a):
    a = np.asarray(a, np.float32)
    return dict(gh=int(a.shape[0]), gw=int(a.shape[1]), dim=int(a.shape[2]), data=f32s(a))


def mat(a, cols=None):
    a = np.asarray(a, np.float32)
    a = a.reshape(a.shape[0], -1) if (a.size or cols is None) else a.reshape(0, cols)
    return dict(rows=int(a.shape[0]), cols=int(a.shape[1]), data=f32s(a))


def geo5(g):
    g = tr._geo_dict(g)
    area = g["area"]
    area = int(area) if float(area).is_integer() else float(area)
    return dict(area=area, fill=float(g["fill"]), aspect=float(g["aspect"]), hu1=float(g["hu1"]),
                solidity=float(g["solidity"]))


def geo_out(g, label):
    return dict(label=label, area=int(g.area), fill=g.fill, aspect=g.aspect, hu1=g.hu1, solidity=g.solidity,
                theta=g.theta, hullArea=g.hullArea)


def peak(pk):
    return None if pk is None else dict(row=int(pk[0]), col=int(pk[1]), index=int(pk[2]))


def case(name, inputs, expected, kind=None):
    c = dict(name=name)
    if kind is not None:
        c["kind"] = kind
    c["inputs"] = inputs
    c["expected"] = expected
    return c


@contextmanager
def guard(label, kinds, allow_exact=()):
    """Run reference code under decision-margin recording; fail on near-ties, hazards or unexpected exact ties."""
    with tr.recording() as m:
        yield m
    bad = {k: v for k, v in m.min_gap.items() if k in kinds and v[0] < GAP}
    assert not bad, "%s: near-tie decisions %r" % (label, bad)
    assert not m.flags, "%s: hazards %r" % (label, m.flags)
    ex = {k: n for k, n in m.exact.items() if k in kinds}
    unexpected = {k: n for k, n in ex.items() if k not in allow_exact}
    assert not unexpected, "%s: unexpected exact ties %r" % (label, unexpected)
    gaps = {k: "%.1e" % v[0] for k, v in sorted(m.min_gap.items()) if k in kinds}
    DIAG.append("%-30s min gaps %s  exact ties %s" % (label, gaps, ex))


# ======================================================================================================== stats
def sec_stats():
    rng = np.random.default_rng(SEED + 1)
    cases = []

    def add(name, x, qs):
        x = [float(v) for v in x]
        qs = [float(q) for q in qs]
        cases.append(case(name, dict(x=x, qs=qs), dict(
            percentiles=[tr.percentile(x, q) for q in qs], median=tr.median(x), mad=tr.mad(x),
            robustSigma=tr.robust_sigma(x), mean=tr.mean(x), std=tr.std(x))))

    add("n1", [3.25], [0, 5, 50, 99, 100])
    add("n2_even", [4.0, 1.0], [0, 25, 50, 75, 100])
    add("even_duplicates", [7, 2, 2, 1, 9, 2, 5, 7], [0, 10, 40, 50, 90, 100])
    add("odd_negative", [-3.5, 0.25, 10, -1, 4.75, 0.25, 2], [0, 5, 50, 95, 100])
    add("random37", rng.normal(0, 1, 37), [0, 5, 40, 50, 95, 99, 100])
    add("random40", rng.uniform(-10, 10, 40), [0, 5, 33.3, 50, 66.7, 99, 100])
    return dict(cases=cases)


# ======================================================================================================== half
def _f32bits(v):
    with np.errstate(over="ignore"):
        return int(np.array([v], np.float32).view(np.uint32)[0])


def sec_half():
    rng = np.random.default_rng(SEED + 2)
    entries = [
        ("zero", 0.0), ("neg_zero", -0.0), ("one", 1.0), ("minus_two", -2.0), ("max_f16_65504", 65504.0),
        ("below_overflow_tie_65519.99", 65519.99), ("overflow_tie_65520_to_inf", 65520.0), ("large_to_inf", 1e10),
        ("large_to_neg_inf", -1e10), ("tie_to_even_down_1+2^-11", 1 + 2 ** -11),
        ("tie_to_even_up_1+3*2^-11", 1 + 3 * 2 ** -11), ("just_above_tie", 1 + 2 ** -11 + 2 ** -23),
        ("just_below_tie", 1 + 2 ** -11 - 2 ** -23), ("min_subnormal_2^-24", 2 ** -24),
        ("subnormal_tie_to_zero_2^-25", 2 ** -25), ("subnormal_tie_to_even_3*2^-25", 3 * 2 ** -25),
        ("subnormal_just_above_tie", 2 ** -25 + 2 ** -40), ("max_subnormal", 2 ** -14 - 2 ** -24),
        ("min_normal_2^-14", 2 ** -14), ("neg_subnormal", -5 * 2 ** -24), ("f32_subnormal_to_zero", 1e-40),
        ("neg_tiny_to_neg_zero", -1e-30), ("point_one", 0.1), ("one_third", 1 / 3),
        ("tie_2049_to_2048", 2049.0), ("tie_2051_to_2052", 2051.0), ("neg_1.5", -1.5)]
    labels = [n for n, _ in entries] + ["pos_inf", "neg_inf", "nan_canonical"]
    bits = [_f32bits(v) for _, v in entries] + [0x7F800000, 0xFF800000, 0x7FC00000]
    rnd = rng.normal(0, 3, 16).astype(np.float32)
    labels += ["random_%d" % i for i in range(rnd.size)]
    bits += [int(b) for b in rnd.view(np.uint32)]
    with np.errstate(over="ignore"):
        h = tr.f16_bits(np.array(bits, np.uint32).view(np.float32))
    back = tr.f16_from_bits(h).view(np.uint32)
    enc = case("encode", dict(f32Bits=bits, labels=labels), dict(f16Bits=ints(h), decodedF32Bits=ints(back)),
               kind="encode")
    dec_bits = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xC000,
                0x3555, 0x8001, 0xFBFF, 0x83FF]
    extra = rng.integers(0, 0x7C00, 16)
    dec_bits += [int(b) | (0x8000 if i % 3 == 0 else 0) for i, b in enumerate(extra)]
    dec = case("decode", dict(f16Bits=dec_bits), dict(f32Bits=ints(tr.f16_from_bits(dec_bits).view(np.uint32))),
               kind="decode")
    return dict(cases=[enc, dec])


# ===================================================================================================== binomial
def sec_binomial():
    conf = 0.95
    alpha_m = [1, 20, 29, 59, 299]
    cp = [(0, 1), (0, 20), (0, 59), (1, 20), (2, 29), (5, 100), (10, 10), (0, 299), (3, 299), (19, 20), (1, 2)]
    two = [(0, 10), (10, 10), (5, 10), (1, 20), (19, 20), (0, 1), (1, 1), (50, 100), (297, 299), (3, 7)]
    binc = [(0.3, 2.0, 5.0), (0.9, 2.0, 5.0), (0.01, 1.0, 299.0), (0.5, 50.0, 51.0), (0.99, 3.0, 1.0),
            (0.2, 0.5, 0.5), (0.75, 4.0, 4.0), (0.05, 3.0, 297.0)]
    binv = [(0.95, 1.0, 20.0), (0.95, 3.0, 27.0), (0.025, 5.0, 6.0), (0.975, 6.0, 5.0), (0.95, 4.0, 296.0),
            (0.5, 2.0, 2.0), (0.05, 10.0, 3.0)]
    b_used = [1, 2, 3, 4, 5]
    tw = [tr.two_sided(k, n, conf) for k, n in two]
    inputs = dict(conf=conf, alphaM=alpha_m, cpUpper=[dict(r=r, n=n) for r, n in cp],
                  twoSided=[dict(k=k, n=n) for k, n in two], betaInc=[dict(x=x, a=a, b=b) for x, a, b in binc],
                  betaInv=[dict(p=p, a=a, b=b) for p, a, b in binv], withinPartBUsed=b_used)
    expected = dict(alpha=[tr.alpha_order(m, conf) for m in alpha_m], cpUpper=[tr.cp_upper(r, n, conf) for r, n in cp],
                    twoSided=[dict(lower=lo, upper=hi) for lo, hi in tw],
                    betaInc=[tr.beta_inc(x, a, b) for x, a, b in binc],
                    betaInv=[tr.beta_inv(p, a, b) for p, a, b in binv],
                    withinPart=[tr.alpha_within(b, conf) for b in b_used])
    return dict(cases=[case("bounds", inputs, expected)])


# ======================================================================================================== auroc
def sec_auroc():
    rng = np.random.default_rng(SEED + 4)
    items = [("perfect", [0.9, 0.8, 0.7], [0.1, 0.2, 0.3, 0.4]),
             ("inverted", [0.1, 0.2], [0.5, 0.6, 0.7]),
             ("ties", [1.0, 2.0, 3.0, 3.0], [2.0, 3.0, 0.0]),
             ("all_equal", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
             ("single", [2.0], [1.0]),
             ("random_rounded", np.round(rng.normal(1, 1, 40), 1), np.round(rng.normal(0, 1, 50), 1))]
    return dict(cases=[case(n, dict(pos=floats(p), neg=floats(q)), dict(auroc=tr.auroc(p, q))) for n, p, q in items])


# ======================================================================================================== image
def _exact_sample(im, u, v):
    """Exact rational s1.4 sample (Fractions) for dyadic geometry."""
    hgt, wid = im.shape[:2]
    x, y = u - Fraction(1, 2), v - Fraction(1, 2)
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    xa, xb = min(max(x0, 0), wid - 1), min(max(x0 + 1, 0), wid - 1)
    ya, yb = min(max(y0, 0), hgt - 1), min(max(y0 + 1, 0), hgt - 1)
    return [(1 - fx) * (1 - fy) * int(im[ya, xa, c]) + fx * (1 - fy) * int(im[ya, xb, c])
            + (1 - fx) * fy * int(im[yb, xa, c]) + fx * fy * int(im[yb, xb, c]) for c in range(im.shape[2])]


def _assert_exact(values, points, im, label):
    vals = np.asarray(values, np.float64).reshape(-1, im.shape[2])
    assert vals.shape[0] == len(points)
    for k, (u, v) in enumerate(points):
        ex = _exact_sample(im, u, v)
        for c in range(im.shape[2]):
            assert Fraction(float(vals[k, c])) == ex[c], "%s: float64 value is not exact at %d" % (label, k)


def sec_image():
    rng = np.random.default_rng(SEED + 5)
    cases = []
    # ---- s1.1 downscale (random RGBA with stride padding)
    buf16 = rng.integers(0, 256, 12 * 72, dtype=np.uint8)
    buf12 = rng.integers(0, 256, 9 * 52, dtype=np.uint8)
    for name, buf, w, h, stride, f in [("downscale_f4", buf16, 16, 12, 72, 4), ("downscale_f2", buf16, 16, 12, 72, 2),
                                       ("downscale_f3_odd", buf12, 12, 9, 52, 3)]:
        out = tr.downscale_rgba(buf, w, h, stride, f)
        cases.append(case(name, dict(rgba=dict(w=w, h=h, c=4, rowStride=stride, data=ints(buf)), factor=f),
                          dict(rgb=img(out)), kind="downscale"))
    # ---- s1.2 / s1.3 grey + sharpness
    rgb = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    rgb[4:9, 5:12] = np.clip(rgb[4:9, 5:12].astype(int) // 3 + 160, 0, 255)
    g = tr.grey(rgb)
    rects = [(0, 0, 16, 12), (3, 2, 11, 9), (-4, -3, 6, 5), (10, 7, 30, 30), (5, 5, 8, 7), (5, 5, 8, 8),
             (8, 8, 8, 10), (1, 1, 15, 11)]
    res = [tr.sharpness(g, *r) for r in rects]
    cases.append(case("sharpness_rects",
                      dict(rgb=img(rgb), rects=[dict(x0=a, y0=b, x1=c, y1=d) for a, b, c, d in rects]),
                      dict(grey=floats(g), count=[c for _, c in res], sharpness=[v for v, _ in res]),
                      kind="sharpness"))
    # ---- s1.4 samples, crop-resize, rotated crop
    im = rng.integers(0, 256, (8, 10, 3), dtype=np.uint8)
    samples = [(0.5, 0.5), (5.0, 4.0), (-3.0, 2.2), (9.99, 7.9), (12.0, -1.0), (3.3, 6.7), (1.25, 2.75)]
    dyadic_crops = [(1.0, 0.5, 6.0, 4), (2.0, 1.0, 4.0, 8), (0.0, 0.0, 8.0, 16), (-1.5, -1.0, 12.0, 6)]
    other_crops = [(2.3, 1.7, 5.1, 5), (6.35, 3.1, 3.7, 3)]
    dyadic_rot = [(5.0, 4.0, 4.0, 0.0, 4)]
    other_rot = [(5.0, 4.0, 6.0, 0.3, 5), (4.5, 3.5, 5.0, -1.1, 4), (5.07, 3.93, 8.0, 1.9, 6)]
    crops_out, rot_out = [], []
    for (x0, y0, side, n) in dyadic_crops:
        vals = tr.crop_resize(im, x0, y0, side, n)
        us = [Fraction(x0) + (j + Fraction(1, 2)) * Fraction(side) / n for j in range(n)]
        vs = [Fraction(y0) + (i + Fraction(1, 2)) * Fraction(side) / n for i in range(n)]
        _assert_exact(vals, [(u, v) for v in vs for u in us], im, "crop %r" % ((x0, y0, side, n),))
        crops_out.append(dict(values=floats(vals), bytes=ints(tr.crop_bytes(vals))))
    for (cx, cy, side, th, n) in dyadic_rot:
        vals = tr.crop_resize_rotated(im, cx, cy, side, th, n)
        a = [(j + Fraction(1, 2)) * Fraction(side) / n - Fraction(side) / 2 for j in range(n)]
        _assert_exact(vals, [(Fraction(cx) + aj, Fraction(cy) + bi) for bi in a for aj in a], im, "rot0")
        rot_out.append(dict(values=floats(vals), bytes=ints(tr.crop_bytes(vals))))
    with guard("image/non-dyadic crops", {"cropByteRounding"}):
        for (x0, y0, side, n) in other_crops:
            vals = tr.crop_resize(im, x0, y0, side, n)
            crops_out.append(dict(values=floats(vals), bytes=ints(tr.crop_bytes(vals))))
        for (cx, cy, side, th, n) in other_rot:
            vals = tr.crop_resize_rotated(im, cx, cy, side, th, n)
            rot_out.append(dict(values=floats(vals), bytes=ints(tr.crop_bytes(vals))))
    ties = sum(1 for c in crops_out[:len(dyadic_crops)] for v in c["values"] if (v - math.floor(v)) == 0.5)
    DIAG.append("image: exact .5 rounding ties in the dyadic crops = %d" % ties)
    assert ties >= 1, "no exact rounding tie in the dyadic crops"
    cases.append(case("crop_and_rotate", dict(
        rgb=img(im), samples=[dict(u=u, v=v) for u, v in samples],
        crops=[dict(x0=a, y0=b, side=s, n=n) for a, b, s, n in dyadic_crops + other_crops],
        rotated=[dict(cx=a, cy=b, side=s, theta=t, n=n) for a, b, s, t, n in dyadic_rot + other_rot]), dict(
        samples=[floats(tr.sample(im, u, v)) for u, v in samples], crops=crops_out, rotated=rot_out), kind="crop"))
    return dict(cases=cases)


# ========================================================================================================= mask
def _ellipse(h, w, cx, cy, rx, ry, ang=0.0):
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = xx + 0.5 - cx, yy + 0.5 - cy
    c, s = math.cos(ang), math.sin(ang)
    u, v = x * c + y * s, -x * s + y * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def _paint(frame, mask, colour, rng=None, sigma=0.0):
    col = np.array(colour, np.float64)
    vals = np.broadcast_to(col, frame[mask].shape).copy()
    if rng is not None and sigma > 0:
        vals += rng.normal(0, sigma, vals.shape)
    frame[mask] = np.clip(np.round(vals), 0, 255).astype(np.uint8)


def _mask_params(w, h, gh, gw, f=4):
    mp = tr.MaskParams()
    params = dict(sigmaMin=mp.sigmaMin, kSigma=mp.kSigma, minBlobPx=mp.minBlobPx, borderMargin=mp.borderMargin,
                  minAreaFrac=mp.minAreaFrac, maxAreaFrac=mp.maxAreaFrac, multipleRatio=mp.multipleRatio,
                  cropMargin=mp.cropMargin, coreThreshold=mp.coreThreshold, analysisFactor=f, fullW=f * w,
                  fullH=f * h, gh=gh, gw=gw)
    return mp, params


def _scene_case(name, frame, sheet_frames, sheet, gh, gw, expect_sanity, expect_per=()):
    h, w = frame.shape[:2]
    mp, params = _mask_params(w, h, gh, gw)
    kinds = {"foreground", "mainObjectArea", "areaFrac", "multipleArea", "multipleInside", "covSamplePoint"}
    with guard("mask/" + name, kinds, allow_exact={"multipleArea", "foreground"}):
        if sheet_frames is not None:
            sm = tr.fit_sheet(sheet_frames, None, mp.sigmaMin)
        else:
            sm = tr.SheetModel(np.array(sheet[0], np.float64), np.array(sheet[1], np.float64), None)
        seg, geoms, mo, per, san = tr.analyse_scene(frame, sm, mp, gh, gw)
    assert san == expect_sanity, "%s: main-object sanity %s != %s" % (name, san, expect_sanity)
    got = [p["sanity"] for p in per]
    for want in expect_per:
        assert want in got, "%s: expected a component with sanity %s, got %s" % (name, want, got)
    DIAG.append("mask/%s: areas=%s main=%s sanity=%s per=%s rawAreas=%s" % (
        name, [c.area for c in seg.components], 0 if mo is None else mo.label, san,
        [(p["label"], p["sanity"], p["coreCount"]) for p in per], seg.rawAreas))
    inputs = dict(frame=img(frame))
    if sheet_frames is not None:
        inputs["sheetFrames"] = [img(f) for f in sheet_frames]
    else:
        inputs["sheet"] = dict(mu=floats(sm.mu), sigma=floats(sm.sigma))
    inputs["params"] = params
    expected = dict(sheet=dict(mu=floats(sm.mu), sigma=floats(sm.sigma), mad=None if sm.mad is None else floats(sm.mad)),
                    d2=floats(seg.d2), foreground=ints(seg.foreground), afterOpen=ints(seg.afterOpen),
                    afterClose=ints(seg.afterClose), rawAreas=[int(a) for a in seg.rawAreas], labels=ints(seg.labels),
                    components=[c.to_json() for c in seg.components],
                    geometry=[geo_out(g, c.label) for g, c in zip(geoms, seg.components)],
                    mainObject=0 if mo is None else mo.label,
                    perComponent=[dict(label=p["label"], crop=p["crop"].to_json(), coreCount=p["coreCount"],
                                       sanity=p["sanity"]) for p in per],
                    sanity=san)
    return case(name, inputs, expected, kind="scene")


NO_CORE_GRID = 3


def sec_mask():
    rng = np.random.default_rng(SEED + 6)
    cases = []
    # ---- scene 1: sheet fit from 2 frames + a test frame with 4+ shapes and specks
    h, w = 60, 80
    bg = np.array([196.0, 202.0, 188.0])
    sig = np.array([2.0, 4.0, 6.0])

    def background():
        return np.clip(np.round(bg + rng.normal(0, 1, (h, w, 3)) * sig), 0, 255).astype(np.uint8)

    sheets = [background(), background()]
    fr = background()
    _paint(fr, _ellipse(h, w, 28.3, 30.6, 14.0, 8.0, math.radians(25)), (40, 90, 160), rng, 3.0)
    fr[30, 28] = np.round(bg).astype(np.uint8)                                  # 1-px hole (closed by closing)
    ring = _ellipse(h, w, 62.5, 17.5, 9.0, 9.0) & ~_ellipse(h, w, 62.5, 17.5, 5.5, 5.5)
    _paint(fr, ring, (180, 60, 50), rng, 3.0)
    _paint(fr, _ellipse(h, w, 1.0, 47.0, 7.0, 7.0), (60, 60, 60), rng, 3.0)     # touches the left border
    sq = np.zeros((h, w), bool)
    sq[8:14, 8:14] = True                                                       # 36 px < 1 % of 4800 -> TOO_SMALL
    _paint(fr, sq, (30, 160, 60), rng, 3.0)
    sp2 = np.zeros((h, w), bool)
    sp2[52:54, 70:72] = True                                                    # removed by the opening
    _paint(fr, sp2, (20, 20, 20))
    sp4 = np.zeros((h, w), bool)
    sp4[49:53, 50:54] = True                                                    # survives opening, < minBlobPx
    _paint(fr, sp4, (20, 20, 20))
    line = np.zeros((h, w), bool)
    for k in range(17):
        line[5 + k // 2, 40 + k] = True                                         # 1 px wide -> removed by opening
    _paint(fr, line, (20, 20, 20))
    cases.append(_scene_case("sheet_fit_scene", fr, sheets, None, 6, 6, tr.OK,
                             [tr.OK, tr.TOUCHES_BORDER, tr.TOO_SMALL]))
    mu, sg = (200.0, 205.0, 210.0), (3.0, 4.5, 6.0)
    base = np.array(mu, np.uint8)
    # ---- scene 2: MULTIPLE (neighbour area exactly multipleRatio * main area, centroid inside the crop square)
    h, w = 56, 64
    fr = np.empty((h, w, 3), np.uint8)
    fr[:] = base
    fr[20:31, 10:54] = (40, 40, 40)          # 44 x 11 = 484 px
    fr[4:15, 26:37] = (50, 100, 150)         # 11 x 11 = 121 px = 0.25 * 484 (exact tie -> counts)
    fr[40:49, 53:60] = (90, 30, 30)          # 7 x 9 = 63 px elsewhere
    cases.append(_scene_case("multiple", fr, None, (mu, sg), 6, 6, tr.MULTIPLE, [tr.MULTIPLE]))
    # ---- scene 3: NO_OBJECT (only a border-touching blob and a speck)
    h, w = 30, 40
    fr = np.empty((h, w, 3), np.uint8)
    fr[:] = base
    _paint(fr, _ellipse(h, w, 38.0, 14.0, 6.0, 5.0), (30, 30, 90))
    fr[5:8, 10:13] = (30, 30, 90)
    cases.append(_scene_case("no_object", fr, None, (mu, sg), 6, 6, tr.NO_OBJECT, [tr.TOUCHES_BORDER]))
    # ---- scene 4: TOO_LARGE
    h, w = 50, 64
    fr = np.empty((h, w, 3), np.uint8)
    fr[:] = base
    fr[2:47, 2:62] = (60, 120, 60)           # 60 x 45 = 2700 px = 84.4 % (not touching the 2-px border band)
    cases.append(_scene_case("too_large", fr, None, (mu, sg), 6, 6, tr.TOO_LARGE, [tr.TOO_LARGE]))
    # ---- scene 5: NO_CORE (a thin ring whose patch cells are all < 50 % covered)
    h, w = 40, 52
    fr = np.empty((h, w, 3), np.uint8)
    fr[:] = base
    ring = _ellipse(h, w, 25.8, 19.4, 14.2, 14.2) & ~_ellipse(h, w, 25.8, 19.4, 10.6, 10.6)
    _paint(fr, ring, (150, 40, 40))
    cases.append(_scene_case("no_core", fr, None, (mu, sg), NO_CORE_GRID, NO_CORE_GRID, tr.NO_CORE, [tr.NO_CORE]))
    return dict(cases=cases)


# =================================================================================================== crop_patch
def sec_crop_patch():
    h, w, f, gh, gw = 18, 24, 4, 6, 6
    lab = np.zeros((h, w), np.int32)
    lab[_ellipse(h, w, 9.0, 8.8, 4.8, 3.2, 0.4)] = 1
    lab[_ellipse(h, w, 2.4, 2.1, 3.1, 2.3, 0.2)] = 2
    lab[_ellipse(h, w, 21.6, 15.4, 2.4, 1.9)] = 3
    lab[1:17, 16:19] = 4                                  # 16 rows -> side 76.8 > fullH 72 -> side limited
    comps = {c.label: c for c in tr.components_of(lab, 4, 2)}
    params = dict(analysisFactor=f, fullW=f * w, fullH=f * h, cropMargin=0.10, coreThreshold=0.5, gh=gh, gw=gw)
    cases = []
    specs = [("interior", 1, None), ("clamp_top_left", 2, None), ("clamp_bottom_right", 3, None),
             ("side_limited", 4, None), ("explicit_outside_left_top", 2, tr.Crop(-3.1, -2.6, 40.3)),
             ("explicit_beyond_right_bottom", 3, tr.Crop(70.3, 50.1, 37.3))]
    with guard("crop_patch", {"covSamplePoint"}):
        for name, label, crop in specs:
            inputs = dict(labels=dict(w=w, h=h, data=ints(lab)), label=label, params=params)
            if crop is None:
                c = comps[label]
                inputs["component"] = dict(minX=c.minX, minY=c.minY, maxX=c.maxX, maxY=c.maxY)
                crop = tr.crop_square(c, f, f * w, f * h, 0.10)
            else:
                inputs["crop"] = crop.to_json()
            ps = tr.patch_sets(lab, label, crop, f, gh, gw, 0.5)
            DIAG.append("crop_patch/%s: crop=%s core=%d S=%d B=%d" % (name, crop, ps.core.sum(), ps.S.sum(),
                                                                     ps.B.sum()))
            assert ps.core.any()
            cases.append(case(name, inputs, dict(crop=crop.to_json(), cov=floats(ps.cov), core=ints(ps.core),
                                                 S=ints(ps.S), B=ints(ps.B))))
    return dict(cases=cases)


# ========================================================================================================== knn
def sec_knn():
    rng = np.random.default_rng(SEED + 8)
    feats = rng.normal(0, 1, (3, 4, 8)).astype(np.float32)
    bank = rng.normal(0, 1, (10, 8)).astype(np.float32)
    cases = []
    specs = [("plain", bank, [0] * 10), ("masked", bank, [1, 0, 0, 1, 1, 0, 0, 0, 1, 0]),
             ("all_excluded", bank, [1] * 10)]
    q = feats.reshape(-1, 8)
    b2 = np.concatenate([bank[:5], q[[0, 5, 11]], q[[5]]]).astype(np.float32)
    specs.append(("exact_matches_and_duplicates", b2, [0] * b2.shape[0]))
    for name, b, ex in specs:
        ms = tr.knn_min_sq(feats, b, np.array(ex, bool))
        cases.append(case(name, dict(feats=fmap(feats), bank=mat(b), excluded=ex, knnMask=tr.KNN_MASK),
                          dict(minSq=floats(ms), d=floats(np.sqrt(np.maximum(0.0, ms))), allExcluded=all(ex))))
    return dict(cases=cases)


# ====================================================================================================== scoring
def _score_case(name, dmap, core, tau, sens, rule, top_fraction=0.01):
    gh, gw = dmap.shape
    core = np.asarray(core, bool)
    S = tr.dilate(core, 1)
    fs = tr.frame_score(dmap, S, rule, top_fraction)
    thr = tau * sens
    an = int(np.sum(fs.sm[core] > thr))
    ncore = int(core.sum())
    DIAG.append("scoring/%s: raw=%r s=%r peak=%s topN=%d anomalous=%d/%d |S|=%d" % (
        name, fs.raw, fs.raw / thr, fs.peak, fs.topN, an, ncore, S.sum()))
    return case(name, dict(gh=gh, gw=gw, dmap=floats(dmap), core=ints(core), S=ints(S), tau=tau, sensitivity=sens,
                           scoreRule=rule, params=dict(topFraction=top_fraction)),
                dict(sm=floats(fs.sm), raw=fs.raw, s=fs.raw / thr, topN=fs.topN, peak=peak(fs.peak), coreCount=ncore,
                     anomalousCount=an, anomalousFraction=an / ncore, areaPct=100.0 * an / ncore))


def sec_scoring():
    rng = np.random.default_rng(SEED + 9)
    cases = []
    with guard("scoring", {"peakArgmax"}, allow_exact={"peakArgmax"}):
        d6 = rng.integers(0, 24, (6, 6)) / 8.0
        core = np.zeros((6, 6), bool)
        core[2:4, 2:4] = True
        core[1, 2] = True
        cases.append(_score_case("smoothed_basic", d6, core, 1.25, 1.0, tr.SMOOTHED_MAX))
        cases.append(_score_case("top1_small", d6, core, 1.25, 1.0, tr.TOP1_MEAN))
        cases.append(_score_case("sensitivity", d6, core, 0.875, 1.5, tr.SMOOTHED_MAX))
        dt = np.full((6, 6), 0.5)
        dt[1, 1] = dt[1, 4] = 3.0                          # mirror-symmetric twin peaks -> lowest index wins
        core_t = np.zeros((6, 6), bool)
        core_t[1:3, 1:5] = True
        cases.append(_score_case("peak_tie", dt, core_t, 1.0, 1.0, tr.SMOOTHED_MAX))
        de = rng.integers(0, 8, (6, 6)) / 8.0              # all <= 0.875
        de[2:5, 2:5] = 1.25                                # max sm = 1.25 = tau * sensitivity exactly
        core_e = np.zeros((6, 6), bool)
        core_e[2:5, 2:5] = True
        cases.append(_score_case("exact_threshold", de, core_e, 1.0, 1.25, tr.SMOOTHED_MAX))
        d12 = rng.integers(0, 40, (12, 12)) / 8.0
        d12[5, 6] = d12[7, 3] = 5.5                        # tied top values (mean unaffected)
        core12 = _ellipse(12, 12, 6.2, 5.9, 5.3, 4.4, 0.5)    # |S| > 100 -> topN = 2
        cases.append(_score_case("top1_large", d12, core12, 2.75, 1.0, tr.TOP1_MEAN))
        cases.append(_score_case("smoothed_large", d12, core12, 2.25, 1.25, tr.SMOOTHED_MAX))
        dc = rng.integers(0, 24, (6, 6)) / 8.0
        core_c = np.zeros((6, 6), bool)
        core_c[0, 0] = core_c[0, 1] = core_c[1, 0] = True
        cases.append(_score_case("core_at_corner", dc, core_c, 1.625, 1.0, tr.SMOOTHED_MAX))    # one core sm == 1.625 exactly
    return dict(cases=cases)


# ======================================================================================================== teach
def grid_cov(gh, gw, cx, cy, rx, ry, ang):
    """Fraction of the 4x4 sample points of each grid cell inside an ellipse (grid units), like spec s4."""
    cov = np.zeros((gh, gw))
    c, s = math.cos(ang), math.sin(ang)
    k = (np.arange(4) + 0.5) / 4.0
    for r in range(gh):
        for col in range(gw):
            x = col + k[None, :] - cx
            y = r + k[:, None] - cy
            u, v = x * c + y * s, -x * s + y * c
            cov[r, col] = np.sum((u / rx) ** 2 + (v / ry) ** 2 <= 1.0) / 16.0
    return cov


class World:
    """Synthetic part seen by a 'backbone': patch feature = cov-weighted mix of a slowly changing object appearance
    (A + a(t) V1 + b(t) V2) and a static background, plus noise. The two corner cells carry one constant outlier
    vector whenever they are background, so the pooled bank has exact duplicate rows (an exact coreset tie)."""

    def __init__(self, seed, gh=6, gw=6, d=8, period=12000.0, a_sigma=0.6, v_sigma=0.15, shape=None):
        rng = np.random.default_rng(seed)
        self.gh, self.gw, self.d, self.period, self.a_sigma = gh, gw, d, period, a_sigma
        self.A = np.abs(rng.normal(1.0, a_sigma, (gh, gw, d)))
        self.V1 = rng.normal(0, v_sigma, (gh, gw, d))
        self.V2 = rng.normal(0, v_sigma, (gh, gw, d))
        self.BG = np.abs(rng.normal(0.3, 0.15, (gh, gw, d)))
        self.const_cells = [(0, gw - 1), (gh - 1, 0)]
        self.const_vec = (2.5 + 0.1 * np.arange(d)).astype(np.float32)
        self._shape = shape

    def shape(self, t):
        if self._shape is not None:
            return self._shape(t)
        return (2.7 + 0.2 * math.sin(2 * math.pi * t / 9000.0), 2.6 + 0.15 * math.cos(2 * math.pi * t / 7000.0),
                2.0, 1.6, 0.3 + 0.2 * math.sin(2 * math.pi * t / 12000.0))

    def cov(self, t):
        return grid_cov(self.gh, self.gw, *self.shape(t))

    def other_object(self, rng):
        return np.abs(rng.normal(1.0, self.a_sigma, self.A.shape))

    def features(self, t, rng, cov=None, noise=0.03, scale=1.0, A=None):
        cov = self.cov(t) if cov is None else cov
        a = math.sin(2 * math.pi * t / self.period)
        b = math.cos(2 * math.pi * t / (0.7 * self.period))
        obj = ((self.A if A is None else A) + a * self.V1 + b * self.V2) * scale
        c3 = cov[..., None]
        f = (c3 * obj + (1.0 - c3) * self.BG + rng.normal(0, noise, obj.shape)).astype(np.float32)
        for (r, c) in self.const_cells:
            if cov[r, c] == 0:
                f[r, c] = self.const_vec
        return f

    @staticmethod
    def geometry(rng):
        return dict(area=int(round(1500 + rng.normal(0, 60))), fill=0.62 + rng.normal(0, 0.05),
                    aspect=0.80 + rng.normal(0, 0.01), hu1=0.170 + rng.normal(0, 0.002),
                    solidity=0.93 + rng.normal(0, 0.02))


def teach_params_json(tp, gh, gw, d):
    keys = ["keepPercentile", "minKept", "kMin", "kMax", "kcEps", "segments", "ratio", "minK", "maxRows", "tauFactor",
            "coreThreshold", "scoreRule", "topFraction", "knnMask", "kGeo", "fillFloor", "aspectFloor",
            "solidityFloor", "hu1FloorFrac", "areaFactor", "coverageCut", "posPercentile", "negPercentile",
            "idMinGap", "idMadK"]
    out = dict(gh=gh, gw=gw, dim=d)
    for k in keys:
        v = getattr(tp, k)
        out[k] = float(v) if isinstance(v, float) else v
    return out


def frame_json(fr):
    if not fr.sane:
        return dict(tMs=int(fr.tMs), sane=False, sharpness=None, cov=None, geometry=None, features=None)
    return dict(tMs=int(fr.tMs), sane=True, sharpness=float(fr.sharpness), cov=floats(fr.cov),
                geometry=geo5(fr.geometry), features=fmap(fr.features))


TEACH_KINDS = {"keepSharpness", "kcFirstSharpness", "kcArgmax", "kcEps", "segmentFloor", "bankRatioFloor",
               "coresetArgmax"}


def twin_expected(twin, tp, negatives_given):
    ti = twin.teachInfo
    noneg = tr.tau_id(twin.positives, [], tp)
    withneg = None
    if negatives_given:
        wn = tr.tau_id(twin.positives, twin.negSims, tp)
        withneg = dict(tauId=wn["tauId"], rule=wn["rule"], margin=wn["margin"], overlap=wn["overlap"],
                       negSims=floats(twin.negSims))
    return dict(
        accepted=ti["accepted"], sharpnessCut=ti["sharpnessCut"], kept=ti["kept"],
        keyframes=[dict(index=k["index"], frame=k["frame"], tMs=k["tMs"], segment=k["segment"],
                        sharpness=k["sharpness"]) for k in twin.keyframes],
        kcMinDist=floats(ti["kcMinDist"]), kcStop=ti["kcStop"], kcNext=ti["kcNext"],
        t0=ti["t0"], t1=ti["t1"], segmentsUsed=twin.segmentsUsed, pooledRows=ti["pooledRows"],
        bankRows=ti["bankRows"], coresetIndices=ti["coresetIndices"], bank=mat(twin.bank), bankKf=ints(twin.bankKf),
        bankSeg=ints(twin.bankSeg), globals=mat(twin.globals), lsoMode=twin.lsoMode, lso=floats(twin.lso),
        lsoKeyframes=[int(k) for k in twin.lsoKeyframes], tauTeach=twin.tauTeach, tau=twin.tau,
        positives=floats(twin.positives),
        identity=dict(noNegatives=dict(tauId=noneg["tauId"], rule=noneg["rule"], margin=None, overlap=None),
                      withNegatives=withneg),
        geometry=twin.geometry.to_json(), coverageCut=twin.coverageCut)


def twin_json_for_verdicts(twin):
    return dict(gh=twin.gh, gw=twin.gw, dim=twin.dim, bank=mat(twin.bank), bankKf=ints(twin.bankKf),
                bankSeg=ints(twin.bankSeg), globals=mat(twin.globals), kfFeats=mat(twin.kfFeats),
                kfCov=mat(twin.kfCov), tau=twin.tau, tauId=twin.tauId, coverageCut=twin.coverageCut,
                geometry=twin.geometry.to_json())


def _teach_case(name, frames, tp, negatives, gh, gw, d, expect):
    inputs = dict(params=teach_params_json(tp, gh, gw, d),
                  negatives=mat(np.zeros((0, d), np.float32) if negatives is None else negatives, cols=d),
                  frames=[frame_json(f) for f in frames])
    with guard("teach/" + name, TEACH_KINDS,
               allow_exact={"coresetArgmax", "kcArgmax", "keepSharpness", "bankRatioFloor"}):
        try:
            twin = tr.teach(frames, tp, negatives)
        except tr.TeachError as e:
            assert expect == "fail", "%s: unexpected teach failure %s" % (name, e.code)
            DIAG.append("teach/%s: error %s accepted=%d kept=%d" % (name, e.code, len(e.info["accepted"]),
                                                                    len(e.info["kept"])))
            return case(name, inputs, dict(error=e.code, accepted=e.info["accepted"], kept=e.info["kept"]),
                        kind="fail"), None
    assert expect != "fail", name
    ti = twin.teachInfo
    DIAG.append("teach/%s: kept=%d K=%d stop=%s next=%s segs=%s used=%d pooled=%d bank=%d lso=%s mode=%s tau=%.4f "
                "tauId=%.5f pos=[%.5f..%.5f] neg=%s" % (
                    name, len(ti["kept"]), len(twin.keyframes), ti["kcStop"], ti["kcNext"],
                    [k["segment"] for k in twin.keyframes], twin.segmentsUsed, ti["pooledRows"], ti["bankRows"],
                    ["%.3f" % v for v in twin.lso], twin.lsoMode, twin.tau, twin.tauId, min(twin.positives),
                    max(twin.positives), ["%.4f" % v for v in twin.negSims]))
    return case(name, inputs, twin_expected(twin, tp, negatives is not None and len(negatives) > 0), kind="ok"), twin


def sec_teach():
    cases = []
    # ---- main: 36 frames over 12 s, 4 insane, 6x6x8 features, negatives
    world = World(SEED + 10)
    rng = np.random.default_rng(SEED + 11)
    n = 36
    ts = [int(round(i * 12000 / (n - 1))) + (int(rng.integers(-20, 21)) if 0 < i < n - 1 else 0) for i in range(n)]
    insane = {4, 13, 14, 27}
    frames = []
    for i, t in enumerate(ts):
        if i in insane:
            frames.append(tr.TeachFrame(t, False))
            continue
        cov = world.cov(t)
        frames.append(tr.TeachFrame(t, True, float(rng.uniform(200, 900)), cov, World.geometry(rng),
                                    world.features(t, rng, cov)))
    negs = []
    for k in range(3):                                    # three other parts presented as negatives
        t = 3000.0 * (k + 1)
        cov = world.cov(t)
        f = world.features(t, rng, cov, A=world.other_object(rng))
        negs.append(tr.global_descriptor(f.reshape(-1, world.d), (cov >= 0.5).ravel()).astype(np.float32))
    negs = np.array(negs, np.float32)
    tp = tr.TeachParams(kMin=4, kMax=8, ratio=0.25, minK=16, maxRows=40)
    c_main, twin = _teach_case("main", frames, tp, negs, 6, 6, 8, "ok")
    cases.append(c_main)
    assert twin.lsoMode == "SEGMENT" and twin.segmentsUsed >= 2
    assert twin.teachInfo["bankRows"] == tp.maxRows, "maxRows should bind in the main case"

    # ---- lso_fallback: 13 early frames (5 views) + 1 late exact duplicate of frame 0 -> keyframes in 1 segment
    wf = World(SEED + 20, gh=4, gw=4, d=8, v_sigma=0.0, shape=lambda t: (1.9, 2.0, 1.4, 1.2, 0.2))
    rngf = np.random.default_rng(SEED + 21)
    views = [np.abs(rngf.normal(1.0, 0.6, (4, 4, 8))) for _ in range(5)]
    covf = wf.cov(0.0)
    fr_f = []
    sharp = [700, 800, 720, 710, 690, 300, 310, 320, 330, 340, 350, 360, 370, 650]
    for i in range(14):
        t = 100 * i if i < 13 else 10000
        v = 0 if i == 13 else i % 5
        noise = 0.0 if i in (0, 13) else 0.02
        feats = wf.features(t, np.random.default_rng(SEED + 100 + (0 if i == 13 else i)), covf, noise=noise,
                            A=views[v])
        fr_f.append(tr.TeachFrame(t, True, float(sharp[i]) + 0.25 * i, covf, World.geometry(rngf), feats))
    fr_f[13].features = fr_f[0].features.copy()
    fr_f[13].geometry = dict(fr_f[0].geometry)
    tpf = tr.TeachParams(kMin=4, kMax=8, minK=16, maxRows=2400)
    c_fb, twin_fb = _teach_case("lso_fallback", fr_f, tpf, None, 4, 4, 8, "ok")
    cases.append(c_fb)
    assert twin_fb.lsoMode == "KEYFRAME" and twin_fb.teachInfo["kcStop"] == "EPS", twin_fb.teachInfo["kcStop"]
    assert 13 in twin_fb.teachInfo["kept"]

    # ---- lso_skip: a one-row bank; keyframes in that row's segment are skipped
    ws = World(SEED + 30, gh=4, gw=4, d=4, v_sigma=0.7, shape=lambda t: (2.0, 2.0, 1.3, 1.1, 0.0))
    rngs = np.random.default_rng(SEED + 31)
    fr_s = []
    for i in range(10):
        t = 1111 * i
        cov = ws.cov(t)
        fr_s.append(tr.TeachFrame(t, True, float(rngs.uniform(300, 600)), cov, World.geometry(rngs),
                                  ws.features(t, rngs, cov)))
    tps = tr.TeachParams(keepPercentile=0.0, kMin=4, kMax=6, ratio=0.001, minK=1, maxRows=2400)
    c_sk, twin_sk = _teach_case("lso_skip", fr_s, tps, None, 4, 4, 4, "ok")
    cases.append(c_sk)
    assert twin_sk.teachInfo["bankRows"] == 1 and len(twin_sk.lso) < len(twin_sk.keyframes)
    assert twin_sk.lsoMode == "SEGMENT"

    # ---- not_enough_frames
    rngn = np.random.default_rng(SEED + 40)
    fr_n = []
    for i in range(10):
        if i in (2, 5, 8):
            fr_n.append(tr.TeachFrame(400 * i, False))
            continue
        cov = np.array([[0.25, 0.5], [0.75, 1.0]])
        fr_n.append(tr.TeachFrame(400 * i, True, float(rngn.uniform(100, 500)), cov, World.geometry(rngn),
                                  rngn.normal(0, 1, (2, 2, 2)).astype(np.float32)))
    c_nf, _ = _teach_case("not_enough_frames", fr_n, tr.TeachParams(), None, 2, 2, 2, "fail")
    cases.append(c_nf)
    return dict(cases=cases), twin, world, tp


# ===================================================================================================== verdicts
VERDICT_KINDS = {"identity", "geometryBound", "geometryArea", "passThreshold", "coverageCut", "anomalousPatch",
                 "peakArgmax", "topkViews", "voteWindow", "voteMedian"}


def sec_verdicts(twin, world, tp):
    jp = tr.JudgeParams(coreThreshold=tp.coreThreshold, topFraction=tp.topFraction)
    gm = twin.geometry
    good_geo = dict(area=1510, fill=gm.fill[0] + 0.3 * gm.fill[1], aspect=gm.aspect[0] - 0.5 * gm.aspect[1],
                    hu1=gm.hu1[0] + 0.2 * gm.hu1[1], solidity=gm.solidity[0] - 0.4 * gm.solidity[1])
    t_q = 6100.0
    cov_q = world.cov(t_q)
    core_cells = [tuple(int(v) for v in x) for x in np.argwhere(cov_q >= 0.5)]
    cells_a = [(2, 2), (2, 3), (3, 2)]               # L-shaped 3-patch defect inside the core
    cells_b = [(1, 2), (2, 1), (2, 2)]               # another L (for the two-crop vote)
    assert all(c in core_cells for c in cells_a + cells_b), core_cells

    def feats(seed, amp=0.0, scale=1.0, A=None, cells=None):
        """Query features; a defect subtracts amp * (the crop's own global direction) from a few core patches: the
        core sum keeps its direction (identity unchanged) while those patches move away from every bank row."""
        f = world.features(t_q, np.random.default_rng(seed), cov_q, scale=scale, A=A)
        if amp:
            u = tr.global_descriptor(f.reshape(-1, world.d), (cov_q >= 0.5).ravel())
            for r, c in (cells or cells_a):
                f[r, c] -= (amp * u).astype(np.float32)
        return f

    def s_of(seed, amp, cells=None):
        return tr.judge(twin, feats(seed, amp, cells=cells), cov_q, good_geo, tr.OK, 1.0, tr.SMOOTHED_MAX,
                        tr.CORESET, jp).s

    def k_for(seed, target, cells=None):
        """Defect amplitude giving s ~ target (bisection; s grows with the amplitude)."""
        lo, hi = 0.0, 8.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if s_of(seed, mid, cells) < target:
                lo = mid
            else:
                hi = mid
        return round(0.5 * (lo + hi), 4)

    queries = []

    def q(name, sanity=tr.OK, f=None, cov=cov_q, geo=good_geo, sens=1.0, rule=tr.SMOOTHED_MAX, bank=tr.CORESET,
          voting=False, extra=()):
        queries.append(dict(name=name, sanity=sanity, f=f, cov=None if f is None else cov,
                            geo=None if f is None else geo, sens=sens, rule=rule, bank=bank, voting=voting,
                            extra=list(extra)))

    f_pass = feats(501)
    f_def = feats(502, k_for(502, 1.22))
    q("pass", f=f_pass)
    q("defect", f=f_def)
    q("defect_sensitivity_pass", f=f_def, sens=1.5)
    q("defect_top1_mean", f=f_def, rule=tr.TOP1_MEAN)
    q("pass_topk_views", f=f_pass, bank=tr.TOPK_VIEWS)
    q("defect_topk_views", f=f_def, bank=tr.TOPK_VIEWS)
    other = world.other_object(np.random.default_rng(SEED + 55))
    q("identity_with_shape_failure", f=feats(503, A=other), geo=dict(good_geo, area=5000))
    q("shape", f=feats(504), geo=dict(good_geo, fill=gm.fill[0] + 4.5 * gm.fill[1],
                                      solidity=gm.solidity[0] - 3.6 * gm.solidity[1]))
    q("coverage", f=feats(505, scale=1.8))
    for reason in (tr.NO_OBJECT, tr.TOUCHES_BORDER, tr.TOO_SMALL, tr.TOO_LARGE, tr.MULTIPLE):
        q("reframe_" + reason.lower(), sanity=reason)
    q("reframe_no_core", f=feats(506), cov=np.full(cov_q.shape, 7.0 / 16.0))
    f_b = feats(510, k_for(510, 1.08))
    f_lo = feats(511, k_for(511, 0.85))
    f_hi = feats(517, k_for(517, 1.25))
    f_b2 = feats(513, k_for(513, 1.12, cells_b), cells=cells_b)
    q("vote_to_pass", f=f_b, voting=True, extra=[(feats(514), cov_q), (f_lo, cov_q)])
    q("vote_stays_defect", f=f_b, voting=True, extra=[(f_hi, cov_q), (feats(514), cov_q)])
    q("vote_two_crops_higher", f=f_b, voting=True, extra=[(f_b2, cov_q)])
    q("vote_disabled", f=f_b, voting=False, extra=[(feats(514), cov_q)])
    q("vote_not_borderline", f=f_def, voting=True, extra=[(feats(514), cov_q)])

    results = []
    with guard("verdicts", VERDICT_KINDS, allow_exact={"voteMedian"}):
        for qq in queries:
            j = tr.judge(twin, qq["f"], qq["cov"], qq["geo"], qq["sanity"], qq["sens"], qq["rule"], qq["bank"], jp)
            if qq["voting"]:
                j = tr.vote(twin, j, qq["extra"], qq["sens"], qq["rule"], qq["bank"], jp)
            results.append(j)
    expect = {"pass": (tr.PASS, None), "defect": (tr.DEFECT, None), "defect_sensitivity_pass": (tr.PASS, None),
              "defect_top1_mean": (tr.DEFECT, None), "pass_topk_views": (tr.PASS, None),
              "defect_topk_views": (tr.DEFECT, None),
              "identity_with_shape_failure": (tr.NOT_ENROLLED, tr.IDENTITY), "shape": (tr.NOT_ENROLLED, tr.SHAPE),
              "coverage": (tr.NOT_ENROLLED, tr.COVERAGE), "reframe_no_core": (tr.REFRAME, tr.NO_CORE),
              "vote_to_pass": (tr.DEFECT, None), "vote_stays_defect": (tr.DEFECT, None),
              "vote_two_crops_higher": (tr.DEFECT, None), "vote_disabled": (tr.DEFECT, None),
              "vote_not_borderline": (tr.DEFECT, None)}
    final = {"vote_to_pass": tr.PASS, "vote_stays_defect": tr.DEFECT, "vote_two_crops_higher": tr.DEFECT}
    for qq, j in zip(queries, results):
        want = expect.get(qq["name"], (tr.REFRAME, qq["sanity"]))
        assert (j.verdict, j.reason) == want, "%s: got %s/%s want %s" % (qq["name"], j.verdict, j.reason, want)
        if qq["name"] in final:
            assert j.vote is not None and j.finalVerdict == final[qq["name"]], (qq["name"], j.vote, j.finalVerdict)
        elif qq["voting"]:
            assert j.vote is None, qq["name"]
        DIAG.append("verdicts/%-28s %-12s %-9s sim=%s s=%s a=%s geo=%s vote=%s" % (
            qq["name"], j.verdict, j.reason, None if j.sim is None else "%.5f" % j.sim,
            None if j.s is None else "%.4f" % j.s, None if j.anomalousFraction is None else
            "%.3f" % j.anomalousFraction, j.geoFailures,
            None if j.vote is None else (["%.4f" % v for v in j.vote["s"]], j.vote["chosen"], j.vote["verdict"])))
    shape_q = [j for qq, j in zip(queries, results) if qq["name"] == "identity_with_shape_failure"][0]
    assert shape_q.geoFailures == ["area"], shape_q.geoFailures
    vt = [j for qq, j in zip(queries, results) if qq["name"] == "vote_two_crops_higher"][0]
    assert vt.vote["chosen"] == 1 and len(vt.vote["s"]) == 2

    def qjson(qq):
        return dict(name=qq["name"], sanity=qq["sanity"], features=None if qq["f"] is None else fmap(qq["f"]),
                    cov=None if qq["cov"] is None else floats(qq["cov"]),
                    geometry=None if qq["geo"] is None else geo5(qq["geo"]), sensitivity=qq["sens"],
                    scoreRule=qq["rule"], bankRule=qq["bank"], voting=qq["voting"],
                    extraCrops=[dict(features=fmap(f), cov=floats(c)) for f, c in qq["extra"]])

    def rjson(j):
        v = None
        if j.vote is not None:
            v = dict(s=floats(j.vote["s"]), chosen=j.vote["chosen"], finalS=j.vote["finalS"], verdict=j.vote["verdict"],
                     reason=j.vote["reason"], peak=peak(j.vote["peak"]),
                     anomalousFraction=j.vote["anomalousFraction"], areaPct=j.vote["areaPct"])
        return dict(verdict=j.verdict, reason=j.reason, sim=j.sim, idOk=j.idOk, geoOk=j.geoOk,
                    geoFailures=j.geoFailures, raw=j.raw, s=j.s, peak=peak(j.peak), coreCount=j.coreCount,
                    anomalousFraction=j.anomalousFraction, areaPct=j.areaPct,
                    dmap=None if j.dmap is None else floats(j.dmap), views=j.views, vote=v,
                    finalVerdict=j.finalVerdict)

    params = dict(coreThreshold=jp.coreThreshold, topFraction=jp.topFraction, topKViews=jp.topKViews,
                  knnMask=jp.knnMask, voteLo=jp.voteLo, voteHi=jp.voteHi, voteMaxExtra=jp.voteMaxExtra)
    return dict(cases=[case("main", dict(twin=twin_json_for_verdicts(twin), params=params,
                                         queries=[qjson(qq) for qq in queries]),
                            dict(results=[rjson(j) for j in results]))])


# ================================================================================================== calibration
def sec_calibration(twin, tp):
    rng = np.random.default_rng(SEED + 60)
    cp = tr.CalParams()
    params = dict(conf=cp.conf, kGeo=cp.kGeo, fillFloor=cp.fillFloor, aspectFloor=cp.aspectFloor,
                  solidityFloor=cp.solidityFloor, hu1FloorFrac=cp.hu1FloorFrac, areaFactor=cp.areaFactor,
                  posPercentile=cp.posPercentile, negPercentile=cp.negPercentile, idMinGap=cp.idMinGap,
                  idMadK=cp.idMadK, latencyWindow=cp.latencyWindow)
    gm = twin.geometry

    def geo():
        return dict(area=int(round(gm.meanArea + rng.normal(0, 50))), fill=gm.fill[0] + rng.normal(0, 0.04),
                    aspect=gm.aspect[0] + rng.normal(0, 0.012), hu1=gm.hu1[0] + rng.normal(0, 0.0025),
                    solidity=gm.solidity[0] + rng.normal(0, 0.015))

    def samples_a():
        out = []
        for i in range(32):
            if i in (3, 17, 25):
                out.append(tr.CalSample(False))
                continue
            id_ok = i not in (6, 21)
            geo_ok = i not in (10, 21)
            sim = float(rng.uniform(0.990, 0.999)) if id_ok else float(rng.uniform(0.90, 0.95))
            raw = float(rng.uniform(0.30, 0.95)) * twin.tauTeach
            if i == 12:
                raw = 1.08 * twin.tauTeach            # valid and above tau_teach -> tau_cal = this raw
            if i == 21:
                raw = 2.0 * twin.tauTeach             # invalid (identity + geometry) -> ignored
            out.append(tr.CalSample(True, id_ok, geo_ok, raw, sim, geo()))
        return out

    def samples_b():
        out = []
        for i in range(20):
            out.append(tr.CalSample(True, True, True, float(rng.uniform(0.2, 0.9)) * twin.tauTeach,
                                    float(rng.uniform(0.985, 0.999)), geo()))
        return out

    lat = [float(v) for v in np.round(rng.gamma(9.0, 4.0, 600), 2)]
    cases = []
    for name, neg, samples, sens in [("with_negatives", twin.negSims, samples_a(), [1.0, 0.95, 0.9, 1.2]),
                                     ("no_negatives", [], samples_b(), [1.0, 0.5, 0.8])]:
        with guard("calibration/" + name, {"certificateValid"}, allow_exact={"certificateValid"}):
            cal = tr.calibrate(twin.tauTeach, twin.positives, neg, twin.segmentsUsed, samples, cp, sens, lat)
        DIAG.append("calibration/%s: n=%d sane=%d m=%d tauCal=%.4f (teach %.4f) alpha=%.4f tauId=%.5f valid=%s cert=%s" % (
            name, cal["n"], cal["nSanityOk"], cal["m"], cal["tauCal"], twin.tauTeach, cal["alpha"],
            cal["identity"]["tauId"], cal["valid"],
            {k: (v["k"], v["n"], round(v["upper"], 4)) for k, v in cal["certificate"].items() if isinstance(v, dict)}))
        cert = cal["certificate"]
        inputs = dict(params=params, tauTeach=twin.tauTeach, positives=floats(twin.positives), negSims=floats(neg),
                      segmentsUsed=twin.segmentsUsed,
                      samples=[dict(sanityOk=s.sanityOk, idOk=s.idOk, geoOk=s.geoOk, raw=s.raw, sim=s.sim,
                                    geometry=None if s.geometry is None else geo5(s.geometry)) for s in samples],
                      sensitivities=sens, latencyMs=lat)
        expected = dict(n=cal["n"], nSanityOk=cal["nSanityOk"], m=cal["m"], tauCal=cal["tauCal"], calMax=cal["calMax"],
                        alpha=cal["alpha"], identity=dict(tauId=cal["identity"]["tauId"], rule=cal["identity"]["rule"],
                                                          margin=cal["identity"]["margin"],
                                                          overlap=cal["identity"]["overlap"]),
                        geometry=cal["geometry"].to_json(),
                        certificate=dict(sanity=cert["sanity"], identity=cert["identity"], geometry=cert["geometry"],
                                         identityMargin=cert["identityMargin"], withinPart=cert["withinPart"],
                                         latencyP50=cert["latencyP50"], latencyP95=cert["latencyP95"]),
                        valid=cal["valid"])
        cases.append(case(name, inputs, expected))
    return dict(cases=cases)


# ====================================================================================================== tracker
TW, TH = 320, 180


def _det(cx, cy, hw, hh, sharp):
    x0, x1 = max(0, int(math.floor(cx - hw))), min(TW - 1, int(math.ceil(cx + hw)) - 1)
    y0, y1 = max(0, int(math.floor(cy - hh))), min(TH - 1, int(math.ceil(cy + hh)) - 1)
    if x1 < x0 or y1 < y0:
        return None
    return dict(area=(x1 - x0 + 1) * (y1 - y0 + 1), minX=x0, minY=y0, maxX=x1, maxY=y1, cx=(x0 + x1 + 1) / 2.0,
                cy=(y0 + y1 + 1) / 2.0, touchesBorder=bool(x0 < 2 or y0 < 2 or x1 >= TW - 2 or y1 >= TH - 2),
                sharpness=float(sharp))


def _tracker_params(**kw):
    p = tr.TrackerParams(w=TW, h=TH, **kw)
    if p.linePos is None:
        p.linePos = 0.5 * (TW if p.lineAxis == "X" else TH)
    return p


def _tracker_case(name, p, ts, parts, expect_fired):
    frames = []
    for i, t in enumerate(ts):
        dets = []
        for part in parts:
            d = part(i, t)
            if d is not None:
                dets.append(d)
        frames.append(dict(tMs=int(t), detections=dets))
    kinds = {"gate", "pairOrder", "lineSide", "steadySpeed", "steadySharp", "bestCropQuality"}
    out, fired, exited = [], [], []
    with guard("tracker/" + name, kinds, allow_exact={"lineSide", "bestCropQuality", "steadySharp"}):
        trk = tr.Tracker(p)
        for i, fr in enumerate(frames):
            r = trk.step(fr["tMs"], [tr.Detection(**d) for d in fr["detections"]])
            for e in r["events"]:
                if e["type"] == "FIRED":
                    fired.append(dict(frame=i, trackId=e["trackId"], trigger=e["trigger"], judgeFrame=e["judgeFrame"]))
                else:
                    exited.append(dict(frame=i, trackId=e["trackId"], judged=e["judged"]))
            out.append(dict(assign=r["assign"], events=r["events"], tracks=[tk.state() for tk in trk.tracks]))
    got = [(f["frame"], f["trackId"], f["trigger"], f["judgeFrame"]) for f in fired]
    DIAG.append("tracker/%s: fired=%s exited=%s" % (name, got, [(e["frame"], e["trackId"], e["judged"])
                                                                  for e in exited]))
    assert [(f[1], f[2]) for f in got] == expect_fired, "%s: fired %s != %s" % (name, got, expect_fired)
    params = dict(w=p.w, h=p.h, gateBase=p.gateBase, gateArea=p.gateArea, gateSpeed=p.gateSpeed, velAlpha=p.velAlpha,
                  minHits=p.minHits, maxMissed=p.maxMissed, lineEnabled=p.lineEnabled, lineAxis=p.lineAxis,
                  linePos=float(p.linePos), lineDirection=p.lineDirection, crossFrames=p.crossFrames,
                  steadyEnabled=p.steadyEnabled, vStill=p.vStill, holdMs=float(p.holdMs),
                  holdSharpRatio=p.holdSharpRatio, bestK=p.bestK)
    return case(name, dict(params=params, frames=frames), dict(frames=out, fired=fired, exited=exited))


def sec_tracker():
    cases = []
    t33 = [int(round(33.3 * i)) for i in range(60)]
    # 1. two parts crossing at speed (opposite directions, ANY) with occlusion gaps + a one-frame spurious blob
    def part_a(i, t):
        if i in (5, 6):
            return None
        return _det(20 + 0.35 * t, 60, 12, 10, 420 + 60 * math.sin(0.9 * i))

    def part_b(i, t):
        if i < 2 or i in (20, 21, 22):
            return None
        return _det(300 - 0.3 * t, 125 + 0.01 * t, 10, 10, 380 + 50 * math.cos(0.7 * i))

    def spurious(i, t):
        return _det(250, 30, 3, 3, 100.0) if i == 8 else None

    cases.append(_tracker_case("two_parts_crossing", _tracker_params(steadyEnabled=False), t33[:42],
                               [part_a, part_b, spurious], [(1, "LINE"), (2, "LINE")]))
    # 2. a part that crosses by one frame, reverses, crosses again (incl. a frame exactly on the line), fires once,
    #    reverses and re-crosses without a second event
    xs = [100, 110, 120, 130, 140, 150, 164, 154, 144, 134, 128, 138, 148, 158, 168, 160, 170, 180, 190, 182, 172,
          162, 152, 147, 157, 167, 177, 187, 197]

    def part_r(i, t):
        return _det(xs[i], 90, 10, 8, 400 + 30 * math.sin(1.3 * i)) if i < len(xs) else None

    cases.append(_tracker_case("reverse_recross", _tracker_params(steadyEnabled=False), t33[:len(xs) + 7], [part_r],
                               [(1, "LINE")]))
    # 3. steady hold with a blur dip (the hold clock restarts), both triggers on: STEADY fires first, the later line
    #    crossing is ignored. Timestamps 31.25 ms apart so the hold reaches exactly holdMs = 500.
    t31 = [int(round(31.25 * i)) for i in range(64)]
    xs3 = [20, 40, 58, 74, 87, 97, 104, 108, 110, 111] + [111.5] * 34
    jitter = {14: 112.0, 23: 112.0, 30: 112.0}

    def part_s(i, t):
        if i < len(xs3):
            x = jitter.get(i, xs3[i])
        else:
            x = 111.5 + 12 * (i - len(xs3) + 1)
        sharp = 250.0 if i == 20 else 480 + 8 * math.sin(0.8 * i)
        return _det(x, 100, 12, 9, sharp)

    cases.append(_tracker_case("steady_blur_dip", _tracker_params(), t31, [part_s], [(1, "STEADY")]))
    # 4. ID switch after the line (POSITIVE): the part fires, its detection jumps beyond the gate, the new track is
    #    first confirmed downstream and never fires; the old track exits judged.
    def part_i(i, t):
        x = 80 + 10 * i if i <= 13 else 262 + 10 * (i - 14)
        return _det(x, 90, 10, 8, 450 + 20 * math.cos(i))

    cases.append(_tracker_case("id_switch_after_line", _tracker_params(steadyEnabled=False, lineDirection="POSITIVE"),
                               t33[:34], [part_i], [(1, "LINE")]))
    # 5. NEGATIVE direction: a right-to-left part fires, a wrong-way part never fires, a part that always touches the
    #    top border fires with no buffered crop (judged REFRAME TOUCHES_BORDER).
    def part_p(i, t):
        return _det(280 - 0.3 * t, 70, 10, 8, 400 + 25 * math.sin(i))

    def part_q(i, t):
        return _det(40 + 0.3 * t, 135, 10, 8, 380 + 25 * math.cos(i))

    def part_top(i, t):
        return _det(300 - 0.35 * t, 8, 14, 9, 300 + 10 * math.sin(2 * i))

    cases.append(_tracker_case("negative_and_border", _tracker_params(steadyEnabled=False, lineDirection="NEGATIVE"),
                               t33[:30], [part_p, part_q, part_top], [(1, "LINE"), (3, "LINE")]))
    # 6. axis Y, POSITIVE (downwards): the falling part fires, the rising part (first confirmed downstream) never does
    def part_d(i, t):
        return _det(200, 20 + 0.25 * t, 9, 9, 350 + 20 * math.sin(i))

    def part_u(i, t):
        return _det(80, 170 - 0.25 * t, 9, 9, 360 + 20 * math.cos(i))

    cases.append(_tracker_case("axis_y_positive", _tracker_params(steadyEnabled=False, lineAxis="Y",
                                                                  lineDirection="POSITIVE"),
                               t33[:24], [part_d, part_u], [(1, "LINE")]))
    return dict(cases=cases)


# ===================================================================================================== governor
def sec_governor():
    gp = tr.GovernorParams()
    params = dict(cooldownMs=gp.cooldownMs, statusL2=gp.statusL2, statusL1=gp.statusL1, headroomL2=gp.headroomL2,
                  headroomL1=gp.headroomL1, fpsL0=gp.fpsL0, fpsL1=gp.fpsL1, fpsL2=gp.fpsL2, fpsIdle=gp.fpsIdle)
    seqs = {
        "hysteresis": [(0, 0, 0.30, False), (1000, 2, 0.30, False), (2000, 1, 0.90, False), (3000, 1, 0.85, False),
                       (10000, 1, 0.80, False), (20000, 3, 0.20, False), (21000, 1, 0.75, False),
                       (50999, 1, 0.72, False), (51000, 1, 0.70, False), (52000, 0, None, False),
                       (60000, 0, 0.10, True), (81999, 0, 0.10, True), (82000, 0, 0.10, True),
                       (83000, 0, 0.10, False), (84000, 6, None, False), (85000, 4, 0.99, False),
                       (86000, 0, 0.90, False), (87000, 0, 0.10, False), (100000, 1, 0.50, True),
                       (117000, 0, 0.10, False)],
        "nan_and_idle": [(0, 0, None, True), (500, 0, None, False), (1000, 2, None, False), (1500, 1, None, False),
                         (2000, 1, 0.69, False), (31500, 1, None, True), (32000, 0, 0.70, False),
                         (33000, 5, None, False), (34000, 1, 0.851, False), (35000, 1, 0.849, False),
                         (65000, 1, 0.849, True), (65500, 0, None, False)],
    }
    cases = []
    for name, seq in seqs.items():
        g = tr.Governor(gp)
        steps = [g.step(t, st, hr, idle) for t, st, hr, idle in seq]
        DIAG.append("governor/%s: %s" % (name, [(s["level"], s["fps"]) for s in steps]))
        cases.append(case(name, dict(params=params, samples=[dict(tMs=t, status=st, headroom=hr, idle=idle)
                                                             for t, st, hr, idle in seq]), dict(steps=steps)))
    return dict(cases=cases)


# ==================================================================================================== twin_json
PIPELINES = {
    "r18_default": {
        "backboneId": "r18_320_f32", "backboneSha256": "3f0a1c9e5b7d2468ace13579bdf02468ace13579bdf02468ace13579bdf0246a",
        "inputSize": 320, "gh": 40, "gw": 40, "dim": 128, "precision": "fp32", "l2NormalizePatches": False,
        "preprocessVersion": 1,
        "mask": {"analysisFactor": 4, "kSigma": 4.0, "sigmaMin": 3.0, "minBlobPx": 30, "borderMargin": 2,
                 "cropMargin": 0.10, "coreThreshold": 0.5},
        "knnGraphId": "knn_p1600_d128", "scoreRule": "SMOOTHED_MAX", "bankRule": "CORESET", "rotation": "NONE"},
    "dinov2_challengers": {
        "backboneId": "dinov2s14_448_f16", "backboneSha256": "b7e2d4f6a8c0e2d4f6a8c0e2d4f6a8c0e2d4f6a8c0e2d4f6a8c0e2d4f6a8c0e2",
        "inputSize": 448, "gh": 32, "gw": 32, "dim": 384, "precision": "fp16", "l2NormalizePatches": True,
        "preprocessVersion": 2,
        "mask": {"analysisFactor": 4, "kSigma": 3.5, "sigmaMin": 2.5, "minBlobPx": 40, "borderMargin": 3,
                 "cropMargin": 0.125, "coreThreshold": 0.5625},
        "knnGraphId": "knn_p1024_d384", "scoreRule": "TOP1_MEAN", "bankRule": "TOPK_VIEWS", "rotation": "AUGMENT4"},
}


def sec_twin_json():
    cases = []
    for name, pl in PIPELINES.items():
        cases.append(case(name, dict(pipeline=pl), dict(canonical=tr.canonical_json(pl), fingerprint=tr.fingerprint(pl)),
                          kind="pipeline"))
    vals = np.array([[0.0, -0.0, 1.0, 1 + 2 ** -11], [65504.0, 1e-7, -2.5, 0.1], [3.14159, -1e5, 2 ** -24, 1 / 3]],
                    np.float32)
    raw = tr.f16_file_bytes(vals)
    back = tr.parse_f16(raw)
    assert back.shape == (3, 4)
    with np.errstate(over="ignore"):
        bits = tr.f16_bits(vals)
    cases.append(case("f16_file", dict(rows=3, cols=4, values=f32s(vals)), dict(f16Bits=ints(bits), bytesHex=raw.hex()),
                      kind="f16File"))
    return dict(cases=cases)


# ======================================================================================================== build
def build():
    DIAG.clear()
    doc = dict(meta=dict(spec="docs/verification/twin-spec.md v1", generator="tools/lab/make_golden_twin.py",
                         seed=SEED, tolerances=TOL))
    doc["stats"] = sec_stats()
    doc["half"] = sec_half()
    doc["binomial"] = sec_binomial()
    doc["auroc"] = sec_auroc()
    doc["image"] = sec_image()
    doc["mask"] = sec_mask()
    doc["crop_patch"] = sec_crop_patch()
    doc["knn"] = sec_knn()
    doc["scoring"] = sec_scoring()
    teach_sec, twin, world, tp = sec_teach()
    doc["teach"] = teach_sec
    doc["verdicts"] = sec_verdicts(twin, world, tp)
    doc["calibration"] = sec_calibration(twin, tp)
    doc["tracker"] = sec_tracker()
    doc["governor"] = sec_governor()
    doc["twin_json"] = sec_twin_json()
    return doc


JSON_KW = dict(allow_nan=False, separators=(",", ":"), ensure_ascii=True)


def dumps(doc) -> str:
    return json.dumps(doc, **JSON_KW)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="write the JSON here (UTF-8, no BOM); without it only the checks run")
    ap.add_argument("--diag", action="store_true", help="print decision margins and case summaries")
    a = ap.parse_args()
    doc = build()
    if a.diag:
        print("\n".join(DIAG))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, **JSON_KW)
        print("wrote %s (%d bytes, %d sections)" % (a.out, os.path.getsize(a.out), len(doc) - 1))
    else:
        print("built %d bytes (no --out given, nothing written)" % len(dumps(doc).encode("utf-8")))


if __name__ == "__main__":
    main()
