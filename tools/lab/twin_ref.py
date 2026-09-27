"""
Kaizen Eye 2 - numpy reference of the Visual Twin maths (docs/verification/twin-spec.md v1, sections 0-14).

Written only from the spec, independently of the Kotlin port in native/core. Used by
  * tools/lab/make_golden_twin.py  -> testdata/golden_twin.json (the Kotlin unit tests check against it),
  * tools/lab/twin_eval.py         -> offline A/B evaluation (imports the functions / classes below).

Precision: float64 everywhere, except stored feature rows (float32 copies, binary16-rounded as the spec says).
Reductions accumulate sequentially in index order (np.cumsum) wherever the golden file can depend on it. Large k-NN /
coreset problems switch to a BLAS path (|a|^2 + |b|^2 - 2ab in float64, ~1e-12 relative agreement); the golden
inputs keep every discrete decision >= 1e-6 (relative) away from flipping, so both paths make the same choices.
`recording()` collects those decision margins (make_golden_twin.py asserts on them).

API overview (all names follow the spec / the Kotlin port's camelCase where they are data):
  stats      percentile, median, mad, robust_sigma, mean, std, auroc
  binary16   f16_bits, f16_from_bits, f16_round, write_f16, read_f16
  binomial   beta_inc, beta_inv, alpha_order, cp_upper, two_sided, alpha_within
  image      downscale_rgba, grey, sharpness, sample, crop_resize, crop_resize_rotated, crop_bytes
  mask       fit_sheet, sq_distance, foreground, erode, dilate, open_close, label_components, segment,
             main_object, crop_square, sanity, geometry_features, analyse_scene
  patches    patch_cov, patch_sets, patch_sets_from_cov, global_descriptor, l2_normalize_patches
  k-NN       knn_min_sq, knn_dist, greedy_coreset
  scoring    masked_smooth, frame_score, score_crop
  teach      TeachFrame, TeachParams, Twin, teach, tau_id, geometry_model, geometry_gate
  verdict    JudgeParams, Judgement, judge, vote, judge_best_of (CANONICAL), score_crop, view_bank
  calibrate  CalSample, calibrate, apply_calibration
  live       Detection, TrackerParams, Tracker, GovernorParams, Governor
  storage    canonical_json, fingerprint, twin_json, save_twin, load_twin, write_f16, read_f16
  rotation   canonical_crops (CANONICAL), augment4 + teach(augment=...) (AUGMENT4)

SPEC-QUESTION list (reading chosen closest to the text; also in the H2a report):
 1. s7.6 coreset: the spec says "add the row with the largest min-distance, ties -> lowest index" without excluding
    already-selected rows. Chosen: every pooled row is a candidate (identical to the legacy patchcore_ref); if all
    min-distances are 0 the lowest index (possibly already selected) is appended again. Golden data never reaches it.
 2. s7.6 "All stored feature rows are rounded through binary16": the negative globals (negatives.f16) are stored
    feature rows too, so they are binary16-rounded before the negative similarities of s7.3 are computed.
 3. s5/s7: the global descriptor g is float64 (sum and normalisation); a stored (keyframe) global is rounded
    float64 -> binary16 directly (single rounding, numpy astype(float16)). Golden data is checked to give the same
    bits via float64 -> float32 -> binary16, so a port that rounds through float32 still matches.
 4. s7.3 the tauIdRule string for the with-negatives rule is not named by the spec: "midpoint" is used here.
 5. s7.7 if every keyframe is skipped in LSO (no lso value) tau is undefined: teach fails with error "NO_LSO".
 6. s9 borderline vote: extra crops are only re-scored (no sanity / identity / geometry gates); an extra crop with an
    empty core is not "available". The chosen crop is the first one in buffer order (primary first) whose s equals
    the median value (for 2 values the higher one); ties therefore go to the lowest index.
 7. s9 NO_CORE: judge() receives the s2.6 steps 1-5 result as `sanity` and applies step 6 (empty core) itself.
 8. s11.3 "consecutive matched frames": unmatched (missed) frames neither extend nor reset the downstream run.
    For direction ANY the upstream side is fixed at the confirming frame. LINE is evaluated before STEADY within a
    frame (the spec only says "the first one wins").
 9. s11.4 "its own maximum sharpness so far" = max over every matched frame of the track, including the frame that
    created it and the current frame.
10. s11.5 the best-crop buffer is updated with the current frame before the triggers are evaluated, so a LINE fire
    may be judged on the firing frame itself.
11. s12 the cooldown clock starts at the first sample whose target is below the current level and is reset by any
    sample whose target is >= the level; the level drops when t - since >= cooldownMs (inclusive) to that sample's
    target. (If the lower target changes value during the cooldown, the clock is NOT restarted; golden data avoids
    this pattern because "the lower target has held continuously" can also be read per target value.)
12. s8 canonical JSON: doubles outside [1e-3, 1e7) print differently in Python ("1e-05") and Kotlin ("1.0E-5");
    canonical_json refuses them. The fingerprint is lowercase hex.
13. s2.6 area fractions are computed as a float64 division area / (W*H) and compared with minAreaFrac/maxAreaFrac
    (not area < frac*W*H); golden data keeps areas away from the boundaries.
14. s10.3 geometry pass count = sanity-ok samples with geoOk (independent of idOk); identity likewise.
15. s10.1 "pos = teach positives U sims of sanity-ok samples" is taken literally: sanity-ok samples that FAILED the
    identity gate still add their (low) sim to pos, which can pull P5(pos) below P99(neg) (overlap = true).
16. s7.4 the geometry bounds are closed intervals (m - kGeo*s <= phi <= m + kGeo*s); area likewise.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------------------------------------------------- enums
PASS, DEFECT, NOT_ENROLLED, REFRAME = "PASS", "DEFECT", "NOT_ENROLLED", "REFRAME"
OK, NO_OBJECT, TOUCHES_BORDER, TOO_SMALL, TOO_LARGE, MULTIPLE, NO_CORE = (
    "OK", "NO_OBJECT", "TOUCHES_BORDER", "TOO_SMALL", "TOO_LARGE", "MULTIPLE", "NO_CORE")
IDENTITY, SHAPE, COVERAGE = "IDENTITY", "SHAPE", "COVERAGE"
SMOOTHED_MAX, TOP1_MEAN = "SMOOTHED_MAX", "TOP1_MEAN"
CORESET, TOPK_VIEWS = "CORESET", "TOPK_VIEWS"
LINE, STEADY = "LINE", "STEADY"
KNN_MASK = 1e30
GEO_FEATURES = ("fill", "aspect", "hu1", "solidity")


# ================================================================================================ decision margins
class Margins:
    """How close each discrete decision came to flipping. gap = |a-b| / max(|a|, |b|, scale)."""

    def __init__(self):
        self.min_gap = {}      # kind -> (gap, detail)
        self.exact = {}        # kind -> number of exact ties (a == b bit for bit)
        self.flags = {}        # kind -> count of hazards that must never happen in golden data

    def flag(self, kind, count=1):
        if count:
            self.flags[kind] = self.flags.get(kind, 0) + int(count)

    def rec(self, kind, a, b, scale=0.0, detail=None):
        a, b = float(a), float(b)
        if a == b:
            self.exact[kind] = self.exact.get(kind, 0) + 1
            return
        if math.isinf(a) or math.isinf(b):
            return
        den = max(abs(a), abs(b), scale)
        gap = abs(a - b) / den if den > 0 else math.inf
        if kind not in self.min_gap or gap < self.min_gap[kind][0]:
            self.min_gap[kind] = (gap, detail)

    def rec_array(self, kind, a, b, scale=0.0, detail=None):
        a = np.asarray(a, np.float64).ravel()
        if a.size == 0:
            return
        b = np.broadcast_to(np.asarray(b, np.float64), a.shape)
        eq = a == b
        if eq.any():
            self.exact[kind] = self.exact.get(kind, 0) + int(eq.sum())
        a, b = a[~eq], b[~eq]
        if a.size:
            den = np.maximum(np.maximum(np.abs(a), np.abs(b)), scale)
            g = np.abs(a - b) / den
            i = int(np.argmin(g))
            self.rec(kind, a[i], b[i], scale, detail)

    def rec_floor(self, kind, v, detail=None):
        """A floor()/round() decision on v: distance of v to the nearest integer (relative to max(1, |v|))."""
        v = np.asarray(v, np.float64).ravel()
        if v.size:
            self.rec_array(kind, v, np.round(v), 1.0, detail)

    def top2(self, kind, values, scale=0.0, detail=None):
        """argmax decision: gap between the best and the second best value (exact tie if the max is shared)."""
        v = np.asarray(values, np.float64).ravel()
        if v.size < 2:
            return
        i = int(np.argmax(v))
        best = v[i]
        rest = np.delete(v, i)
        self.rec(kind, best, float(rest.max()), scale, detail)

    def report(self, limit=1e-6):
        bad = {k: g for k, g in self.min_gap.items() if g[0] < limit}
        return bad


_MARGINS: Optional[Margins] = None


@contextmanager
def recording():
    global _MARGINS
    prev, _MARGINS = _MARGINS, Margins()
    try:
        yield _MARGINS
    finally:
        _MARGINS = prev


def _rec(kind, a, b, scale=0.0, detail=None):
    if _MARGINS is not None:
        _MARGINS.rec(kind, a, b, scale, detail)


def _rec_array(kind, a, b, scale=0.0, detail=None):
    if _MARGINS is not None:
        _MARGINS.rec_array(kind, a, b, scale, detail)


def _rec_floor(kind, v, detail=None):
    if _MARGINS is not None:
        _MARGINS.rec_floor(kind, v, detail)


def _top2(kind, values, scale=0.0, detail=None):
    if _MARGINS is not None:
        _MARGINS.top2(kind, values, scale, detail)


# ======================================================================================================= s0 stats
def seqsum(a) -> float:
    """Sum in index order (float64), like a plain Kotlin loop."""
    a = np.asarray(a, np.float64).ravel()
    return float(np.cumsum(a)[-1]) if a.size else 0.0


def seqdot(a, b) -> float:
    return seqsum(np.asarray(a, np.float64).ravel() * np.asarray(b, np.float64).ravel())


def rowdots(m, v) -> np.ndarray:
    """m @ v with every row's dot product accumulated sequentially (identical rows -> identical results)."""
    m = np.asarray(m, np.float64)
    if m.shape[0] == 0:
        return np.zeros(0)
    return np.cumsum(m * np.asarray(v, np.float64)[None, :], axis=1)[:, -1]


def percentile(x, q) -> float:
    """numpy 'linear' percentile, computed exactly as the spec writes it: h = (n-1)*q/100."""
    s = np.sort(np.asarray(x, np.float64).ravel())
    n = s.size
    if n < 1:
        raise ValueError("percentile of an empty set")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    h = (n - 1) * q / 100.0
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return float(s[lo] + (h - lo) * (s[hi] - s[lo]))


def median(x) -> float:
    return percentile(x, 50.0)


def mad(x) -> float:
    x = np.asarray(x, np.float64).ravel()
    return median(np.abs(x - median(x)))


def robust_sigma(x) -> float:
    return 1.4826 * mad(x)


def mean(x) -> float:
    x = np.asarray(x, np.float64).ravel()
    return seqsum(x) / x.size


def std(x) -> float:
    """Population standard deviation (ddof = 0), two-pass."""
    x = np.asarray(x, np.float64).ravel()
    m = mean(x)
    d = x - m
    return math.sqrt(seqsum(d * d) / x.size)


def auroc(pos, neg) -> float:
    """Mann-Whitney U / (|pos| |neg|), ties count 0.5 (s14)."""
    p = np.asarray(pos, np.float64).ravel()[:, None]
    n = np.asarray(neg, np.float64).ravel()[None, :]
    u = float(np.sum(p > n)) + 0.5 * float(np.sum(p == n))
    return u / (p.size * n.size)


# ==================================================================================================== s0 binary16
def f16_bits(x) -> np.ndarray:
    """IEEE binary16 bits (RNE) of float32 (or float64, single rounding) values."""
    a = np.asarray(x)
    if a.dtype != np.float64:
        a = a.astype(np.float32)
    with np.errstate(over="ignore"):
        return a.astype(np.float16).view(np.uint16)


def f16_from_bits(bits) -> np.ndarray:
    """binary16 bits -> float32 values (exact widening)."""
    return np.asarray(bits, np.uint16).view(np.float16).astype(np.float32)


def f16_round(x) -> np.ndarray:
    """Round through binary16 and return float32 (float32 input: f32->f16; float64 input: f64->f16 directly)."""
    a = np.asarray(x)
    if a.dtype != np.float64:
        a = a.astype(np.float32)
    with np.errstate(over="ignore"):
        return a.astype(np.float16).astype(np.float32)


F16_MAGIC = b"KZF16v1\n"


def f16_file_bytes(arr) -> bytes:
    """s8 .f16 format: magic, int32 LE rows, int32 LE cols (= product of the remaining dims), binary16 LE values."""
    a = np.asarray(arr)
    if a.dtype != np.float64:
        a = a.astype(np.float32)
    rows = int(a.shape[0]) if a.ndim >= 1 else 1
    cols = int(np.prod(a.shape[1:])) if a.ndim >= 2 else (1 if a.ndim == 1 else 1)
    with np.errstate(over="ignore"):
        body = a.astype(np.float16).astype("<f2").tobytes()
    return F16_MAGIC + struct.pack("<ii", rows, cols) + body


def write_f16(path, arr):
    with open(path, "wb") as fh:
        fh.write(f16_file_bytes(arr))


def parse_f16(buf: bytes) -> np.ndarray:
    if buf[:8] != F16_MAGIC:
        raise ValueError("not a KZF16v1 file")
    rows, cols = struct.unpack("<ii", buf[8:16])
    n = rows * cols
    body = buf[16:16 + 2 * n]
    if len(body) != 2 * n:
        raise ValueError("truncated .f16 file")
    return np.frombuffer(body, dtype="<f2").astype(np.float16).reshape(rows, cols)


def read_f16(path) -> np.ndarray:
    with open(path, "rb") as fh:
        return parse_f16(fh.read())


# ================================================================================================= s10.2 binomial
def _betacf(a, b, x, maxit=10000, eps=1e-15, fpmin=1e-300):
    """Continued fraction for the incomplete beta function (Numerical Recipes betacf, modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            return h
    raise ArithmeticError("betacf did not converge (a=%g b=%g x=%g)" % (a, b, x))


def beta_inc(x, a, b) -> float:
    """Regularised incomplete beta I_x(a, b); symmetry I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b
    return bt * _betacf(a, b, x) / a


def beta_inv(p, a, b, tol=1e-12) -> float:
    """x in [0, 1] with I_x(a, b) = p, by bisection until the bracket is narrower than tol (returns its middle)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo >= tol:
        mid = 0.5 * (lo + hi)
        if beta_inc(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def alpha_order(m, conf=0.95) -> float:
    """Order-statistic bound: alpha = 1 - (1 - conf)^(1/m)."""
    return 1.0 - (1.0 - conf) ** (1.0 / m)


def alpha_within(b_used, conf=0.95) -> float:
    return alpha_order(b_used, conf)


def cp_upper(r, n, conf=0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate with r events out of n."""
    if r >= n:
        return 1.0
    if r == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    return beta_inv(conf, r + 1, n - r)


def two_sided(k, n, conf=0.95):
    """Two-sided Clopper-Pearson interval for k successes out of n."""
    lower = 0.0 if k == 0 else beta_inv((1.0 - conf) / 2.0, k, n - k + 1)
    upper = 1.0 if k == n else beta_inv((1.0 + conf) / 2.0, k + 1, n - k)
    return lower, upper


# ======================================================================================================= s1 image
def downscale_rgba(data, w, h, row_stride, f=4) -> np.ndarray:
    """RGBA_8888 (with row stride in bytes) -> f x f box-averaged RGB uint8 [h/f, w/f, 3] (integer maths)."""
    if w % f or h % f:
        raise ValueError("W and H must be divisible by f")
    buf = np.frombuffer(bytes(data), np.uint8) if isinstance(data, (bytes, bytearray)) else np.asarray(data, np.uint8)
    img = buf[:h * row_stride].reshape(h, row_stride)[:, :w * 4].reshape(h, w, 4)[:, :, :3].astype(np.int64)
    s = img.reshape(h // f, f, w // f, f, 3).sum(axis=(1, 3))
    ff = f * f
    return ((s + ff // 2) // ff).astype(np.uint8)    # == floor((s + f^2/2) / f^2) for integer s, also for odd f


def grey(rgb) -> np.ndarray:
    a = np.asarray(rgb, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sharpness(g, x0, y0, x1, y1):
    """Variance of the Laplacian over [x0,x1) x [y0,y1) (s1.3). Returns (value, count of qualifying pixels)."""
    g = np.asarray(g, np.float64)
    hgt, wid = g.shape
    xa, xb = max(x0, 1), min(x1, wid - 1)
    ya, yb = max(y0, 1), min(y1, hgt - 1)
    count = max(0, xb - xa) * max(0, yb - ya)
    if count < 9:
        return 0.0, count
    lap = (g[ya - 1:yb - 1, xa:xb] + g[ya + 1:yb + 1, xa:xb] + g[ya:yb, xa - 1:xb - 1] + g[ya:yb, xa + 1:xb + 1]
           - 4.0 * g[ya:yb, xa:xb])
    m = seqsum(lap) / count
    d = lap - m
    return seqsum(d * d) / count, count


def _bilinear(img, u, v):
    """Vectorised s1.4 sample at continuous points (u, v) (arrays of the same shape) -> [..., C] float64."""
    im = np.asarray(img, np.float64)
    if im.ndim == 2:
        im = im[:, :, None]
    hgt, wid = im.shape[:2]
    x = np.asarray(u, np.float64) - 0.5
    y = np.asarray(v, np.float64) - 0.5
    xf, yf = np.floor(x), np.floor(y)
    fx, fy = x - xf, y - yf
    xi, yi = xf.astype(np.int64), yf.astype(np.int64)
    xa, xb = np.clip(xi, 0, wid - 1), np.clip(xi + 1, 0, wid - 1)
    ya, yb = np.clip(yi, 0, hgt - 1), np.clip(yi + 1, 0, hgt - 1)
    w00 = ((1.0 - fx) * (1.0 - fy))[..., None]
    w01 = (fx * (1.0 - fy))[..., None]
    w10 = ((1.0 - fx) * fy)[..., None]
    w11 = (fx * fy)[..., None]
    return w00 * im[ya, xa] + w01 * im[ya, xb] + w10 * im[yb, xa] + w11 * im[yb, xb]


def sample(img, u, v) -> np.ndarray:
    return _bilinear(img, np.float64(u), np.float64(v))


def crop_resize(img, x0, y0, side, n) -> np.ndarray:
    """n x n x C float64 crop of the square [x0, x0+side) x [y0, y0+side) (s1.4)."""
    j = np.arange(n, dtype=np.float64)
    u = x0 + (j + 0.5) * side / n
    v = y0 + (j + 0.5) * side / n
    uu, vv = np.meshgrid(u, v)          # uu[i, j] = u_j, vv[i, j] = v_i
    return _bilinear(img, uu, vv)


def crop_resize_rotated(img, cx, cy, side, theta, n) -> np.ndarray:
    """Rotated square crop around (cx, cy) (s1.4, rotation challenger s6.5)."""
    j = np.arange(n, dtype=np.float64)
    a = (j + 0.5) * side / n - side / 2.0
    aa, bb = np.meshgrid(a, a)          # aa[i, j] = a_j, bb[i, j] = b_i
    c, s = math.cos(theta), math.sin(theta)
    return _bilinear(img, cx + aa * c - bb * s, cy + aa * s + bb * c)


def crop_bytes(values) -> np.ndarray:
    """clamp(floor(v + 0.5), 0, 255) -> uint8."""
    v = np.asarray(values, np.float64)
    _rec_floor("cropByteRounding", v + 0.5)
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


# ================================================================================================= s2 sheet / mask
@dataclass
class MaskParams:
    sigmaMin: float = 3.0
    kSigma: float = 4.0
    minBlobPx: int = 30
    borderMargin: int = 2
    minAreaFrac: float = 0.01
    maxAreaFrac: float = 0.80
    multipleRatio: float = 0.25
    cropMargin: float = 0.10
    coreThreshold: float = 0.5
    analysisFactor: int = 4


@dataclass
class SheetModel:
    mu: np.ndarray        # [3] float64
    sigma: np.ndarray     # [3] float64
    mad: Optional[np.ndarray] = None


def fit_sheet(frames, roi=None, sigma_min=3.0) -> SheetModel:
    """Per-channel exact median / MAD over the ROI pixels of all K frames (s2.1). roi = (x0, y0, x1, y1) or None."""
    mus, sigmas, mads = [], [], []
    for c in range(3):
        vals = []
        for fr in frames:
            a = np.asarray(fr)
            if roi is not None:
                x0, y0, x1, y1 = roi
                a = a[y0:y1, x0:x1]
            vals.append(a[..., c].astype(np.float64).ravel())
        v = np.concatenate(vals)
        mu = median(v)
        md = median(np.abs(v - mu))
        mus.append(mu)
        mads.append(md)
        sigmas.append(max(1.4826 * md, sigma_min))
    return SheetModel(np.array(mus), np.array(sigmas), np.array(mads))


def sq_distance(frame, sheet: SheetModel) -> np.ndarray:
    """d^2(x, y) = sum_c ((I_c - mu_c) / sigma_c)^2 (float64), channels in order R, G, B."""
    a = np.asarray(frame, np.float64)
    t0 = (a[..., 0] - sheet.mu[0]) / sheet.sigma[0]
    t1 = (a[..., 1] - sheet.mu[1]) / sheet.sigma[1]
    t2 = (a[..., 2] - sheet.mu[2]) / sheet.sigma[2]
    return t0 * t0 + t1 * t1 + t2 * t2


def foreground(d2, k_sigma=4.0) -> np.ndarray:
    thr = k_sigma * k_sigma
    _rec_array("foreground", d2, thr)
    return np.asarray(d2) > thr


def dilate(mask, r=1) -> np.ndarray:
    """(2r+1)^2 square dilation; neighbours outside the image are ignored."""
    m = np.asarray(mask, bool)
    hgt, wid = m.shape
    p = np.zeros((hgt + 2 * r, wid + 2 * r), bool)
    p[r:r + hgt, r:r + wid] = m
    out = np.zeros_like(m)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + hgt, dx:dx + wid]
    return out


def erode(mask, r=1) -> np.ndarray:
    """(2r+1)^2 square erosion; neighbours outside the image are ignored (treated as 1)."""
    m = np.asarray(mask, bool)
    hgt, wid = m.shape
    p = np.ones((hgt + 2 * r, wid + 2 * r), bool)
    p[r:r + hgt, r:r + wid] = m
    out = np.ones_like(m)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out &= p[dy:dy + hgt, dx:dx + wid]
    return out


def open_close(mask):
    """Open (erode, dilate) then close (dilate, erode) with a 3x3 square. Returns (afterOpen, afterClose)."""
    opened = dilate(erode(mask))
    closed = erode(dilate(opened))
    return opened, closed


def label_components(mask):
    """8-connected labelling; labels 1.. in the order of each component's first pixel in a row-major scan.
    Returns (labels int32 [H, W], areas list in label order)."""
    m = np.asarray(mask, bool)
    hgt, wid = m.shape
    labels = np.zeros((hgt, wid), np.int32)
    areas = []
    nxt = 0
    ys, xs = np.nonzero(m)                       # row-major order
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0, x0]:
            continue
        nxt += 1
        labels[y0, x0] = nxt
        stack = [(y0, x0)]
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            for yy in (y - 1, y, y + 1):
                if yy < 0 or yy >= hgt:
                    continue
                for xx in (x - 1, x, x + 1):
                    if 0 <= xx < wid and m[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = nxt
                        stack.append((yy, xx))
        areas.append(area)
    return labels, areas


@dataclass
class Component:
    label: int
    area: int
    minX: int
    minY: int
    maxX: int
    maxY: int
    cx: float
    cy: float
    touchesBorder: bool

    def to_json(self):
        return dict(label=self.label, area=self.area, minX=self.minX, minY=self.minY, maxX=self.maxX, maxY=self.maxY,
                    cx=self.cx, cy=self.cy, touchesBorder=self.touchesBorder)


def _pixels_by_label(labels, n):
    """Row-major pixel coordinates of every label 1..n -> list of (xs, ys) int64 arrays."""
    lab = np.asarray(labels)
    wid = lab.shape[1]
    flat = lab.ravel()
    idx = np.flatnonzero(flat)
    lv = flat[idx]
    order = np.argsort(lv, kind="stable")
    idx, lv = idx[order], lv[order]
    bounds = np.searchsorted(lv, np.arange(1, n + 2))
    out = []
    for k in range(n):
        ii = idx[bounds[k]:bounds[k + 1]]
        out.append((ii % wid, ii // wid))
    return out


def components_of(labels, n, border_margin=2):
    lab = np.asarray(labels)
    hgt, wid = lab.shape
    comps = []
    for k, (xs, ys) in enumerate(_pixels_by_label(lab, n)):
        a = int(xs.size)
        m = border_margin
        tb = bool(np.any((xs < m) | (ys < m) | (xs >= wid - m) | (ys >= hgt - m)))
        comps.append(Component(k + 1, a, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
                               float(int(xs.sum())) / a + 0.5, float(int(ys.sum())) / a + 0.5, tb))
    return comps


@dataclass
class Segmentation:
    d2: np.ndarray
    foreground: np.ndarray
    afterOpen: np.ndarray
    afterClose: np.ndarray
    rawAreas: list
    labels: np.ndarray            # final label map (after speck removal + renumbering)
    components: list              # [Component] in label order


def segment(frame_rgb, sheet: SheetModel, params: MaskParams = MaskParams()) -> Segmentation:
    """s2.2 - s2.4 on an analysis-resolution RGB frame."""
    d2 = sq_distance(frame_rgb, sheet)
    fg = foreground(d2, params.kSigma)
    opened, closed = open_close(fg)
    raw, raw_areas = label_components(closed)
    remap = np.zeros(len(raw_areas) + 1, np.int32)
    nxt = 0
    for i, a in enumerate(raw_areas):
        if a >= params.minBlobPx:
            nxt += 1
            remap[i + 1] = nxt
    labels = remap[raw]
    comps = components_of(labels, nxt, params.borderMargin)
    return Segmentation(d2, fg, opened, closed, list(raw_areas), labels, comps)


def main_object(comps) -> Optional[Component]:
    """Largest component that does not touch the border (ties -> lowest label)."""
    best = None
    for c in comps:
        if c.touchesBorder:
            continue
        if best is None or c.area > best.area:
            best = c
    cands = [c.area for c in comps if not c.touchesBorder]
    _top2("mainObjectArea", cands)
    return best


# ---------------------------------------------------------------------------------------------- s2.7 geometry
@dataclass
class Geometry:
    area: float
    fill: float
    aspect: float
    hu1: float
    solidity: float
    theta: float = 0.0
    hullArea: float = 0.0

    def features(self):
        return dict(area=self.area, fill=self.fill, aspect=self.aspect, hu1=self.hu1, solidity=self.solidity)


def _hull_area_rows(xs, ys) -> float:
    """Convex hull area of the pixel-corner points (only the extreme corners of every row matter)."""
    pts = set()
    for y in np.unique(ys).tolist():
        row = xs[ys == y]
        lo, hi = int(row.min()), int(row.max()) + 1
        pts.update(((lo, y), (lo, y + 1), (hi, y), (hi, y + 1)))
    p = sorted(pts)
    if len(p) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = lower[:-1] + upper[:-1]
    s = 0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def geometry_features(xs, ys) -> Geometry:
    """Scale/rotation/translation-invariant shape features of a pixel set (integer coordinates, row-major order)."""
    xs = np.asarray(xs, np.int64)
    ys = np.asarray(ys, np.int64)
    a = int(xs.size)
    xbar = float(int(xs.sum())) / a
    ybar = float(int(ys.sum())) / a
    dx = xs.astype(np.float64) - xbar
    dy = ys.astype(np.float64) - ybar
    mu20, mu02, mu11 = seqsum(dx * dx), seqsum(dy * dy), seqsum(dx * dy)
    hu1 = (mu20 + mu02) / (float(a) * float(a))
    ca, cc, cb = mu20 / a, mu02 / a, mu11 / a
    mid = (ca + cc) / 2.0
    rad = math.hypot((ca - cc) / 2.0, cb)
    l1, l2 = mid + rad, max(mid - rad, 0.0)
    aspect = 1.0 if l1 == 0 else math.sqrt(l2 / l1)
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    if _MARGINS is not None:   # a nearly (but not exactly) degenerate principal angle makes 'fill' ill-conditioned
        if 0.0 < abs(mu20 - mu02) + abs(2.0 * mu11) < 1e-6 * (mu20 + mu02):
            _MARGINS.flag("thetaIllConditioned")
    ct, st = math.cos(theta), math.sin(theta)
    pu = xs * ct + ys * st
    pv = -xs * st + ys * ct
    wu = float(pu.max() - pu.min() + 1.0)
    wv = float(pv.max() - pv.min() + 1.0)
    fill = a / (wu * wv)
    hull = _hull_area_rows(xs, ys)
    return Geometry(float(a), fill, aspect, hu1, a / hull, theta, hull)


def component_geometry(labels, comp: Component) -> Geometry:
    ys, xs = np.nonzero(np.asarray(labels) == comp.label)
    return geometry_features(xs, ys)


# =================================================================================================== s3 crop square
@dataclass
class Crop:
    x0: float
    y0: float
    side: float

    def to_json(self):
        return dict(x0=self.x0, y0=self.y0, side=self.side)


def crop_square(comp, f, full_w, full_h, margin=0.10) -> Crop:
    """Full-resolution crop square of a component (bbox in analysis pixels)."""
    bx0, by0 = float(f * comp.minX), float(f * comp.minY)
    bx1, by1 = float(f * (comp.maxX + 1)), float(f * (comp.maxY + 1))
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    side = max(bx1 - bx0, by1 - by0) * (1.0 + 2.0 * margin)
    side = min(side, float(full_w), float(full_h))
    x0 = min(max(cx - side / 2.0, 0.0), full_w - side)
    y0 = min(max(cy - side / 2.0, 0.0), full_h - side)
    return Crop(x0, y0, side)


# ==================================================================================================== s4 patch sets
def patch_cov(labels, label, crop: Crop, f, gh, gw) -> np.ndarray:
    """cov[r, c] = fraction of the 4x4 sample points of patch cell (r, c) that land on `label` (s4). [gh, gw]."""
    lab = np.asarray(labels)
    hgt, wid = lab.shape
    k = np.arange(4, dtype=np.float64)
    cc = np.arange(gw, dtype=np.float64)
    rr = np.arange(gh, dtype=np.float64)
    u = crop.x0 + (cc[:, None] + (k[None, :] + 0.5) / 4.0) * crop.side / gw     # [gw, 4]
    v = crop.y0 + (rr[:, None] + (k[None, :] + 0.5) / 4.0) * crop.side / gh     # [gh, 4]
    _rec_floor("covSamplePoint", u / f)
    _rec_floor("covSamplePoint", v / f)
    ax = np.floor(u / f).astype(np.int64)
    ay = np.floor(v / f).astype(np.int64)
    okx = (ax >= 0) & (ax < wid)
    oky = (ay >= 0) & (ay < hgt)
    axc, ayc = np.clip(ax, 0, wid - 1), np.clip(ay, 0, hgt - 1)
    # hit[r, i, c, j]
    hit = (lab[ayc[:, :, None, None], axc[None, None, :, :]] == label)
    hit &= oky[:, :, None, None] & okx[None, None, :, :]
    return hit.sum(axis=(1, 3)).astype(np.float64) / 16.0


@dataclass
class PatchSets:
    cov: np.ndarray    # [gh, gw] float64
    core: np.ndarray   # [gh, gw] bool
    S: np.ndarray      # [gh, gw] bool (3x3 dilation of core)
    B: np.ndarray      # [gh, gw] bool (5x5 dilation of core)


def patch_sets_from_cov(cov, core_threshold=0.5) -> PatchSets:
    c = np.asarray(cov, np.float64)
    core = c >= core_threshold
    return PatchSets(c, core, dilate(core, 1), dilate(core, 2))


def patch_sets(labels, label, crop: Crop, f, gh, gw, core_threshold=0.5) -> PatchSets:
    return patch_sets_from_cov(patch_cov(labels, label, crop, f, gh, gw), core_threshold)


# ============================================================================================ s2.6 sanity + scene
def sanity(chosen: Optional[Component], comps, w, h, params: MaskParams, full_w, full_h,
           core_count: Optional[int] = None) -> str:
    """REFRAME reason (or OK) for the chosen component, checked in the s2.6 order."""
    if chosen is None:
        return NO_OBJECT
    if chosen.touchesBorder:
        return TOUCHES_BORDER
    frac = chosen.area / float(w * h)
    _rec("areaFrac", frac, params.minAreaFrac)
    _rec("areaFrac", frac, params.maxAreaFrac)
    if frac < params.minAreaFrac:
        return TOO_SMALL
    if frac > params.maxAreaFrac:
        return TOO_LARGE
    f = params.analysisFactor
    sq = crop_square(chosen, f, full_w, full_h, params.cropMargin)
    ax0, ay0, aside = sq.x0 / f, sq.y0 / f, sq.side / f
    for d in comps:
        if d.label == chosen.label:
            continue
        need = params.multipleRatio * chosen.area
        _rec("multipleArea", d.area, need)
        if d.area >= need:
            _rec("multipleInside", d.cx, ax0)
            _rec("multipleInside", d.cx, ax0 + aside)
            _rec("multipleInside", d.cy, ay0)
            _rec("multipleInside", d.cy, ay0 + aside)
            if ax0 <= d.cx < ax0 + aside and ay0 <= d.cy < ay0 + aside:
                return MULTIPLE
    if core_count is not None and core_count == 0:
        return NO_CORE
    return OK


def analyse_scene(frame_rgb, sheet: SheetModel, params: MaskParams, gh, gw, full_w=None, full_h=None):
    """Segmentation + geometry + main object + per-component sanity (incl. NO_CORE on a gh x gw grid)."""
    seg = segment(frame_rgb, sheet, params)
    h, w = seg.labels.shape
    f = params.analysisFactor
    full_w = f * w if full_w is None else full_w
    full_h = f * h if full_h is None else full_h
    geoms = []
    per = []
    for c in seg.components:
        geoms.append(component_geometry(seg.labels, c))
        sq = crop_square(c, f, full_w, full_h, params.cropMargin)
        ps = patch_sets(seg.labels, c.label, sq, f, gh, gw, params.coreThreshold)
        cc = int(ps.core.sum())
        per.append(dict(label=c.label, crop=sq, coreCount=cc,
                        sanity=sanity(c, seg.components, w, h, params, full_w, full_h, cc)))
    mo = main_object(seg.components)
    if mo is None:
        san = NO_OBJECT
    else:
        san = per[mo.label - 1]["sanity"]
    return seg, geoms, mo, per, san


# ==================================================================================================== s5 k-NN
def l2_normalize_patches(feat) -> np.ndarray:
    """DINOv2 patch vectors are L2-normalised per patch before anything else (s5); float32 in, float32 out.
    A zero vector stays zero."""
    f = np.asarray(feat, np.float32)
    d = f.shape[-1]
    rows = f.reshape(-1, d).astype(np.float64)
    nrm = np.sqrt(np.cumsum(rows * rows, axis=1)[:, -1])
    out = np.where(nrm[:, None] > 0, rows / np.where(nrm > 0, nrm, 1.0)[:, None], rows)
    return out.astype(np.float32).reshape(f.shape)


def global_descriptor(feat, core=None) -> np.ndarray:
    """g = L2normalise(sum of the core patch vectors) in float64 (all patches if the core is empty)."""
    f = np.asarray(feat, np.float32)
    d = f.shape[-1]
    rows = f.reshape(-1, d).astype(np.float64)
    if core is not None:
        c = np.asarray(core, bool).ravel()
        if c.any():
            rows = rows[c]
    s = np.cumsum(rows, axis=0)[-1]
    nrm = math.sqrt(seqsum(s * s))
    return s / nrm if nrm > 0 else s


_DIRECT_LIMIT = 4_000_000


def _pairwise_sq(q, b):
    """Squared distances [P, M] in float64. Exact (sequential over D) for small problems, BLAS path otherwise."""
    if q.shape[0] * b.shape[0] * q.shape[1] <= _DIRECT_LIMIT:
        diff = q[:, None, :] - b[None, :, :]
        return np.cumsum(diff * diff, axis=2)[:, :, -1]
    qq = np.einsum("ij,ij->i", q, q)[:, None]
    bb = np.einsum("ij,ij->i", b, b)[None, :]
    return np.maximum(qq + bb - 2.0 * (q @ b.T), 0.0)


def knn_min_sq(feats, bank, excluded=None, knn_mask=KNN_MASK, chunk=1024) -> np.ndarray:
    """minSq(p) = min over bank rows of ||f_p - b||^2 (+ knn_mask for excluded rows). float64 [P]."""
    f = np.asarray(feats, np.float32)
    d = f.shape[-1]
    q = f.reshape(-1, d).astype(np.float64)
    b = np.asarray(bank, np.float32).reshape(-1, d).astype(np.float64)
    if b.shape[0] == 0:
        return np.full(q.shape[0], np.inf)
    add = None
    if excluded is not None:
        ex = np.asarray(excluded).astype(bool).ravel()
        if ex.any():
            add = np.where(ex, knn_mask, 0.0)
    out = np.empty(q.shape[0])
    for s in range(0, q.shape[0], chunk):
        d2 = _pairwise_sq(q[s:s + chunk], b)
        if add is not None:
            d2 = d2 + add[None, :]
        out[s:s + chunk] = d2.min(axis=1)
    return out


def knn_dist(feats, bank, excluded=None, knn_mask=KNN_MASK) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, knn_min_sq(feats, bank, excluded, knn_mask)))


def greedy_coreset(x, k, start=0, record=True) -> np.ndarray:
    """Greedy k-centre on float32 rows with float64 squared Euclidean distances; start at `start`, add the row with
    the largest min-distance (ties -> lowest index, every row is a candidate - SPEC-QUESTION 1)."""
    xs = np.asarray(x, np.float32)
    x64 = xs.reshape(xs.shape[0], -1).astype(np.float64)
    n = x64.shape[0]
    k = min(int(k), n)
    sel = np.empty(k, np.int64)
    if k == 0:
        return sel
    exact = x64.size <= _DIRECT_LIMIT
    norms = None if exact else np.einsum("ij,ij->i", x64, x64)
    scale = float(np.median(np.einsum("ij,ij->i", x64, x64))) if (record and _MARGINS is not None) else 0.0

    def dist_to(j):
        if exact:
            diff = x64 - x64[j]
            return np.cumsum(diff * diff, axis=1)[:, -1]
        return np.maximum(norms + norms[j] - 2.0 * (x64 @ x64[j]), 0.0)

    sel[0] = start
    mind = dist_to(start)
    for i in range(1, k):
        j = int(np.argmax(mind))
        if record:
            _top2("coresetArgmax", mind, scale)
        sel[i] = j
        np.minimum(mind, dist_to(j), out=mind)
    return sel


# ================================================================================================= s6 scoring
def masked_smooth(dmap, S) -> np.ndarray:
    """sm(p) = mean of d(q) over the 3x3 neighbourhood of p intersected with S, for p in S (0 elsewhere)."""
    d = np.asarray(dmap, np.float64)
    s = np.asarray(S, bool)
    gh, gw = d.shape
    dp = np.zeros((gh + 2, gw + 2))
    sp = np.zeros((gh + 2, gw + 2))
    dp[1:-1, 1:-1] = np.where(s, d, 0.0)
    sp[1:-1, 1:-1] = s
    tot = np.zeros((gh, gw))
    cnt = np.zeros((gh, gw))
    for dy in range(3):              # row-major neighbour order, adding exact zeros for q not in S
        for dx in range(3):
            tot = tot + dp[dy:dy + gh, dx:dx + gw]
            cnt = cnt + sp[dy:dy + gh, dx:dx + gw]
    return np.where(s, tot / np.maximum(cnt, 1.0), 0.0)


@dataclass
class FrameScore:
    raw: float
    sm: np.ndarray
    peak: tuple            # (row, col, index)
    topN: int


def frame_score(dmap, S, rule=SMOOTHED_MAX, top_fraction=0.01) -> FrameScore:
    d = np.asarray(dmap, np.float64)
    s = np.asarray(S, bool)
    gh, gw = d.shape
    ns = int(s.sum())
    if ns == 0:
        raise ValueError("empty score set S")
    sm = masked_smooth(d, s)
    flat = np.where(s, sm, -np.inf).ravel()
    p = int(np.argmax(flat))
    _top2("peakArgmax", flat[s.ravel()])
    if rule == SMOOTHED_MAX:
        raw, top_n = float(flat[p]), 0
    elif rule == TOP1_MEAN:
        top_n = max(1, int(math.ceil(top_fraction * ns)))
        vals = np.sort(d[s])[::-1]
        raw = seqsum(vals[:top_n]) / top_n
    else:
        raise ValueError("unknown score rule " + str(rule))
    return FrameScore(raw, sm, (p // gw, p % gw, p), top_n)


# ============================================================================================ s7 teach + the Twin
@dataclass
class TeachParams:
    keepPercentile: float = 40.0
    minKept: int = 8
    kMin: int = 16
    kMax: int = 32
    kcEps: float = 0.002
    segments: int = 5
    ratio: float = 0.10
    minK: int = 256
    maxRows: int = 2400
    tauFactor: float = 1.4
    coreThreshold: float = 0.5
    scoreRule: str = SMOOTHED_MAX
    topFraction: float = 0.01
    knnMask: float = KNN_MASK
    kGeo: float = 3.0
    fillFloor: float = 0.04
    aspectFloor: float = 0.04
    solidityFloor: float = 0.03
    hu1FloorFrac: float = 0.04
    areaFactor: float = 2.5
    coverageCut: float = 0.5
    posPercentile: float = 5.0
    negPercentile: float = 99.0
    idMinGap: float = 0.02
    idMadK: float = 3.0


@dataclass
class TeachFrame:
    tMs: int
    sane: bool
    sharpness: Optional[float] = None
    cov: Optional[np.ndarray] = None          # [gh, gw] (or flat P)
    geometry: Optional[dict] = None           # {area, fill, aspect, hu1, solidity} (or a Geometry)
    features: Optional[np.ndarray] = None     # [gh, gw, D] float32 (or flat)


@dataclass
class GeoModel:
    kGeo: float
    meanArea: float
    areaFactor: float
    fill: tuple
    aspect: tuple
    hu1: tuple
    solidity: tuple

    def to_json(self):
        return dict(kGeo=self.kGeo, meanArea=self.meanArea, areaFactor=self.areaFactor, fill=list(self.fill),
                    aspect=list(self.aspect), hu1=list(self.hu1), solidity=list(self.solidity))


def _geo_dict(g):
    if g is None:
        return None
    if isinstance(g, Geometry):
        return g.features()
    return dict(g)


def geometry_model(geoms, params) -> GeoModel:
    """s7.4: per feature mean and max(std, floor); floors fill/aspect/solidity absolute, hu1 relative to its mean."""
    gs = [_geo_dict(g) for g in geoms]
    out = {}
    for name in GEO_FEATURES:
        v = np.array([g[name] for g in gs], np.float64)
        m, s = mean(v), std(v)
        floor = {"fill": params.fillFloor, "aspect": params.aspectFloor, "solidity": params.solidityFloor,
                 "hu1": params.hu1FloorFrac * m}[name]
        out[name] = (m, max(s, floor))
    area = mean(np.array([g["area"] for g in gs], np.float64))
    return GeoModel(params.kGeo, area, params.areaFactor, out["fill"], out["aspect"], out["hu1"], out["solidity"])


def geometry_gate(model: GeoModel, geom):
    """Returns (ok, failing feature names in the order fill, aspect, hu1, solidity, area)."""
    g = _geo_dict(geom)
    fails = []
    for name in GEO_FEATURES:
        m, s = getattr(model, name)
        lo, hi = m - model.kGeo * s, m + model.kGeo * s
        _rec("geometryBound", g[name], lo, abs(m))
        _rec("geometryBound", g[name], hi, abs(m))
        if not (lo <= g[name] <= hi):
            fails.append(name)
    lo, hi = model.meanArea / model.areaFactor, model.areaFactor * model.meanArea
    _rec("geometryArea", g["area"], lo)
    _rec("geometryArea", g["area"], hi)
    if not (lo <= g["area"] <= hi):
        fails.append("area")
    return len(fails) == 0, fails


def tau_id(pos, neg, params) -> dict:
    """s7.3 identity threshold."""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    if neg.size >= 1:
        lo = percentile(pos, params.posPercentile)
        hi = percentile(neg, params.negPercentile)
        margin = lo - hi
        return dict(tauId=(lo + hi) / 2.0, rule="midpoint", margin=margin, overlap=bool(margin <= 0))
    t = percentile(pos, params.posPercentile) - max(params.idMinGap, params.idMadK * 1.4826 * mad(pos))
    return dict(tauId=t, rule="no-negatives", margin=None, overlap=None)


class TeachError(Exception):
    def __init__(self, code, **info):
        super().__init__(code)
        self.code = code
        self.info = info


@dataclass
class Twin:
    gh: int
    gw: int
    dim: int
    bank: np.ndarray            # [M, D] float32, binary16-rounded
    bankKf: np.ndarray          # [M] int
    bankSeg: np.ndarray         # [M] int
    globals: np.ndarray         # [K, D] float32, binary16-rounded
    kfFeats: np.ndarray         # [K, P, D] float32, binary16-rounded
    kfCov: np.ndarray           # [K, P] float32, binary16-rounded
    negatives: np.ndarray       # [Nneg, D] float32, binary16-rounded
    keyframes: list             # [{index, frame, tMs, segment, sharpness}]
    lso: list
    lsoKeyframes: list
    lsoMode: str
    positives: list
    negSims: list
    tau: float
    tauTeach: float
    tauFactor: float
    tauId: float
    tauIdRule: str
    identityMargin: Optional[float]
    identityOverlap: Optional[bool]
    geometry: GeoModel
    coverageCut: float
    calibrated: bool = False
    sensitivity: float = 1.0
    calMax: Optional[float] = None
    segmentsUsed: int = 0
    teachInfo: dict = field(default_factory=dict)   # accepted, kept, cut, kc*, coreset indices, t0, t1, counts

    def kf_patch_sets(self, k, core_threshold=0.5) -> PatchSets:
        return patch_sets_from_cov(self.kfCov[k].astype(np.float64).reshape(self.gh, self.gw), core_threshold)


def _as_fmap(features, gh, gw, d):
    return np.asarray(features, np.float32).reshape(gh * gw, d)


def teach(frames: Sequence[TeachFrame], params: TeachParams = TeachParams(), negatives=None,
          knn: Optional[Callable] = None, gh=None, gw=None, augment: Optional[Callable] = None) -> Twin:
    """Build a Twin from a recording (s7). Frames in time order; features [gh, gw, D] float32 of sane frames.
    knn(feats[P,D], bank[M,D], excluded[M] bool | None, knn_mask) -> minSq[P] replaces the exact search (e.g. a
    float32 / LiteRT-graph emulation). augment(k, frame_index) -> [(features, cov), ...] adds extra maps to the
    pool for keyframe k (the AUGMENT4 challenger of s6.5: the keyframe crop at 90/180/270 degrees); their B-set
    rows are pooled right after the keyframe's own rows, with bankKf = k."""
    knn = knn or knn_min_sq
    accepted = [i for i, fr in enumerate(frames) if fr.sane]
    if not accepted:
        raise TeachError("NOT_ENOUGH_FRAMES", accepted=[], kept=[], cut=None)
    sharp = [float(frames[i].sharpness) for i in accepted]
    cut = percentile(sharp, params.keepPercentile)
    for v in sharp:
        _rec("keepSharpness", v, cut)
    kept = [i for i in accepted if frames[i].sharpness >= cut]
    if len(kept) < params.minKept:
        raise TeachError("NOT_ENOUGH_FRAMES", accepted=accepted, kept=kept, cut=cut)

    f0 = np.asarray(frames[kept[0]].features, np.float32)
    if gh is None:
        gh, gw = int(f0.shape[0]), int(f0.shape[1])
    d = int(f0.shape[-1])
    feats = {i: _as_fmap(frames[i].features, gh, gw, d) for i in kept}
    covs = {i: np.asarray(frames[i].cov, np.float64).reshape(gh, gw) for i in kept}
    psets = {i: patch_sets_from_cov(covs[i], params.coreThreshold) for i in kept}

    # 4. keyframes: greedy k-centre on the kept frames' global descriptors, delta = 1 - g_a . g_b
    g = np.stack([global_descriptor(feats[i], psets[i].core) for i in kept])
    nk = len(kept)
    sh_kept = [float(frames[i].sharpness) for i in kept]
    first = int(np.argmax(sh_kept))
    _top2("kcFirstSharpness", sh_kept)
    sel = [first]
    chosen = np.zeros(nk, bool)
    chosen[first] = True
    mind = 1.0 - rowdots(g, g[first])
    kc_min, stop, nxt = [], None, None
    while True:
        if len(sel) == params.kMax:
            stop = "KMAX"
        elif len(sel) == nk:
            stop = "ALL"
        cand = np.flatnonzero(~chosen)
        if cand.size:
            vals = mind[cand]
            jj = int(np.argmax(vals))
            j, best = int(cand[jj]), float(vals[jj])
        if stop is not None:
            nxt = best if cand.size else None
            break
        _top2("kcArgmax", vals, 1.0)
        if len(sel) >= params.kMin:
            _rec("kcEps", best, params.kcEps, 1.0)
            if best < params.kcEps:
                stop, nxt = "EPS", best
                break
        sel.append(j)
        chosen[j] = True
        kc_min.append(best)
        mind = np.minimum(mind, 1.0 - rowdots(g, g[j]))
    kf_frames = [kept[s] for s in sel]
    kk_count = len(kf_frames)

    # 5. segments
    t0, t1 = int(frames[kept[0]].tMs), int(frames[kept[-1]].tMs)
    nseg = params.segments
    seg = []
    for fi in kf_frames:
        x = nseg * (int(frames[fi].tMs) - t0) / (t1 - t0 + 1e-9)
        if x != 0.0 and round(x) < nseg:        # t = t0 is exact; x ~ B (last kept frame) is clamped to B-1
            _rec_floor("segmentFloor", x)
        seg.append(min(nseg - 1, int(math.floor(x))))
    seg = np.array(seg, np.int64)
    segments_used = int(np.unique(seg).size)

    # 6. bank: pool B-set patches of the keyframes (index order, row-major), greedy coreset, then binary16
    rows, row_kf = [], []
    for k, fi in enumerate(kf_frames):
        idx = np.flatnonzero(psets[fi].B.ravel())
        rows.append(feats[fi][idx])
        row_kf.extend([k] * idx.size)
        for af, ac in (augment(k, fi) if augment is not None else []):
            aidx = np.flatnonzero(patch_sets_from_cov(np.asarray(ac, np.float64).reshape(gh, gw),
                                                      params.coreThreshold).B.ravel())
            rows.append(_as_fmap(af, gh, gw, d)[aidx])
            row_kf.extend([k] * aidx.size)
    pooled = np.concatenate(rows).astype(np.float32)
    row_kf = np.array(row_kf, np.int64)
    n_pool = pooled.shape[0]
    _rec_floor("bankRatioFloor", params.ratio * n_pool)
    kbank = min(params.maxRows, max(params.minK, int(math.floor(params.ratio * n_pool))))
    kbank = min(kbank, n_pool)
    core_idx = greedy_coreset(pooled, kbank, start=0)
    bank = f16_round(pooled[core_idx])
    bank_kf = row_kf[core_idx]
    bank_seg = seg[bank_kf]

    kf_feats = np.stack([f16_round(feats[fi]) for fi in kf_frames])                 # [K, P, D]
    kf_cov = np.stack([f16_round(covs[fi].ravel().astype(np.float32)) for fi in kf_frames])
    g_kf64 = g[sel]
    if _MARGINS is not None:   # SPEC-QUESTION 3: f64->f16 must equal f64->f32->f16 on golden data
        a = f16_bits(g_kf64)
        b = f16_bits(g_kf64.astype(np.float32))
        _MARGINS.flag("globalDoubleRounding", int(np.sum(a != b)))
    globals16 = f16_round(g_kf64)                                                   # [K, D]

    # 7. leave-segment-out scores (fallback: leave-one-keyframe-out)
    fallback = segments_used < 2
    lso, lso_kf = [], []
    for k in range(kk_count):
        allowed = (bank_kf != k) if fallback else (bank_seg != seg[k])
        if not allowed.any():
            continue
        dm = np.sqrt(np.maximum(0.0, knn(kf_feats[k], bank, ~allowed, params.knnMask))).reshape(gh, gw)
        ps = patch_sets_from_cov(kf_cov[k].astype(np.float64).reshape(gh, gw), params.coreThreshold)
        lso.append(frame_score(dm, ps.S, params.scoreRule, params.topFraction).raw)
        lso_kf.append(k)
    if not lso:
        raise TeachError("NO_LSO", accepted=accepted, kept=kept, cut=cut)
    tau_teach = params.tauFactor * max(lso)

    # 8. identity positives (binary16 globals) and tau_id
    g16 = globals16.astype(np.float64)
    positives = []
    for k in range(kk_count):
        others = [j for j in range(kk_count) if (j != k if fallback else seg[j] != seg[k])]
        positives.append(max(seqdot(g16[k], g16[j]) for j in others))
    neg = np.zeros((0, d), np.float32) if negatives is None else np.asarray(negatives, np.float32).reshape(-1, d)
    neg16 = f16_round(neg)
    neg_sims = [max(seqdot(neg16[i], g16[k]) for k in range(kk_count)) for i in range(neg16.shape[0])]
    tid = tau_id(positives, neg_sims, params)

    # 9. geometry model from all kept frames
    geo = geometry_model([frames[i].geometry for i in kept], params)

    keyframes = [dict(index=k, frame=fi, tMs=int(frames[fi].tMs), segment=int(seg[k]),
                      sharpness=float(frames[fi].sharpness)) for k, fi in enumerate(kf_frames)]
    info = dict(accepted=accepted, kept=kept, sharpnessCut=cut, kcMinDist=kc_min, kcStop=stop, kcNext=nxt,
                t0=t0, t1=t1, pooledRows=n_pool, bankRows=int(kbank), coresetIndices=[int(i) for i in core_idx],
                framesSeen=len(frames), framesAccepted=len(accepted), framesKept=len(kept),
                durationMs=int(frames[-1].tMs) - int(frames[0].tMs) if frames else 0)
    return Twin(gh=gh, gw=gw, dim=d, bank=bank, bankKf=bank_kf, bankSeg=bank_seg, globals=globals16,
                kfFeats=kf_feats, kfCov=kf_cov, negatives=neg16, keyframes=keyframes, lso=lso, lsoKeyframes=lso_kf,
                lsoMode="KEYFRAME" if fallback else "SEGMENT", positives=positives, negSims=neg_sims,
                tau=tau_teach, tauTeach=tau_teach, tauFactor=params.tauFactor, tauId=tid["tauId"],
                tauIdRule=tid["rule"], identityMargin=tid["margin"], identityOverlap=tid["overlap"], geometry=geo,
                coverageCut=params.coverageCut, segmentsUsed=segments_used, teachInfo=info)


# ================================================================================================ s9 verdicts
@dataclass
class JudgeParams:
    coreThreshold: float = 0.5
    topFraction: float = 0.01
    topKViews: int = 3
    knnMask: float = KNN_MASK
    voteLo: float = 1.0
    voteHi: float = 1.15
    voteMaxExtra: int = 2


@dataclass
class CropScore:
    sets: PatchSets
    dmap: np.ndarray        # [gh, gw]
    sm: np.ndarray
    raw: float
    s: float
    peak: tuple
    coreCount: int
    anomalousCount: int
    anomalousFraction: float
    views: Optional[list]
    g: np.ndarray


def view_bank(twin: Twin, g_query, k, core_threshold=0.5):
    """TOPK_VIEWS: the k keyframes with the highest g_q . g_k (ties -> lowest index) and their B-set rows."""
    g16 = twin.globals.astype(np.float64)
    sims = np.array([seqdot(g16[j], g_query) for j in range(g16.shape[0])])
    order = sorted(range(len(sims)), key=lambda j: (-sims[j], j))
    if _MARGINS is not None and len(order) > k:
        _MARGINS.rec("topkViews", sims[order[k - 1]], sims[order[k]], 1.0)
    views = order[:k]
    rows = []
    for j in views:
        ps = twin.kf_patch_sets(j, core_threshold)
        rows.append(twin.kfFeats[j][np.flatnonzero(ps.B.ravel())])
    return views, np.concatenate(rows)


def score_crop(twin: Twin, feat, cov, sensitivity=1.0, score_rule=SMOOTHED_MAX, bank_rule=CORESET,
               params: JudgeParams = JudgeParams(), knn=None) -> Optional[CropScore]:
    """s6 for one crop. None when the core is empty (NO_CORE)."""
    knn = knn or knn_min_sq
    gh, gw, d = twin.gh, twin.gw, twin.dim
    sets = patch_sets_from_cov(np.asarray(cov, np.float64).reshape(gh, gw), params.coreThreshold)
    core_count = int(sets.core.sum())
    if core_count == 0:
        return None
    fm = _as_fmap(feat, gh, gw, d)
    g = global_descriptor(fm, sets.core)
    views = None
    if bank_rule == CORESET:
        bank = twin.bank
    elif bank_rule == TOPK_VIEWS:
        views, bank = view_bank(twin, g, params.topKViews, params.coreThreshold)
    else:
        raise ValueError("unknown bank rule " + str(bank_rule))
    dm = np.sqrt(np.maximum(0.0, knn(fm, bank, None, params.knnMask))).reshape(gh, gw)
    fs = frame_score(dm, sets.S, score_rule, params.topFraction)
    thr = twin.tau * sensitivity
    s = fs.raw / thr
    core_sm = fs.sm[sets.core]
    _rec_array("anomalousPatch", core_sm, thr, 0.0)
    an = int(np.sum(core_sm > thr))
    return CropScore(sets, dm, fs.sm, fs.raw, s, fs.peak, core_count, an, an / core_count, views, g)


@dataclass
class Judgement:
    verdict: str
    reason: Optional[str] = None
    sim: Optional[float] = None
    idOk: Optional[bool] = None
    geoOk: Optional[bool] = None
    geoFailures: Optional[list] = None
    raw: Optional[float] = None
    s: Optional[float] = None
    peak: Optional[tuple] = None
    coreCount: Optional[int] = None
    anomalousFraction: Optional[float] = None
    areaPct: Optional[float] = None
    dmap: Optional[np.ndarray] = None
    views: Optional[list] = None
    vote: Optional[dict] = None
    finalVerdict: Optional[str] = None
    score: Optional[CropScore] = None


def _decide(s, a, coverage_cut):
    """Steps 3-5 of s9 for a score s and anomalous fraction a."""
    _rec("passThreshold", s, 1.0, 1.0)
    if s <= 1.0:
        return PASS, None
    _rec("coverageCut", a, coverage_cut, 1.0)
    if a > coverage_cut:
        return NOT_ENROLLED, COVERAGE
    return DEFECT, None


def judge(twin: Twin, feat, cov, geometry, sanity_reason=OK, sensitivity=1.0, score_rule=SMOOTHED_MAX,
          bank_rule=CORESET, params: JudgeParams = JudgeParams(), knn=None) -> Judgement:
    """Verdict for one presentation (s9 steps 1-5, no voting)."""
    if sanity_reason != OK:
        return Judgement(REFRAME, sanity_reason, finalVerdict=REFRAME)
    sc = score_crop(twin, feat, cov, sensitivity, score_rule, bank_rule, params, knn)
    if sc is None:
        return Judgement(REFRAME, NO_CORE, finalVerdict=REFRAME)
    g16 = twin.globals.astype(np.float64)
    sim = max(seqdot(g16[k], sc.g) for k in range(g16.shape[0]))
    _rec("identity", sim, twin.tauId, 1.0)
    id_ok = sim >= twin.tauId
    geo_ok, fails = geometry_gate(twin.geometry, geometry)
    if not id_ok:
        verdict, reason = NOT_ENROLLED, IDENTITY
    elif not geo_ok:
        verdict, reason = NOT_ENROLLED, SHAPE
    else:
        verdict, reason = _decide(sc.s, sc.anomalousFraction, twin.coverageCut)
    return Judgement(verdict, reason, sim, bool(id_ok), bool(geo_ok), fails, sc.raw, sc.s, sc.peak, sc.coreCount,
                     sc.anomalousFraction, 100.0 * sc.anomalousFraction, sc.dmap, sc.views, None, verdict, sc)


def judge_best_of(twin: Twin, candidates, geometry, sanity_reason=OK, sensitivity=1.0, score_rule=SMOOTHED_MAX,
                  bank_rule=CORESET, params: JudgeParams = JudgeParams(), knn=None) -> Judgement:
    """Rotation = CANONICAL (s6.5): judge each (features, cov) candidate - the crop at theta and at theta + pi -
    and keep the one with the lower raw score (ties -> the first candidate). Empty-core candidates are skipped."""
    best = None
    for feat, cov in candidates:
        j = judge(twin, feat, cov, geometry, sanity_reason, sensitivity, score_rule, bank_rule, params, knn)
        if j.verdict == REFRAME and sanity_reason != OK:
            return j
        if j.raw is None:
            best = best or j
            continue
        if best is None or best.raw is None or j.raw < best.raw:
            best = j
    return best


def vote(twin: Twin, primary: Judgement, extra_crops, sensitivity=1.0, score_rule=SMOOTHED_MAX, bank_rule=CORESET,
         params: JudgeParams = JudgeParams(), knn=None) -> Judgement:
    """Borderline voting (s9): returns the judgement with .vote / .finalVerdict filled when a vote ran."""
    if primary.verdict != DEFECT:
        return primary
    _rec("voteWindow", primary.s, params.voteLo, 1.0)
    _rec("voteWindow", primary.s, params.voteHi, 1.0)
    if not (params.voteLo < primary.s <= params.voteHi):
        return primary
    crops = [primary.score]
    for fc in list(extra_crops)[:params.voteMaxExtra]:
        feat, cov = fc
        sc = score_crop(twin, feat, cov, sensitivity, score_rule, bank_rule, params, knn)
        if sc is not None:
            crops.append(sc)
    svals = [c.s for c in crops]
    order = sorted(svals)
    med = order[len(order) // 2]
    if _MARGINS is not None:
        for i in range(len(order) - 1):
            _MARGINS.rec("voteMedian", order[i], order[i + 1], 1.0)
    chosen = svals.index(med)
    c = crops[chosen]
    v, reason = _decide(med, c.anomalousFraction, twin.coverageCut)
    primary.vote = dict(s=svals, chosen=chosen, finalS=med, verdict=v, reason=reason, peak=c.peak,
                        anomalousFraction=c.anomalousFraction, areaPct=100.0 * c.anomalousFraction)
    primary.finalVerdict = v
    return primary


# ========================================================================================= s10 calibration
@dataclass
class CalSample:
    sanityOk: bool
    idOk: bool = False
    geoOk: bool = False
    raw: Optional[float] = None
    sim: Optional[float] = None
    geometry: Optional[dict] = None


@dataclass
class CalParams:
    conf: float = 0.95
    kGeo: float = 3.0
    fillFloor: float = 0.04
    aspectFloor: float = 0.04
    solidityFloor: float = 0.03
    hu1FloorFrac: float = 0.04
    areaFactor: float = 2.5
    posPercentile: float = 5.0
    negPercentile: float = 99.0
    idMinGap: float = 0.02
    idMadK: float = 3.0
    latencyWindow: int = 512


def latency_stats(latencies, window=512):
    lat = list(latencies)[-window:]
    if not lat:
        return None, None
    return percentile(lat, 50.0), percentile(lat, 95.0)


def calibrate(tau_teach, positives, neg_sims, segments_used, samples: Sequence[CalSample],
              params: CalParams = CalParams(), sensitivities=(), latencies=()) -> dict:
    """s10.1 / s10.3: tau_cal (never lowered), re-derived tau_id and geometry model, certificate numbers."""
    n = len(samples)
    sane = [s for s in samples if s.sanityOk]
    valid = [s for s in sane if s.idOk and s.geoOk]
    m = len(valid)
    cal_max = max(s.raw for s in valid) if valid else None
    tau_cal = tau_teach if cal_max is None else max(tau_teach, cal_max)
    alpha = alpha_order(m, params.conf) if m else None
    pos = list(positives) + [s.sim for s in sane]
    tid = tau_id(pos, neg_sims, params)
    geo = geometry_model([s.geometry for s in sane], params) if sane else None

    def gate(k, nn):
        return dict(k=k, n=nn, rejections=nn - k, upper=cp_upper(nn - k, nn, params.conf) if nn else None)

    p50, p95 = latency_stats(latencies, params.latencyWindow)
    cert = dict(sanity=gate(len(sane), n), identity=gate(sum(1 for s in sane if s.idOk), len(sane)),
                geometry=gate(sum(1 for s in sane if s.geoOk), len(sane)), identityMargin=tid["margin"],
                withinPart=alpha_within(segments_used, params.conf), latencyP50=p50, latencyP95=p95)
    valid_flags = [bool(cal_max is None or tau_cal * sens >= cal_max) for sens in sensitivities]
    if cal_max is not None:
        for sens in sensitivities:
            _rec("certificateValid", tau_cal * sens, cal_max, 1.0)
    return dict(n=n, nSanityOk=len(sane), m=m, tauCal=tau_cal, calMax=cal_max, alpha=alpha, identity=tid,
                geometry=geo, certificate=cert, valid=valid_flags)


def apply_calibration(twin: Twin, cal: dict) -> Twin:
    twin.tau = cal["tauCal"]
    twin.calMax = cal["calMax"]
    twin.calibrated = True
    twin.tauId = cal["identity"]["tauId"]
    twin.tauIdRule = cal["identity"]["rule"]
    twin.identityMargin = cal["identity"]["margin"]
    twin.identityOverlap = cal["identity"]["overlap"]
    if cal["geometry"] is not None:
        twin.geometry = cal["geometry"]
    return twin


# ============================================================================================ s11 tracker
@dataclass
class Detection:
    area: int
    minX: int
    minY: int
    maxX: int
    maxY: int
    cx: float
    cy: float
    touchesBorder: bool
    sharpness: float


@dataclass
class TrackerParams:
    w: int = 320
    h: int = 180
    gateBase: float = 12.0
    gateArea: float = 0.75
    gateSpeed: float = 0.5
    velAlpha: float = 0.5
    minHits: int = 2
    maxMissed: int = 5
    lineEnabled: bool = True
    lineAxis: str = "X"
    linePos: Optional[float] = None       # default 0.5 * w (or 0.5 * h for axis Y)
    lineDirection: str = "ANY"
    crossFrames: int = 2
    steadyEnabled: bool = True
    vStill: float = 0.02
    holdMs: float = 500.0
    holdSharpRatio: float = 0.7
    bestK: int = 3


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    vx: float
    vy: float
    area: int
    bbox: tuple
    hits: int
    missed: int
    lastT: float
    confirmed: bool = False
    judged: bool = False
    maxSharp: float = 0.0
    upstream: Optional[int] = None
    lineDisabled: bool = False
    seenUpstream: bool = False
    downCount: int = 0
    stillSince: Optional[float] = None
    buffer: list = field(default_factory=list)     # [(q, frameIndex)], best first

    def state(self):
        return dict(id=self.id, cx=self.cx, cy=self.cy, vx=self.vx, vy=self.vy, hits=self.hits, missed=self.missed,
                    confirmed=self.confirmed, judged=self.judged)


class Tracker:
    """s11: greedy association, virtual photo-eye (LINE), steady hold (STEADY), best-crop buffer."""

    def __init__(self, params: TrackerParams = TrackerParams()):
        self.p = params
        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame_index = -1
        lp = params.linePos
        if lp is None:
            lp = 0.5 * (params.w if params.lineAxis == "X" else params.h)
        self.line_pos = lp

    def _side(self, tr: Track):
        coord = tr.cx if self.p.lineAxis == "X" else tr.cy
        _rec("lineSide", coord, self.line_pos, 1.0)
        return 1 if coord > self.line_pos else -1

    def _quality(self, det: Detection):
        p = self.p
        dist = math.hypot(det.cx - p.w / 2.0, det.cy - p.h / 2.0)
        centrality = 1.0 - min(1.0, dist / (0.5 * math.hypot(p.w, p.h)))
        return det.sharpness * centrality

    def _buffer_add(self, tr: Track, q, fi):
        if _MARGINS is not None:
            for bq, _ in tr.buffer:
                _MARGINS.rec("bestCropQuality", q, bq, 1e-9)
        tr.buffer.append((q, fi))
        tr.buffer.sort(key=lambda e: (-e[0], e[1]))
        del tr.buffer[self.p.bestK:]

    def _line(self, tr: Track):
        p = self.p
        side = self._side(tr)
        if tr.upstream is None:            # first confirmed frame
            if p.lineDirection == "ANY":
                tr.upstream = side
            else:
                tr.upstream = -1 if p.lineDirection == "POSITIVE" else 1
                if side != tr.upstream:
                    tr.lineDisabled = True
        if tr.lineDisabled:
            return False
        if side == tr.upstream:
            tr.seenUpstream = True
            tr.downCount = 0
            return False
        if tr.seenUpstream:
            tr.downCount += 1
            return tr.downCount >= p.crossFrames
        return False

    def _steady(self, tr: Track, det: Detection, t):
        p = self.p
        speed = math.hypot(tr.vx, tr.vy)
        _rec("steadySpeed", speed, p.vStill, 1e-3)
        _rec("steadySharp", det.sharpness, p.holdSharpRatio * tr.maxSharp)
        ok = (not det.touchesBorder) and speed < p.vStill and det.sharpness >= p.holdSharpRatio * tr.maxSharp
        if not ok:
            tr.stillSince = None
            return False
        if tr.stillSince is None:
            tr.stillSince = t
        return t - tr.stillSince >= p.holdMs

    def step(self, t, detections: Sequence[Detection]):
        """Process one frame. Returns dict(assign=[trackId per detection], events=[...])."""
        p = self.p
        self.frame_index += 1
        fi = self.frame_index
        live = sorted(self.tracks, key=lambda tr: tr.id)
        pairs = []
        for tr in live:
            dt = t - tr.lastT
            px, py = tr.cx + tr.vx * dt, tr.cy + tr.vy * dt
            gate = p.gateBase + p.gateArea * math.sqrt(tr.area) + p.gateSpeed * math.hypot(tr.vx, tr.vy) * dt
            for j, det in enumerate(detections):
                dist = math.hypot(det.cx - px, det.cy - py)
                _rec("gate", dist, gate, 1.0)
                if dist <= gate:
                    pairs.append((dist, tr.id, j))
        pairs.sort()
        if _MARGINS is not None:            # only pairs that compete (same track or same detection) matter
            for ia in range(len(pairs)):
                for ib in range(ia + 1, len(pairs)):
                    a, b = pairs[ia], pairs[ib]
                    if a[1] == b[1] or a[2] == b[2]:
                        _MARGINS.rec("pairOrder", a[0], b[0], 1.0)
        by_id = {tr.id: tr for tr in live}
        t_used, d_used, match = set(), set(), {}
        for dist, tid, j in pairs:
            if tid in t_used or j in d_used:
                continue
            t_used.add(tid)
            d_used.add(j)
            match[tid] = j
        assign = [0] * len(detections)
        fired, exited = [], []
        for tr in live:
            if tr.id not in match:
                continue
            j = match[tr.id]
            det = detections[j]
            assign[j] = tr.id
            dt = t - tr.lastT
            if dt != 0:
                nvx, nvy = (det.cx - tr.cx) / dt, (det.cy - tr.cy) / dt
                if tr.hits == 1:
                    tr.vx, tr.vy = nvx, nvy
                else:
                    a = p.velAlpha
                    tr.vx, tr.vy = a * nvx + (1.0 - a) * tr.vx, a * nvy + (1.0 - a) * tr.vy
            tr.cx, tr.cy = det.cx, det.cy
            tr.area = det.area
            tr.bbox = (det.minX, det.minY, det.maxX, det.maxY)
            tr.hits += 1
            tr.missed = 0
            tr.lastT = t
            tr.confirmed = tr.hits >= p.minHits
            tr.maxSharp = max(tr.maxSharp, det.sharpness)
            if not tr.confirmed:
                continue
            if not det.touchesBorder:
                self._buffer_add(tr, self._quality(det), fi)
            line_fire = self._line(tr) if p.lineEnabled else False
            steady_fire = self._steady(tr, det, t) if p.steadyEnabled else False
            if tr.judged:
                continue
            trig = LINE if line_fire else (STEADY if steady_fire else None)
            if trig is None:
                continue
            tr.judged = True
            if trig == STEADY:
                judge_frame = fi
            else:
                judge_frame = tr.buffer[0][1] if tr.buffer else None
            fired.append(dict(type="FIRED", trackId=tr.id, trigger=trig, judgeFrame=judge_frame,
                              buffer=[e[1] for e in tr.buffer],
                              reframe=None if judge_frame is not None else TOUCHES_BORDER))
        keep = []
        for tr in live:
            if tr.id in match:
                keep.append(tr)
                continue
            tr.missed += 1
            if tr.missed > p.maxMissed:
                exited.append(dict(type="EXITED", trackId=tr.id, judged=tr.judged, confirmed=tr.confirmed))
            else:
                keep.append(tr)
        for j, det in enumerate(detections):
            if j in d_used:
                continue
            tr = Track(self.next_id, det.cx, det.cy, 0.0, 0.0, det.area, (det.minX, det.minY, det.maxX, det.maxY),
                       1, 0, t, confirmed=1 >= p.minHits, maxSharp=det.sharpness)
            self.next_id += 1
            assign[j] = tr.id
            keep.append(tr)
        self.tracks = sorted(keep, key=lambda tr: tr.id)
        events = sorted(fired, key=lambda e: e["trackId"]) + sorted(exited, key=lambda e: e["trackId"])
        return dict(assign=assign, events=events)


# ============================================================================================ s12 governor
@dataclass
class GovernorParams:
    cooldownMs: int = 30000
    statusL2: int = 3
    statusL1: int = 2
    headroomL2: float = 0.85
    headroomL1: float = 0.70
    fpsL0: int = 30
    fpsL1: int = 24
    fpsL2: int = 15
    fpsIdle: int = 2


class Governor:
    """s12 thermal governor: rise immediately, fall after the lower target held for cooldownMs."""
    VLM = ("AUTO", "ON_TAP", "PAUSED")

    def __init__(self, params: GovernorParams = GovernorParams()):
        self.p = params
        self.level = 0
        self.low_since = None
        self.log = []

    def target(self, status, headroom):
        p = self.p
        hr = None if headroom is None or (isinstance(headroom, float) and math.isnan(headroom)) else headroom
        if status >= p.statusL2 or (hr is not None and hr > p.headroomL2):
            return 2
        if status == p.statusL1 or (hr is not None and hr >= p.headroomL1):
            return 1
        return 0

    def step(self, t, status, headroom, idle):
        p = self.p
        tgt = self.target(status, headroom)
        changed = False
        if tgt > self.level:
            self.level, self.low_since, changed = tgt, None, True
        elif tgt < self.level:
            if self.low_since is None:
                self.low_since = t
            if t - self.low_since >= p.cooldownMs:
                self.level, self.low_since, changed = tgt, None, True
        else:
            self.low_since = None
        if changed:
            self.log.append(dict(tMs=t, level=self.level, status=status, headroom=headroom, idle=idle))
        lv = self.level
        fps = p.fpsIdle if idle else (p.fpsL0, p.fpsL1, p.fpsL2)[lv]
        return dict(target="L%d" % tgt, level="L%d" % lv, changed=changed, fps=fps, voting=lv == 0,
                    vlm=self.VLM[lv], banner=lv == 2)


# ============================================================================================ s8 storage
def _canon_value(v):
    if isinstance(v, bool) or v is None or isinstance(v, str) or isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite number in canonical JSON")
        if v != 0.0 and not (1e-3 <= abs(v) < 1e7):
            raise ValueError("SPEC-QUESTION 12: double %r outside [1e-3, 1e7) has no agreed canonical form" % v)
        return v
    if isinstance(v, dict):
        return {str(k): _canon_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon_value(x) for x in v]
    raise TypeError("unsupported type in canonical JSON: %r" % type(v))


def canonical_json(obj) -> str:
    """Keys sorted at every level, no whitespace, shortest round-trip numbers (4.0 stays 4.0, 320 stays 320)."""
    return json.dumps(_canon_value(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(pipeline) -> str:
    return hashlib.sha256(canonical_json(pipeline).encode("utf-8")).hexdigest()


def twin_json(twin: Twin, pipeline: dict, twin_id="lab", name="lab twin", created_at_ms=0) -> dict:
    """The s8 twin.json object for a Twin (keyframe file names follow keyframes/kf_NN.jpg)."""
    ti = twin.teachInfo
    return {
        "schema": 1, "id": twin_id, "name": name, "createdAtMs": created_at_ms,
        "pipeline": pipeline, "fingerprint": fingerprint(pipeline),
        "teach": {"framesSeen": ti.get("framesSeen", 0), "framesAccepted": ti.get("framesAccepted", 0),
                  "framesKept": ti.get("framesKept", 0), "keyframes": len(twin.keyframes),
                  "segmentsUsed": twin.segmentsUsed, "durationMs": ti.get("durationMs", 0),
                  "bankRows": int(twin.bank.shape[0]), "pooledRows": ti.get("pooledRows", 0)},
        "keyframes": [{"index": k["index"], "tMs": k["tMs"], "segment": k["segment"], "sharpness": k["sharpness"],
                       "file": "keyframes/kf_%02d.jpg" % k["index"]} for k in twin.keyframes],
        "bankKf": [int(v) for v in twin.bankKf],
        "lso": [float(v) for v in twin.lso],
        "positives": [float(v) for v in twin.positives],
        "negatives": {"count": int(twin.negatives.shape[0]), "similarities": [float(v) for v in twin.negSims]},
        "thresholds": {
            "tau": twin.tau, "tauTeach": twin.tauTeach, "tauFactor": twin.tauFactor, "calibrated": twin.calibrated,
            "tauId": twin.tauId, "tauIdRule": twin.tauIdRule, "identityMargin": twin.identityMargin,
            "coverageCut": twin.coverageCut, "sensitivity": twin.sensitivity,
            "geometry": {"kGeo": twin.geometry.kGeo, "meanArea": twin.geometry.meanArea,
                         "fill": list(twin.geometry.fill), "aspect": list(twin.geometry.aspect),
                         "hu1": list(twin.geometry.hu1), "solidity": list(twin.geometry.solidity)}},
        "certificate": None,
    }


def save_twin(directory, twin: Twin, pipeline: dict, **meta):
    """Write twin.json + the .f16 files (keyframe JPEGs are the caller's business)."""
    tmp = str(directory) + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "twin.json"), "w", encoding="utf-8") as fh:
        doc = twin_json(twin, pipeline, **meta)
        doc["bankSeg"] = [int(v) for v in twin.bankSeg]          # lab extra (derivable from bankKf + keyframes)
        json.dump(doc, fh, indent=1)
    write_f16(os.path.join(tmp, "bank.f16"), twin.bank)
    write_f16(os.path.join(tmp, "globals.f16"), twin.globals)
    write_f16(os.path.join(tmp, "kf_feats.f16"), twin.kfFeats)
    write_f16(os.path.join(tmp, "kf_cov.f16"), twin.kfCov)
    write_f16(os.path.join(tmp, "negatives.f16"), twin.negatives.reshape(-1, twin.dim))
    if os.path.exists(directory):
        import shutil
        shutil.rmtree(directory)
    os.replace(tmp, directory)


def load_twin(directory) -> Twin:
    with open(os.path.join(directory, "twin.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    pl = doc["pipeline"]
    gh, gw, d = pl["gh"], pl["gw"], pl["dim"]
    th = doc["thresholds"]
    ge = th["geometry"]
    bank = read_f16(os.path.join(directory, "bank.f16")).astype(np.float32)
    kf = [dict(index=k["index"], frame=None, tMs=k["tMs"], segment=k["segment"], sharpness=k["sharpness"])
          for k in doc["keyframes"]]
    seg_of = np.array([k["segment"] for k in kf], np.int64)
    bank_kf = np.array(doc["bankKf"], np.int64)
    return Twin(gh=gh, gw=gw, dim=d, bank=bank, bankKf=bank_kf, bankSeg=seg_of[bank_kf],
                globals=read_f16(os.path.join(directory, "globals.f16")).astype(np.float32),
                kfFeats=read_f16(os.path.join(directory, "kf_feats.f16")).astype(np.float32).reshape(-1, gh * gw, d),
                kfCov=read_f16(os.path.join(directory, "kf_cov.f16")).astype(np.float32),
                negatives=read_f16(os.path.join(directory, "negatives.f16")).astype(np.float32),
                keyframes=kf, lso=doc["lso"], lsoKeyframes=[], lsoMode="", positives=doc["positives"],
                negSims=doc["negatives"]["similarities"], tau=th["tau"], tauTeach=th["tauTeach"],
                tauFactor=th["tauFactor"], tauId=th["tauId"], tauIdRule=th["tauIdRule"],
                identityMargin=th["identityMargin"], identityOverlap=None,
                geometry=GeoModel(ge["kGeo"], ge["meanArea"], ge.get("areaFactor", 2.5), tuple(ge["fill"]),
                                  tuple(ge["aspect"]), tuple(ge["hu1"]), tuple(ge["solidity"])),
                coverageCut=th["coverageCut"], calibrated=th["calibrated"], sensitivity=th["sensitivity"],
                segmentsUsed=doc["teach"]["segmentsUsed"])


# ================================================================================ s6.5 rotation challengers
def canonical_crops(full_rgb, crop: Crop, theta, n):
    """CANONICAL: rotated crops around the crop-square centre at theta and theta + pi (float values)."""
    cx, cy = crop.x0 + crop.side / 2.0, crop.y0 + crop.side / 2.0
    return (crop_resize_rotated(full_rgb, cx, cy, crop.side, theta, n),
            crop_resize_rotated(full_rgb, cx, cy, crop.side, theta + math.pi, n))


def augment4(crop_u8):
    """AUGMENT4: the crop at 0, 90, 180, 270 degrees (counter-clockwise), for adding keyframes to the bank."""
    return [np.rot90(crop_u8, k).copy() for k in range(4)]
