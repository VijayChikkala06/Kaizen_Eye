"""
Kaizen Eye - reference implementation of the scoring maths (numpy, float32).

This file is the SPEC for the TypeScript port in mobile/src/core/patchcore.ts, which mirrors these functions, and
`make_golden.py` writes test vectors (testdata/golden_core.json) so the TypeScript port can be checked number-for-number.

Pipeline (all training-free):
  enrol(frames)   frames = list of [gh, gw, D] feature maps of GOOD parts
                  -> greedy k-centre coreset (memory bank)
                  -> leave-one-frame-out calibration  ->  tau
  score(feat)     -> patch distance map d [gh, gw],  s_norm = max(d) / tau
                  s_norm > 1.0  means "further from normal than anything seen in enrolment"

Measured choices (random-weight ResNet18, synthetic parts; re-check with real photos):
  * frame score = max of the patch distances. AUROC good-vs-defect: max 0.92, 3x3-smoothed max 0.92,
    99th percentile only 0.87 (a 99th percentile over 1,024 patches is the 10th-largest value, so small defects
    barely move it). Smoothing is available (smooth=True) but off by default.
  * calibration = leave-one-out on the final coreset (cheapest). 2-fold and 4-fold hold-out land on the same
    false-reject/detection curve at ~2x the enrolment compute, so they are not worth it by default.
"""
import math

import numpy as np

# App settings (mobile/src/config.ts), chosen with tools/lab/handheld_eval.py: frame score on the 3x3-smoothed
# distance map, outer 10% of the patch grid unscored. The functions below default to the README spec (off).
APP_SMOOTH = True
APP_BORDER_FRACTION = 0.1


def app_border(grid):
    """Unscored patch rows/columns for a grid of `grid` patches per side (JS Math.round semantics)."""
    return int(math.floor(APP_BORDER_FRACTION * grid + 0.5))


def sqdist(a, b):
    """Squared euclidean distances between rows of a [N,D] and b [M,D] -> [N,M]."""
    aa = (a * a).sum(1)[:, None]
    bb = (b * b).sum(1)[None, :]
    return np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)


def greedy_coreset(x, k, start=0):
    """Greedy k-centre. Deterministic: starts at `start`, ties -> lowest index (np.argmax)."""
    n = x.shape[0]
    k = min(k, n)
    sel = np.empty(k, np.int64)
    sel[0] = start
    mind = ((x - x[start]) ** 2).sum(1)
    for i in range(1, k):
        j = int(np.argmax(mind))
        sel[i] = j
        np.minimum(mind, ((x - x[j]) ** 2).sum(1), out=mind)
    return sel


def nn_dist(p, bank, chunk=2048):
    """Distance from every patch in p [P,D] to its nearest neighbour in bank [M,D] -> [P]."""
    out = np.empty(p.shape[0], np.float32)
    for s in range(0, p.shape[0], chunk):
        out[s:s + chunk] = np.sqrt(sqdist(p[s:s + chunk], bank).min(1))
    return out


def smooth3(m):
    """3x3 mean filter with edge replication (suppresses single-patch noise)."""
    p = np.pad(m, 1, mode="edge")
    h, w = m.shape
    return sum(p[i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0


def frame_score(dmap, smooth=False, border=0):
    """Raw frame score = max patch distance (optionally of the 3x3-smoothed map), ignoring the outer `border`
    patch rows/columns. Smoothing is applied to the full map before the border is cut off."""
    m = smooth3(dmap) if smooth else dmap
    if border:
        m = m[border:-border, border:-border]
    return float(m.max())


class Profile:
    def __init__(self, bank, bank_frame, tau, gh, gw, loo_scores, smooth=False, border=0):
        self.bank, self.bank_frame, self.tau = bank, bank_frame, float(tau)
        self.gh, self.gw, self.loo_scores = gh, gw, loo_scores
        self.smooth, self.border = bool(smooth), int(border)   # frame-score settings used at calibration

    def save(self, path):
        np.savez_compressed(path, bank=self.bank, bank_frame=self.bank_frame, tau=self.tau,
                            gh=self.gh, gw=self.gw, loo=np.array(self.loo_scores, np.float32),
                            smooth=self.smooth, border=self.border)

    @staticmethod
    def load(path):
        z = np.load(path)
        return Profile(z["bank"], z["bank_frame"], float(z["tau"]), int(z["gh"]), int(z["gw"]), list(z["loo"]),
                       bool(z["smooth"]) if "smooth" in z else False, int(z["border"]) if "border" in z else 0)


def enrol(frames, k=None, ratio=0.05, margin=1.0, folds=0, min_k=256, smooth=False, border=0):
    """Build the memory bank + threshold from ~20 good frames. Returns a Profile.

    Calibration (default, folds=0): leave-one-frame-out on the final coreset. Each enrolled frame is scored
    against the bank WITHOUT its own entries; tau = max over frames of those scores * margin.
    Optional folds >= 2: K-fold hold-out (bank rebuilt per fold from the other frames). Measured: same
    false-reject/detection curve as leave-one-out, at about twice the compute.
    smooth / border: frame-score settings (see frame_score); the same settings are used again by score().
    """
    gh, gw, d = frames[0].shape
    n = len(frames)
    x = np.concatenate([f.reshape(-1, d) for f in frames]).astype(np.float32)
    fid = np.repeat(np.arange(n), gh * gw)
    k = k or max(min_k, int(ratio * len(x)))          # bank size = max(min_k, floor(ratio * N))
    sel = greedy_coreset(x, k)
    bank, bank_frame = x[sel], fid[sel]

    held = []
    if folds >= 2:
        folds = min(folds, n)
        for fold in range(folds):
            test = [i for i in range(n) if i % folds == fold]
            train = np.isin(fid, [i for i in range(n) if i % folds != fold])
            xt = x[train]
            b = xt[greedy_coreset(xt, max(min_k, int(ratio * len(xt))))]
            for i in test:
                held.append(frame_score(nn_dist(frames[i].reshape(-1, d), b).reshape(gh, gw), smooth, border))
    else:
        for f in range(n):
            keep = bank_frame != f
            if keep.sum():
                held.append(frame_score(nn_dist(frames[f].reshape(-1, d), bank[keep]).reshape(gh, gw), smooth, border))
    tau = max(held) * margin
    return Profile(bank, bank_frame, tau, gh, gw, held, smooth, border)


def score(feat, prof, sensitivity=1.0, smooth=None, border=None):
    """feat [gh,gw,D] -> (patch distance map [gh,gw], normalised score). sensitivity multiplies tau.
    smooth / border default to the settings the profile was calibrated with."""
    gh, gw, d = feat.shape
    smooth = prof.smooth if smooth is None else smooth
    border = prof.border if border is None else border
    dmap = nn_dist(feat.reshape(-1, d).astype(np.float32), prof.bank).reshape(gh, gw)
    return dmap, frame_score(dmap, smooth, border) / (prof.tau * sensitivity)
