#!/usr/bin/env python3
r"""
Kaizen Eye 2 - offline A/B harness for the Visual Twin (docs/verification/twin-spec.md v1) on an evaluation export
folder written by the phone (format: docs/verification/export-format.md).

  .venv\Scripts\python tools\lab\twin_eval.py --export <export dir> --backbone r18 --out out\eval_r18
  .venv\Scripts\python tools\lab\twin_eval.py --export <export dir> --backbone dinov2 --out out\eval_dinov2
  options: --model <tflite> (override the backbone file), --variants all|default, --threads 4, --sensitivity 1.0

What it does (all spec maths comes from tools/lab/twin_ref.py: teach, judge, calibrate, patch_cov, crop_resize,
crop_resize_rotated, geometry_features, sharpness, auroc, two_sided, ...):
  1. Loads manifest + teach/ + parts/ + negatives/. Decodes every crop whose sanity is OK and resizes it to the
     backbone input N with the spec crop-resize (1.4, full crop, side = cropSize) when cropSize != N.
  2. Recomputes features on the laptop with the backbone TFLite (ai_edge_litert): r18 = the shipped
     mobile/assets/models/backbone_r18_320.tflite, dinov2 = tools/dinov2/out/dinov2_s14_448_fp32.tflite with
     per-patch L2 normalisation (spec 5). cov for the backbone grid (spec 4) and the geometry features (2.7) are
     recomputed from the exported component mask + crop square + analysis factor and checked against the phone's
     values; sharpness (1.3) is recomputed on the unrotated backbone-size crop.
  3. Variants: score rule {SMOOTHED_MAX, TOP1_MEAN} x bank {CORESET, TOPK_VIEWS} x rotation {NONE, CANONICAL,
     AUGMENT4} x tau rule {LSO = tauFactor x max leave-segment-out score (spec 7.7), CALIBRATED = spec 10.1 over the
     `calib` parts (tau never lowered, tau_id and geometry re-derived); CALIBRATED only when calib parts exist}.
     ROTATION APPROXIMATION: the phone re-crops the full-resolution frame with cropResizeRotated (spec 6.5); the
     laptop only has the stored axis-aligned crop, so it rotates that crop about its centre (spec 1.4 bilinear
     sampling of the decoded JPEG, padded with the sheet colour manifest.sheet.mu where the rotated square leaves the
     stored one). This resamples twice and replaces real sheet texture / neighbours in the corners by a flat colour.
     cov of a rotated crop = the spec-4 4x4 sampling pushed through the same rotation (exact, from the mask).
  4. Judges every part with the spec-9 verdict rules (no borderline voting) and reports per variant:
     AUROC good-vs-defect (normalised score s), AUROC good-vs-wrong (identity sim; look-alikes separately),
     PASS rate on held-out good (and on rotated), NOT_ENROLLED rate on wrong (look-alikes separately), DEFECT rate
     on seeded defects, each rate with a two-sided 95 % Clopper-Pearson interval (twin_ref.two_sided).
     REFRAME counts as a failure in every rate (it is not the wanted verdict); AUROCs use the parts with a score.
     Winner: highest AUROC good-vs-defect, then highest AUROC good-vs-wrong; exact ties -> the variant with the
     fewest non-default choices (defaults SMOOTHED_MAX, CORESET, NONE, LSO), i.e. defaults win ties.
  Outputs: <out>/report.md, <out>/results.json (all metrics + per-part verdicts of the default and the winner).

SPEC-QUESTION (choices made here, not in the spec / twin_ref):
  a. CANONICAL cov: the spec rotates the crop (6.5) but defines cov only for axis-aligned squares (4); the same 4x4
     sample points are pushed through the rotation (rotated_cov below; identical to twin_ref.patch_cov at 0 rad).
  b. CANONICAL sharpness for the teach keep step: computed on the unrotated crop (so all rotation variants keep the
     same frames). A query's winning orientation (lower raw) also supplies sim, a and peak.
  c. AUGMENT4: twin_ref.teach has no hook, so teach steps 1-5 and 9 come from twin_ref.teach and steps 6-8 are
     re-run here with twin_ref primitives on 4K keyframes (each keyframe at 0/90/180/270 deg via twin_ref.augment4,
     rotated copies inherit the segment; LSO / positives over all 4K; KEYFRAME fallback excludes all copies of the
     same source keyframe). Queries are not rotated. With TOPK_VIEWS the rotated copies are ordinary views.
  d. CALIBRATED: calib parts are judged by the uncalibrated Twin of the same variant (same score / bank / rotation),
     then twin_ref.calibrate + apply_calibration; the evaluation parts are judged by the calibrated Twin.
"""
import argparse
import copy
import glob
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import twin_ref as tr  # noqa: E402  (the spec reference)

BACKBONES = {
    "r18": dict(path=os.path.join(ROOT, "mobile", "assets", "models", "backbone_r18_320.tflite"), l2=False),
    "dinov2": dict(path=os.path.join(ROOT, "tools", "dinov2", "out", "dinov2_s14_448_fp32.tflite"), l2=True),
}
SCORE_RULES = ["SMOOTHED_MAX", "TOP1_MEAN"]
BANK_RULES = ["CORESET", "TOPK_VIEWS"]
ROTATIONS = ["NONE", "CANONICAL", "AUGMENT4"]
TAU_RULES = ["LSO", "CALIBRATED"]
DEFAULT = ("SMOOTHED_MAX", "CORESET", "NONE", "LSO")
EVAL_LABELS = ["good", "defect", "wrong", "rotated", "lookalike"]


# ================================================================================================== export loading
def rle_decode(mask):
    """{"x","y","w","h","rle"} -> uint8 [h, w] window (runs alternate 0/1, starting with a 0-run)."""
    out = np.zeros(int(mask["w"]) * int(mask["h"]), np.uint8)
    pos, val = 0, 0
    for r in mask["rle"]:
        r = int(r)
        if val:
            out[pos:pos + r] = 1
        pos += r
        val ^= 1
    if pos != out.size:
        raise ValueError(f"mask rle sums to {pos}, expected {out.size}")
    return out.reshape(int(mask["h"]), int(mask["w"]))


def load_export(path):
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("schema") != 1:
        raise SystemExit(f"unsupported export schema {man.get('schema')}")
    items = {"teach": [], "parts": [], "negatives": []}
    for folder in items:
        for jp in sorted(glob.glob(os.path.join(path, folder, "*.json"))):
            with open(jp, encoding="utf-8") as fh:
                d = json.load(fh)
            d["_name"] = os.path.splitext(os.path.basename(jp))[0]
            d["_folder"] = folder
            img = os.path.splitext(jp)[0] + ".jpg"
            d["_jpg"] = img if os.path.exists(img) else None
            items[folder].append(d)
    items["teach"].sort(key=lambda d: (d["tMs"], d.get("index", 0), d["_name"]))
    return man, items


def decode(path):
    with open(path, "rb") as fh:
        return np.asarray(Image.open(io.BytesIO(fh.read())).convert("RGB"), np.uint8)


# ======================================================================================================== backbone
class Backbone:
    def __init__(self, path, l2, threads=4):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.size = int(self.inp["shape"][1])
        outs = self.it.get_output_details()
        self.out = [o for o in outs if len(o["shape"]) == 4][0]      # the patch map (DINOv2 also has [1, D] CLS)
        _, self.gh, self.gw, self.dim = (int(v) for v in self.out["shape"])
        self.l2 = l2
        self.path = path
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, crop_u8):
        """uint8 [N, N, 3] -> float32 [gh, gw, D] (per-patch L2-normalised for DINOv2, spec 5 / 0)."""
        t = time.perf_counter()
        self.it.set_tensor(self.inp["index"], np.ascontiguousarray(crop_u8, np.float32)[None])
        self.it.invoke()
        f = self.it.get_tensor(self.out["index"])[0].astype(np.float32).copy()
        if self.l2:
            n = np.sqrt((f.astype(np.float64) ** 2).sum(-1, keepdims=True))
            f = np.where(n > 0, f / np.where(n > 0, n, 1.0), f).astype(np.float32)
        self.calls += 1
        self.seconds += time.perf_counter() - t
        return f


# ============================================================================================= crops, cov, geometry
def resize_to(img, n):
    cs = img.shape[0]
    if cs == n:
        return img
    return tr.crop_bytes(tr.crop_resize(img, 0.0, 0.0, float(cs), n))


def rotated_crop(img, mu, theta, n):
    """cropResizeRotated about the stored crop's centre, same side (spec 6.5 on the stored crop, not the frame);
    points outside the stored square read the sheet colour mu (padding)."""
    cs = img.shape[0]
    p = int(math.ceil(cs * (math.sqrt(2.0) - 1.0) / 2.0)) + 2
    pad = np.empty((cs + 2 * p, cs + 2 * p, 3), np.float64)
    pad[...] = np.asarray(mu, np.float64)[None, None, :]
    pad[p:p + cs, p:p + cs] = img
    return tr.crop_bytes(tr.crop_resize_rotated(pad, p + cs / 2.0, p + cs / 2.0, float(cs), theta, n))


def rotated_cov(lab, crop, theta, f, gh, gw):
    """SPEC-QUESTION a: spec-4 cov with the sample points pushed through the rotation of cropResizeRotated."""
    k = (np.arange(4, dtype=np.float64) + 0.5) / 4.0
    a = (np.arange(gw, dtype=np.float64)[:, None] + k[None, :]).ravel() * crop.side / gw - crop.side / 2.0
    b = (np.arange(gh, dtype=np.float64)[:, None] + k[None, :]).ravel() * crop.side / gh - crop.side / 2.0
    aa, bb = np.meshgrid(a, b)                       # aa[i, j] = a_j (columns), bb[i, j] = b_i (rows)
    c, s = math.cos(theta), math.sin(theta)
    cx, cy = crop.x0 + crop.side / 2.0, crop.y0 + crop.side / 2.0
    u, v = cx + aa * c - bb * s, cy + aa * s + bb * c
    ax, ay = np.floor(u / f).astype(np.int64), np.floor(v / f).astype(np.int64)
    hgt, wid = lab.shape
    ok = (ax >= 0) & (ax < wid) & (ay >= 0) & (ay < hgt)
    hit = np.zeros(ax.shape, bool)
    hit[ok] = lab[ay[ok], ax[ok]] == 1
    return hit.reshape(gh, 4, gw, 4).sum(axis=(1, 3)).astype(np.float64) / 16.0


class Item:
    """One exported crop (teach frame, negative or part) with everything recomputed on the laptop."""

    def __init__(self, d, man):
        self.d = d
        self.name = d["_folder"] + "/" + d["_name"]
        self.sanity = d.get("sanity", tr.NO_OBJECT)
        self.label = d.get("label")
        self.ok = self.sanity == tr.OK and d.get("_jpg") is not None and d.get("mask") is not None
        self.img = None
        self.lab = None
        self.geom = None
        if d.get("mask") is not None and d.get("crop") is not None:
            f = int(man["analysisFactor"])
            aw, ah = int(man["frameW"]) // f, int(man["frameH"]) // f
            m = d["mask"]
            win = rle_decode(m)
            self.lab = np.zeros((ah, aw), np.int32)
            x0, y0 = int(m["x"]), int(m["y"])
            self.lab[y0:y0 + win.shape[0], x0:x0 + win.shape[1]] = win
            ys, xs = np.nonzero(self.lab == 1)
            self.geom = tr.geometry_features(xs, ys) if xs.size else None
            c = d["crop"]
            self.crop = tr.Crop(float(c["x0"]), float(c["y0"]), float(c["side"]))
        self.cache = {}

    def image(self):
        if self.img is None:
            self.img = decode(self.d["_jpg"])
        return self.img

    def geometry(self):
        return None if self.geom is None else self.geom.features()


class FeatureStore:
    """Features / cov per (item, orientation) for one backbone; orientation keys: 'r0', 'r90', 'r180', 'r270',
    'can0' (theta), 'can180' (theta + pi)."""

    def __init__(self, bb, man):
        self.bb, self.man = bb, man
        self.f = int(man["analysisFactor"])
        self.mu = man.get("sheet", {}).get("mu", [0.0, 0.0, 0.0])

    def get(self, it: Item, key):
        if key in it.cache:
            return it.cache[key]
        n, gh, gw = self.bb.size, self.bb.gh, self.bb.gw
        if key.startswith("r"):
            base = resize_to(it.image(), n)
            k = int(key[1:]) // 90
            crop = np.rot90(base, k).copy() if k else base
            if "cov_r0" not in it.cache:
                it.cache["cov_r0"] = tr.patch_cov(it.lab, 1, it.crop, self.f, gh, gw)
            cov = np.rot90(it.cache["cov_r0"], k).copy() if k else it.cache["cov_r0"]
        else:
            theta = it.geom.theta + (math.pi if key == "can180" else 0.0)
            crop = rotated_crop(it.image(), self.mu, theta, n)
            cov = rotated_cov(it.lab, it.crop, theta, self.f, gh, gw)
        feat = self.bb(crop)
        it.cache[key] = (feat, cov)
        if key == "r0":
            it.cache["sharp"] = tr.sharpness(tr.grey(crop), 0, 0, n, n)[0]
        return it.cache[key]

    def sharpness(self, it):
        if "sharp" not in it.cache:
            self.get(it, "r0")
        return it.cache["sharp"]


# ========================================================================================================= teach
class KnnCache:
    """k-NN results shared across variants that use the same (query features, bank) pair (judging only)."""

    def __init__(self):
        self.memo = {}
        self.hits = self.misses = 0

    @staticmethod
    def _h(a):
        a = np.ascontiguousarray(np.asarray(a, np.float32))
        return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest() + str(a.shape)

    def __call__(self, feats, bank, excluded=None, knn_mask=tr.KNN_MASK):
        if excluded is not None:
            return tr.knn_min_sq(feats, bank, excluded, knn_mask)
        key = (self._h(feats), self._h(bank))
        if key in self.memo:
            self.hits += 1
            return self.memo[key]
        self.misses += 1
        v = tr.knn_min_sq(feats, bank, None, knn_mask)
        self.memo[key] = v
        return v


def teach_frames(store, teach_items, orient):
    frames = []
    for it in teach_items:
        sane = it.ok
        if sane:
            feat, cov = store.get(it, orient)
            if not tr.patch_sets_from_cov(cov).core.any():
                sane = False                                  # NO_CORE on this grid / orientation
        if not sane:
            frames.append(tr.TeachFrame(int(it.d["tMs"]), False))
            continue
        frames.append(tr.TeachFrame(int(it.d["tMs"]), True, store.sharpness(it), cov, it.geometry(), feat))
    return frames


def negative_globals(store, neg_items, orient):
    out = []
    for it in neg_items:
        if not it.ok:
            continue
        feat, cov = store.get(it, orient)
        core = tr.patch_sets_from_cov(cov).core
        out.append(tr.global_descriptor(feat, core if core.any() else None))
    return np.array(out, np.float32) if out else None


def augment_twin(twin, store, teach_items, frames, params):
    """SPEC-QUESTION c: AUGMENT4 - keyframes at 0/90/180/270 deg added before the coreset; teach steps 6-8 re-run
    with twin_ref primitives on the 4K keyframes."""
    gh, gw, d = twin.gh, twin.gw, twin.dim
    K = len(twin.keyframes)
    src_items = [teach_items[kf["frame"]] for kf in twin.keyframes]
    seg0 = np.array([kf["segment"] for kf in twin.keyframes], np.int64)
    feats, covs, src = [], [], []
    for rot in (0, 90, 180, 270):
        for k, it in enumerate(src_items):
            feat, cov = store.get(it, "r%d" % rot)
            feats.append(np.asarray(feat, np.float32).reshape(gh * gw, d))
            covs.append(np.asarray(cov, np.float64).reshape(gh, gw))
            src.append(k)
    src = np.array(src, np.int64)
    seg = seg0[src]
    kk = len(feats)
    psets = [tr.patch_sets_from_cov(c, params.coreThreshold) for c in covs]
    rows, row_kf = [], []
    for j in range(kk):
        idx = np.flatnonzero(psets[j].B.ravel())
        rows.append(feats[j][idx])
        row_kf.extend([j] * idx.size)
    pooled = np.concatenate(rows).astype(np.float32)
    row_kf = np.array(row_kf, np.int64)
    n_pool = pooled.shape[0]
    kbank = min(params.maxRows, max(params.minK, int(math.floor(params.ratio * n_pool))), n_pool)
    core_idx = tr.greedy_coreset(pooled, kbank, start=0, record=False)
    bank = tr.f16_round(pooled[core_idx])
    bank_kf = row_kf[core_idx]
    bank_seg = seg[bank_kf]
    kf_feats = np.stack([tr.f16_round(f) for f in feats])
    kf_cov = np.stack([tr.f16_round(c.ravel().astype(np.float32)) for c in covs])
    g = np.stack([tr.global_descriptor(feats[j], psets[j].core) for j in range(kk)])
    globals16 = tr.f16_round(g)
    fallback = twin.segmentsUsed < 2
    lso, lso_kf = [], []
    for j in range(kk):
        allowed = (src[bank_kf] != src[j]) if fallback else (bank_seg != seg[j])
        if not allowed.any():
            continue
        dm = np.sqrt(np.maximum(0.0, tr.knn_min_sq(kf_feats[j], bank, ~allowed, params.knnMask))).reshape(gh, gw)
        ps = tr.patch_sets_from_cov(kf_cov[j].astype(np.float64).reshape(gh, gw), params.coreThreshold)
        lso.append(tr.frame_score(dm, ps.S, params.scoreRule, params.topFraction).raw)
        lso_kf.append(j)
    tau_teach = params.tauFactor * max(lso)
    g16 = globals16.astype(np.float64)
    positives = []
    for j in range(kk):
        others = [i for i in range(kk) if (src[i] != src[j] if fallback else seg[i] != seg[j])]
        positives.append(max(tr.seqdot(g16[j], g16[i]) for i in others) if others else 1.0)
    neg16 = twin.negatives.reshape(-1, d)
    neg_sims = [max(tr.seqdot(neg16[i], g16[j]) for j in range(kk)) for i in range(neg16.shape[0])]
    tid = tr.tau_id(positives, neg_sims, params)
    tw = copy.copy(twin)
    tw.bank, tw.bankKf, tw.bankSeg = bank, bank_kf, bank_seg
    tw.globals, tw.kfFeats, tw.kfCov = globals16, kf_feats, kf_cov
    tw.keyframes = [dict(twin.keyframes[src[j]], index=j, rotation=90 * (j // K)) for j in range(kk)]
    tw.lso, tw.lsoKeyframes, tw.positives, tw.negSims = lso, lso_kf, positives, neg_sims
    tw.tau = tw.tauTeach = tau_teach
    tw.tauId, tw.tauIdRule = tid["tauId"], tid["rule"]
    tw.identityMargin, tw.identityOverlap = tid["margin"], tid["overlap"]
    tw.teachInfo = dict(twin.teachInfo, pooledRows=n_pool, bankRows=int(kbank), augmented=True)
    return tw


# ========================================================================================================= judge
def judge_item(tw, store, it, rotation, score_rule, bank_rule, sensitivity, knn):
    """-> twin_ref Judgement (CANONICAL: the orientation with the lower raw wins, SPEC-QUESTION b)."""
    if not it.ok:
        return tr.judge(tw, None, None, None, sanity_reason=it.sanity if it.sanity != tr.OK else tr.NO_OBJECT)
    keys = ("can0", "can180") if rotation == "CANONICAL" else ("r0",)
    best = None
    for key in keys:
        feat, cov = store.get(it, key)
        j = tr.judge(tw, feat, cov, it.geometry(), tr.OK, sensitivity, score_rule, bank_rule, knn=knn)
        if best is None or (j.raw is not None and (best.raw is None or j.raw < best.raw)):
            best = j
    return best


def calibrated(tw, store, calib_items, rotation, score_rule, bank_rule, sensitivity, knn):
    samples = []
    for it in calib_items:
        j = judge_item(tw, store, it, rotation, score_rule, bank_rule, sensitivity, knn)
        ok = j.verdict != tr.REFRAME
        samples.append(tr.CalSample(ok, bool(j.idOk), bool(j.geoOk), j.raw, j.sim, it.geometry() if ok else None))
    cal = tr.calibrate(tw.tauTeach, tw.positives, tw.negSims, tw.segmentsUsed, samples)
    return tr.apply_calibration(copy.copy(tw), cal), cal


# ======================================================================================================= metrics
def rate(k, n):
    if n == 0:
        return None
    lo, hi = tr.two_sided(k, n)
    return dict(k=k, n=n, rate=k / n, lo=lo, hi=hi)


def metrics(results):
    """results: list of (label, Judgement)."""
    by = {lab: [j for l2, j in results if l2 == lab] for lab in EVAL_LABELS}

    def scores(lab, attr):
        return [getattr(j, attr) for j in by[lab] if getattr(j, attr) is not None]

    def auc(pos, neg):
        return tr.auroc(pos, neg) if pos and neg else None

    def count(lab, verdict):
        return rate(sum(1 for j in by[lab] if j.verdict == verdict), len(by[lab]))

    return dict(
        n={lab: len(by[lab]) for lab in EVAL_LABELS},
        auroc_defect=auc(scores("defect", "s"), scores("good", "s")),
        auroc_identity=auc(scores("good", "sim"), scores("wrong", "sim")),
        auroc_identity_lookalike=auc(scores("good", "sim"), scores("lookalike", "sim")),
        pass_good=count("good", tr.PASS), pass_rotated=count("rotated", tr.PASS),
        notenrolled_wrong=count("wrong", tr.NOT_ENROLLED), notenrolled_lookalike=count("lookalike", tr.NOT_ENROLLED),
        defect_defect=count("defect", tr.DEFECT))


def non_defaults(v):
    return sum(1 for a, b in zip(v, DEFAULT) if a != b)


def pick_winner(table):
    """table: {variant tuple: metrics}. Highest AUROC defect, then identity; exact ties -> fewest non-defaults,
    then the order of SCORE_RULES x BANK_RULES x ROTATIONS x TAU_RULES (defaults first)."""
    order = {v: i for i, v in enumerate(table)}

    def key(v):
        m = table[v]
        a = m["auroc_defect"] if m["auroc_defect"] is not None else -1.0
        b = m["auroc_identity"] if m["auroc_identity"] is not None else -1.0
        return (-a, -b, non_defaults(v), order[v])
    return min(table, key=key)


def fmt_rate(r):
    if r is None:
        return "n/a"
    return f"{r['k']}/{r['n']} = {100.0 * r['rate']:.0f} % [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"


def fmt_auc(a):
    return "n/a" if a is None else f"{a:.3f}"


def jv(j):
    return dict(verdict=j.verdict, reason=j.reason, s=None if j.s is None else float(j.s),
                raw=None if j.raw is None else float(j.raw), sim=None if j.sim is None else float(j.sim),
                a=None if j.anomalousFraction is None else float(j.anomalousFraction))


# ========================================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="r18")
    ap.add_argument("--model", help="backbone .tflite (default: the one for --backbone)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", choices=["all", "default"], default="all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sensitivity", type=float, default=1.0)
    ap.add_argument("--title", default="", help="free text for the report header (e.g. SYNTHETIC)")
    a = ap.parse_args()
    t0 = time.time()
    man, raw = load_export(a.export)
    cfg = BACKBONES[a.backbone]
    bb = Backbone(a.model or cfg["path"], cfg["l2"], a.threads)
    store = FeatureStore(bb, man)
    teach_items = [Item(d, man) for d in raw["teach"]]
    neg_items = [Item(d, man) for d in raw["negatives"]]
    part_items = [Item(d, man) for d in raw["parts"]]
    calib_items = [it for it in part_items if it.label == "calib"]
    eval_items = [it for it in part_items if it.label in EVAL_LABELS]
    print(f"[load] {a.export}: teach {len(teach_items)} (ok {sum(it.ok for it in teach_items)}), negatives "
          f"{len(neg_items)}, parts {len(part_items)} (calib {len(calib_items)}, eval {len(eval_items)}); backbone "
          f"{os.path.basename(bb.path)} {bb.size}px -> {bb.gh}x{bb.gw}x{bb.dim}")

    # consistency of the export with the laptop recomputation (cov on the phone grid, geometry, theta)
    checks = dict(covCompared=0, covMismatch=0, covMaxAbs=0.0, geoCompared=0, geoMaxRel=0.0, thetaMaxAbs=0.0)
    pl = man.get("pipeline", {})
    for it in teach_items + neg_items + part_items:
        if it.lab is None:
            continue
        d = it.d
        if d.get("cov") is not None and pl.get("gh") and len(d["cov"]) == pl["gh"] * pl["gw"]:
            c = tr.patch_cov(it.lab, 1, it.crop, int(man["analysisFactor"]), pl["gh"], pl["gw"]).ravel()
            diff = float(np.abs(c - np.asarray(d["cov"], np.float64)).max())
            checks["covCompared"] += 1
            checks["covMismatch"] += int(diff > 0)
            checks["covMaxAbs"] = max(checks["covMaxAbs"], diff)
        if d.get("geometry") and it.geom is not None:
            g = it.geometry()
            for kname in ("area", "fill", "aspect", "hu1", "solidity"):
                ref = float(d["geometry"][kname])
                checks["geoMaxRel"] = max(checks["geoMaxRel"], abs(g[kname] - ref) / max(abs(ref), 1e-12))
            checks["geoCompared"] += 1
            if d.get("theta") is not None:
                checks["thetaMaxAbs"] = max(checks["thetaMaxAbs"], abs(it.geom.theta - float(d["theta"])))
    print(f"[checks] cov {checks['covMismatch']}/{checks['covCompared']} mismatching (max |d| {checks['covMaxAbs']}), "
          f"geometry max rel {checks['geoMaxRel']:.2e}, theta max |d| {checks['thetaMaxAbs']:.2e}")

    variants = []
    rot_list = ["NONE"] if a.variants == "default" else ROTATIONS
    sr_list = ["SMOOTHED_MAX"] if a.variants == "default" else SCORE_RULES
    br_list = ["CORESET"] if a.variants == "default" else BANK_RULES
    tau_list = ["LSO"] if (a.variants == "default" or not calib_items) else TAU_RULES
    knn = KnnCache()
    table, per_part, twins = {}, {}, {}
    for rotation in rot_list:
        orient = "can0" if rotation == "CANONICAL" else "r0"
        frames = teach_frames(store, teach_items, orient)
        negs = negative_globals(store, neg_items, orient)
        for sr in sr_list:
            params = tr.TeachParams(scoreRule=sr)
            try:
                twin = tr.teach(frames, params, negatives=negs)
            except tr.TeachError as e:
                print(f"[teach] {rotation}/{sr}: FAILED {e.code}")
                continue
            if rotation == "AUGMENT4":
                twin = augment_twin(twin, store, teach_items, frames, params)
            for br in br_list:
                cal_twin, cal = (None, None)
                if "CALIBRATED" in tau_list:
                    cal_twin, cal = calibrated(twin, store, calib_items, rotation, sr, br, a.sensitivity, knn)
                for tau_rule in tau_list:
                    tw = twin if tau_rule == "LSO" else cal_twin
                    v = (sr, br, rotation, tau_rule)
                    res = [(it.label, judge_item(tw, store, it, rotation, sr, br, a.sensitivity, knn))
                           for it in eval_items]
                    m = metrics(res)
                    m["twin"] = dict(tau=tw.tau, tauTeach=tw.tauTeach, tauId=tw.tauId, tauIdRule=tw.tauIdRule,
                                     keyframes=len(tw.keyframes), bankRows=int(tw.bank.shape[0]),
                                     segmentsUsed=tw.segmentsUsed, framesKept=tw.teachInfo.get("framesKept"),
                                     calibrated=bool(tw.calibrated),
                                     calValid=None if cal is None or tau_rule == "LSO" else f"{cal['m']}/{cal['n']}")
                    table[v] = m
                    per_part[v] = {it.name: jv(j) for it, (_, j) in zip(eval_items, res)}
                    variants.append(v)
                    print(f"[{time.time() - t0:5.0f}s] {'/'.join(v):44s} AUROC def {fmt_auc(m['auroc_defect'])} "
                          f"id {fmt_auc(m['auroc_identity'])} | PASS good {fmt_rate(m['pass_good'])} | NE wrong "
                          f"{fmt_rate(m['notenrolled_wrong'])} | DEFECT {fmt_rate(m['defect_defect'])}", flush=True)
    if not table:
        raise SystemExit("no variant could be evaluated (teach failed everywhere)")
    winner = pick_winner(table)
    default = DEFAULT if DEFAULT in table else variants[0]

    # phone vs laptop (default variant, only meaningful when the phone ran the same backbone)
    phone_cmp = None
    same_bb = (pl.get("inputSize") == bb.size and pl.get("gh") == bb.gh and pl.get("dim") == bb.dim)
    if same_bb:
        agree = n = 0
        ds, dsim = [], []
        for it in eval_items:
            ph = it.d.get("phone")
            if not ph:
                continue
            lp = per_part[default][it.name]
            n += 1
            agree += int(ph["verdict"] == lp["verdict"])
            if ph.get("s") is not None and lp["s"] is not None:
                ds.append(abs(ph["s"] - lp["s"]))
            if ph.get("sim") is not None and lp["sim"] is not None:
                dsim.append(abs(ph["sim"] - lp["sim"]))
        if n:
            phone_cmp = dict(n=n, agree=agree, sMedAbs=float(np.median(ds)) if ds else None,
                             sMaxAbs=float(max(ds)) if ds else None, simMedAbs=float(np.median(dsim)) if dsim else None,
                             simMaxAbs=float(max(dsim)) if dsim else None)
    os.makedirs(a.out, exist_ok=True)
    out = dict(export=os.path.abspath(a.export), backbone=a.backbone, model=os.path.abspath(bb.path),
               title=a.title, sensitivity=a.sensitivity, calibParts=len(calib_items), checks=checks,
               phoneVsLaptop=phone_cmp,
               default="/".join(default), winner="/".join(winner),
               variants={"/".join(v): table[v] for v in variants},
               parts={it.name: dict(label=it.label, phone=it.d.get("phone"), default=per_part[default][it.name],
                                    winner=per_part[winner][it.name]) for it in eval_items},
               runtime=dict(seconds=round(time.time() - t0, 1), backboneCalls=bb.calls,
                            backboneSeconds=round(bb.seconds, 1), knnCacheHits=knn.hits, knnCacheMisses=knn.misses))
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    write_report(os.path.join(a.out, "report.md"), out, table, variants, winner, default, man, bb, a)
    print(f"[done] {len(variants)} variants in {time.time() - t0:.0f} s; default {'/'.join(default)}; "
          f"winner {'/'.join(winner)}; report {os.path.join(a.out, 'report.md')}")


def write_report(path, out, table, variants, winner, default, man, bb, a):
    L = []
    title = (a.title + " - ") if a.title else ""
    L.append(f"# {title}Twin A/B report ({a.backbone})\n")
    L.append(f"- export: `{out['export']}` (twin `{man.get('twinId')}`, crops {man.get('cropSize')} px, analysis "
             f"factor {man.get('analysisFactor')})")
    L.append(f"- backbone: `{os.path.basename(bb.path)}` {bb.size} px -> {bb.gh}x{bb.gw}x{bb.dim}"
             f"{' (per-patch L2)' if bb.l2 else ''}; sensitivity {a.sensitivity}")
    n = table[variants[0]]["n"]
    L.append("- evaluation parts: " + ", ".join(f"{k} {v}" for k, v in n.items()) +
             f"; calib parts (CALIBRATED tau rule only): {out['calibParts']}")
    c = out["checks"]
    L.append(f"- export consistency: cov (phone grid) {c['covMismatch']}/{c['covCompared']} mismatching, geometry max "
             f"rel diff {c['geoMaxRel']:.1e}, theta max |diff| {c['thetaMaxAbs']:.1e}")
    if out["phoneVsLaptop"]:
        p = out["phoneVsLaptop"]
        L.append(f"- phone vs laptop (default variant, same backbone, laptop features from the JPEG crops): verdicts "
                 f"agree {p['agree']}/{p['n']}; |s| diff median {p['sMedAbs']:.3g} max {p['sMaxAbs']:.3g}; |sim| diff "
                 f"median {p['simMedAbs']:.3g} max {p['simMaxAbs']:.3g}")
    L.append(f"- rotation variants rotate the stored crop about its centre (sheet-colour padding) - an approximation "
             f"of the phone's full-frame re-crop")
    L.append(f"- **winner: `{'/'.join(winner)}`** (AUROC good-vs-defect, then identity AUROC; exact ties -> defaults); "
             f"default: `{'/'.join(default)}`")
    wm = table[winner]
    ties = [v for v in variants if table[v]["auroc_defect"] == wm["auroc_defect"]
            and table[v]["auroc_identity"] == wm["auroc_identity"]]
    if len(ties) > 1:
        L.append(f"- note: {len(ties)} of {len(variants)} variants tie on both AUROCs "
                 f"({fmt_auc(wm['auroc_defect'])} / {fmt_auc(wm['auroc_identity'])}), so the winner is decided by the "
                 f"defaults-win-ties rule; compare the operating-point rates below before changing a default")
    L.append("")
    L.append("| variant (score/bank/rotation/tau) | tau | tau_id | AUROC def | AUROC id | AUROC id look | PASS good | "
             "PASS rotated | NOT_ENR wrong | NOT_ENR look | DEFECT |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for v in variants:
        m = table[v]
        tag = " **(winner)**" if v == winner else (" (default)" if v == default else "")
        L.append(f"| `{'/'.join(v)}`{tag} | {m['twin']['tau']:.3f} | {m['twin']['tauId']:.4f} | "
                 f"{fmt_auc(m['auroc_defect'])} | {fmt_auc(m['auroc_identity'])} | "
                 f"{fmt_auc(m['auroc_identity_lookalike'])} | {fmt_rate(m['pass_good'])} | "
                 f"{fmt_rate(m['pass_rotated'])} | {fmt_rate(m['notenrolled_wrong'])} | "
                 f"{fmt_rate(m['notenrolled_lookalike'])} | {fmt_rate(m['defect_defect'])} |")
    L.append("\nRates: k/n = % [two-sided 95 % Clopper-Pearson]; REFRAME counts as a failure. AUROC per spec 14 "
             "(ties 0.5). Twin summary per variant in results.json.")
    rt = out["runtime"]
    L.append(f"\nRuntime {rt['seconds']} s ({rt['backboneCalls']} backbone runs, {rt['backboneSeconds']} s; k-NN cache "
             f"{rt['knnCacheHits']} hits / {rt['knnCacheMisses']} misses).")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
