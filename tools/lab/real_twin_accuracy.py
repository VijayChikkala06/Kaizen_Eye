"""
Kaizen Eye 2 - real-capture accuracy study: can the Visual Twin tell two look-alike REAL parts apart?

Input: twin folders pulled from the phone (each holds the taught keyframe crops + twin.json with the keyframe segments):
    adb exec-out run-as com.kaizeneye.v2 tar -c -C files twins > twins.tar   (then extract)
Three twins are compared pairwise: every twin is "enrolled" in turn, every twin (and augmented copies of it) is a query.

For each enrolled twin X and query set Q the script measures, for a chosen backbone (R18 = the old default, DINOv2 = the new one)
and rotation handling (NONE = old, CANONICAL = principal axis horizontal, best of theta / theta + pi):
    raw   the spec score  max over S of the 3x3-smoothed nearest-patch distance          (defect / local anomaly)
    fit   the FIT statistic  mean over the core of the nearest-patch distance            (does the WHOLE crop look like X?)
against a coreset bank of X's keyframes (10 %, min 256, max 2400 rows, greedy k-centre - twin-spec 7.6) and reports
    * the spread of the SAME part re-presented (augmented keyframes: random rotation, shift, scale, brightness, blur),
    * the closest other-object sample, AUROC(same vs other), and the leave-segment-out (LSO) value the teach would derive tau from.

  .venv\\Scripts\\python tools\\lab\\real_twin_accuracy.py --twins D:\\kz-tmp\\twins --backbone dino --rotation canonical \\
        --twin A=20260926-234828-06b8 --twin B=20260926-235659-96bf --twin C=20260926-233918-6e6e

Approximations (it is a study, not the reference implementation - that is twin_ref.py): the crops carry no mask, so the object mask is
re-estimated from the crop border colour; canonical crops rotate the stored 320 px crop about its centre.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tools", "dinov2", "out")


# ------------------------------------------------------------------------------------------------ backbones
class Backbone:
    def __init__(self, kind):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        self.kind = kind
        if kind == "dino":
            self.sess = ort.InferenceSession(os.path.join(OUT, "dinov2_s14_448.onnx"), so, providers=["CPUExecutionProvider"])
            self.size, self.g, self.dim = 448, 32, 384
        else:
            self.sess = ort.InferenceSession(os.path.join(OUT, "backbone_r18_320.onnx"), so, providers=["CPUExecutionProvider"])
            self.size, self.g, self.dim = 320, 40, 128
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, img):
        x = np.asarray(Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR)).astype(np.float32)[None]
        p = self.sess.run(None, {self.inp: x})[0][0].reshape(self.g * self.g, -1).astype(np.float64)
        if self.kind == "dino":                                   # twin-spec 5: per-patch L2 normalisation
            p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
        return p.astype(np.float32)


# ------------------------------------------------------------------------------------------------ image helpers
def bg_colour(im):
    b = np.concatenate([im[:5].reshape(-1, 3), im[-5:].reshape(-1, 3), im[:, :5].reshape(-1, 3), im[:, -5:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(b, axis=0))


def foreground(img):
    from scipy import ndimage as ndi

    a = img.astype(np.float64)
    b = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    d = np.sqrt((((a - np.median(b, axis=0)) / np.maximum(b.std(axis=0), 3.0)) ** 2).sum(-1))
    m = ndi.binary_closing(ndi.binary_opening(d > 6.0, iterations=2), iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    m = lab == 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(m)


def principal_angle(mask):
    ys, xs = np.nonzero(mask)
    x, y = xs - xs.mean(), ys - ys.mean()
    return 0.5 * math.atan2(2 * (x * y).mean(), (x * x).mean() - (y * y).mean())


def rot(im, deg):
    return np.asarray(Image.fromarray(im).rotate(deg, resample=Image.BILINEAR, fillcolor=bg_colour(im)))


def augment(img, rng):
    """The same part presented again: any rotation, +-8 px shift, +-4 % scale, brightness / contrast, sometimes a little blur."""
    h, w = img.shape[:2]
    im = Image.fromarray(img).rotate(rng.uniform(0, 360), resample=Image.BILINEAR, fillcolor=bg_colour(img))
    nw = int(round(w * rng.uniform(0.96, 1.04)))
    canvas = Image.new("RGB", (w, h), bg_colour(img))
    canvas.paste(im.resize((nw, nw), Image.BILINEAR), ((w - nw) // 2, (h - nw) // 2))
    dx, dy = rng.integers(-8, 9, 2)
    im = canvas.transform(canvas.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR, fillcolor=bg_colour(img))
    a = np.asarray(im).astype(np.float64)
    a = (a - a.mean()) * rng.uniform(0.92, 1.08) + a.mean() * rng.uniform(0.9, 1.1)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return np.asarray(im)


def cov_grid(mask, size, g):
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
    k = size // g
    return m.reshape(g, k, g, k).mean(axis=(1, 3))


# ------------------------------------------------------------------------------------------------ maths
def nn_dist(q, bank):
    q, bank = q.astype(np.float64), bank.astype(np.float64)
    d2 = np.maximum((q ** 2).sum(1)[:, None] + (bank ** 2).sum(1)[None, :] - 2 * q @ bank.T, 0)
    return np.sqrt(d2.min(1))


def smooth_max(d, S, g):
    dd, ss = np.where(S, d, 0.0), S.astype(np.float64)
    pad = lambda x: np.pad(x, 1)
    sm, cnt = np.zeros_like(dd), np.zeros_like(dd)
    for i in range(3):
        for j in range(3):
            sm += pad(dd)[i:i + g, j:j + g]
            cnt += pad(ss)[i:i + g, j:j + g]
    return float((sm / np.maximum(cnt, 1))[S].max())


def greedy_coreset(rows, k):
    sel, mind = [0], ((rows - rows[0]) ** 2).sum(1)
    for _ in range(k - 1):
        j = int(np.argmax(mind))
        sel.append(j)
        mind = np.minimum(mind, ((rows - rows[j]) ** 2).sum(1))
    return np.array(sel)


def auroc(same, other):
    same, other = np.asarray(same), np.asarray(other)
    return float(((other[:, None] > same[None, :]).mean() + 0.5 * (other[:, None] == same[None, :]).mean()))


class Item:
    def __init__(self, bb, img):
        self.img = img
        self.mask = foreground(img)
        self.f = bb(img)
        c = cov_grid(self.mask, bb.size, bb.g)
        self.S, self.core = c > 0, c >= 0.5
        self.g = bb.g

    def stats(self, bank):
        d = nn_dist(self.f, bank).reshape(self.g, self.g)
        return dict(raw=smooth_max(d, self.S, self.g), fit=float(d[self.core].mean() if self.core.any() else d[self.S].mean()))


def variants(bb, img, canonical):
    """The crops a query is judged on: the crop itself (NONE) or the two canonical orientations."""
    if not canonical:
        return [Item(bb, img)]
    th = math.degrees(principal_angle(foreground(img)))
    return [Item(bb, rot(img, th)), Item(bb, rot(img, th + 180))]


def best(stats_list, key):
    return min(s[key] for s in stats_list)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twins", required=True)
    ap.add_argument("--twin", action="append", required=True, help="LABEL=folder-name (2 or more)")
    ap.add_argument("--backbone", choices=["dino", "r18"], default="dino")
    ap.add_argument("--rotation", choices=["none", "canonical"], default="canonical")
    ap.add_argument("--variants", type=int, default=5, help="augmented copies per keyframe")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    canonical = a.rotation == "canonical"
    rng = np.random.default_rng(a.seed)
    bb = Backbone(a.backbone)
    names = [t.split("=")[0] for t in a.twin]
    folders = dict(t.split("=") for t in a.twin)
    imgs, segs = {}, {}
    for n in names:
        kd = os.path.join(a.twins, folders[n], "keyframes")
        imgs[n] = [np.asarray(Image.open(os.path.join(kd, f)).convert("RGB").resize((320, 320), Image.BILINEAR)) for f in sorted(os.listdir(kd))]
        segs[n] = np.array([k["segment"] for k in json.load(open(os.path.join(a.twins, folders[n], "twin.json")))["keyframes"]])
    queries = {n: [augment(im, rng) for im in imgs[n] for _ in range(a.variants)] for n in names}
    print(f"backbone {a.backbone}, rotation {a.rotation}, {a.variants} augmented copies per keyframe\n")
    qcache = {n: [variants(bb, q, canonical) for q in queries[n]] for n in names}
    kfitems = {n: [variants(bb, im, canonical)[0] for im in imgs[n]] for n in names}
    for e in names:
        kf = kfitems[e]
        rows = np.concatenate([it.f[it.S.reshape(-1)] for it in kf]).astype(np.float64)
        kidx = np.concatenate([np.full(int(it.S.sum()), i) for i, it in enumerate(kf)])
        sel = greedy_coreset(rows, min(2400, max(256, int(0.10 * len(rows)))))
        bank, bk = rows[sel].astype(np.float32), kidx[sel]
        lso = [kf[i].stats(bank[segs[e][bk] != segs[e][i]]) for i in range(len(kf))]
        lso_raw, lso_fit = np.array([s["raw"] for s in lso]), np.array([s["fit"] for s in lso])
        res = {q: [[v.stats(bank) for v in vs] for vs in qcache[q]] for q in names}
        print(f"=== enrolled {e}   LSO raw max {lso_raw.max():.3f} (median {np.median(lso_raw):.3f})   LSO fit max {lso_fit.max():.3f} (median {np.median(lso_fit):.3f})")
        for key in ("raw", "fit"):
            same = np.array([best(s, key) for s in res[e]])
            line = f"  {key}: same part p50 {np.median(same):.3f} p95 {np.percentile(same, 95):.3f} max {same.max():.3f}"
            for q in names:
                if q == e:
                    continue
                oth = np.array([best(s, key) for s in res[q]])
                line += f" | vs {q}: min {oth.min():.3f} med {np.median(oth):.3f} AUROC {auroc(same, oth):.3f}"
            print(line)
        # ---- the shipped rules applied to this study (pass rates in %)
        def rate(vals, tau):
            return 100.0 * np.mean(np.asarray(vals) <= tau)

        tau_old = 1.4 * lso_raw.max()                                   # spec rule
        tau_raw = 1.15 * lso_raw.max()                                  # accuracy mode
        tau_fit = 1.10 * lso_fit.max()                                  # no negatives
        neg = [min(it.stats(bank)["fit"] for it in kfitems[q]) for q in names if q != e]   # closest keyframe of every OTHER twin
        hi, lo = lso_fit.max(), min(neg)
        tau_mid = min(1.10 * hi, (hi + lo) / 2) if lo > hi else 1.10 * hi
        print(f"  shipped rules: tau_raw old {tau_old:.3f} / new {tau_raw:.3f}; tau_fit {tau_fit:.3f} (no negatives) / {tau_mid:.3f} (other twins as negatives)")
        rows = [("same part", e)] + [(f"other {q}", q) for q in names if q != e]
        for label, q in rows:
            raw = [best(s, "raw") for s in res[q]]
            fit = [best(s, "fit") for s in res[q]]
            both = [(r <= tau_raw and f <= tau_fit) for r, f in zip(raw, fit)]
            both_mid = [(r <= tau_raw and f <= tau_mid) for r, f in zip(raw, fit)]
            print(f"    {label:<10} PASS: old score gate {rate(raw, tau_old):5.1f} %  | new score gate {rate(raw, tau_raw):5.1f} %  | + fit gate {100.0 * np.mean(both):5.1f} %  | + fit gate with negatives {100.0 * np.mean(both_mid):5.1f} %")
        print()


if __name__ == "__main__":
    sys.exit(main())
