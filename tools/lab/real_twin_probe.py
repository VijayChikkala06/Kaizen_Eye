"""
Real-data probe of the Visual Twin's discrimination, using the keyframe crops that the phone saved when teaching
(twins pulled from the app's private storage: `adb exec-out run-as com.kaizeneye.v2 tar -c -C files twins > twins.tar`).

It answers: how far apart are two DIFFERENT real objects (e.g. two black earbud cases) compared with the spread of the
SAME object, for the current scoring (patch k-NN score + global-descriptor identity) and for candidate improvements?

  .venv\\Scripts\\python tools\\lab\\real_twin_probe.py --twins D:\\kz-tmp\\twins --a 20260926-234828-06b8 --b 20260926-235659-96bf

Everything here is an approximation of the app (the crops carry no mask, so a foreground mask is re-estimated from the
crop's border colour); it is an experiment harness, not the reference implementation (that is twin_ref.py).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX = os.path.join(ROOT, "tools", "dinov2", "out", "backbone_r18_320.onnx")
N = 320
G = 40


def load_session():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 6
    s = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return lambda img: s.run(None, {name: img.astype(np.float32)[None]})[0][0]  # [40,40,128]


def foreground(img):
    """Object mask of a teach crop: colour distance from the crop's border colour (median), Otsu-ish threshold."""
    a = img.astype(np.float64)
    b = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    bg = np.median(b, axis=0)
    sd = np.maximum(b.std(axis=0), 3.0)
    d = np.sqrt((((a - bg) / sd) ** 2).sum(-1))
    m = d > 6.0
    # keep the biggest connected blob
    from scipy import ndimage as ndi

    m = ndi.binary_opening(m, iterations=2)
    m = ndi.binary_closing(m, iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(m)


def patch_cov(mask):
    return mask.reshape(G, N // G, G, N // G).mean(axis=(1, 3))  # [40,40]


def principal_angle(mask):
    ys, xs = np.nonzero(mask)
    x = xs - xs.mean()
    y = ys - ys.mean()
    cxx, cyy, cxy = (x * x).mean(), (y * y).mean(), (x * y).mean()
    return 0.5 * np.arctan2(2 * cxy, cxx - cyy)


def rotate(img, theta_rad, fill=None):
    """Rotate so the principal axis becomes horizontal (theta = principal angle)."""
    im = Image.fromarray(img)
    return np.asarray(im.rotate(np.degrees(theta_rad), resample=Image.BILINEAR, fillcolor=tuple(int(v) for v in np.median(np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3)]), axis=0))))


def smooth_score(d, S):
    """SMOOTHED_MAX: mean of d over the 3x3 neighbourhood inside S, max over S."""
    dd = np.where(S, d, 0.0)
    ss = S.astype(np.float64)
    pad = lambda x: np.pad(x, 1)
    sm = np.zeros_like(dd)
    cnt = np.zeros_like(dd)
    for i in range(3):
        for j in range(3):
            sm += pad(dd)[i:i + G, j:j + G]
            cnt += pad(ss)[i:i + G, j:j + G]
    sm = sm / np.maximum(cnt, 1)
    return float(sm[S].max()) if S.any() else float("nan")


def nn_dist(q, bank):
    """q [P,d], bank [R,d] -> per-row Euclidean nn distance."""
    q2 = (q ** 2).sum(1)[:, None]
    b2 = (bank ** 2).sum(1)[None, :]
    d2 = np.maximum(q2 + b2 - 2 * q @ bank.T, 0)
    return np.sqrt(d2.min(1))


def load_twin(root, tid):
    kd = os.path.join(root, tid, "keyframes")
    return [np.asarray(Image.open(os.path.join(kd, f)).convert("RGB").resize((N, N), Image.BILINEAR)) for f in sorted(os.listdir(kd))]


def gap(f, core):
    g = f[core.reshape(-1)].sum(0) if core.any() else f.sum(0)
    return g / (np.linalg.norm(g) + 1e-12)


def seg_of(k, n):
    return min(4, int(5 * k / n))


def raw_scores(q_items, bank_items, exclude_seg=None, segs=None):
    """raw SMOOTHED_MAX of each query item against the pooled S-patches of bank_items (optionally leaving a segment out)."""
    out = []
    for qi, q in enumerate(q_items):
        rows = [b["f"][b["S"].reshape(-1)] for bi, b in enumerate(bank_items)
                if exclude_seg is None or segs[bi] != exclude_seg[qi]]
        bank = np.concatenate(rows)
        d = nn_dist(q["f"], bank).reshape(G, G)
        out.append(smooth_score(d, q["S"]))
    return np.array(out)


def baseline(data):
    A, B = data["A"], data["B"]
    segs = [seg_of(k, len(A)) for k in range(len(A))]
    lso = raw_scores(A, A, exclude_seg=segs, segs=segs)
    tau = 1.4 * lso.max()
    rb = raw_scores(B, A)
    print("== BASELINE (current algorithm, approximated)")
    print("A leave-segment-out raw: min %.3f median %.3f max %.3f -> tau(1.4x max)=%.3f" % (lso.min(), np.median(lso), lso.max(), tau))
    print("B (other case) raw vs A bank: min %.3f median %.3f max %.3f   -> PASS if raw<=tau: %d/%d" % (rb.min(), np.median(rb), rb.max(), (rb <= tau).sum(), len(rb)))
    if "C" in data:
        rc = raw_scores(data["C"], A)
        print("C (jeans)  raw vs A bank: min %.3f median %.3f max %.3f   -> PASS: %d/%d" % (rc.min(), np.median(rc), rc.max(), (rc <= tau).sum(), len(rc)))
    gA = [gap(x["f"], x["core"]) for x in A]
    pos = np.array([max(gA[k] @ gA[j] for j in range(len(A)) if segs[j] != segs[k]) for k in range(len(A))])
    neg = np.array([max(gap(b["f"], b["core"]) @ ga for ga in gA) for b in B])
    print("identity GAP sim: A-vs-A(other seg) min %.3f med %.3f | B-vs-A min %.3f med %.3f max %.3f" % (pos.min(), np.median(pos), neg.min(), np.median(neg), neg.max()))
    mad = np.median(np.abs(pos - np.median(pos)))
    tid = np.percentile(pos, 5) - max(0.02, 3 * 1.4826 * mad)
    print("tau_id (no negatives) = %.3f -> B passes identity: %d/%d" % (tid, (neg >= tid).sum(), len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", default=None)
    a = ap.parse_args()
    emb = load_session()
    sets = {"A": load_twin(a.twins, a.a), "B": load_twin(a.twins, a.b)}
    if a.c:
        sets["C"] = load_twin(a.twins, a.c)
    data = {}
    for k, imgs in sets.items():
        items = []
        for im in imgs:
            m = foreground(im)
            f = emb(im).reshape(G * G, -1)
            cov = patch_cov(m)
            items.append(dict(img=im, mask=m, f=f, S=(cov > 0.0), core=(cov >= 0.5), theta=principal_angle(m)))
        data[k] = items
        print(k, len(items), "mask area %:", np.round(np.mean([it["mask"].mean() * 100 for it in items]), 1))
    baseline(data)


if __name__ == "__main__":
    sys.exit(main())
