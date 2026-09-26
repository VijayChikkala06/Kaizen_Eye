#!/usr/bin/env python3
"""
Tune Kaizen Eye on REAL phone photos, on the laptop, in seconds.

  python tools/lab/eval_folder.py --model out/backbone_r18_256.tflite --good photos/good --test photos/test

  photos/good : photos of GOOD parts. The first --n-enrol (sorted by name) enrol the model; any extra
                good photos are scored as unseen good parts (that is your false-reject check).
  photos/test : anything to score. Name files  bad_*.jpg  or  good_*.jpg  to get an accuracy summary and a
                suggested sensitivity (margin). Heat-map overlays go to --out.

Rules of thumb: same lighting, same distance, same mount as the demo; lock exposure on the phone camera.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import patchcore_ref as pc
from embedder import TfliteEmbedder
from viz import load_square, overlay

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def listing(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(EXT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--good", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--n-enrol", type=int, default=20)
    ap.add_argument("--folds", type=int, default=0, help="0 = leave-one-out on the coreset (default)")
    ap.add_argument("--ratio", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=1.0, help="sensitivity multiplier on tau (lower = stricter)")
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--spec", action="store_true", help="README spec scoring (no smoothing, no border) instead of the app's")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    emb = TfliteEmbedder(a.model)
    S = emb.size
    good = listing(a.good)
    if len(good) < 5:
        raise SystemExit(f"need at least 5 good photos in {a.good}")
    enrol_paths, extra = good[: a.n_enrol], good[a.n_enrol:]
    imgs = [load_square(p, S) for p in enrol_paths]
    feats = [emb(i) for i in imgs]
    smooth, border = (False, 0) if a.spec else (pc.APP_SMOOTH, pc.app_border(feats[0].shape[0]))
    prof = pc.enrol(feats, ratio=a.ratio, folds=a.folds, margin=1.0, smooth=smooth, border=border)
    print(f"scoring: {'README spec' if a.spec else f'app settings (smoothing {smooth}, border {border} patches)'}")
    tau0 = prof.tau                                   # base threshold (margin 1.0)
    print(f"enrolled {len(imgs)} photos -> bank {prof.bank.shape}, base tau={tau0:.3f}  "
          f"(enrol held-out scores {min(prof.loo_scores):.2f}..{max(prof.loo_scores):.2f})")

    rows = []                                          # (name, label or None, raw score)
    def run(paths, default_label):
        for p in paths:
            im = load_square(p, S)
            d, s = pc.score(emb(im), prof)
            raw = s * tau0
            name = os.path.basename(p)
            low = name.lower()
            label = "bad" if low.startswith("bad") else "good" if low.startswith("good") else default_label
            rows.append((name, label, raw))
            overlay(im, d, tau0 * a.margin, os.path.join(a.out, os.path.splitext(name)[0] + ".png"))

    run(extra, "good")
    if a.test:
        run(listing(a.test), None)

    print(f"\n{'file':38s} {'label':>6s} {'score':>7s}  verdict (margin {a.margin:.2f})")
    for name, label, raw in sorted(rows, key=lambda r: -r[2]):
        s = raw / (tau0 * a.margin)
        print(f"{name[:38]:38s} {str(label or '-'):>6s} {s:7.2f}  {'REJECT' if s > 1 else 'ok'}")

    lab = [(l, r) for _, l, r in rows if l]
    if lab and any(l == "bad" for l, _ in lab) and any(l == "good" for l, _ in lab):
        print("\nmargin sweep (labelled files only):   margin  false-rejects  misses")
        best = []
        for m in np.arange(0.70, 1.55, 0.05):
            fr = sum(1 for l, r in lab if l == "good" and r / (tau0 * m) > 1)
            ms = sum(1 for l, r in lab if l == "bad" and r / (tau0 * m) <= 1)
            best.append((fr + ms, m))
            print(f"                                       {m:5.2f}  {fr:9d}  {ms:9d}")
        e = min(b[0] for b in best)
        ok = [m for err, m in best if err == e]
        print(f"\nfewest errors ({e}) at margin {min(ok):.2f}..{max(ok):.2f}  ->  start the app slider near {np.median(ok):.2f}")
    print(f"\noverlays: {a.out}/")


if __name__ == "__main__":
    main()
