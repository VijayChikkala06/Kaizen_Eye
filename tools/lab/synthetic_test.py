#!/usr/bin/env python3
"""End-to-end mechanics test with synthetic parts:  embed -> enrol(20) -> score good vs 4 defect types.
   python tools/lab/synthetic_test.py --model out/backbone_r18_256.tflite
NOTE: with --random-weights models the numbers only prove the pipeline works. Use the real
pretrained model (and real photos via eval_folder.py) for anything you quote."""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import patchcore_ref as pc
from embedder import TfliteEmbedder
from synth_data import KINDS, make_defect, make_good
from viz import overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-enrol", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/overlays")
    ap.add_argument("--spec", action="store_true", help="README spec scoring (no smoothing, no border) instead of the app's")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    emb = TfliteEmbedder(a.model)
    S = emb.size

    t = time.time()
    enrol_imgs = [make_good(rng, S) for _ in range(a.n_enrol)]
    feats = [emb(im) for im in enrol_imgs]
    t_embed = (time.time() - t) / len(feats)
    t = time.time()
    smooth, border = (False, 0) if a.spec else (pc.APP_SMOOTH, pc.app_border(feats[0].shape[0]))
    prof = pc.enrol(feats, margin=a.margin, smooth=smooth, border=border)
    print(f"enrol: {len(feats)} frames -> bank {prof.bank.shape}, tau={prof.tau:.3f}, "
          f"LOO scores min/max = {min(prof.loo_scores):.3f}/{max(prof.loo_scores):.3f}  "
          f"(embed {t_embed*1000:.0f} ms/frame on this laptop CPU, coreset+LOO {time.time()-t:.1f} s)")

    stride = S // prof.gh
    good = []
    for i in range(a.n_test):
        im = make_good(rng, S)
        d, s = pc.score(emb(im), prof)
        good.append(s)
        if i == 0:
            overlay(im, d, prof.tau, f"{a.out}/good.png")
    good = np.array(good)
    print(f"\n{'case':10s} {'mean score':>10s} {'min':>7s} {'detected>1.0':>13s} {'peak in defect box':>19s}")
    print(f"{'good':10s} {good.mean():10.2f} {good.min():7.2f} {'(false rej) %d/%d' % ((good > 1).sum(), len(good)):>13s}")
    for kind in KINDS:
        sc, hit = [], 0
        for i in range(a.n_test):
            im, (x0, y0, x1, y1) = make_defect(rng, kind, S)
            d, s = pc.score(emb(im), prof)
            sc.append(s)
            r, c = np.unravel_index(np.argmax(pc.smooth3(d)), d.shape)
            px, py = (c + 0.5) * stride, (r + 0.5) * stride
            hit += (x0 - 16 <= px <= x1 + 16) and (y0 - 16 <= py <= y1 + 16)
            if i == 0:
                overlay(im, d, prof.tau, f"{a.out}/{kind}.png")
        sc = np.array(sc)
        print(f"{kind:10s} {sc.mean():10.2f} {sc.min():7.2f} {'%d/%d' % ((sc > 1).sum(), len(sc)):>13s} {'%d/%d' % (hit, len(sc)):>19s}")
    print(f"\noverlays written to {a.out}/")


if __name__ == "__main__":
    main()
