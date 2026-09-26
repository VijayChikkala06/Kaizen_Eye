#!/usr/bin/env python3
"""Write golden test vectors for the TypeScript port in mobile/src/core (checked by mobile/scripts/verify-core.ts):
   python tools/lab/make_golden.py --out testdata/golden_core.json
Use --out rather than `> file`: PowerShell 5.1 redirection writes UTF-16 with a BOM, which JSON.parse rejects.
The TypeScript greedy coreset / k-NN / calibration must reproduce these numbers (indices exactly, floats to ~1e-4).
Bank size rule: k = max(minK, floor(ratio * N)).  Tie-break: lowest index wins (same as a strict `>` comparison)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import patchcore_ref as pc

ap = argparse.ArgumentParser()
ap.add_argument("--out", help="write the JSON to this file (UTF-8, no BOM) instead of stdout")
a = ap.parse_args()

rng = np.random.default_rng(42)
F, gh, gw, D = 6, 4, 4, 8
base = rng.normal(0, 1, (gh, gw, D)).astype(np.float32)
frames = [(base + rng.normal(0, 0.15, (gh, gw, D))).astype(np.float32) for _ in range(F)]
test = (base + rng.normal(0, 0.15, (gh, gw, D))).astype(np.float32)
test[1, 2, :4] += 2.0                                           # the "defect"

RATIO, MIN_K, FOLDS, MARGIN = 0.25, 8, 0, 1.10
prof = pc.enrol(frames, ratio=RATIO, min_k=MIN_K, folds=FOLDS, margin=MARGIN)
x = np.concatenate([f.reshape(-1, D) for f in frames])
coreset = pc.greedy_coreset(x, max(MIN_K, int(RATIO * len(x))))
dmap, snorm = pc.score(test, prof)

out = dict(
    params=dict(gh=gh, gw=gw, D=D, ratio=RATIO, minK=MIN_K, folds=FOLDS, margin=MARGIN, start=0),
    frames=[f.tolist() for f in frames],
    coreset_indices=coreset.tolist(),
    loo_scores=[float(v) for v in prof.loo_scores],
    tau=prof.tau,
    test_frame=test.tolist(),
    patch_dist=dmap.tolist(),
    raw_score=float(pc.frame_score(dmap)),
    normalised_score=float(snorm),
)
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"wrote {a.out}")
else:
    json.dump(out, sys.stdout)
