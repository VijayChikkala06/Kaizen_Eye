#!/usr/bin/env python3
r"""
Kaizen Eye 2 - on-device parity fixture for the DINOv2 export: one fixed input and the PyTorch outputs as raw
little-endian binaries, so the app (which cannot run PyTorch) can check its LiteRT output against the reference
(plan B5 "parity vs PyTorch"; spec 13 thresholds).

  .venv\Scripts\python tools\dinov2\write_parity_fixture.py      (needs torch_reference.npz written by
                                                                  `export_dinov2.py onnx` into %KZ_DINOV2_WORK%,
                                                                  default D:\kz-tmp\h2b\dinov2-work)
Writes to tools/dinov2/out/:
  dinov2_parity_input_448.rgb   uint8 [448, 448, 3] RGB row-major (feed as float32 0..255, NHWC [1,448,448,3])
  dinov2_parity_patch.f32       float32 LE [32, 32, 384] row-major (torch `patch`, NOT L2-normalised)
  dinov2_parity_cls.f32         float32 LE [384] (torch `cls`)
  dinov2_parity_fixture.json    description + the laptop TFLite/ORT agreement on this input
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REF = os.path.join(os.environ.get("KZ_DINOV2_WORK", r"D:\kz-tmp\h2b\dinov2-work"), "torch_reference.npz")
KEY = "photo_good_00"


def main():
    z = np.load(REF)
    img = z["image__" + KEY].astype(np.uint8)
    patch = z["patch__" + KEY].astype("<f4")
    cls = z["cls__" + KEY].astype("<f4")
    assert img.shape == (448, 448, 3) and patch.shape == (1, 32, 32, 384) and cls.shape == (1, 384)
    img.tofile(os.path.join(OUT, "dinov2_parity_input_448.rgb"))
    patch.tofile(os.path.join(OUT, "dinov2_parity_patch.f32"))
    cls.tofile(os.path.join(OUT, "dinov2_parity_cls.f32"))
    lap = {}
    pj = os.path.join(OUT, "dinov2_parity.json")
    if os.path.exists(pj):
        with open(pj, encoding="utf-8") as fh:
            par = json.load(fh)["parity"]
        lap = {k: v.get(KEY) for k, v in par.items()}
    meta = {
        "source": "demo_photos/good/good_00.jpg, centre square resized to 448 (tools/lab/viz.load_square)",
        "input": {"file": "dinov2_parity_input_448.rgb", "dtype": "uint8", "shape": [448, 448, 3],
                  "feed": "as float32 0..255, NHWC [1,448,448,3] RGB"},
        "reference": {"patch": {"file": "dinov2_parity_patch.f32", "dtype": "float32 little-endian",
                                "shape": [32, 32, 384], "note": "final-norm patch tokens, not L2-normalised"},
                      "cls": {"file": "dinov2_parity_cls.f32", "dtype": "float32 little-endian", "shape": [384]},
                      "producer": "tools/dinov2/dinov2_s14.py (PyTorch 2.13 CPU, float32)"},
        "checks": {"fp32 candidates (spec 13)": "all finite, cosine >= 0.999 and relative L2 <= 1 % vs this reference",
                   "plan G5": "mean per-patch cosine >= 0.99 and CLS cosine >= 0.99"},
        "laptop_agreement_on_this_input": lap,
    }
    with open(os.path.join(OUT, "dinov2_parity_fixture.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote fixture:", img.nbytes + patch.nbytes + cls.nbytes, "bytes")


if __name__ == "__main__":
    sys.exit(main())
