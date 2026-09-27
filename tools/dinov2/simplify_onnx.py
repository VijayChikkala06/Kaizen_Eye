#!/usr/bin/env python3
r"""
Kaizen Eye 2 - fold an ONNX model's static shape computations with onnxsim (e.g. the Shape -> Slice -> Resize chain of
the ResNet18 export) so an NPU execution provider (ORT-QNN) sees a fully static graph. Values are unchanged.

  .venv-convert\Scripts\python tools\dinov2\simplify_onnx.py tools\dinov2\out\backbone_r18_320.onnx tools\dinov2\out\backbone_r18_320_sim.onnx
"""
import collections
import sys

import onnx
from onnxsim import simplify


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    before = collections.Counter(n.op_type for n in m.graph.node)
    ms, ok = simplify(m)
    if not ok:
        raise SystemExit("onnxsim: simplified model failed its check")
    onnx.checker.check_model(ms)
    onnx.save(ms, dst)
    print("before:", dict(before))
    print("after: ", dict(collections.Counter(n.op_type for n in ms.graph.node)))
    print("wrote", dst)


if __name__ == "__main__":
    main()
