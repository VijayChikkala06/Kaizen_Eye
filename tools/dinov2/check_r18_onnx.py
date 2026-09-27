#!/usr/bin/env python3
r"""
Kaizen Eye 2 - parity check of the ResNet18@320 ONNX export (ONNX-Runtime-QNN fallback arm) against the shipped TFLite.

  .venv\Scripts\python tools\dinov2\check_r18_onnx.py [--out tools\dinov2\out\r18_parity.json]

Compares, on fixed inputs (random uint8 seed 0 + two demo photos when present):
  shipped  mobile/assets/models/backbone_r18_320.tflite   (ai_edge_litert, the frozen reference)
  onnx     tools/dinov2/out/backbone_r18_320.onnx         (onnxruntime CPU)
  rebuilt  tools/dinov2/out/backbone_r18_320.tflite       (ai_edge_litert; same converter run that wrote the .onnx)
Metrics: cosine over the whole [1,40,40,128] map, mean per-patch cosine, max |diff|, max |diff| / max |ref|.
Also times each runtime on this laptop CPU (median of 10 after 2 warm-ups). Read-only: never writes mobile/.
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIPPED = os.path.join(ROOT, "mobile", "assets", "models", "backbone_r18_320.tflite")
OUT = os.path.join(ROOT, "tools", "dinov2", "out")
SIZE = 320


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def compare(got, ref):
    g = got.reshape(-1, got.shape[-1]).astype(np.float64)
    r = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
    cos = float((g * r).sum() / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    pc = (g * r).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(r, axis=1) + 1e-30)
    mad = float(np.abs(g - r).max())
    return dict(cosine=cos, mean_patch_cosine=float(pc.mean()), min_patch_cosine=float(pc.min()),
                max_abs_diff=mad, max_abs_diff_rel=mad / (float(np.abs(r).max()) + 1e-30))


def tflite_runner(path):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]

    def run(x):
        it.set_tensor(i["index"], x)
        it.invoke()
        return it.get_tensor(o["index"]).copy()

    return run


def onnx_runner(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return lambda x: s.run(None, {name: x})[0]


def median_ms(fn, x, n=10):
    for _ in range(2):
        fn(x)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(ts)


def inputs():
    rng = np.random.default_rng(0)
    xs = {"random_u8_seed0": rng.integers(0, 256, (1, SIZE, SIZE, 3)).astype(np.float32)}
    sys.path.insert(0, os.path.join(ROOT, "tools", "lab"))
    try:
        from viz import load_square

        for p in sorted(glob.glob(os.path.join(ROOT, "demo_photos", "*", "*.jpg")))[:2]:
            xs["photo_" + os.path.basename(p)] = load_square(p, SIZE).astype(np.float32)[None]
    except Exception as e:  # noqa: BLE001 - photos are optional
        print("(demo photos skipped:", e, ")")
    return xs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=os.path.join(OUT, "backbone_r18_320.onnx"))
    ap.add_argument("--rebuilt", default=os.path.join(OUT, "backbone_r18_320.tflite"))
    ap.add_argument("--out", default=os.path.join(OUT, "r18_parity.json"))
    a = ap.parse_args()

    runs = {"shipped_tflite": tflite_runner(SHIPPED), "onnx_ort": onnx_runner(a.onnx)}
    if os.path.exists(a.rebuilt):
        runs["rebuilt_tflite"] = tflite_runner(a.rebuilt)
    res = {"files": {k: {"path": os.path.relpath(p, ROOT), "sha256": sha256(p), "bytes": os.path.getsize(p)}
                     for k, p in (("shipped_tflite", SHIPPED), ("onnx", a.onnx), ("rebuilt_tflite", a.rebuilt))
                     if os.path.exists(p)},
           "cases": {}, "timing_ms_median10_4threads": {}}
    xs = inputs()
    for name, x in xs.items():
        ref = runs["shipped_tflite"](x)
        case = {}
        for k, fn in runs.items():
            if k != "shipped_tflite":
                case[k + "_vs_shipped"] = compare(fn(x), ref)
        res["cases"][name] = case
        for k, v in case.items():
            print(f"{name:28s} {k:28s} cos {v['cosine']:.8f}  mean patch cos {v['mean_patch_cosine']:.8f}  "
                  f"max|d| {v['max_abs_diff']:.3e}  rel {v['max_abs_diff_rel']:.3e}")
    x0 = next(iter(xs.values()))
    for k, fn in runs.items():
        res["timing_ms_median10_4threads"][k] = round(median_ms(fn, x0), 2)
    print("timing (ms, median of 10, 4 threads):", res["timing_ms_median10_4threads"])
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", a.out)
    worst = min(v["cosine"] for c in res["cases"].values() for v in c.values())
    if worst < 0.9999:
        raise SystemExit(f"FAILED: cosine {worst:.6f} < 0.9999")


if __name__ == "__main__":
    main()
