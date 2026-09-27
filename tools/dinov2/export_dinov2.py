#!/usr/bin/env python3
r"""
Kaizen Eye 2 - export DINOv2 ViT-S/14 (facebook/dinov2-small) at a fixed 448x448 input: torch -> ONNX -> TFLite.

Stages (run each under the machine-wide lock, one heavy process at a time; see docs/verification/dinov2-export.md):
  onnx    (.venv-convert)  build the module (dinov2_s14.py), save torch reference outputs on fixed inputs,
                           export ONNX opset 17 with static shapes, simplify with onnxsim, check with onnxruntime.
  tflite  (.venv-convert)  onnx2tf (TensorFlow CPU, builtin ops only) -> fp32 .tflite + fp16-weight .tflite.
  check   (.venv)          ai_edge_litert + onnxruntime parity vs the torch reference, TFLite op list with NPU/HTP
                           flags, laptop CPU timing (median of 10), SHA-256s.

  powershell -NoProfile -Command "& native\tools\with-lock.ps1 -- .venv-convert\Scripts\python.exe tools\dinov2\export_dinov2.py onnx"
  powershell -NoProfile -Command "& native\tools\with-lock.ps1 -- .venv-convert\Scripts\python.exe tools\dinov2\export_dinov2.py tflite"
  powershell -NoProfile -Command "& native\tools\with-lock.ps1 -- .venv\Scripts\python.exe tools\dinov2\export_dinov2.py check"

Model contract (same convention as mobile/assets/models/backbone_r18_320.tflite):
  input   image  NHWC float32 [1,448,448,3], RGB 0..255 (ImageNet mean/std baked into the graph)
  outputs patch  NHWC float32 [1,32,32,384]  final-norm patch tokens, NOT L2-normalised (the app normalises)
          cls    float32 [1,384]             final-norm CLS token
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
# intermediates (raw ONNX, TF SavedModel, onnx2tf log, torch reference) - outside the repo by default
WORK = os.environ.get("KZ_DINOV2_WORK", r"D:\kz-tmp\h2b\dinov2-work")
WEIGHTS = r"D:\kz-tmp\dinov2-weights\model.safetensors"
SIZE, GRID, DIM = 448, 32, 384
NAME = "dinov2_s14_448"
ONNX_RAW = os.path.join(WORK, NAME + "_raw.onnx")
ONNX = os.path.join(OUT, NAME + ".onnx")
TFL32 = os.path.join(OUT, NAME + "_fp32.tflite")
TFL16 = os.path.join(OUT, NAME + "_fp16w.tflite")
REF = os.path.join(WORK, "torch_reference.npz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def parity_inputs():
    """Fixed inputs: random uint8 (seed 0), two demo photos (centre square), a flat grey image."""
    rng = np.random.default_rng(0)
    ims = {"random_u8_seed0": rng.integers(0, 256, (SIZE, SIZE, 3)).astype(np.uint8)}
    sys.path.insert(0, os.path.join(ROOT, "tools", "lab"))
    from viz import load_square

    for p in sorted(glob.glob(os.path.join(ROOT, "demo_photos", "*", "*.jpg")))[:2]:
        ims["photo_" + os.path.splitext(os.path.basename(p))[0]] = load_square(p, SIZE)
    ims["flat_grey_128"] = np.full((SIZE, SIZE, 3), 128, np.uint8)
    return ims


# ------------------------------------------------------------------------------------------------------ stage onnx
def fix_gelu_constants(model):
    """Reshape the scalar sqrt(2) divisor of every exported erf-GELU (Div -> Erf -> Add 1 -> Mul x -> Mul 0.5) to a
    1-D [1] tensor. Same values, same broadcast; the reason is onnx2tf 1.29.24's GeLU fusion (ops/Div.py), which
    compares the divisor with `== 1.4142135`: with numpy 1.26 that is False for a 0-d float32 value (compared in
    float64) and True for a 1-D one (compared in float32). Without the fusion onnx2tf emits FlexErf (TF Select op,
    not a LiteRT builtin); with it, one builtin GELU op per block."""
    from onnx import numpy_helper

    inits = {t.name: t for t in model.graph.initializer}
    consts = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    fixed = set()
    for n in model.graph.node:
        if n.op_type != "Div" or len(n.input) != 2 or n.input[1] in fixed:
            continue
        name = n.input[1]
        if name in inits:
            v = numpy_helper.to_array(inits[name])
            if v.ndim == 0 and abs(float(v) - 2 ** 0.5) < 1e-6:
                inits[name].CopyFrom(numpy_helper.from_array(v.reshape(1), name))
                fixed.add(name)
        elif name in consts:
            for att in consts[name].attribute:
                if att.name == "value":
                    v = numpy_helper.to_array(att.t)
                    if v.ndim == 0 and abs(float(v) - 2 ** 0.5) < 1e-6:
                        att.t.CopyFrom(numpy_helper.from_array(v.reshape(1)))
                        fixed.add(name)
    print(f"[onnx] GELU sqrt(2) divisors reshaped to [1]: {len(fixed)} initializer(s)")
    return len(fixed)


def stage_onnx(a):
    import onnx
    import torch

    sys.path.insert(0, HERE)
    import dinov2_s14 as dm

    torch.set_num_threads(4)
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    net = dm.build(a.weights, SIZE)
    ims = parity_inputs()
    ref = {}
    with torch.no_grad():
        for k, im in ims.items():
            pt, ct = net(torch.as_tensor(im.astype(np.float32))[None])
            ref["patch__" + k], ref["cls__" + k], ref["image__" + k] = pt.numpy(), ct.numpy(), im
    np.savez_compressed(REF, **ref)
    print(f"[torch] outputs patch {ref['patch__random_u8_seed0'].shape} cls {ref['cls__random_u8_seed0'].shape}; "
          f"reference saved to {REF}")

    sample = torch.as_tensor(ims["random_u8_seed0"].astype(np.float32))[None]
    torch.onnx.export(net, (sample,), ONNX_RAW, input_names=["image"], output_names=["patch", "cls"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    del net
    m = onnx.load(ONNX_RAW)
    onnx.checker.check_model(m)
    print("[onnx] raw export:", dict(collections.Counter(n.op_type for n in m.graph.node)))
    try:
        from onnxsim import simplify

        ms, ok = simplify(m)
        if not ok:
            raise RuntimeError("onnxsim check failed")
        print("[onnxsim] simplified:", dict(collections.Counter(n.op_type for n in ms.graph.node)))
    except Exception as e:  # noqa: BLE001
        print("[onnxsim] skipped:", type(e).__name__, str(e)[:200])
        ms = m
    fix_gelu_constants(ms)
    onnx.checker.check_model(ms)
    onnx.save(ms, ONNX)
    import onnxruntime as ort

    s = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    for k in ims:
        y = s.run(None, {"image": ref["image__" + k].astype(np.float32)[None]})
        byname = dict(zip([o.name for o in s.get_outputs()], y))
        print(f"[ort {ort.__version__}] {k:24s} patch max|d| {np.abs(byname['patch'] - ref['patch__' + k]).max():.2e}"
              f"  cls max|d| {np.abs(byname['cls'] - ref['cls__' + k]).max():.2e}")
    print(f"[onnx] wrote {ONNX} ({os.path.getsize(ONNX) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------------------------------- stage tflite
def stage_tflite(a):
    tf_dir = os.path.join(WORK, "tf")
    shutil.rmtree(tf_dir, ignore_errors=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PATH"] = os.pathsep.join([sysconfig.get_path("scripts"), os.path.dirname(sys.executable), env.get("PATH", "")])
    # onnx2tf loads this file from the cwd for an internal dummy inference (downloads it when missing); values unused.
    np.save(os.path.join(WORK, "calibration_image_sample_data_20x128x128x3_float32.npy"),
            np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32))
    cmd = [sys.executable, "-m", "onnx2tf", "-i", ONNX, "-o", tf_dir, "-kat", "image"] + a.extra
    print("[onnx2tf]", " ".join(cmd), flush=True)
    t0 = time.time()
    log = os.path.join(WORK, "onnx2tf.log")
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=WORK, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[onnx2tf] exit {r.returncode} after {time.time() - t0:.0f} s (log {log})")
    f32 = glob.glob(os.path.join(tf_dir, "*_float32.tflite"))
    f16 = glob.glob(os.path.join(tf_dir, "*_float16.tflite"))
    if r.returncode != 0 or len(f32) != 1:
        with open(log, encoding="utf-8", errors="replace") as fh:
            print("".join(fh.readlines()[-60:]))
        raise SystemExit("[onnx2tf] conversion failed")
    shutil.copyfile(f32[0], TFL32)
    print(f"[onnx2tf] wrote {TFL32} ({os.path.getsize(TFL32) / 1e6:.1f} MB)")
    if len(f16) == 1:
        shutil.copyfile(f16[0], TFL16)
        print(f"[onnx2tf] wrote {TFL16} ({os.path.getsize(TFL16) / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------------------------------- stage check
# Heuristic HTP/NPU flags for TFLite op types (see docs/verification/dinov2-export.md for the reasoning).
HTP_NOTES = {
    "GELU": "fine if the delegate maps GELU natively (QNN HTP has Gelu); otherwise it is decomposed",
    "ERF": "erf decomposition (GELU not fused) - often unsupported or slow on HTP",
    "EXP": "part of a decomposed softmax/GELU - check it is not on the hot path",
    "TANH": "tanh-GELU decomposition",
    "SOFTMAX": "softmax over 1025 tokens x 6 heads per block (1025x1025 score maps) - large, HTP handles it but it "
               "is the memory-bandwidth hot spot",
    "BATCH_MATMUL": "activation x activation matmul (attention) - supported on HTP, fp16 precision",
    "GATHER": "gather-heavy op - usually falls back to CPU",
    "GATHER_ND": "gather-heavy op - usually falls back to CPU",
    "CAST": "dtype cast - check for float64/int64",
    "SHAPE": "dynamic shape computation",
    "WHERE": "dynamic/select op",
    "DENSIFY": "sparse weights",
    "DEQUANTIZE": "fp16-weight model: fp16 -> fp32 weight dequantisation at load/first run",
    "TRANSPOSE": "layout shuffles (6-D or large transposes are slow on HTP)",
    "RESHAPE": "cheap, but many of them split HTP graphs if they are >4-D",
    "SQUARED_DIFFERENCE": "LayerNorm decomposition (MEAN/SQUARED_DIFFERENCE/RSQRT)",
    "RSQRT": "LayerNorm decomposition",
    "MEAN": "LayerNorm decomposition (reduction over the channel axis)",
    "STRIDED_SLICE": "slicing (CLS split)",
    "CUSTOM": "custom op - not runnable on the NPU",
    "FLEX": "TF Select (Flex) op - not in the builtin runtime",
}
BAD = {"ERF", "GATHER", "GATHER_ND", "SHAPE", "WHERE", "CUSTOM", "DENSIFY"}


def tflite_info(path, threads=4):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    ins, outs = it.get_input_details(), it.get_output_details()
    try:
        ops = collections.Counter(d["op_name"] for d in it._get_ops_details())
    except Exception as e:  # noqa: BLE001 - private API
        ops = collections.Counter({"?": 1})
        print("[ops] unavailable:", e)
    dtypes = sorted({str(t["dtype"].__name__) for t in it.get_tensor_details()})
    dyn = [d["name"] for d in ins + outs if -1 in list(d.get("shape_signature", d["shape"]))]
    return it, ins, outs, ops, dtypes, dyn


def run_tflite(it, ins, outs, x):
    it.set_tensor(ins[0]["index"], x)
    it.invoke()
    got = {}
    for o in outs:
        v = it.get_tensor(o["index"]).copy()
        if list(v.shape) == [1, GRID, GRID, DIM]:
            got["patch"] = v
        elif list(v.shape) == [1, DIM]:
            got["cls"] = v
        else:
            got[o["name"]] = v
    return got


def metrics(patch, cls, rp, rc):
    a = patch.reshape(-1, DIM).astype(np.float64)
    b = rp.reshape(-1, DIM).astype(np.float64)
    pc = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-30)
    ca, cb = cls.reshape(-1).astype(np.float64), rc.reshape(-1).astype(np.float64)
    return dict(mean_patch_cos=float(pc.mean()), min_patch_cos=float(pc.min()),
                cls_cos=float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-30)),
                patch_max_abs=float(np.abs(a - b).max()),
                patch_max_abs_rel=float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30)),
                l2n_patch_max_dist=float(np.linalg.norm(an - bn, axis=1).max()))


def median_ms(fn, n=10):
    fn()
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return round(statistics.median(ts), 1)


def stage_check(a):
    import onnx
    import onnxruntime as ort

    z = np.load(REF)
    keys = [k[len("image__"):] for k in z.files if k.startswith("image__")]
    res = {"model": NAME, "inputs": keys, "files": {}, "parity": {}, "ops": {}, "timing_ms_median10": {},
           "thresholds": {"mean_patch_cos": 0.99, "cls_cos": 0.99}}
    for p in (ONNX, TFL32, TFL16):
        if os.path.exists(p):
            res["files"][os.path.basename(p)] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}

    om = onnx.load(ONNX, load_external_data=False)
    res["ops"]["onnx"] = dict(collections.Counter(n.op_type for n in om.graph.node))
    res["onnx_io"] = {"inputs": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in om.graph.input],
                      "outputs": [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in om.graph.output]}
    del om
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    onames = [o.name for o in s.get_outputs()]
    res["parity"]["onnx_ort_" + ort.__version__] = {}
    for k in keys:
        y = dict(zip(onames, s.run(None, {"image": z["image__" + k].astype(np.float32)[None]})))
        res["parity"]["onnx_ort_" + ort.__version__][k] = metrics(y["patch"], y["cls"], z["patch__" + k], z["cls__" + k])
    x0 = z["image__" + keys[0]].astype(np.float32)[None]
    res["timing_ms_median10"]["onnx_ort_cpu_4threads"] = median_ms(lambda: s.run(None, {"image": x0}))
    del s

    ok = True
    for label, path in (("tflite_fp32", TFL32), ("tflite_fp16w", TFL16)):
        if not os.path.exists(path):
            continue
        it, ins, outs, ops, dtypes, dyn = tflite_info(path)
        res["ops"][label] = dict(sorted(ops.items(), key=lambda kv: -kv[1]))
        res.setdefault("tflite_io", {})[label] = {
            "inputs": [(d["name"], [int(v) for v in d["shape"]], d["dtype"].__name__) for d in ins],
            "outputs": [(d["name"], [int(v) for v in d["shape"]], d["dtype"].__name__) for d in outs],
            "tensor_dtypes": dtypes, "dynamic_io": dyn}
        try:
            res["tflite_io"][label]["signatures"] = {k: {kk: list(vv) for kk, vv in v.items()}
                                                    for k, v in it.get_signature_list().items()}
        except Exception:  # noqa: BLE001
            pass
        res["parity"][label] = {}
        for k in keys:
            got = run_tflite(it, ins, outs, z["image__" + k].astype(np.float32)[None])
            m = metrics(got["patch"], got["cls"], z["patch__" + k], z["cls__" + k])
            res["parity"][label][k] = m
            ok &= m["mean_patch_cos"] >= 0.99 and m["cls_cos"] >= 0.99
        res["timing_ms_median10"][label + "_cpu_4threads"] = median_ms(lambda: run_tflite(it, ins, outs, x0))
        it1 = tflite_info(path, threads=1)
        res["timing_ms_median10"][label + "_cpu_1thread"] = median_ms(lambda: run_tflite(it1[0], it1[1], it1[2], x0), n=5)
        flags = {op: HTP_NOTES.get(op, "") for op in ops if op in HTP_NOTES}
        res.setdefault("htp_flags", {})[label] = {
            "likely_unfriendly": sorted(op for op in ops if op in BAD or op.startswith("FLEX")),
            "float64_tensors": "float64" in dtypes, "int64_tensors": "int64" in dtypes,
            "dynamic_io": dyn, "notes": flags}
        del it, it1

    res["pass"] = bool(ok)
    with open(os.path.join(OUT, "dinov2_parity.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    for label, per in res["parity"].items():
        for k, m in per.items():
            print(f"{label:22s} {k:24s} mean patch cos {m['mean_patch_cos']:.6f} min {m['min_patch_cos']:.6f} "
                  f"cls cos {m['cls_cos']:.6f} max|d| {m['patch_max_abs']:.2e} (rel {m['patch_max_abs_rel']:.2e})")
    for label, o in res["ops"].items():
        print(f"[ops] {label}: {o}")
    print("[htp]", json.dumps(res.get("htp_flags", {}), indent=1)[:3000])
    print("[timing ms]", res["timing_ms_median10"])
    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["onnx", "tflite", "check"])
    ap.add_argument("--weights", default=WEIGHTS)
    a, a.extra = ap.parse_known_args()   # unknown flags are passed to onnx2tf (tflite stage)
    {"onnx": stage_onnx, "tflite": stage_tflite, "check": stage_check}[a.stage](a)


if __name__ == "__main__":
    main()
