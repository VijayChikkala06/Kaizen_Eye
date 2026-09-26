#!/usr/bin/env python3
r"""
Kaizen Eye - PatchCore nearest-neighbour search as a tiny LiteRT (.tflite) model.

The app used to find, for each patch vector of an image, the nearest of K memory-bank rows in JavaScript (349 ms on
a phone for 1,024 patches x 1,024 rows). As a matrix product this is 2*P*D*K FLOP (1 GFLOP for the 320 px backbone:
P = 1,600 patches, K = 2,400 rows), which XNNPACK / a GPU / an NPU do in milliseconds. The model has NO weights - the
bank is an input - so one model serves every profile.

  inputs   feats     float32 [P, D]   patch features of one image (the backbone output, row-major)
           bankT     float32 [D, K]   memory-bank rows, transposed (--op bmm, default)
           bank      float32 [K, D]   the same, not transposed      (--op fc: FULLY_CONNECTED weight layout)
           bankNorm  float32 [K]      |b_k|^2 + mask; use a huge value (1e30) for padded / excluded rows
  output   minD2     float32 [P]      min_k |f_p - b_k|^2  =  max(0, |f_p|^2 + min_k(bankNorm_k - 2 f_p.b_k))

Same maths as tools/lab/patchcore_ref.py sqdist (|a|^2 + |b|^2 - 2 a.b, clamped at 0). Builtin ops only:
MUL, BATCH_MATMUL (or FULLY_CONNECTED), ADD, REDUCE_MIN, SUM, ADD+RELU. The TF 2.19 GPU-compatibility analyzer
accepts both forms; NNAPI maps BATCH_MATMUL from feature level 6 (Android 12 drivers), FULLY_CONNECTED from API 27.

Why BATCH_MATMUL is the default (laptop, ai_edge_litert XNNPACK, P=1024 D=128 K=1536):
  bmm  whole graph delegated to XNNPACK in ONE partition:   5.7 ms @1 thread, 1.6 ms @4 threads
  fc   XNNPACK does not take FULLY_CONNECTED with runtime (non-constant) weights, so it runs on the builtin
       kernel between two XNNPACK partitions:                11.9 ms @1 thread, 11.2 ms @4 threads
  (BatchMatMul with adj_y=True is rewritten to FULLY_CONNECTED by the converter, so it is no alternative.)

Profiles with fewer bank rows than the model's K pad the bank with zero columns whose bankNorm is 1e30. The app ships
K buckets matching its bank sizes (k = max(256, floor(0.05 * frames * 1600)) = 80 per enrolment photo: 10 photos ->
800, 20 -> 1,600, 30 (the cap) -> 2,400) and uses the smallest that fits (mobile/src/ml/knn.ts); banks larger than the
biggest bucket are split into chunks and the per-chunk minima combined.

Build (TensorFlow venv, Windows: .venv-convert; defaults P 1600, D 128, K 800 1600 2400 -> mobile\assets\models):
  .venv-convert\Scripts\python tools\make_knn_model.py
Check / time an existing model with any interpreter (ai_edge_litert in .venv, or TensorFlow):
  .venv\Scripts\python tools\make_knn_model.py --check mobile\assets\models\knn_p1600_d128_k2400.tflite
Options: --op fc|bmm, --P, --D, --threads 1 4, --no-realistic (skip the demo-photo check), --gpu-report.
"""
import argparse
import glob
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK = np.float32(1e30)  # bankNorm value that makes a bank row never win (padding / leave-one-out exclusion)


def model_name(P, D, K, op):
    return f"knn_p{P}_d{D}_k{K}{'' if op == 'bmm' else '_' + op}.tflite"


# ---------------------------------------------------------------------------------------------------------- build
def build(P, D, K, op):
    """-> .tflite bytes. Needs TensorFlow (2.19 tested)."""
    import tensorflow as tf

    bank_spec = tf.TensorSpec([K, D], tf.float32, name="bank") if op == "fc" else tf.TensorSpec([D, K], tf.float32, name="bankT")

    @tf.function(input_signature=[tf.TensorSpec([P, D], tf.float32, name="feats"), bank_spec,
                                  tf.TensorSpec([K], tf.float32, name="bankNorm")])
    def knn(feats, bank, bankNorm):
        f2 = feats * -2.0                                          # MUL [P,D]  (x -2 is exact in floating point)
        if op == "fc":
            cross = tf.matmul(f2, bank, transpose_b=True)          # FULLY_CONNECTED -> [P,K] = -2 f.b
        else:
            cross = tf.raw_ops.BatchMatMulV2(x=f2, y=bank)         # rank-2 BATCH_MATMUL [P,D]x[D,K] -> [P,K]
        near = tf.reduce_min(cross + bankNorm, axis=1)             # ADD (broadcast) + REDUCE_MIN -> [P]
        fnorm = tf.reduce_sum(feats * feats, axis=1)               # MUL + SUM -> [P]
        return {"minD2": tf.nn.relu(near + fnorm)}                 # ADD + RELU (clamp >= 0)

    conv = tf.lite.TFLiteConverter.from_concrete_functions([knn.get_concrete_function()], knn)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # builtin ops only, float32, no quantisation
    return conv.convert()


# ------------------------------------------------------------------------------------------------ interpreter glue
def make_interpreter(model_path, threads):
    try:
        from ai_edge_litert.interpreter import Interpreter
        which = "ai_edge_litert"
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
        which = "tensorflow.lite"
    it = Interpreter(model_path=model_path, num_threads=threads)
    it.allocate_tensors()
    return it, which


def io_map(it):
    """Map inputs by name (feats / bank / bankT / bankNorm), falling back to shape."""
    ins = it.get_input_details()
    out = it.get_output_details()[0]
    m = {}
    for d in ins:
        n = d["name"]
        key = "bankNorm" if "bankNorm" in n else "bankT" if "bankT" in n else "bank" if "bank" in n else "feats" if "feats" in n else None
        m[key] = d
    if None in m or len(m) != 3:
        raise SystemExit(f"cannot identify inputs: {[d['name'] for d in ins]}")
    return m, out


def run(it, m, out, feats, bank, bank_norm):
    it.set_tensor(m["feats"]["index"], feats)
    if "bank" in m:
        it.set_tensor(m["bank"]["index"], bank)
    else:
        it.set_tensor(m["bankT"]["index"], np.ascontiguousarray(bank.T))
    it.set_tensor(m["bankNorm"]["index"], bank_norm)
    it.invoke()
    return it.get_tensor(out["index"]).copy()


def ops_of(it):
    try:
        return [o["op_name"] for o in it._get_ops_details()]
    except Exception:  # noqa: BLE001 - private API, informative only
        return ["?"]


# ------------------------------------------------------------------------------------------------ reference + data
def ref_min_d2(feats, bank, excluded=None):
    """float64 brute force: min_k |f - b_k|^2 over the rows not excluded."""
    f = feats.astype(np.float64)
    b = bank.astype(np.float64)
    if excluded is not None:
        b = b[~excluded]
    d2 = (f * f).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * f @ b.T
    return np.maximum(d2.min(1), 0.0)


def padded_inputs(bank, K, excluded=None):
    kk, D = bank.shape
    assert kk <= K
    bp = np.zeros((K, D), np.float32)
    bp[:kk] = bank
    norm = np.full(K, MASK, np.float32)
    norm[:kk] = (bank.astype(np.float64) ** 2).sum(1).astype(np.float32)
    if excluded is not None:
        norm[:kk][excluded] = MASK
    return bp, norm


def errors(got, ref):
    """rel. L2 error, max abs error / max(ref), max element rel. error (floor: 1e-3 * max(ref)), cosine."""
    got = got.astype(np.float64)
    scale = max(float(ref.max()), 1e-30)
    rel_l2 = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
    max_abs = float(np.abs(got - ref).max() / scale)
    max_rel = float((np.abs(got - ref) / np.maximum(ref, 1e-3 * scale)).max())
    cos = float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
    # the app scores with sqrt(minD2): error in distance units, relative to the largest distance
    sq = float(np.abs(np.sqrt(np.maximum(got, 0)) - np.sqrt(ref)).max() / max(np.sqrt(scale), 1e-30))
    return dict(rel_l2=rel_l2, max_abs=max_abs, max_rel=max_rel, cos=cos, sqrt_max=sq)


def realistic_data(P, D):
    """Backbone features of the demo photos: bank = greedy coreset of the 20 good frames (the app's recipe),
    queries = the 11 test photos (good + defects)."""
    sys.path.insert(0, os.path.join(ROOT, "tools", "lab"))
    from patchcore_ref import greedy_coreset  # noqa: E402
    from viz import load_square  # noqa: E402

    grid = {1024: 256, 1600: 320}.get(P)  # the shipped backbone whose output has P patches
    bb = os.path.join(ROOT, "mobile", "assets", "models", f"backbone_r18_{grid}.tflite")
    good = sorted(glob.glob(os.path.join(ROOT, "demo_photos", "good", "*.jpg")))
    test = sorted(glob.glob(os.path.join(ROOT, "demo_photos", "test", "*.jpg")))
    if not grid or not good or not test or not os.path.exists(bb):
        return None
    it, _ = make_interpreter(bb, 4)
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    size = int(i["shape"][1])

    def emb(path):
        it.set_tensor(i["index"], load_square(path, size).astype(np.float32)[None])
        it.invoke()
        return it.get_tensor(o["index"])[0].reshape(-1, D).copy()

    frames = [emb(p) for p in good]
    per = frames[0].shape[0]
    x = np.concatenate(frames)
    k = max(256, int(0.05 * len(x)))
    sel = greedy_coreset(x, k)
    return dict(bank=x[sel], bank_frame=sel // per, frames=frames, queries=[emb(p) for p in test], k=k)


# ------------------------------------------------------------------------------------------------------- checks
def check(model_path, threads=(1, 4), realistic=True, runs=30):
    it, which = make_interpreter(model_path, 1)
    m, out = io_map(it)
    P, D = (int(v) for v in m["feats"]["shape"])
    K = int(m["bankNorm"]["shape"][0])
    print(f"\n== {os.path.basename(model_path)}  ({os.path.getsize(model_path)} bytes, {which})")
    print(f"   inputs  {[(d['name'], [int(v) for v in d['shape']]) for d in it.get_input_details()]}")
    print(f"   output  {out['name']} {[int(v) for v in out['shape']]}")
    print(f"   ops     {ops_of(it)}")
    rng = np.random.default_rng(0)
    worst = 0.0

    # 1) random data, full + partially filled + masked bank
    for label, kk, mask_frac in (("random, full bank", K, 0.0), ("random, 60% filled", int(0.6 * K), 0.0),
                                 ("random, LOO mask", K, 0.05)):
        f = rng.random((P, D), dtype=np.float32)
        b = rng.random((kk, D), dtype=np.float32)
        ex = (rng.random(kk) < mask_frac) if mask_frac else None
        bp, norm = padded_inputs(b, K, ex)
        e = errors(run(it, m, out, f, bp, norm), ref_min_d2(f, b, ex))
        worst = max(worst, e["max_rel"])
        print(f"   {label:24s} rel_l2 {e['rel_l2']:.2e}  max_abs/max {e['max_abs']:.2e}  max_rel {e['max_rel']:.2e}"
              f"  sqrt-dist max err {e['sqrt_max']:.2e}  cos {e['cos']:.9f}")

    # 2) realistic: demo-photo features, the app's bank recipe, inspection + leave-one-out masks
    if realistic:
        data = realistic_data(P, D)
        if data is None:
            print("   (realistic check skipped: demo photos / backbone not found)")
        else:
            bank = data["bank"]
            if len(bank) <= K:
                es = []
                for q in data["queries"]:
                    bp, norm = padded_inputs(bank, K)
                    es.append(errors(run(it, m, out, q, bp, norm), ref_min_d2(q, bank)))
                for fi, fr in enumerate(data["frames"][:5]):  # leave-one-out: exclude the frame's own rows
                    ex = data["bank_frame"] == fi
                    bp, norm = padded_inputs(bank, K, ex)
                    es.append(errors(run(it, m, out, fr, bp, norm), ref_min_d2(fr, bank, ex)))
                agg = {k: max(e[k] for e in es) for k in ("rel_l2", "max_abs", "max_rel", "sqrt_max")}
                worst = max(worst, agg["max_rel"])
                print(f"   realistic k={len(bank)} (11 photos + 5 LOO)  rel_l2 {agg['rel_l2']:.2e}  "
                      f"max_abs/max {agg['max_abs']:.2e}  max_rel {agg['max_rel']:.2e}  sqrt-dist max err {agg['sqrt_max']:.2e}")
            else:
                print(f"   (realistic bank k={len(bank)} > K={K}: skipped for this bucket)")

    # 3) timing
    f = rng.random((P, D), dtype=np.float32)
    bp, norm = padded_inputs(rng.random((K, D), dtype=np.float32), K)
    for th in threads:
        it2, _ = make_interpreter(model_path, th)
        m2, out2 = io_map(it2)
        for _ in range(3):
            run(it2, m2, out2, f, bp, norm)
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            run(it2, m2, out2, f, bp, norm)
            ts.append((time.perf_counter() - t) * 1000)
        print(f"   {th} thread{'s' if th > 1 else ' '}: median {statistics.median(ts):6.2f} ms   min {min(ts):6.2f} ms"
              f"   ({2 * P * D * K / 1e6:.0f} MFLOP matmul)")
    return worst


def gpu_report(model_bytes):
    import tensorflow as tf
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_content=model_bytes, gpu_compatibility=True)
    txt = buf.getvalue()
    lines = [ln for ln in txt.splitlines() if "GPU" in ln or "Op#" in ln]
    return "\n".join("   " + ln for ln in lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=int, default=1600, help="patches per image (40x40 grid of the 320 px backbone)")
    ap.add_argument("--D", type=int, default=128, help="feature dim")
    ap.add_argument("--K", type=int, nargs="+", default=[800, 1600, 2400], help="bank-size buckets")
    ap.add_argument("--op", choices=["bmm", "fc"], default="bmm")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "mobile", "assets", "models"))
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--no-realistic", action="store_true", help="skip the demo-photo check")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--gpu-report", action="store_true", help="print TF's GPU-delegate compatibility analysis")
    ap.add_argument("--check", nargs="+", metavar="MODEL", help="only check / time existing .tflite files")
    a = ap.parse_args()

    if a.check:
        for pth in a.check:
            check(pth, a.threads, not a.no_realistic)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    for K in a.K:
        path = os.path.join(a.out_dir, model_name(a.P, a.D, K, a.op))
        data = build(a.P, a.D, K, a.op)
        with open(path, "wb") as fh:
            fh.write(data)
        print(f"wrote {path} ({len(data)} bytes)")
        if a.gpu_report:
            print(gpu_report(data))
        if not a.no_check:
            worst = check(path, a.threads, not a.no_realistic)
            if worst > 1e-3:
                raise SystemExit(f"FAILED: max element relative error {worst:.2e} > 1e-3")


if __name__ == "__main__":
    main()
