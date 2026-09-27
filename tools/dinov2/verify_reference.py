#!/usr/bin/env python3
r"""
Kaizen Eye 2 - independent check of tools/dinov2/dinov2_s14.py (the PyTorch module that gets exported).

A second, deliberately different implementation of DINOv2 ViT-S/14 in float64 numpy:
  * its own safetensors reader (seek + read per tensor, no shared code with dinov2_s14.read_safetensors);
  * its own bicubic position-embedding resize (PyTorch's cubic-convolution kernel, A = -0.75, align_corners=False,
    source scale = 1 / scale_factor, clamped taps), applied as two separable [g, 37] weight matrices;
  * patch embedding as im2col + matmul (no convolution), normalisation as x/255 - mean / std;
  * fused qkv projection (the facebookresearch layout: concat(Wq, Wk, Wv)), q * 0.125 AFTER the projection,
    numerically stable softmax, exact erf GELU (scipy), explicit LayerNorm, LayerScale NOT folded.
It then compares the torch module (float32, folded scale/LayerScale) with it on fixed inputs and checks two
structural properties: the resize is the identity at the pretrained 37x37 grid, and the torch module equals
the numpy one at 518 px as well (no interpolation path).

  .venv\Scripts\python tools\dinov2\verify_reference.py [--weights D:\kz-tmp\dinov2-weights\model.safetensors]
Pass: min per-patch cosine >= 0.9999 and CLS cosine >= 0.9999 (float32 vs float64), else exit 2.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_WEIGHTS = r"D:\kz-tmp\dinov2-weights\model.safetensors"


# ------------------------------------------------------------------------------------------------ weights (own reader)
def load_weights_f64(path):
    out = {}
    with open(path, "rb") as fh:
        hlen = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(hlen))
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            assert info["dtype"] == "F32", info
            s, e = info["data_offsets"]
            fh.seek(8 + hlen + s)
            raw = fh.read(e - s)
            out[name] = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(info["shape"])
    return out


# ------------------------------------------------------------------------------------------------ bicubic (own code)
def cubic_weights_matrix(n_in, n_out, scale_factor, a=-0.75):
    """[n_out, n_in] matrix M with out = M @ in (one axis), PyTorch bicubic semantics."""
    m = np.zeros((n_out, n_in), np.float64)
    step = 1.0 / scale_factor
    for i in range(n_out):
        src = step * (i + 0.5) - 0.5
        i0 = int(math.floor(src))
        t = src - i0
        ws = [((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
              ((a + 2) * t - (a + 3)) * t * t + 1,
              ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1,
              ((a * (2 - t) - 5 * a) * (2 - t) + 8 * a) * (2 - t) - 4 * a]
        for j, wj in enumerate(ws):
            idx = min(max(i0 - 1 + j, 0), n_in - 1)
            m[i, idx] += wj
    return m


def resize_pos(pos_patch, grid, offset=0.1):
    """pos_patch [37*37, D] -> [grid*grid, D] (float64)."""
    mm, d = pos_patch.shape
    m = int(round(math.sqrt(mm)))
    if grid == m:
        return pos_patch.copy()
    w = cubic_weights_matrix(m, grid, (grid + offset) / m)
    p = pos_patch.reshape(m, m, d)
    y = np.einsum("im,mnd->ind", w, p)
    y = np.einsum("jn,ind->ijd", w, y)
    return y.reshape(grid * grid, d)


# ------------------------------------------------------------------------------------------------ forward (float64)
def layer_norm(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def forward_np(W, img_u8, size):
    """img [S,S,3] uint8/float -> (patch [g,g,384], cls [384]) float64."""
    g = size // 14
    x = np.asarray(img_u8, np.float64) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    cols = x.reshape(g, 14, g, 14, 3).transpose(0, 2, 4, 1, 3).reshape(g * g, 3 * 14 * 14)   # (c, ky, kx)
    wpe = W["embeddings.patch_embeddings.projection.weight"].reshape(384, -1)
    tok = cols @ wpe.T + W["embeddings.patch_embeddings.projection.bias"]
    pos = W["embeddings.position_embeddings"][0]
    pos_full = np.concatenate([pos[:1], resize_pos(pos[1:], g)], 0)
    x = np.concatenate([W["embeddings.cls_token"].reshape(1, 384), tok], 0) + pos_full      # [T, 384]
    T = x.shape[0]
    for i in range(12):
        p = f"encoder.layer.{i}."
        h = layer_norm(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
        wqkv = np.concatenate([W[p + "attention.attention." + n + ".weight"] for n in ("query", "key", "value")], 0)
        bqkv = np.concatenate([W[p + "attention.attention." + n + ".bias"] for n in ("query", "key", "value")], 0)
        qkv = (h @ wqkv.T + bqkv).reshape(T, 3, 6, 64).transpose(1, 2, 0, 3)                # [3, H, T, 64]
        q, k, v = qkv[0] * (64 ** -0.5), qkv[1], qkv[2]
        s = q @ k.transpose(0, 2, 1)                                                          # [H, T, T]
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(T, 384)
        o = o @ W[p + "attention.output.dense.weight"].T + W[p + "attention.output.dense.bias"]
        x = x + W[p + "layer_scale1.lambda1"] * o
        h = layer_norm(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
        h = h @ W[p + "mlp.fc1.weight"].T + W[p + "mlp.fc1.bias"]
        h = 0.5 * h * (1.0 + erf(h / math.sqrt(2.0)))
        h = h @ W[p + "mlp.fc2.weight"].T + W[p + "mlp.fc2.bias"]
        x = x + W[p + "layer_scale2.lambda1"] * h
    x = layer_norm(x, W["layernorm.weight"], W["layernorm.bias"])
    return x[1:].reshape(g, g, 384), x[0]


def compare(patch_a, cls_a, patch_b, cls_b):
    a = patch_a.reshape(-1, 384).astype(np.float64)
    b = patch_b.reshape(-1, 384).astype(np.float64)
    pc = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    ca, cb = cls_a.reshape(-1).astype(np.float64), cls_b.reshape(-1).astype(np.float64)
    return dict(mean_patch_cos=float(pc.mean()), min_patch_cos=float(pc.min()),
                cls_cos=float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb))),
                patch_max_abs=float(np.abs(a - b).max()), patch_max_abs_rel=float(np.abs(a - b).max() / np.abs(b).max()),
                cls_max_abs=float(np.abs(ca - cb).max()))


def test_images(size):
    rng = np.random.default_rng(0)
    ims = {"random_u8_seed0": rng.integers(0, 256, (size, size, 3)).astype(np.uint8)}
    photos = sorted(glob.glob(os.path.join(ROOT, "demo_photos", "*", "*.jpg")))
    if photos:
        sys.path.insert(0, os.path.join(ROOT, "tools", "lab"))
        from viz import load_square

        ims["photo_" + os.path.basename(photos[0])] = load_square(photos[0], size)
    return ims


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "dinov2_reference_check.json"))
    a = ap.parse_args()
    import torch

    sys.path.insert(0, HERE)
    import dinov2_s14 as dm

    torch.set_num_threads(4)
    W = load_weights_f64(a.weights)
    res = {"weights": a.weights, "checks": {}}

    # structural: the resize at the pretrained grid is the identity; 448 resize vs torch F.interpolate
    pos = W["embeddings.position_embeddings"][0][1:]
    res["checks"]["resize_identity_at_37"] = float(np.abs(resize_pos(pos, 37) - pos).max())
    tp = dm.interpolate_patch_pos(pos.astype(np.float32), 32).numpy().astype(np.float64)
    res["checks"]["pos_resize_448_torch_vs_numpy_max_abs"] = float(np.abs(tp - resize_pos(pos, 32)).max())
    res["checks"]["pos_resize_448_numpy_max_abs_value"] = float(np.abs(resize_pos(pos, 32)).max())
    print("pos-embed resize: identity@37 max|d| %.2e, torch vs numpy @32 max|d| %.2e (values up to %.3f)" % (
        res["checks"]["resize_identity_at_37"], res["checks"]["pos_resize_448_torch_vs_numpy_max_abs"],
        res["checks"]["pos_resize_448_numpy_max_abs_value"]))

    ok = True
    for size in (448, 518):
        net = dm.build(a.weights, size)
        for name, im in test_images(size).items():
            t0 = time.time()
            with torch.no_grad():
                pt, ct = net(torch.as_tensor(im, dtype=torch.float32)[None])
            pn, cn = forward_np(W, im, size)
            c = compare(pt.numpy()[0], ct.numpy()[0], pn, cn)
            c["seconds"] = round(time.time() - t0, 1)
            res["checks"][f"torch_vs_numpy_{size}_{name}"] = c
            print(f"{size} {name:26s} mean patch cos {c['mean_patch_cos']:.9f}  min {c['min_patch_cos']:.9f}  "
                  f"cls cos {c['cls_cos']:.9f}  patch max|d| {c['patch_max_abs']:.2e} (rel {c['patch_max_abs_rel']:.2e})")
            ok &= c["min_patch_cos"] >= 0.9999 and c["cls_cos"] >= 0.9999
        del net
    res["pass"] = bool(ok)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", a.out, "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
