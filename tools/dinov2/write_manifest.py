#!/usr/bin/env python3
r"""
Kaizen Eye 2 - write tools/dinov2/out/dinov2_s14_448.json (model card in the backbone_r18_320.json style) and
tools/dinov2/out/SHA256SUMS.txt (sha256sum format, every file in out/ except SHA256SUMS.txt itself).

  .venv\Scripts\python tools\dinov2\write_manifest.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
WEIGHTS_DIR = r"D:\kz-tmp\dinov2-weights"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fileinfo(name):
    p = os.path.join(OUT, name)
    return {"file": name, "bytes": os.path.getsize(p), "sha256": sha256(p)} if os.path.exists(p) else None


def main():
    parity = {}
    pj = os.path.join(OUT, "dinov2_parity.json")
    if os.path.exists(pj):
        with open(pj, encoding="utf-8") as fh:
            parity = json.load(fh)
    w = os.path.join(WEIGHTS_DIR, "model.safetensors")
    c = os.path.join(WEIGHTS_DIR, "config.json")
    meta = {
        "backbone": "dinov2_vits14",
        "input_size": 448,
        "grid_h": 32,
        "grid_w": 32,
        "dim": 384,
        "patch_size": 14,
        "input": "NHWC float32 RGB 0..255",
        "output": "NHWC float32 [1,grid_h,grid_w,dim]",
        "outputs": {
            "patch": "NHWC float32 [1,32,32,384]: final-LayerNorm patch tokens, row-major p = r*32 + c, NOT L2-normalised",
            "cls": "float32 [1,384]: final-LayerNorm CLS token"},
        "tflite_output_order": "output index 0 = patch [1,32,32,384], index 1 = cls [1,384] (tensor names "
                               "StatefulPartitionedCall:1 / :0; signature 'serving_default' outputs 'patch', 'cls') "
                               "- identify by shape",
        "l2_normalize_patches": True,
        "preprocess": "ImageNet mean (0.485,0.456,0.406) / std (0.229,0.224,0.225) on x/255, baked into the graph",
        "pos_embed": "pretrained 37x37 grid resized once to 32x32 (bicubic, align_corners=False, scale_factor "
                     "(32+0.1)/37, antialias off - the facebookresearch/dinov2 reference), stored as a constant",
        "pretrained": True,
        "license": "Apache-2.0",
        "source": "https://huggingface.co/facebook/dinov2-small",
        "weights_file": "model.safetensors",
        "weights_bytes": os.path.getsize(w) if os.path.exists(w) else None,
        "weights_sha256": sha256(w) if os.path.exists(w) else None,
        "config_sha256": sha256(c) if os.path.exists(c) else None,
        "files": {k: fileinfo(n) for k, n in (("tflite_fp32", "dinov2_s14_448_fp32.tflite"),
                                               ("tflite_fp16w", "dinov2_s14_448_fp16w.tflite"),
                                               ("onnx", "dinov2_s14_448.onnx"),
                                               ("knn", "knn_p1024_d384_k2400.tflite"))},
        "precision": {"tflite_fp32": "float32 weights and activations",
                      "tflite_fp16w": "float16 weights (DEQUANTIZE to float32 at run time), float32 activations"},
        "parity_vs_torch": {k: {kk: {m: round(v[m], 7) for m in ("mean_patch_cos", "min_patch_cos", "cls_cos")}
                                for kk, v in per.items()} for k, per in parity.get("parity", {}).items()},
        "laptop_cpu_ms_median10": parity.get("timing_ms_median10"),
        "exporter": "tools/dinov2/export_dinov2.py (torch 2.13 -> ONNX opset 17 -> onnxsim -> onnx2tf 1.29.24 / TF 2.19.1)",
    }
    with open(os.path.join(OUT, "dinov2_s14_448.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    lines = []
    for n in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, n)
        if os.path.isfile(p) and n != "SHA256SUMS.txt":
            lines.append(f"{sha256(p)}  {n}")
    with open(os.path.join(OUT, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
