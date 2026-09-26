#!/usr/bin/env python3
"""
Kaizen Eye - backbone converter.  NO TRAINING: takes a pretrained ImageNet CNN and turns it into a
single-output "patch feature" model that runs on the phone through LiteRT.

  input  : NHWC float32 [1, S, S, 3], RGB, values 0..255   (normalisation is baked into the graph)
  output : NHWC float32 [1, gh, gw, D]                      (one feature vector per image patch)

PatchCore-style processing is baked into the graph, so the Android side only does k-NN:
  1. taps from several stages of the backbone
  2. 3x3 average pooling ("locally aware" patch features)
  3. bilinear upsample to the first tap's grid, concatenate along channels
  4. optional fixed channel-group averaging (384 -> 128 ...). Fixed weights, nothing is learned.

Examples
  python tools/convert_backbone.py --backbone resnet18 --size 256 --dim 128 --out out/backbone_r18_256.tflite
  python tools/convert_backbone.py --backbone mobilenet_v3_small --size 320 --dim 56 --out out/backbone_mnv3_320.tflite
  python tools/convert_backbone.py --random-weights ...   # offline smoke test only (features are meaningless)
Add --onnx to also write a .onnx file (backup path: ONNX Runtime Mobile).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFeatureNet(nn.Module):
    def __init__(self, backbone: str, size: int, dim: int, pretrained: bool):
        super().__init__()
        import torchvision as tv

        if backbone == "resnet18":
            w = tv.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = tv.models.resnet18(weights=w)
            stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)  # stride 4
            # stage 0 ends at stride 8 (tap 0: 128 ch), stage 1 ends at stride 16 (tap 1: 256 ch)
            self.stages = nn.ModuleList([nn.Sequential(stem, m.layer2), m.layer3])
            self.tap_channels = [128, 256]
            self.tap_strides = [8, 16]
        elif backbone == "mobilenet_v3_small":
            w = tv.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            m = tv.models.mobilenet_v3_small(weights=w)
            f = m.features
            self.stages = nn.ModuleList(
                [nn.Sequential(*f[0:4]), nn.Sequential(*f[4:9]), nn.Sequential(*f[9:12])]
            )
            self.tap_channels = [24, 48, 96]
            self.tap_strides = [8, 16, 32]
        else:
            raise SystemExit(f"unknown backbone {backbone}")

        self.size = size
        self.total = sum(self.tap_channels)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0)
        self.aggr = nn.AvgPool2d(3, stride=1, padding=1)

        if dim in (0, self.total):
            self.dim, self.proj = self.total, None
        else:
            if self.total % dim != 0:
                raise SystemExit(f"--dim must divide {self.total} (or be 0 for no reduction)")
            g = self.total // dim
            wgt = torch.zeros(dim, self.total, 1, 1)
            for i in range(dim):
                wgt[i, i * g:(i + 1) * g] = 1.0 / g
            self.proj = nn.Conv2d(self.total, dim, 1, bias=False)
            self.proj.weight.data.copy_(wgt)
            self.proj.weight.requires_grad_(False)
            self.dim = dim

        self.grid = size // self.tap_strides[0]
        self.eval()

    def forward(self, x):  # x: [1, S, S, 3] float32 0..255 RGB
        x = x.permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        feats = []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        gh, gw = feats[0].shape[-2], feats[0].shape[-1]
        outs = []
        for f in feats:
            f = self.aggr(f)
            if f.shape[-2] != gh or f.shape[-1] != gw:
                f = F.interpolate(f, size=(gh, gw), mode="bilinear", align_corners=False)
            outs.append(f)
        y = torch.cat(outs, dim=1)
        if self.proj is not None:
            y = self.proj(y)
        return y.permute(0, 2, 3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--size", type=int, default=256, help="square input size, multiple of 32")
    ap.add_argument("--dim", type=int, default=128, help="output channels (0 = no reduction)")
    ap.add_argument("--out", default="out/backbone.tflite")
    ap.add_argument("--random-weights", action="store_true", help="skip ImageNet download (smoke test only)")
    ap.add_argument("--onnx", action="store_true", help="also export ONNX (backup runtime path)")
    a = ap.parse_args()
    if a.size % 32:
        raise SystemExit("--size must be a multiple of 32")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.manual_seed(0)
    net = PatchFeatureNet(a.backbone, a.size, a.dim, pretrained=not a.random_weights)
    sample = torch.randint(0, 256, (1, a.size, a.size, 3)).float()
    with torch.no_grad():
        ref = net(sample).numpy()
    print(f"[torch] output {ref.shape}  grid={net.grid}x{net.grid}  dim={net.dim}")

    import litert_torch

    edge = litert_torch.convert(net, (sample,))
    edge.export(a.out)
    print(f"[litert] wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)")

    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=a.out)
    it.allocate_tensors()
    i0, o0 = it.get_input_details()[0], it.get_output_details()
    print("[litert] input ", i0["shape"], i0["dtype"].__name__, "| outputs:", [(o["shape"].tolist(), o["dtype"].__name__) for o in o0])
    it.set_tensor(i0["index"], sample.numpy())
    it.invoke()
    got = it.get_tensor(o0[0]["index"])
    err = float(np.abs(got - ref).max())
    rel = err / (float(np.abs(ref).max()) + 1e-9)
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9))
    print(f"[verify] max|diff|={err:.3e}  relative={rel:.3e}  cosine={cos:.6f}")
    if got.shape != ref.shape or rel > 1e-3:
        print("!! tflite output does not match torch output - do not ship this model")
        sys.exit(2)

    try:
        ops = collections.Counter(d["op_name"] for d in it._get_ops_details())
        print("[ops]", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
        if ops.get("TRANSPOSE", 0) > 2:
            print("   note: several TRANSPOSE ops - fine on CPU, check GPU/NPU latency")
    except Exception as e:  # private API, may change
        print("[ops] unavailable:", e)

    meta = dict(
        backbone=a.backbone, input_size=a.size, grid_h=int(net.grid), grid_w=int(net.grid), dim=int(net.dim),
        input="NHWC float32 RGB 0..255", output="NHWC float32 [1,grid_h,grid_w,dim]",
        taps=net.tap_strides, pretrained=not a.random_weights,
    )
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[meta]", meta)

    if a.onnx:
        p = os.path.splitext(a.out)[0] + ".onnx"
        try:
            torch.onnx.export(net, (sample,), p, input_names=["image"], output_names=["features"], opset_version=17, dynamo=False)
            import onnxruntime as ort

            s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            y = s.run(None, {"image": sample.numpy()})[0]
            print(f"[onnx] wrote {p}  max|diff| vs torch = {np.abs(y - ref).max():.3e}")
        except Exception as e:
            print("[onnx] export failed:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    main()
