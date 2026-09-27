"""
Kaizen Eye 2 - minimal, self-contained DINOv2 ViT-S/14 (facebook/dinov2-small, Apache-2.0) for a FIXED input size.

No transformers / timm / torch.hub: the safetensors file is parsed here (8-byte little-endian header length + JSON
header + raw little-endian tensors) and loaded into a plain PyTorch module that mirrors the reference implementation
(facebookresearch/dinov2 `vision_transformer.py`, `layers/{attention,block,layer_scale,mlp}.py`):

  image NHWC float32 [1, S, S, 3], RGB 0..255
    -> (x - 255*mean) / (255*std)           ImageNet mean/std baked in (same convention as backbone_r18_320)
    -> Conv2d(3, 384, k=14, s=14)           patch embedding -> [1, 384, S/14, S/14] -> tokens [1, (S/14)^2, 384]
    -> [CLS + pos[0]] ++ (tokens + pos[1:])  position embeddings interpolated ONCE here (bicubic, from the 37x37
                                            pretrained grid, exactly like the reference: scale_factor
                                            (g + 0.1)/37, antialias off) and stored as constants
    -> 12 pre-norm blocks: x += ls1 * attn(norm1(x)); x += ls2 * mlp(norm2(x))   (6 heads x 64, exact-erf GELU)
    -> final LayerNorm (eps 1e-6)
  outputs  patch NHWC [1, S/14, S/14, 384]  final-norm patch tokens, NOT L2-normalised (the app normalises)
           cls        [1, 384]              final-norm CLS token

Export-friendly rewrites (checked against an independent float64 numpy implementation in verify_reference.py):
  * the attention scale 1/sqrt(64) = 0.125 (a power of two) is folded into the query weights and bias -> bit-exact;
  * LayerScale (gamma) is folded into the preceding projection (attn.proj, mlp.fc2) weights and biases -> fp32
    rounding-level differences only;
  * separate q/k/v Linear layers (the HF checkpoint layout) keep every tensor <= 4-D (no 5-D qkv reshape).
"""
import json
import math
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HF_URL = "https://huggingface.co/facebook/dinov2-small"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PATCH = 14
DIM = 384
HEADS = 6
DEPTH = 12
MLP_HIDDEN = 1536
LN_EPS = 1e-6
PRETRAINED_GRID = 37           # 518 / 14
INTERPOLATE_OFFSET = 0.1       # reference kludge (dinov2 hubconf default interpolate_offset=0.1, antialias=False)

_DT = {"F32": np.float32, "F16": np.float16, "BF16": None, "F64": np.float64, "I64": np.int64, "I32": np.int32}


def read_safetensors(path):
    """-> {name: np.ndarray} (copies, native byte order). Parses the format directly (no safetensors package)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    (n,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8:8 + n].decode("utf-8"))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = _DT.get(info["dtype"])
        if dt is None:
            raise ValueError(f"unsupported dtype {info['dtype']} for {name}")
        s, e = info["data_offsets"]
        arr = np.frombuffer(blob, dtype=np.dtype(dt).newbyteorder("<"), count=(e - s) // np.dtype(dt).itemsize,
                            offset=base + s)
        out[name] = arr.reshape(info["shape"]).astype(dt, copy=True)
    return out


def interpolate_patch_pos(pos_patch, grid):
    """pos_patch [M*M, D] (pretrained 37x37 grid) -> [grid*grid, D], the reference way (float32, bicubic,
    align_corners=False, scale_factor=(grid+0.1)/M, no antialias). Returned as a float32 torch tensor."""
    mm, d = pos_patch.shape
    m = int(round(math.sqrt(mm)))
    assert m * m == mm
    if grid == m:
        return torch.as_tensor(pos_patch, dtype=torch.float32)
    t = torch.as_tensor(pos_patch, dtype=torch.float32).reshape(1, m, m, d).permute(0, 3, 1, 2)
    sf = float(grid + INTERPOLATE_OFFSET) / m
    y = F.interpolate(t, scale_factor=(sf, sf), mode="bicubic", antialias=False)
    assert tuple(y.shape[-2:]) == (grid, grid), y.shape
    return y.permute(0, 2, 3, 1).reshape(grid * grid, d).contiguous()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=LN_EPS)
        self.q = nn.Linear(DIM, DIM)
        self.k = nn.Linear(DIM, DIM)
        self.v = nn.Linear(DIM, DIM)
        self.proj = nn.Linear(DIM, DIM)
        self.norm2 = nn.LayerNorm(DIM, eps=LN_EPS)
        self.fc1 = nn.Linear(DIM, MLP_HIDDEN)
        self.act = nn.GELU()                    # exact (erf) GELU, like the reference
        self.fc2 = nn.Linear(MLP_HIDDEN, DIM)

    def forward(self, x):                       # x [1, T, D]
        t = x.shape[1]
        hd = DIM // HEADS
        h = self.norm1(x)
        q = self.q(h).reshape(1, t, HEADS, hd).permute(0, 2, 1, 3)       # [1, H, T, hd] (scale folded into q)
        k = self.k(h).reshape(1, t, HEADS, hd).permute(0, 2, 3, 1)       # [1, H, hd, T]
        v = self.v(h).reshape(1, t, HEADS, hd).permute(0, 2, 1, 3)       # [1, H, T, hd]
        a = torch.softmax(torch.matmul(q, k), dim=-1)                    # [1, H, T, T]
        o = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(1, t, DIM)
        x = x + self.proj(o)                                             # LayerScale folded into proj
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))              # LayerScale folded into fc2
        return x


class DinoV2S14(nn.Module):
    """Fixed-size DINOv2 ViT-S/14. forward(image NHWC [1,S,S,3] 0..255) -> (patch [1,g,g,384], cls [1,384])."""

    def __init__(self, weights, size=448):
        super().__init__()
        if size % PATCH:
            raise ValueError("size must be a multiple of 14")
        self.size, self.grid = size, size // PATCH
        w = weights
        self.register_buffer("mean", torch.tensor(MEAN, dtype=torch.float32).view(1, 1, 1, 3) * 255.0)
        self.register_buffer("std", torch.tensor(STD, dtype=torch.float32).view(1, 1, 1, 3) * 255.0)
        self.patch_embed = nn.Conv2d(3, DIM, PATCH, stride=PATCH)
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM, eps=LN_EPS)

        def p(name):
            return torch.as_tensor(w[name], dtype=torch.float32)

        with torch.no_grad():
            self.patch_embed.weight.copy_(p("embeddings.patch_embeddings.projection.weight"))
            self.patch_embed.bias.copy_(p("embeddings.patch_embeddings.projection.bias"))
            pos = w["embeddings.position_embeddings"][0]                     # [1 + 37*37, 384]
            cls = w["embeddings.cls_token"].reshape(1, 1, DIM)
            self.register_buffer("cls_pos", torch.as_tensor(cls + pos[None, :1], dtype=torch.float32))  # [1,1,D]
            self.register_buffer("patch_pos", interpolate_patch_pos(pos[1:], self.grid)[None])           # [1,g*g,D]
            scale = (DIM // HEADS) ** -0.5                                   # 0.125, exact power of two
            for i, b in enumerate(self.blocks):
                pre = f"encoder.layer.{i}."
                b.norm1.weight.copy_(p(pre + "norm1.weight"))
                b.norm1.bias.copy_(p(pre + "norm1.bias"))
                b.q.weight.copy_(p(pre + "attention.attention.query.weight") * scale)
                b.q.bias.copy_(p(pre + "attention.attention.query.bias") * scale)
                b.k.weight.copy_(p(pre + "attention.attention.key.weight"))
                b.k.bias.copy_(p(pre + "attention.attention.key.bias"))
                b.v.weight.copy_(p(pre + "attention.attention.value.weight"))
                b.v.bias.copy_(p(pre + "attention.attention.value.bias"))
                g1 = p(pre + "layer_scale1.lambda1")
                b.proj.weight.copy_(p(pre + "attention.output.dense.weight") * g1[:, None])
                b.proj.bias.copy_(p(pre + "attention.output.dense.bias") * g1)
                b.norm2.weight.copy_(p(pre + "norm2.weight"))
                b.norm2.bias.copy_(p(pre + "norm2.bias"))
                b.fc1.weight.copy_(p(pre + "mlp.fc1.weight"))
                b.fc1.bias.copy_(p(pre + "mlp.fc1.bias"))
                g2 = p(pre + "layer_scale2.lambda1")
                b.fc2.weight.copy_(p(pre + "mlp.fc2.weight") * g2[:, None])
                b.fc2.bias.copy_(p(pre + "mlp.fc2.bias") * g2)
            self.norm.weight.copy_(p("layernorm.weight"))
            self.norm.bias.copy_(p("layernorm.bias"))
        for prm in self.parameters():
            prm.requires_grad_(False)
        self.eval()

    def forward(self, image):                                        # [1, S, S, 3] float32 0..255 RGB
        x = (image - self.mean) / self.std
        x = self.patch_embed(x.permute(0, 3, 1, 2))                  # [1, D, g, g]
        x = x.permute(0, 2, 3, 1).reshape(1, self.grid * self.grid, DIM)   # row-major patches (r*g + c)
        x = torch.cat([self.cls_pos, x + self.patch_pos], dim=1)     # [1, 1 + g*g, D]
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        cls = x[:, :1].reshape(1, DIM)                               # Slice, not Gather (NPU-friendlier)
        patch = x[:, 1:].reshape(1, self.grid, self.grid, DIM)
        return patch, cls


def build(weights_path, size=448):
    return DinoV2S14(read_safetensors(weights_path), size)
