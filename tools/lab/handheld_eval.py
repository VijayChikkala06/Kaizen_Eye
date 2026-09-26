#!/usr/bin/env python3
r"""
Kaizen Eye - HANDHELD capture benchmark for the no-training PatchCore pipeline (patchcore_ref.py + the app's
whole-image check).

Why: real users enrol ~20 HANDHELD photos. Framing shift / scale / rotation / lighting / blur and different
clutter at the frame borders inflate the leave-one-out threshold tau, so real defects pass. This harness
simulates that capture process on synthetic and real-texture "parts", inserts defects ON THE OBJECT (so they move
with it), and measures scoring/backbone variants with the same maths as the app.

  pip install scikit-image                  # real-texture parts come from its bundled sample images
  python tools/lab/handheld_eval.py --model mobile/assets/models/backbone_r18_256.tflite --out out/handheld
  python tools/lab/handheld_eval.py --model a.tflite --model b.tflite --variants base,b2,b2+mad,b2+pn --regimes casual
  python tools/lab/handheld_eval.py --model a.tflite --bench          # cost: GMACs, latency, P, D, K

Capture simulation (independent per photo): 800x600 phone photo of a 1280^2 scene (object on a desk), similarity
transform (shift / rotation / zoom), lighting gain + contrast (+ gradient), Gaussian blur, sensor noise, JPEG;
then the centre square is resized to the model input exactly like viz.load_square (PIL bilinear).
Regimes: tripod (reference), careful, casual (+ changing edge clutter), casual_nc (casual motion, no clutter),
f_* = careful + ONE casual factor (f_shift, f_shift7, f_rot, f_scale, f_geom, f_blur, f_light, f_desk, f_clutter),
*_fill = the part fills the frame (object 1.5x larger).
Defects: sticker, pen, scratch, missing (hole / bite, background shows through), stain; sizes S/M/L = 3/6/10 %
of the image side. Protocol per part x regime: 20 enrol, 30 good test, 30 per defect type (10 per size).

Variants (tokens joined by '+'):
  base   spec scoring (k = max(256, 5 % of patches), greedy coreset, LOO, tau = max LOO) + app whole-image check
  b1,b2  ignore the outer 1 / 2 patch rows+cols in the frame max (enrol LOO and inspection)
  mad    drop enrol frames with LOO > median + 3*MAD (max 20 %), rebuild bank + tau without them
  2nd    tau = 2nd-largest LOO score (instead of the max)      p90  tau = 90th percentile of LOO
  l2     L2-normalise every patch vector (cosine-like distances)
  sm     frame score = max of the 3x3-smoothed distance map
  pn     per-position normalisation: distance / (3x3-smoothed mean LOO distance at that position), floored at
         0.5 x its median (pnm: floor = median). LOO uses the normaliser WITHOUT the held-out frame.
  aug    enrolment augmentation: +A shifted/rotated/zoomed copies of every enrol photo (same k as without)
  noglob patch score only (no whole-image check)
Outputs: <out>/results_<tag>.csv (per model x variant x regime x part + pooled "ALL"), <out>/scores_<tag>.json.
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patchcore_ref as pc  # noqa: E402

PHOTO_W, PHOTO_H = 800, 600      # simulated phone photo (4:3); the model sees its centre square
VIEW = PHOTO_H                   # scene px covered by the centre square at nominal distance
CANVAS = 1280                    # scene size (object at the centre)
SIZES = {"S": 0.03, "M": 0.06, "L": 0.10}   # defect size as a fraction of the image side
KINDS = ["sticker", "pen", "scratch", "missing", "stain"]
MODEL_SIZES = (256, 320)
JPEG_Q = (85, 95)
NOISE = (1.0, 2.0)
AUG = dict(shift=0.03, rot=3.0, zoom=(1.0, 1.06))   # enrolment augmentation (phone-side warp of the photo)

_CAREFUL = dict(shift=0.04, rot=3.0, scale=0.04, bright=0.06, contrast=0.06, blur=0.8, grad=0.0, clutter=0, desk="plain")
_CASUAL = dict(shift=0.10, rot=8.0, scale=0.10, bright=0.12, contrast=0.12, blur=1.5, grad=0.05, clutter=3, desk="texture")
REGIMES = {
    "tripod": dict(_CAREFUL, shift=0.01, rot=0.5, scale=0.01, bright=0.03, contrast=0.03, blur=0.4),
    "careful": _CAREFUL,
    "casual": _CASUAL,
    "casual_nc": dict(_CASUAL, clutter=0),                      # casual motion/light/blur, textured desk, no clutter
    # one factor at a time: careful + ONE casual factor (capture guidance)
    "f_shift": dict(_CAREFUL, shift=0.10),
    "f_rot": dict(_CAREFUL, rot=8.0),
    "f_scale": dict(_CAREFUL, scale=0.10),
    "f_blur": dict(_CAREFUL, blur=1.5),
    "f_light": dict(_CAREFUL, bright=0.12, contrast=0.12, grad=0.05),
    "f_desk": dict(_CAREFUL, desk="texture"),                   # textured desk, no clutter
    "f_clutter": dict(_CAREFUL, desk="texture", clutter=3),     # textured desk + changing clutter at the borders
    "f_geom": dict(_CAREFUL, shift=0.10, rot=8.0, scale=0.10),  # all casual geometry, careful photometry, plain desk
    "f_shift7": dict(_CAREFUL, shift=0.07),
    # part FILLS the frame (object 1.5x larger: little or no background visible)
    "careful_fill": dict(_CAREFUL, obj=1.5),
    "casual_fill": dict(_CASUAL, obj=1.5),
}


# ----------------------------------------------------------------------------------------------- scene / parts
def _sk(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from skimage import data
        im = getattr(data, name)()
    if isinstance(im, tuple):
        im = im[0]
    if im.dtype == bool:
        im = im.astype(np.uint8) * 255
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    return np.ascontiguousarray(im[..., :3]).astype(np.uint8)


def _centre(im, fy, fx):
    h, w = im.shape[:2]
    ch, cw = int(round(h * fy)), int(round(w * fx))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return im[y0:y0 + ch, x0:x0 + cw]


def _synth_board():
    import synth_data
    a = np.asarray(synth_data._base(256))
    m = 256 // 5
    return a[int(m * 1.3):256 - int(m * 1.3) + 1, m:256 - m + 1]     # the blue board of the synthetic part


# object texture, object width (fraction of the view), corner radius (fraction of the short side), desk texture
PARTS = {
    "synth": dict(tex=_synth_board, width=0.62, radius=0.17, desk="gravel", desk_rgb=(150, 132, 110)),
    "coffee": dict(tex=lambda: _centre(_sk("coffee"), 0.9, 0.8), width=0.70, radius=0.06, desk="grass",
                   desk_rgb=(120, 118, 124)),
    "astronaut": dict(tex=lambda: _centre(_sk("astronaut"), 0.8, 0.8), width=0.62, radius=0.06, desk="brick",
                      desk_rgb=(165, 150, 128)),
    "coins": dict(tex=lambda: (_sk("coins") * np.array([1.0, 0.92, 0.78])).astype(np.uint8), width=0.68,
                  radius=0.05, desk="gravel", desk_rgb=(95, 105, 120)),
    "page": dict(tex=lambda: _centre(_sk("page"), 1.0, 0.78), width=0.64, radius=0.03, desk="grass",
                 desk_rgb=(140, 110, 85)),
    "ihc": dict(tex=lambda: _centre(_sk("immunohistochemistry"), 0.8, 0.8), width=0.60, radius=0.08,
                desk="brick", desk_rgb=(175, 172, 168)),
}
CLUTTER_SRC = ["rocket", "stereo_motorcycle", "hubble_deep_field", "retina", "colorwheel", "clock", "camera", "moon",
               "cat", "brick", "text", "logo"]
DESK_PLAIN = (186, 182, 175)


def _rounded_mask(w, h, r):
    m = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], radius=max(1, int(r * 2)), fill=255)
    return m.resize((w, h), Image.LANCZOS)


class Scene:
    """Fixed per part x regime: the desk and the pristine object. Photos add clutter + defect + camera."""

    def __init__(self, part, regime, seed):
        cfg = PARTS[part]
        rng = np.random.default_rng(seed)
        tex = cfg["tex"]()
        ow = int(round(cfg["width"] * REGIMES[regime].get("obj", 1.0) * VIEW))
        oh = int(round(ow * tex.shape[0] / tex.shape[1]))
        self.obj = Image.fromarray(tex).resize((ow, oh), Image.BICUBIC).convert("RGB")
        self.mask = _rounded_mask(ow, oh, cfg["radius"] * min(ow, oh))
        self.ox, self.oy = (CANVAS - ow) // 2, (CANVAS - oh) // 2
        yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32) / CANVAS
        if REGIMES[regime]["desk"] == "texture":
            t = _sk(cfg["desk"])[..., 0].astype(np.float32)
            t = np.asarray(Image.fromarray(np.tile(t, (2, 2)).astype(np.uint8)).resize((CANVAS, CANVAS), Image.BILINEAR),
                           np.float32)
            t = (t - t.mean()) / (t.std() + 1e-6)
            base = np.array(cfg["desk_rgb"], np.float32)
            desk = base * (1 + 0.04 * (xx - 0.5))[..., None] + 14.0 * t[..., None]
        else:
            n = rng.normal(0, 1, (CANVAS // 4, CANVAS // 4)).astype(np.float32)
            n = np.asarray(Image.fromarray(((n * 20) + 128).clip(0, 255).astype(np.uint8)).resize(
                (CANVAS, CANVAS), Image.BICUBIC), np.float32) - 128
            desk = np.array(DESK_PLAIN, np.float32) * (1 + 0.03 * (xx - 0.5) - 0.02 * (yy - 0.5))[..., None] \
                + 0.15 * n[..., None]
        self.desk = Image.fromarray(desk.clip(0, 255).astype(np.uint8))
        self.pool = None
        self.n_clutter = REGIMES[regime]["clutter"]
        if self.n_clutter:
            self.pool = [_sk(n) for n in CLUTTER_SRC]

    # -------------------------------------------------------------------------------------------- defects
    def _defect(self, obj, mask, kind, size_px, rng):
        """Draw a defect on copies of the object (RGB) / its alpha mask. Returns (obj, mask, centre in obj px)."""
        ow, oh = obj.size
        s = float(size_px)
        mg = s / 2 + 0.05 * min(ow, oh)
        # on the object AND inside the central 80 % of the view (matters only when the object fills the frame)
        cx = rng.uniform(max(mg, ow / 2 - 0.4 * VIEW), min(ow - mg, ow / 2 + 0.4 * VIEW))
        cy = rng.uniform(max(mg, oh / 2 - 0.4 * VIEW), min(oh - mg, oh / 2 + 0.4 * VIEW))
        edge_visible = max(ow, oh) <= 0.9 * VIEW
        if kind == "missing" and rng.random() < 0.5 and edge_visible:   # a bite out of the object's edge
            side = rng.integers(4)
            if side == 0:
                cx = s * 0.15
            elif side == 1:
                cx = ow - s * 0.15
            elif side == 2:
                cy = s * 0.15
            else:
                cy = oh - s * 0.15
        obj, mask = obj.copy(), mask.copy()
        if kind == "sticker":
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.55, 1.0), rng.uniform(0.6, 1.0))
            w, h = s * rng.uniform(0.8, 1.0), s * rng.uniform(0.6, 1.0)
            lay = Image.new("L", (int(2 * s) + 4, int(2 * s) + 4), 0)
            c = lay.size[0] / 2
            box = [c - w / 2, c - h / 2, c + w / 2, c + h / 2]
            if rng.random() < 0.5:
                ImageDraw.Draw(lay).ellipse(box, fill=255)
            else:
                ImageDraw.Draw(lay).rectangle(box, fill=255)
            lay = lay.rotate(float(rng.uniform(0, 180)), resample=Image.BILINEAR)
            col = Image.new("RGB", lay.size, (int(r * 255), int(g * 255), int(b * 255)))
            obj.paste(col, (int(cx - c), int(cy - c)), lay)
        elif kind in ("pen", "scratch"):
            L = s * (1.3 if kind == "pen" else 1.4)
            phi = rng.uniform(0, math.pi)
            amp = s * rng.uniform(0.1, 0.3) if kind == "pen" else s * rng.uniform(0.0, 0.06)
            fr = rng.uniform(0.5, 1.5)
            t = np.linspace(0, 1, 40)
            u, v = L * (t - 0.5), amp * np.sin(2 * math.pi * fr * t)
            px = cx + u * math.cos(phi) - v * math.sin(phi)
            py = cy + u * math.sin(phi) + v * math.cos(phi)
            lay = Image.new("L", obj.size, 0)
            width = 3 if kind == "pen" else int(rng.integers(2, 4))
            ImageDraw.Draw(lay).line(list(zip(px.tolist(), py.tolist())), fill=255, width=width, joint="curve")
            lay = lay.filter(ImageFilter.GaussianBlur(0.6))
            if kind == "pen":
                col = Image.new("RGB", obj.size, tuple(int(x) for x in rng.integers(15, 45, 3) + np.array([0, 0, 30])))
                a = np.asarray(lay, np.float32) * 0.92
            else:
                col = Image.new("RGB", obj.size, (238, 238, 232))
                a = np.asarray(lay, np.float32) * 0.85
            obj = Image.composite(col, obj, Image.fromarray(a.astype(np.uint8)))
        elif kind == "missing":
            k = int(rng.integers(7, 10))
            ang = np.sort(rng.uniform(0, 2 * math.pi, k))
            rad = s / 2 * rng.uniform(0.75, 1.15, k)
            poly = list(zip((cx + rad * np.cos(ang)).tolist(), (cy + rad * np.sin(ang)).tolist()))
            hole = Image.new("L", obj.size, 0)
            ImageDraw.Draw(hole).polygon(poly, fill=255)
            m = np.asarray(mask, np.float32) * (1 - np.asarray(hole, np.float32) / 255.0)
            mask = Image.fromarray(m.astype(np.uint8))
        elif kind == "stain":
            w, h = s * rng.uniform(0.8, 1.0), s * rng.uniform(0.6, 1.0)
            lay = Image.new("L", obj.size, 0)
            ImageDraw.Draw(lay).ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=255)
            lay = lay.filter(ImageFilter.GaussianBlur(s / 6))
            col = tuple(int(x) for x in (rng.uniform(90, 140), rng.uniform(55, 90), rng.uniform(30, 60)))
            a = np.asarray(lay, np.float32) * rng.uniform(0.45, 0.65)
            obj = Image.composite(Image.new("RGB", obj.size, col), obj, Image.fromarray(a.astype(np.uint8)))
        else:
            raise ValueError(kind)
        return obj, mask, (cx, cy)

    # -------------------------------------------------------------------------------------------- capture
    def photo(self, rp, rng, defect=None):
        """-> (photo uint8 [H, W, 3], defect centre in photo px or None)."""
        cv = self.desk.copy()
        for _ in range(int(rng.integers(1, self.n_clutter + 1)) if self.n_clutter else 0):
            src = self.pool[int(rng.integers(len(self.pool)))]
            w, h = (rng.uniform(0.12, 0.30, 2) * VIEW).astype(int)
            sh, sw = src.shape[:2]
            cs = min(sh, sw, int(max(w, h) * rng.uniform(0.6, 1.4)))
            y0, x0 = int(rng.integers(0, sh - cs + 1)), int(rng.integers(0, sw - cs + 1))
            it = Image.fromarray(src[y0:y0 + cs, x0:x0 + cs]).resize((w, h), Image.BILINEAR)
            m = Image.new("L", (w, h), 0)
            if rng.random() < 0.5:
                ImageDraw.Draw(m).ellipse([0, 0, w - 1, h - 1], fill=255)
            else:
                ImageDraw.Draw(m).rectangle([0, 0, w - 1, h - 1], fill=255)
            a, rho = rng.uniform(0, 2 * math.pi), rng.uniform(0.50, 0.72) * VIEW
            cv.paste(it, (int(CANVAS / 2 + rho * math.cos(a) - w / 2), int(CANVAS / 2 + rho * math.sin(a) - h / 2)), m)
        obj, mask, dc = self.obj, self.mask, None
        if defect is not None:
            kind, size = defect
            obj, mask, (dx, dy) = self._defect(obj, mask, kind, SIZES[size] * VIEW, rng)
            dc = (self.ox + dx, self.oy + dy)
        cv.paste(obj, (self.ox, self.oy), mask)

        # camera: photo px (x, y) -> scene px   (similarity: shift, rotation, zoom)
        tx, ty = rng.uniform(-rp["shift"], rp["shift"], 2) * VIEW
        th = math.radians(rng.uniform(-rp["rot"], rp["rot"]))
        z = 1 + rng.uniform(-rp["scale"], rp["scale"])
        a, b, d, e = math.cos(th) / z, -math.sin(th) / z, math.sin(th) / z, math.cos(th) / z
        c = CANVAS / 2 + tx - (a * PHOTO_W / 2 + b * PHOTO_H / 2)
        f = CANVAS / 2 + ty - (d * PHOTO_W / 2 + e * PHOTO_H / 2)
        ph = cv.transform((PHOTO_W, PHOTO_H), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
        sig = rng.uniform(0, rp["blur"]) * PHOTO_H / 256.0          # blur is specified at model-input scale
        if sig > 0.05:
            ph = ph.filter(ImageFilter.GaussianBlur(sig))
        x = np.asarray(ph, np.float32)
        if rp["grad"]:
            yy, xx = np.mgrid[0:PHOTO_H, 0:PHOTO_W].astype(np.float32)
            p = rng.uniform(0, 2 * math.pi)
            g = 1 + rng.uniform(-rp["grad"], rp["grad"]) * ((xx - PHOTO_W / 2) * math.cos(p) + (yy - PHOTO_H / 2) * math.sin(p)) / (PHOTO_H / 2)
            x *= g[..., None]
        mu = x.mean()
        x = (x - mu) * (1 + rng.uniform(-rp["contrast"], rp["contrast"])) + mu
        x *= 1 + rng.uniform(-rp["bright"], rp["bright"])
        x += rng.normal(0, rng.uniform(*NOISE), x.shape)
        buf = io.BytesIO()
        Image.fromarray(x.clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=int(rng.integers(JPEG_Q[0], JPEG_Q[1] + 1)))
        out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        if dc is not None:                                           # scene -> photo px (invert the affine)
            m = np.array([[a, b], [d, e]])
            dc = tuple(np.linalg.solve(m, np.array([dc[0] - c, dc[1] - f])).tolist())
        return out, dc


def square(photo, size):
    """Exactly viz.load_square after decoding: centre square crop, PIL bilinear resize -> uint8 [size, size, 3]."""
    im = Image.fromarray(photo)
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def augment(photo, rng, n):
    """Phone-side enrolment augmentation: small similarity warps of the photo (reflect-padded), then square()."""
    pad = 96
    P = Image.fromarray(np.pad(photo, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"))
    outs = []
    for _ in range(n):
        th = math.radians(rng.uniform(-AUG["rot"], AUG["rot"]))
        z = rng.uniform(*AUG["zoom"])
        tx, ty = rng.uniform(-AUG["shift"], AUG["shift"], 2) * PHOTO_H
        a, b, d, e = math.cos(th) / z, -math.sin(th) / z, math.sin(th) / z, math.cos(th) / z
        c = pad + PHOTO_W / 2 + tx - (a * PHOTO_W / 2 + b * PHOTO_H / 2)
        f = pad + PHOTO_H / 2 + ty - (d * PHOTO_W / 2 + e * PHOTO_H / 2)
        ph = np.asarray(P.transform((PHOTO_W, PHOTO_H), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR))
        outs.append([square(ph, s) for s in MODEL_SIZES])
    return outs


def make_cell(part, regime, seed, n_enrol, n_good, n_def, n_aug, path):
    """Render one part x regime and save it as an .npz (images at every model size + defect metadata)."""
    pi, ri = list(PARTS).index(part), list(REGIMES).index(regime)
    base = seed * 1000 + (pi * 10 + ri if ri < 10 else 500 + pi * 20 + ri)     # unique per cell, stable for old cells
    scene = Scene(part, regime, base)
    rng = np.random.default_rng(base + 7)
    rp = REGIMES[regime]
    out = {f"{k}{s}": [] for k in ("enrol", "aug", "good", "def") for s in MODEL_SIZES}
    kinds, sizes, cen = [], [], []
    for _ in range(n_enrol):
        ph, _ = scene.photo(rp, rng)
        for s in MODEL_SIZES:
            out[f"enrol{s}"].append(square(ph, s))
        au = augment(ph, np.random.default_rng(int(rng.integers(1 << 30))), n_aug)
        for j, s in enumerate(MODEL_SIZES):
            out[f"aug{s}"].append(np.stack([a[j] for a in au]) if n_aug else np.zeros((0, s, s, 3), np.uint8))
    for _ in range(n_good):
        ph, _ = scene.photo(rp, rng)
        for s in MODEL_SIZES:
            out[f"good{s}"].append(square(ph, s))
    for kind in KINDS:
        for i in range(n_def):
            size = "SML"[i % 3]
            ph, dc = scene.photo(rp, rng, (kind, size))
            for s in MODEL_SIZES:
                out[f"def{s}"].append(square(ph, s))
            kinds.append(kind)
            sizes.append(size)
            cen.append(((dc[0] - (PHOTO_W - PHOTO_H) / 2) / PHOTO_H, dc[1] / PHOTO_H))   # fraction of the square
    arrs = {k: np.stack(v) for k, v in out.items()}
    np.savez(path, **arrs, kind=np.array(kinds), size=np.array(sizes), centre=np.array(cen, np.float32))
    # contact sheet for eyeballing: 4 enrol, 2 good, one of each defect kind (L size)
    tiles = list(arrs["enrol256"][:4]) + list(arrs["good256"][:2]) + [arrs["def256"][KINDS.index(k) * n_def + 2] for k in KINDS]
    sheet = Image.new("RGB", (256 * 6, 256 * 2), (0, 0, 0))
    for i, t in enumerate(tiles[:12]):
        sheet.paste(Image.fromarray(t), ((i % 6) * 256, (i // 6) * 256))
    sheet.save(os.path.splitext(path)[0] + "_preview.jpg", quality=85)
    return path


# ---------------------------------------------------------------------------------------------- embedding
class Embedder:
    """Same .tflite as the phone. Handles float and int8-I/O models."""

    def __init__(self, path, threads=4):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]
        self.size = int(self.i["shape"][1])
        self.grid = int(self.o["shape"][1])
        self.dim = int(self.o["shape"][3])

    def __call__(self, img_u8):
        x = img_u8.astype(np.float32)[None]
        if self.i["dtype"] != np.float32:
            sc, zp = self.i["quantization"]
            info = np.iinfo(self.i["dtype"])
            x = np.clip(np.round(x / sc + zp), info.min, info.max).astype(self.i["dtype"])
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.o["index"])[0]
        if self.o["dtype"] != np.float32:
            sc, zp = self.o["quantization"]
            y = (y.astype(np.float32) - zp) * sc
        return np.array(y, np.float32)


# ---------------------------------------------------------------------------------------------- scoring
def parse_variant(name):
    v = dict(border=0, calib="max", l2=False, smooth=False, pn=0.0, aug=False, glob=True)
    for t in name.split("+"):
        if t == "base":
            pass
        elif t[0] == "b" and t[1:].isdigit():
            v["border"] = int(t[1:])
        elif t in ("mad", "2nd", "p90"):
            v["calib"] = t
        elif t == "l2":
            v["l2"] = True
        elif t == "sm":
            v["smooth"] = True
        elif t in ("pn", "pnm"):
            v["pn"] = 0.5 if t == "pn" else 1.0
        elif t == "aug":
            v["aug"] = True
        elif t == "noglob":
            v["glob"] = False
        else:
            raise SystemExit(f"unknown variant token {t!r} in {name!r}")
    return v


def greedy_coreset_fast(x, k, start=0):
    """Same selection rule as pc.greedy_coreset (start index, argmax ties -> lowest index), via |a|^2+|b|^2-2ab."""
    n = x.shape[0]
    k = min(k, n)
    xx = np.einsum("ij,ij->i", x, x)
    sel = np.empty(k, np.int64)
    sel[0] = start
    mind = np.maximum(xx + xx[start] - 2.0 * (x @ x[start]), 0.0)
    for i in range(1, k):
        j = int(np.argmax(mind))
        sel[i] = j
        d = x @ x[j]
        d *= -2.0
        d += xx
        d += xx[j]
        np.minimum(mind, d, out=mind)
    return sel


def l2n(f):
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6)


def gdesc(f):
    g = f.reshape(-1, f.shape[-1]).mean(0)
    return g / (np.linalg.norm(g) or 1.0)


class Bank:
    def __init__(self, E, EA, frames, ratio=0.05, min_k=256):
        """E [n, gh, gw, D] enrol features, EA [n, A, gh, gw, D] or None, frames = indices kept."""
        n, gh, gw, D = E.shape
        self.frames = list(frames)
        P = gh * gw
        rows, fid = [], []
        for f in self.frames:
            rows.append(E[f].reshape(-1, D))
            fid.append(np.full(P, f))
            if EA is not None:
                for a in range(EA.shape[1]):
                    rows.append(EA[f, a].reshape(-1, D))
                    fid.append(np.full(P, f))
        x = np.ascontiguousarray(np.concatenate(rows), np.float32)
        fid = np.concatenate(fid)
        self.k = max(min_k, int(ratio * len(self.frames) * P))     # k from ORIGINAL frames (aug adds no budget)
        sel = greedy_coreset_fast(x, self.k)
        self.bank, self.bank_frame = x[sel], fid[sel]
        self.loo = np.stack([pc.nn_dist(E[f].reshape(-1, D), self.bank[self.bank_frame != f]).reshape(gh, gw)
                             for f in self.frames])
        self.globals = np.stack([gdesc(E[f]) for f in self.frames])
        gd = 1 - self.globals @ self.globals.T
        np.fill_diagonal(gd, np.inf)
        self.gloo = gd.min(1)
        self.gtau = max(0.02, float(self.gloo.max()))


def pos_norm(maps, floor):
    """maps [n, gh, gw] LOO distance maps -> (normaliser for inspection [gh, gw], leave-own-out normalisers [n, gh, gw])."""
    def fl(m):
        m = pc.smooth3(m)
        return np.maximum(m, floor * float(np.median(m)))
    n = len(maps)
    tot = maps.sum(0)
    return fl(tot / n), np.stack([fl((tot - maps[i]) / (n - 1)) for i in range(n)])


def fscore(m, v):
    """Frame score of distance map(s) m [..., gh, gw] under variant v (already position-normalised if pn)."""
    if v["smooth"]:
        m = np.stack([pc.smooth3(x) for x in m.reshape(-1, *m.shape[-2:])]).reshape(m.shape)
    b = v["border"]
    if b:
        m = m[..., b:-b, b:-b]
    return m.reshape(*m.shape[:-2], -1).max(-1)


def peak_rc(m, v):
    """(row, col) of the hottest 3x3-smoothed patch inside the scored region (like the app's heat-map ring)."""
    s = pc.smooth3(m)
    b = v["border"]
    if b:
        s = np.pad(s[b:-b, b:-b], b, constant_values=-np.inf)
    return np.unravel_index(int(np.argmax(s)), s.shape)


def calib(r, rule):
    r = np.sort(np.asarray(r))
    if rule == "2nd":
        return float(r[-2])
    if rule == "p90":
        return float(np.percentile(r, 90))
    return float(r[-1])


def mad_subset(r, frames, max_frac=0.2):
    r = np.asarray(r)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    out = [i for i in np.argsort(-r) if r[i] > med + 3 * mad][: int(max_frac * len(r))]
    return tuple(sorted(frames[i] for i in range(len(r)) if i not in out))


def run_cell(job):
    """Embed one part x regime with one model and score every variant. Returns a list of per-variant dicts."""
    t0 = time.time()
    model, cell_path, variants, threads = job["model"], job["cell"], job["variants"], job["threads"]
    z = np.load(cell_path)
    emb = Embedder(model, threads)
    S = emb.size
    if S not in MODEL_SIZES:
        raise SystemExit(f"model input {S} not rendered (MODEL_SIZES={MODEL_SIZES})")
    specs = {n: parse_variant(n) for n in variants}
    need_aug = any(v["aug"] for v in specs.values())
    E = np.stack([emb(x) for x in z[f"enrol{S}"]])
    EA = None
    if need_aug:
        A = z[f"aug{S}"]
        EA = np.stack([np.stack([emb(x) for x in A[f]]) for f in range(len(A))])
    t_emb = (time.time() - t0) / (len(E) + (0 if EA is None else EA.shape[0] * EA.shape[1]))
    n, gh, gw, D = E.shape
    feats = {False: (E, EA), True: (l2n(E), None if EA is None else l2n(EA))}

    banks = {}

    def bank(l2, aug, frames):
        key = (l2, aug, tuple(frames))
        if key not in banks:
            e, ea = feats[l2]
            banks[key] = Bank(e, ea if aug else None, frames)
        return banks[key]

    allf = tuple(range(n))
    # 1) base banks, 2) outlier-dropped banks (decided with each variant's own frame score)
    plan = {}
    for name, v in specs.items():
        b0 = bank(v["l2"], v["aug"], allf)
        frames = allf
        if v["calib"] == "mad":
            maps = b0.loo
            if v["pn"]:
                _, nl = pos_norm(maps, v["pn"])
                maps = maps / nl
            frames = mad_subset(fscore(maps, v), list(allf))
            bank(v["l2"], v["aug"], frames)
        plan[name] = (v["l2"], v["aug"], frames)

    # 3) stream the test photos through every bank
    good, dft = z[f"good{S}"], z[f"def{S}"]
    tests = np.concatenate([good, dft])
    dmaps = {k: np.empty((len(tests), gh, gw), np.float32) for k in banks}
    gdist = {k: np.empty(len(tests), np.float32) for k in banks}
    for i, im in enumerate(tests):
        f = emb(im)
        for l2 in {k[0] for k in banks}:
            ff = l2n(f) if l2 else f
            g = gdesc(ff)
            pf = ff.reshape(-1, D)
            for k, bk in banks.items():
                if k[0] != l2:
                    continue
                dmaps[k][i] = pc.nn_dist(pf, bk.bank).reshape(gh, gw)
                gdist[k][i] = float((1 - bk.globals @ g).min())

    # 4) per-variant calibration + scores
    ng = len(good)
    cen = z["centre"]
    res = []
    for name, v in specs.items():
        key = (v["l2"], v["aug"], tuple(plan[name][2]))
        bk = banks[key]
        loo, tm = bk.loo, dmaps[key]
        if v["pn"]:
            na, nl = pos_norm(loo, v["pn"])
            loo, tm = loo / nl, tm / na
        r = fscore(loo, v)
        tau = calib(r, v["calib"])
        patch = fscore(tm, v) / tau
        glob = gdist[key] / bk.gtau if v["glob"] else np.zeros(len(tests), np.float32)
        hit = []
        for j in range(len(dft)):
            rr, cc = peak_rc(tm[ng + j], v)
            py, px = (rr + 0.5) / gh, (cc + 0.5) / gw
            sz = SIZES[str(z["size"][j])]
            hit.append(bool(math.hypot(px - cen[j][0], py - cen[j][1]) <= sz / 2 + 1.5 / gh))
        res.append(dict(variant=name, tau=tau, gtau=bk.gtau, k=bk.k, n_kept=len(bk.frames), loo=r.tolist(),
                        good_patch=patch[:ng].tolist(), good_glob=glob[:ng].tolist(),
                        def_patch=patch[ng:].tolist(), def_glob=glob[ng:].tolist(), hit=hit,
                        P=gh * gw, D=D, emb_ms=t_emb * 1000))
    return dict(cell=os.path.basename(cell_path), kind=z["kind"].tolist(), size=z["size"].tolist(), results=res,
                secs=time.time() - t0)


# ---------------------------------------------------------------------------------------------- metrics
def auroc(neg, pos):
    neg, pos = np.asarray(neg, np.float64), np.asarray(pos, np.float64)
    if not len(neg) or not len(pos):
        return float("nan")
    x = np.concatenate([neg, pos])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):                                     # average ranks for ties
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def summarise(rows, sens=(1.10, 1.00)):
    """rows: list of dicts with good (final score), defect (final), patch-only versions, kind, size, hit."""
    g = np.concatenate([r["good"] for r in rows])
    gp = np.concatenate([r["good_p"] for r in rows])
    d = np.concatenate([r["def"] for r in rows])
    dp = np.concatenate([r["def_p"] for r in rows])
    kind = np.concatenate([r["kind"] for r in rows])
    size = np.concatenate([r["size"] for r in rows])
    hit = np.concatenate([r["hit"] for r in rows])
    out = dict(auroc=auroc(g, d), auroc_patch=auroc(gp, dp), n_good=len(g), n_def=len(d))
    for sz in "SML":
        out[f"auroc_{sz}"] = auroc(g, d[size == sz])
    for s in sens:
        t = f"{s:.2f}"
        out[f"frr@{t}"] = float((g > s).mean())
        out[f"dr@{t}"] = float((d > s).mean())
        for sz in "SML":
            out[f"dr_{sz}@{t}"] = float((d[size == sz] > s).mean())
        for k in KINDS:
            out[f"dr_{k}@{t}"] = float((d[kind == k] > s).mean())
        det = d > s
        out[f"loc@{t}"] = float(hit[det].mean()) if det.any() else float("nan")
    return out


# ---------------------------------------------------------------------------------------------- cost
def bench(models, threads_list=(1, 4), frames=20, reps=15):
    import importlib.util
    spec = importlib.util.spec_from_file_location("convert_backbone", os.path.join(HERE, "..", "convert_backbone.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    rows = []
    for m in models:
        e = Embedder(m, 1)
        meta_p = os.path.splitext(m)[0] + ".json"
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        bb = meta.get("backbone", "resnet18")
        net = cb.PatchFeatureNet(bb, e.size, meta.get("dim", e.dim) if meta.get("dim", e.dim) != 384 else 0, False)
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                net(torch.zeros(1, e.size, e.size, 3))
        gmac = fc.get_total_flops() / 2e9
        P = e.grid * e.grid
        K = max(256, int(0.05 * frames * P))
        row = dict(model=os.path.basename(m), size=e.size, P=P, D=e.dim, K=K, backbone_gmac=round(gmac, 3),
                   knn_mmac=round(P * K * e.dim / 1e6, 1), coreset_gmac=round(K * frames * P * e.dim / 1e9, 2),
                   mb=round(os.path.getsize(m) / 1e6, 1))
        img = np.random.default_rng(0).integers(0, 256, (e.size, e.size, 3)).astype(np.uint8)
        for th in threads_list:
            ee = Embedder(m, th)
            for _ in range(3):
                ee(img)
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                ee(img)
                ts.append(time.perf_counter() - t)
            row[f"ms_{th}t"] = round(1000 * float(np.median(ts)), 1)
        rows.append(row)
        print(row, flush=True)
    return rows


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="backbone .tflite (repeatable)")
    ap.add_argument("--variants", default="base,b1,b2,mad,2nd,b2+mad,b2+2nd,noglob")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--regimes", default="careful,casual")
    ap.add_argument("--n-enrol", type=int, default=20)
    ap.add_argument("--n-good", type=int, default=30)
    ap.add_argument("--n-defect", type=int, default=30, help="per defect type (sizes S/M/L cycled)")
    ap.add_argument("--n-aug", type=int, default=4, help="augmented copies per enrol photo (rendered once)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="out/handheld")
    ap.add_argument("--data", default=None, help="cache dir for rendered photos (default <out>/data)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench", action="store_true", help="only measure cost (GMACs, latency, P/D/K)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.bench:
        rows = bench(a.model)
        with open(os.path.join(a.out, "cost.json"), "w") as fh:
            json.dump(rows, fh, indent=1)
        return
    data = a.data or os.path.join(a.out, "data")
    os.makedirs(data, exist_ok=True)
    parts, regimes, variants = a.parts.split(","), a.regimes.split(","), a.variants.split(",")
    for v in variants:
        parse_variant(v)
    cells = [(p, r) for r in regimes for p in parts]
    cpath = {c: os.path.join(data, f"{c[0]}_{c[1]}_s{a.seed}_e{a.n_enrol}g{a.n_good}d{a.n_defect}a{a.n_aug}.npz") for c in cells}

    import multiprocessing as mp
    todo = [c for c in cells if not os.path.exists(cpath[c])]
    if todo:
        t = time.time()
        with mp.get_context("spawn").Pool(min(len(todo), max(1, a.workers * 2))) as pool:
            pool.starmap(make_cell, [(p, r, a.seed, a.n_enrol, a.n_good, a.n_defect, a.n_aug, cpath[(p, r)]) for p, r in todo])
        print(f"rendered {len(todo)} cells in {time.time() - t:.0f} s -> {data}", flush=True)

    tag = a.tag or time.strftime("%Y%m%d_%H%M%S")
    threads = max(1, (os.cpu_count() or 4) // a.workers)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    raw, table = [], []
    for model in a.model:
        mname = os.path.splitext(os.path.basename(model))[0]
        t = time.time()
        jobs = [dict(model=model, cell=cpath[c], variants=variants, threads=threads) for c in cells]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            outs = pool.map(run_cell, jobs)
        print(f"[{mname}] {len(cells)} cells in {time.time() - t:.0f} s", flush=True)
        for (p, r), o in zip(cells, outs):
            for res in o["results"]:
                raw.append(dict(model=mname, part=p, regime=r, kind=o["kind"], size=o["size"], **res))
        for r in regimes:
            for v in variants:
                rows = []
                for p in parts:
                    x = next(q for q in raw if q["model"] == mname and q["part"] == p and q["regime"] == r and q["variant"] == v)
                    gp, gg, dp, dg = (np.array(x[k]) for k in ("good_patch", "good_glob", "def_patch", "def_glob"))
                    row = dict(good=np.maximum(gp, gg), good_p=gp, **{"def": np.maximum(dp, dg)}, def_p=dp,
                               kind=np.array(x["kind"]), size=np.array(x["size"]), hit=np.array(x["hit"]))
                    rows.append(row)
                    table.append(dict(model=mname, variant=v, regime=r, part=p, tau=x["tau"], gtau=x["gtau"], k=x["k"],
                                      n_kept=x["n_kept"], P=x["P"], D=x["D"], emb_ms=x["emb_ms"], **summarise([row])))
                taus = [q["tau"] for q in table[-len(parts):]]
                table.append(dict(model=mname, variant=v, regime=r, part="ALL", tau=float(np.mean(taus)),
                                  gtau=float(np.mean([q["gtau"] for q in table[-len(parts):]])),
                                  k=int(np.mean([q["k"] for q in table[-len(parts):]])),
                                  n_kept=float(np.mean([q["n_kept"] for q in table[-len(parts):]])),
                                  P=table[-1]["P"], D=table[-1]["D"], emb_ms=float(np.mean([q["emb_ms"] for q in table[-len(parts):]])),
                                  **summarise(rows)))

    keys = list(table[0].keys())
    with open(os.path.join(a.out, f"results_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    with open(os.path.join(a.out, f"scores_{tag}.json"), "w") as fh:
        json.dump(raw, fh)

    hdr = (f"{'model':16s} {'variant':14s} {'tau':>5s} {'AUC':>5s} {'AUCp':>5s} | {'FRR':>4s} {'DR':>4s} {'S':>4s} {'M':>4s} "
           f"{'L':>4s} {'loc':>4s} | {'FRR':>4s} {'DR':>4s} {'S':>4s} {'M':>4s} {'L':>4s}")
    for r in regimes:
        print(f"\n=== regime {r}: pooled over {len(parts)} parts ({parts}); left block sensitivity 1.10, right 1.00 (%)")
        print(hdr)
        for q in table:
            if q["regime"] == r and q["part"] == "ALL":
                print(f"{q['model'][:16]:16s} {q['variant'][:14]:14s} {q['tau']:5.2f} {q['auroc']:5.3f} {q['auroc_patch']:5.3f} | "
                      f"{100 * q['frr@1.10']:4.0f} {100 * q['dr@1.10']:4.0f} {100 * q['dr_S@1.10']:4.0f} {100 * q['dr_M@1.10']:4.0f} "
                      f"{100 * q['dr_L@1.10']:4.0f} {100 * q['loc@1.10']:4.0f} | {100 * q['frr@1.00']:4.0f} {100 * q['dr@1.00']:4.0f} "
                      f"{100 * q['dr_S@1.00']:4.0f} {100 * q['dr_M@1.00']:4.0f} {100 * q['dr_L@1.00']:4.0f}")
    print(f"\nwrote {a.out}/results_{tag}.csv and scores_{tag}.json")


if __name__ == "__main__":
    main()
