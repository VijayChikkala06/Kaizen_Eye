#!/usr/bin/env python3
r"""
Kaizen Eye 2 - SYNTHETIC evaluation export (docs/verification/export-format.md, schema 1).

Renders a fake inspection session with PIL - matte sheet, a textured part ("bracket": brushed-steel plate with two
holes of different size), small pose/scale/lighting jitter, seeded defects (scratch, chipped wedge, marker dot),
wrong objects (other shapes/textures), a look-alike (same outline, different surface), large rotations - and runs
every rendered full-resolution frame through the spec pipeline of tools/lab/twin_ref.py (analysis downscale, sheet
model, mask, components, main object, sanity, crop square, crop-resize, cov, geometry, theta, sharpness). The
synthetic "phone" verdicts come from twin_ref teach + judge with the shipped ResNet18 TFLite on the exact
(un-JPEG'd) 320 px crops. For MECHANICS / parser tests of twin_eval.py and the app's exporter only - NOT evidence.

  .venv\Scripts\python tools\lab\make_synthetic_export.py --out testdata\synthetic_export
  options: --seed 7  --crop-size 448  --no-phone (skip the R18 teach/judge)  --small (fewer parts)

The renderer (render_sheet / sprite_* / place / finish_frame) is also used by tools/lab/make_replay_synth.py.
"""
import argparse
import io
import json
import math
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

R18_TFLITE = os.path.join(ROOT, "mobile", "assets", "models", "backbone_r18_320.tflite")

# ================================================================================================= renderer (PIL)
SS = 2                           # supersampling factor for sprite drawing
SHEET_RGB = (52, 88, 74)         # matte dark-green cutting mat
BRACKET_L, BRACKET_W = 264, 110  # full-res px at scale 1.0
WRONG_KINDS = ["hexnut", "washer", "plate", "coin", "triangle", "bolt"]
DEFECT_KINDS = ["scratch", "chip", "dot"]


def render_sheet(w, h, rng, rgb=SHEET_RGB, grad=3.0, noise=0.6):
    """Matte sheet: flat colour + a gentle illumination gradient (+-grad levels) + static fine texture."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = grad * (0.6 * (xx / w - 0.5) * 2 + 0.4 * (yy / h - 0.5) * 2)
    base = np.array(rgb, np.float32)[None, None, :] + g[..., None]
    return base + rng.normal(0.0, noise, (h, w, 1)).astype(np.float32)          # float32 [h, w, 3]


def _alpha_of(shape_fn, wpx, hpx):
    m = Image.new("L", (int(wpx * SS), int(hpx * SS)), 0)
    shape_fn(ImageDraw.Draw(m), SS)
    return m


def _bevel(rgb, alpha, strength=38.0):
    """Light from the top-left: brighten edges facing it, darken the opposite ones."""
    a = alpha.astype(np.float32) / 255.0
    sh = 2 * SS
    up = np.zeros_like(a)
    up[sh:, sh:] = a[:-sh, :-sh]
    dn = np.zeros_like(a)
    dn[:-sh, :-sh] = a[sh:, sh:]
    return rgb + strength * np.clip(a - up, 0, 1)[..., None] - strength * np.clip(a - dn, 0, 1)[..., None]


def _finish(rgb, alpha):
    rgba = np.dstack([np.clip(rgb, 0, 255), alpha.astype(np.float32)]).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA")
    return im.resize((im.width // SS, im.height // SS), Image.LANCZOS)


def _bracket_alpha(L, W):
    def shape(d, s):
        d.rounded_rectangle([0, 0, L * s - 1, W * s - 1], radius=int(0.32 * W * s), fill=255)
        r1, r2 = 0.17 * W, 0.105 * W
        c1, c2 = (0.22 * L, 0.5 * W), (0.80 * L, 0.5 * W)
        d.ellipse([(c1[0] - r1) * s, (c1[1] - r1) * s, (c1[0] + r1) * s, (c1[1] + r1) * s], fill=0)
        d.ellipse([(c2[0] - r2) * s, (c2[1] - r2) * s, (c2[0] + r2) * s, (c2[1] + r2) * s], fill=0)
    return np.asarray(_alpha_of(shape, L, W), np.uint8)


def sprite_bracket(surface="brushed", defect=None, defect_rng=None, defect_scale=1.0):
    """The taught part (surface 'brushed') or its look-alike ('knurl': same outline, brass + cross-hatch).
    defect in {None, 'scratch', 'chip', 'dot'}; the part's own texture is fixed (same part every time),
    the defect placement comes from defect_rng; defect_scale enlarges the defect. Returns an RGBA sprite at
    full-resolution scale."""
    L, W = BRACKET_L, BRACKET_W
    alpha = _bracket_alpha(L, W).copy()
    h, w = alpha.shape
    trng = np.random.default_rng(1234 if surface == "brushed" else 4321)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if surface == "brushed":
        streak = np.convolve(trng.normal(0, 1, h), np.ones(3) / 3, mode="same").astype(np.float32) * 9.0
        rgb = np.array([150, 153, 158], np.float32)[None, None] + streak[:, None, None]
        rgb = rgb + (18.0 * (0.5 - xx / w) + 10.0 * (0.5 - yy / h))[..., None]
        mark = ((xx - 0.40 * w) ** 2 / (0.05 * w) ** 2 + (yy - 0.30 * h) ** 2 / (0.07 * h) ** 2) < 1.0
        rgb[mark] -= 30.0                                # stamped mark: breaks the 180 deg symmetry
    else:
        hatch = (np.sin((xx + yy) * 2 * math.pi / (7.0 * SS)) + np.sin((xx - yy) * 2 * math.pi / (7.0 * SS))) * 11.0
        rgb = np.array([176, 146, 88], np.float32)[None, None] + hatch[..., None] + (14.0 * (0.5 - xx / w))[..., None]
    rgb = _bevel(rgb + trng.normal(0, 3.0, (h, w, 1)).astype(np.float32), alpha)
    info = {}
    if defect is not None:
        dr = defect_rng if defect_rng is not None else np.random.default_rng(0)
        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
        am = Image.fromarray(alpha, "L")
        d, da, s = ImageDraw.Draw(img), ImageDraw.Draw(am), SS
        if defect == "scratch":
            x0, y0 = dr.uniform(0.34, 0.50) * L, dr.uniform(0.25, 0.75) * W
            ang, ln = dr.uniform(-0.5, 0.5), dr.uniform(0.30, 0.38) * L * defect_scale
            x1, y1 = x0 + ln * math.cos(ang), y0 + ln * math.sin(ang)
            d.line([x0 * s, (y0 + 2) * s, x1 * s, (y1 + 2) * s], fill=(70, 70, 74), width=int(3 * s * defect_scale))
            d.line([x0 * s, y0 * s, x1 * s, y1 * s], fill=(236, 238, 240), width=int(5 * s * defect_scale))
            info = {"type": "scratch", "xy": [x0, y0, x1, y1]}
        elif defect == "chip":
            xc, dep, half = (dr.uniform(0.38, 0.62) * L, dr.uniform(0.32, 0.40) * W * defect_scale,
                             dr.uniform(0.09, 0.12) * L * defect_scale)
            if dr.random() < 0.5:
                poly = [(xc - half, -2), (xc + half, -2), (xc + 0.2 * half, dep)]
            else:
                poly = [(xc - half, W + 2), (xc + half, W + 2), (xc - 0.2 * half, W - dep)]
            da.polygon([(px * s, py * s) for px, py in poly], fill=0)
            info = {"type": "chip", "poly": poly}
        elif defect == "dot":
            xc, yc, r = dr.uniform(0.38, 0.62) * L, dr.uniform(0.35, 0.65) * W, dr.uniform(12.0, 16.0) * defect_scale
            col = (150, 28, 30) if dr.random() < 0.5 else (35, 32, 40)
            d.ellipse([(xc - r) * s, (yc - r) * s, (xc + r) * s, (yc + r) * s], fill=col)
            info = {"type": "dot", "xy": [xc, yc], "r": r}
        else:
            raise ValueError(defect)
        rgb, alpha = np.asarray(img, np.float32), np.asarray(am, np.uint8)
    sp = _finish(rgb, alpha)
    sp.info["defect"] = info
    return sp


def sprite_wrong(kind, rng, scale=1.35):
    """A different object; rng varies its size/colour a little (instances differ)."""
    def j(lo, hi):
        return float(rng.uniform(lo, hi)) * scale

    def c(lo, hi):
        return float(rng.uniform(lo, hi))

    if kind in ("hexnut", "washer", "coin"):
        R = j(62, 74) if kind == "hexnut" else j(52, 64) if kind == "washer" else j(50, 60)
        r = R * (0.40 if kind == "hexnut" else 0.42 if kind == "washer" else 0.0)
        size = 2 * R + 4
        wpx = hpx = size

        def shape(d, s):
            if kind == "hexnut":
                d.polygon([((size / 2 + R * math.cos(math.pi / 3 * k)) * s, (size / 2 + R * math.sin(math.pi / 3 * k)) * s)
                           for k in range(6)], fill=255)
            else:
                d.ellipse([(size / 2 - R) * s, (size / 2 - R) * s, (size / 2 + R) * s, (size / 2 + R) * s], fill=255)
            if r > 0:
                d.ellipse([(size / 2 - r) * s, (size / 2 - r) * s, (size / 2 + r) * s, (size / 2 + r) * s], fill=0)
        colour = {"hexnut": (c(100, 120), c(104, 122), c(112, 128)), "washer": (c(185, 200), c(185, 200), c(178, 192)),
                  "coin": (c(170, 190), c(135, 150), c(70, 85))}[kind]
        tex = "radial" if kind == "hexnut" else "rings"
    elif kind == "plate":
        wpx, hpx = j(200, 240), j(62, 78)

        def shape(d, s):
            d.rounded_rectangle([0, 0, wpx * s - 1, hpx * s - 1], radius=int(0.18 * hpx * s), fill=255)
        colour, tex = (c(195, 215), c(160, 180), c(50, 70)), "speckle"
    elif kind == "triangle":
        a = j(150, 180)
        wpx, hpx = a + 4, a * 0.87 + 4

        def shape(d, s):
            d.polygon([(2 * s, (hpx - 2) * s), ((wpx - 2) * s, (hpx - 2) * s), (wpx / 2 * s, 2 * s)], fill=255)
        colour, tex = (c(165, 185), c(45, 60), c(40, 55)), "matte"
    elif kind == "bolt":
        wpx, hpx = j(250, 290), j(40, 48)

        def shape(d, s):
            d.rectangle([0, 0.15 * hpx * s, 0.8 * wpx * s, 0.85 * hpx * s], fill=255)
            d.rectangle([0.8 * wpx * s, 0, wpx * s - 1, hpx * s - 1], fill=255)
        colour, tex = (c(120, 140), c(122, 142), c(128, 146)), "thread"
    else:
        raise ValueError(kind)
    alpha = np.asarray(_alpha_of(shape, wpx, hpx), np.uint8)
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.broadcast_to(np.array(colour, np.float32)[None, None], (h, w, 3)).copy()
    if tex == "radial":
        rgb += (22.0 * (0.6 - np.hypot(xx - w / 2, yy - h / 2) / (w / 2)))[..., None]
    elif tex == "rings":
        rgb += (9.0 * np.sin(np.hypot(xx - w / 2, yy - h / 2) * 2 * math.pi / (6.0 * SS)))[..., None]
    elif tex == "speckle":
        rgb += rng.normal(0, 7.0, (h, w, 1)).astype(np.float32)
    elif tex == "thread":
        rgb += (16.0 * np.sin(xx * 2 * math.pi / (8.0 * SS)) * (xx < 0.8 * w))[..., None]
    rgb += (12.0 * (0.5 - xx / w))[..., None] + rng.normal(0, 2.5, (h, w, 1)).astype(np.float32)
    return _finish(_bevel(rgb, alpha, 25.0), alpha)


def place(frame_f32, sprite, cx, cy, angle_deg, scale=1.0, gain=1.0, blur_px=0.0):
    """Alpha-composite an RGBA sprite onto a float32 frame [H,W,3] in place, rotated by angle_deg (counter-clockwise
    on screen), scaled, centred at full-res (cx, cy). blur_px >= 2 adds a horizontal motion blur."""
    sp = sprite
    if scale != 1.0:
        sp = sp.resize((max(1, round(sp.width * scale)), max(1, round(sp.height * scale))), Image.LANCZOS)
    sp = sp.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    a = np.asarray(sp, np.float32)
    k = int(round(blur_px))
    if k >= 2:
        from scipy.ndimage import uniform_filter1d

        a = uniform_filter1d(a, size=k, axis=1, mode="constant")
    h, w = a.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = frame_f32.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sub = a[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    al = sub[..., 3:4] / 255.0
    dst = frame_f32[fy0:fy1, fx0:fx1]
    dst[...] = dst * (1 - al) + np.clip(sub[..., :3] * gain, 0, 255) * al


def finish_frame(frame_f32, rng, noise=0.8):
    """Sensor noise + quantisation -> uint8 [H,W,3]."""
    f = frame_f32 + rng.normal(0.0, noise, frame_f32.shape).astype(np.float32)
    return np.clip(np.floor(f + 0.5), 0, 255).astype(np.uint8)


def jpeg_bytes(rgb_u8, quality):
    b = io.BytesIO()
    Image.fromarray(rgb_u8).save(b, "JPEG", quality=quality)
    return b.getvalue()


def decode_jpeg(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.uint8)


# =========================================================================================== export-format helpers
def mask_rle(labels, label, comp):
    """The component's pixels over its inclusive bbox as {"x","y","w","h","rle"} (0-run first, row-major)."""
    x0, y0, x1, y1 = int(comp["minX"]), int(comp["minY"]), int(comp["maxX"]), int(comp["maxY"])
    win = (np.asarray(labels)[y0:y1 + 1, x0:x1 + 1] == label).astype(np.uint8).ravel()
    runs, cur, n = [], 0, 0
    for v in win:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = v, 1
    runs.append(n)
    return {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1, "rle": [int(r) for r in runs]}


def rle_decode(mask):
    """{"x","y","w","h","rle"} -> uint8 [h, w] window (inverse of mask_rle)."""
    out = np.zeros(mask["w"] * mask["h"], np.uint8)
    pos, val = 0, 0
    for r in mask["rle"]:
        if val:
            out[pos:pos + r] = 1
        pos += r
        val ^= 1
    return out.reshape(mask["h"], mask["w"])


def jnum(v):
    """JSON-safe float (None for NaN/inf)."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)


# ======================================================================================================== scenario
FULL_W, FULL_H, FACTOR = 1280, 720, 4


def _pose(rng, angle_range=(-8.0, 8.0), jitter=(90.0, 60.0)):
    return dict(cx=FULL_W / 2 + float(rng.uniform(-jitter[0], jitter[0])),
                cy=FULL_H / 2 + float(rng.uniform(-jitter[1], jitter[1])),
                angle=float(rng.uniform(*angle_range)), scale=float(rng.uniform(0.96, 1.04)),
                gain=float(rng.uniform(0.97, 1.03)), blur=0.0)


def build_scenario(rng, small=False):
    """-> list of shots {"folder", "name", "tMs", "label", "clip", "trackId", "obj": None | {kind, surface, defect,
    seed, pose}}; teach frames first (time order), then negatives, then the parts clip by clip."""
    shots = []
    n_teach = 40 if small else 64
    dt = 187
    for i in range(n_teach):
        t = 1000 + i * dt
        if i < 2:
            obj = None                                                    # NO_OBJECT (hand not there yet)
        elif i == 2:
            obj = dict(kind="bracket", surface="brushed", defect=None,
                       pose=dict(cx=70.0, cy=360.0, angle=0.0, scale=1.0, gain=1.0, blur=0.0))  # TOUCHES_BORDER
        else:
            u = (i - 3) / max(1, n_teach - 4)
            obj = dict(kind="bracket", surface="brushed", defect=None,
                       pose=dict(cx=FULL_W / 2 + 70 * math.sin(2 * math.pi * i / 64),
                                 cy=FULL_H / 2 + 40 * math.sin(2 * math.pi * i / 41),
                                 angle=-22.0 + 44.0 * u, scale=1.0 + 0.03 * math.sin(2 * math.pi * i / 23),
                                 gain=1.0 + 0.02 * math.sin(2 * math.pi * i / 17),
                                 blur=7.0 if i % 9 == 4 else 0.0))            # a few motion-blurred frames
        shots.append(dict(folder="teach", name=f"{i:04d}", index=i, tMs=t, obj=obj))
    t = shots[-1]["tMs"] + 5000
    neg_kinds = ["hexnut", "washer", "plate", "triangle", "coin", "bolt", "hexnut", "plate"][: 4 if small else 8]
    for i, k in enumerate(neg_kinds):
        shots.append(dict(folder="negatives", name=f"{i:04d}", index=i, tMs=t,
                          obj=dict(kind=k, seed=int(rng.integers(1 << 30)), pose=_pose(rng, (0.0, 360.0), (60.0, 40.0)))))
        t += 1500
    plan = [("calib", "calib", 12 if small else 24, "good"), ("good", "good", 12 if small else 24, "good"),
            ("defect", "defect", 6 if small else 15, "defect"), ("wrong", "wrong", 6 if small else 12, "wrong"),
            ("look", "lookalike", 2 if small else 4, "lookalike"), ("rot", "rotated", 5 if small else 10, "rotated")]
    rot_angles = [30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0, -90.0, -135.0]
    for clip, label, n, kind in plan:
        t += 10000
        for j in range(n):
            if kind == "good":
                obj = dict(kind="bracket", surface="brushed", defect=None, pose=_pose(rng))
            elif kind == "defect":
                obj = dict(kind="bracket", surface="brushed", defect=DEFECT_KINDS[j % 3],
                           seed=int(rng.integers(1 << 30)), pose=_pose(rng))
            elif kind == "wrong":
                obj = dict(kind=WRONG_KINDS[j % len(WRONG_KINDS)], seed=int(rng.integers(1 << 30)),
                           pose=_pose(rng, (0.0, 360.0)))
            elif kind == "lookalike":
                obj = dict(kind="bracket", surface="knurl", defect=None, pose=_pose(rng))
            else:
                p = _pose(rng)
                p["angle"] = rot_angles[j % len(rot_angles)] + float(rng.uniform(-4, 4))
                obj = dict(kind="bracket", surface="brushed", defect=None, pose=p)
            shots.append(dict(folder="parts", name=f"{clip}_{j + 1:04d}", clip=clip, trackId=j + 1, label=label,
                              tMs=t, trigger="LINE", obj=obj))
            t += 700
    return shots


_SPRITES = {}


def sprite_for(obj):
    """Cached sprite for a scenario object."""
    if obj["kind"] == "bracket":
        key = ("bracket", obj.get("surface", "brushed"), obj.get("defect"), obj.get("seed"))
        if key not in _SPRITES:
            _SPRITES[key] = sprite_bracket(obj.get("surface", "brushed"), obj.get("defect"),
                                           np.random.default_rng(obj.get("seed", 0)))
    else:
        key = (obj["kind"], obj.get("seed"))
        if key not in _SPRITES:
            _SPRITES[key] = sprite_wrong(obj["kind"], np.random.default_rng(obj["seed"]))
    return _SPRITES[key]


def render_shot(sheet, obj, rng):
    f = sheet.copy()
    if obj is not None:
        p = obj["pose"]
        place(f, sprite_for(obj), p["cx"], p["cy"], p["angle"], p["scale"], p["gain"], p["blur"])
    return finish_frame(f, rng)


# ================================================================================== spec pipeline (tools/lab/twin_ref)
def pipeline_object(backbone_sha, mask_params, input_size=320, grid=40, dim=128, backbone_id="r18_320_f32",
                    knn_graph="knn_p1600_d128", l2=False):
    """The twin.json "pipeline" object (spec 8) of the phone pipeline being simulated."""
    mp = mask_params
    return {"backboneId": backbone_id, "backboneSha256": backbone_sha, "inputSize": input_size, "gh": grid,
            "gw": grid, "dim": dim, "precision": "fp32", "l2NormalizePatches": l2, "preprocessVersion": 1,
            "mask": {"analysisFactor": mp.analysisFactor, "kSigma": float(mp.kSigma), "sigmaMin": float(mp.sigmaMin),
                     "minBlobPx": mp.minBlobPx, "borderMargin": mp.borderMargin, "cropMargin": float(mp.cropMargin),
                     "coreThreshold": float(mp.coreThreshold)},
            "knnGraphId": knn_graph, "scoreRule": "SMOOTHED_MAX", "bankRule": "CORESET", "rotation": "NONE"}


def analysis_of(tr, full_rgb, factor):
    """spec 1.1 on the RGBA version of a full-resolution RGB frame."""
    h, w = full_rgb.shape[:2]
    rgba = np.dstack([full_rgb, np.full((h, w, 1), 255, np.uint8)])
    return tr.downscale_rgba(rgba.ravel(), w, h, 4 * w, factor)


def crop_u8(tr, full_rgb, crop, n):
    return tr.crop_bytes(tr.crop_resize(full_rgb, crop.x0, crop.y0, crop.side, n))


def component_info(tr, full_rgb, seg, comp, geom, crop, sanity, mp, gh, gw, input_size):
    """export-format section 3 CROPINFO (minus tMs) for one component + the crop bytes at input_size."""
    f = mp.analysisFactor
    cov = tr.patch_cov(seg.labels, comp.label, crop, f, gh, gw)
    c_in = crop_u8(tr, full_rgb, crop, input_size)
    sharp = tr.sharpness(tr.grey(c_in), 0, 0, input_size, input_size)[0]
    info = {"sanity": sanity, "sharpness": jnum(sharp),
            "crop": {"x0": float(crop.x0), "y0": float(crop.y0), "side": float(crop.side)},
            "mask": mask_rle(seg.labels, comp.label, comp.to_json()), "cov": [jnum(v) for v in cov.ravel()],
            "geometry": {"area": int(geom.area), "fill": jnum(geom.fill), "aspect": jnum(geom.aspect),
                         "hu1": jnum(geom.hu1), "solidity": jnum(geom.solidity)},
            "theta": jnum(geom.theta)}
    return info, c_in


def cropinfo(tr, full_rgb, small, sheet, mp, gh, gw, input_size):
    """Main object of one frame (spec 2.5-2.7, 3, 4, 1.3). Returns (info, component, crop, crop bytes, seg);
    component/crop/bytes are None for NO_OBJECT."""
    H, W = full_rgb.shape[:2]
    seg, geoms, mo, per, san = tr.analyse_scene(small, sheet, mp, gh, gw, W, H)
    if mo is None:
        return {"sanity": san}, None, None, None, seg
    crop = per[mo.label - 1]["crop"]
    info, c_in = component_info(tr, full_rgb, seg, mo, geoms[mo.label - 1], crop, san, mp, gh, gw, input_size)
    return info, mo, crop, c_in, seg


class R18:
    """The shipped ResNet18 backbone (ai_edge_litert): [320,320,3] uint8 -> [40,40,128] float32."""

    def __init__(self, path=R18_TFLITE, threads=4):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, crop):
        self.it.set_tensor(self.i["index"], crop.astype(np.float32)[None])
        self.it.invoke()
        return self.it.get_tensor(self.o["index"])[0].copy()


def phone_fields(j):
    """twin_ref Judgement -> the export's "phone" object."""
    return {"verdict": j.verdict, "reason": j.reason, "raw": jnum(j.raw), "s": jnum(j.s), "sim": jnum(j.sim),
            "a": jnum(j.anomalousFraction), "peak": None if j.peak is None else [int(j.peak[0]), int(j.peak[1])],
            "ms": None}


def sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


LABEL_ORDER = ["calib", "good", "defect", "wrong", "lookalike", "rotated"]


# ============================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "testdata", "synthetic_export"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--crop-size", type=int, default=448, help="side of the stored JPEG crops (manifest.cropSize)")
    ap.add_argument("--no-phone", action="store_true", help="skip the synthetic phone teach/judge (phone = null)")
    ap.add_argument("--small", action="store_true", help="fewer frames/parts (quick test)")
    a = ap.parse_args()
    import twin_ref as tr

    t_start = time.time()
    rng = np.random.default_rng(a.seed)
    mp = tr.MaskParams()
    f, gh, gw, n_in = mp.analysisFactor, 40, 40, 320
    shots = build_scenario(rng, a.small)
    sheet_f = render_sheet(FULL_W, FULL_H, np.random.default_rng(a.seed + 1))
    empties = [analysis_of(tr, finish_frame(sheet_f, rng), f) for _ in range(3)]
    sheet = tr.fit_sheet(empties, None, mp.sigmaMin)
    print(f"[sheet] mu {np.round(sheet.mu, 2).tolist()} sigma {np.round(sheet.sigma, 2).tolist()}")

    tmp = a.out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ("teach", "parts", "negatives"):
        os.makedirs(os.path.join(tmp, sub))
    bb = None if a.no_phone else R18()
    recs = []
    for s in shots:
        full = render_shot(sheet_f, s["obj"], rng)
        small = analysis_of(tr, full, f)
        info, comp, crop, c_in, _ = cropinfo(tr, full, small, sheet, mp, gh, gw, n_in)
        rec = dict(shot=s, info=info, feat=None)
        if comp is not None:
            with open(os.path.join(tmp, s["folder"], s["name"] + ".jpg"), "wb") as fh:
                fh.write(jpeg_bytes(crop_u8(tr, full, crop, a.crop_size), 95))
            if bb is not None and info["sanity"] == tr.OK:
                rec["feat"] = bb(c_in)
        recs.append(rec)
    counts = {}
    for r in recs:
        counts[r["info"]["sanity"]] = counts.get(r["info"]["sanity"], 0) + 1
    print(f"[render+mask] {len(recs)} shots in {time.time() - t_start:.0f} s; sanity counts: {counts}")

    twin_thr = None
    if bb is not None:
        teach_frames = []
        for r in recs:
            if r["shot"]["folder"] != "teach":
                continue
            i = r["info"]
            ok = i["sanity"] == tr.OK
            teach_frames.append(tr.TeachFrame(int(r["shot"]["tMs"]), ok, i.get("sharpness") if ok else None,
                                              np.array(i["cov"]) if ok else None, i.get("geometry") if ok else None,
                                              r["feat"] if ok else None))
        negs = []
        for r in recs:
            if r["shot"]["folder"] == "negatives" and r["feat"] is not None:
                core = tr.patch_sets_from_cov(np.array(r["info"]["cov"]).reshape(gh, gw)).core
                negs.append(tr.global_descriptor(r["feat"], core))
        twin = tr.teach(teach_frames, tr.TeachParams(), negatives=np.array(negs, np.float32) if negs else None)
        print(f"[phone twin] keyframes {len(twin.keyframes)} bank {twin.bank.shape[0]} tau {twin.tau:.4f} "
              f"tauId {twin.tauId:.4f} ({twin.tauIdRule}) segments {twin.segmentsUsed}")

        def judge(r, tw):
            i = r["info"]
            if i["sanity"] != tr.OK:
                return tr.judge(tw, None, None, None, sanity_reason=i["sanity"])
            return tr.judge(tw, r["feat"], np.array(i["cov"]), i["geometry"])

        # calibration presentations are judged by the uncalibrated Twin, then the Twin is calibrated (spec 10.1)
        cal_samples = []
        for r in recs:
            if r["shot"].get("label") == "calib":
                j = judge(r, twin)
                r["phone"] = phone_fields(j)
                ok = j.verdict != tr.REFRAME
                cal_samples.append(tr.CalSample(ok, bool(j.idOk), bool(j.geoOk), j.raw, j.sim,
                                                r["info"].get("geometry") if ok else None))
        if cal_samples:
            cal = tr.calibrate(twin.tauTeach, twin.positives, twin.negSims, twin.segmentsUsed, cal_samples)
            twin = tr.apply_calibration(twin, cal)
            print(f"[phone calibrate] valid {cal['m']}/{cal['n']} tau {twin.tau:.4f} tauId {twin.tauId:.4f}")
        for r in recs:
            if r["shot"]["folder"] == "parts" and "phone" not in r:
                r["phone"] = phone_fields(judge(r, twin))
        twin_thr = {"tau": twin.tau, "tauTeach": twin.tauTeach, "tauId": twin.tauId, "sensitivity": 1.0,
                    "coverageCut": twin.coverageCut, "calibrated": bool(twin.calibrated)}

    for r in recs:
        s, i = r["shot"], r["info"]
        d = {}
        if s["folder"] in ("teach", "negatives"):
            d["index"] = s["index"]
        if s["folder"] == "parts":
            d.update(clip=s["clip"], trackId=s["trackId"], label=s["label"], trigger=s["trigger"])
        d["tMs"] = int(s["tMs"])
        d.update(i)
        if s["folder"] == "parts":
            d["phone"] = r.get("phone")
        write_json(os.path.join(tmp, s["folder"], s["name"] + ".json"), d)
    pipe = pipeline_object(sha256_file(R18_TFLITE), mp)
    man = {"schema": 1, "createdAtMs": 1790000000000, "twinId": "synthetic-bracket", "twinName": "SYNTHETIC bracket",
           "pipeline": pipe, "fingerprint": tr.fingerprint(pipe), "analysisFactor": f, "inputSize": n_in,
           "cropSize": a.crop_size, "frameW": FULL_W, "frameH": FULL_H,
           "sheet": {"mu": [float(v) for v in sheet.mu], "sigma": [float(v) for v in sheet.sigma]},
           "twin": twin_thr,
           "counts": {k: sum(1 for x in shots if x["folder"] == k) for k in ("teach", "parts", "negatives")},
           "app": {"generator": "tools/lab/make_synthetic_export.py", "seed": a.seed,
                   "twinRefSha256": sha256_file(tr.__file__),
                   "note": "SYNTHETIC renders - mechanics/parser test only, not evidence"}}
    if twin_thr is None:
        del man["twin"]
    write_json(os.path.join(tmp, "manifest.json"), man)
    shutil.rmtree(a.out, ignore_errors=True)
    os.replace(tmp, a.out)
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(a.out) for fn in fns)
    print(f"[export] wrote {a.out}: {size / 1e6:.2f} MB in {time.time() - t_start:.0f} s")
    if bb is not None:
        for lab in LABEL_ORDER:
            vs = {}
            for r in recs:
                if r["shot"]["folder"] == "parts" and r["shot"]["label"] == lab:
                    vs[r["phone"]["verdict"]] = vs.get(r["phone"]["verdict"], 0) + 1
            print(f"   phone {lab:10s} {vs}")


if __name__ == "__main__":
    main()
