"""Synthetic 'good part' generator + four defect types. For MECHANICS testing only (no real photos needed)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BG = (205, 205, 208)


def _base(size):
    im = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(im)
    m = size // 5
    d.rounded_rectangle([m, int(m * 1.3), size - m, size - int(m * 1.3)], radius=size // 12, fill=(35, 85, 160))
    for i in range(5):
        y = int(size * 0.30 + i * size * 0.07)
        d.rectangle([m + 12, y, size - m - 12, y + 4], fill=(235, 235, 240))
    r = int(size * 0.14)
    d.ellipse([int(size * 0.60), int(size * 0.60), int(size * 0.60) + r, int(size * 0.60) + r], fill=(225, 175, 40))
    for i in range(4):
        x0 = m + 14 + i * 18
        d.rectangle([x0, int(size * 0.68), x0 + 10, int(size * 0.68) + 10], fill=(240, 240, 240))
    return im


def make_good(rng, size=256):
    im = _base(size).rotate(float(rng.uniform(-1.5, 1.5)), translate=(int(rng.integers(-3, 4)), int(rng.integers(-3, 4))),
                            resample=Image.BILINEAR, fillcolor=BG)
    a = np.asarray(im, dtype=np.float32) * float(rng.uniform(0.95, 1.05)) + rng.normal(0, 2.0, (size, size, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def make_defect(rng, kind, size=256):
    """Returns (image uint8, bbox (x0,y0,x1,y1) of the defect)."""
    img = Image.fromarray(make_good(rng, size))
    d = ImageDraw.Draw(img)
    x = int(rng.integers(int(size * 0.32), int(size * 0.62)))
    y = int(rng.integers(int(size * 0.34), int(size * 0.58)))
    if kind == "scratch":
        d.line([x, y, x + 34, y + 12], fill=(245, 245, 245), width=2)
        bbox = (x, y, x + 34, y + 12)
    elif kind == "missing":
        d.rectangle([x, y, x + 26, y + 26], fill=BG)
        bbox = (x, y, x + 26, y + 26)
    elif kind == "stain":
        layer = Image.new("RGB", img.size, (120, 60, 40))
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse([x, y, x + 24, y + 24], fill=190)
        img = Image.composite(layer, img, mask.filter(ImageFilter.GaussianBlur(2)))
        bbox = (x, y, x + 24, y + 24)
    elif kind == "shift":
        blk = img.crop((x, y, x + 40, y + 14))
        img.paste(blk, (x + 8, y + 9))
        bbox = (x, y, x + 48, y + 23)
    else:
        raise ValueError(kind)
    return np.asarray(img, dtype=np.uint8), bbox


KINDS = ["scratch", "missing", "stain", "shift"]
