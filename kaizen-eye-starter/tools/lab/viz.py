"""Tiny helpers: load a photo the way the phone will feed it, and draw a heat-map overlay (PIL only)."""
import numpy as np
from PIL import Image, ImageOps


def load_square(path, size):
    """Open a photo (fixes phone EXIF rotation), centre-crop to square, resize -> uint8 [size,size,3]."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _jet(x):
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay(img_u8, dmap, tau, out_path, verdict_text=""):
    """Blend a heat map (normalised so 1.0 == the reject threshold) over the frame and save a PNG."""
    size = img_u8.shape[0]
    hm = Image.fromarray((np.clip(dmap / (2.0 * tau), 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    a = np.asarray(hm, dtype=np.float32) / 255.0
    col = _jet(a) * 255.0
    alpha = (np.clip(a * 2.0 - 0.5, 0, 0.75))[..., None]          # only warm areas show through
    out = img_u8.astype(np.float32) * (1 - alpha) + col * alpha
    Image.fromarray(out.astype(np.uint8)).save(out_path)
