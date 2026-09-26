/**
 * Load a photo the way tools/lab/viz.py `load_square` does: EXIF-upright, centre-crop to a square,
 * resize to the model input size, and hand back float32 RGB 0..255 in NHWC order.
 */
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import decodeJpeg from 'jpeg-js/lib/decoder';

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_LOOKUP = (() => {
  const t = new Uint8Array(256);
  for (let i = 0; i < B64.length; i++) t[B64.charCodeAt(i)] = i;
  t['-'.charCodeAt(0)] = 62;
  t['_'.charCodeAt(0)] = 63;
  return t;
})();

export function base64ToBytes(b64: string): Uint8Array {
  const clean = b64.replace(/^data:[^,]*,/, '').replace(/[^A-Za-z0-9+/\-_]/g, '');
  const n = clean.length;
  const out = new Uint8Array(Math.floor((n * 3) / 4));
  let o = 0;
  for (let i = 0; i < n; i += 4) {
    const a = B64_LOOKUP[clean.charCodeAt(i)];
    const b = B64_LOOKUP[clean.charCodeAt(i + 1)];
    const c = i + 2 < n ? B64_LOOKUP[clean.charCodeAt(i + 2)] : 0;
    const d = i + 3 < n ? B64_LOOKUP[clean.charCodeAt(i + 3)] : 0;
    out[o++] = (a << 2) | (b >> 4);
    if (i + 2 < n) out[o++] = ((b & 15) << 4) | (c >> 2);
    if (i + 3 < n) out[o++] = ((c & 3) << 6) | d;
  }
  return out.subarray(0, o);
}

const DISPLAY_SIZE = 1024;

export interface SquareImage {
  /** Float32 RGB 0..255, NHWC [size*size*3] - the model input. */
  rgb: Float32Array;
  /** Uri of the size x size square JPEG (for previews / heat-map background). */
  uri: string;
  size: number;
}

export async function loadSquare(uri: string, size: number): Promise<SquareImage> {
  // Decode once at full resolution (orientation applied), then centre-crop and step the size down
  // in halves so the final bilinear resize does not alias badly.
  const full = await ImageManipulator.manipulate(uri).renderAsync();
  const s = Math.min(full.width, full.height);
  const ctx = ImageManipulator.manipulate(full).crop({
    originX: Math.floor((full.width - s) / 2),
    originY: Math.floor((full.height - s) / 2),
    width: s,
    height: s,
  });
  // Sharp copy for the screen (the model input itself is only 256 px).
  let cur = s;
  while (cur / 2 >= DISPLAY_SIZE) {
    cur = Math.floor(cur / 2);
    ctx.resize({ width: cur, height: cur });
  }
  const dispSize = Math.min(cur, DISPLAY_SIZE);
  ctx.resize({ width: dispSize, height: dispSize });
  const disp = await ctx.renderAsync();
  const dispSaved = await disp.saveAsync({ format: SaveFormat.JPEG, compress: 0.9 });

  const mctx = ImageManipulator.manipulate(disp);
  cur = dispSize;
  while (cur / 2 >= size * 1.5) {
    cur = Math.floor(cur / 2);
    mctx.resize({ width: cur, height: cur });
  }
  mctx.resize({ width: size, height: size });
  const ref = await mctx.renderAsync();
  const saved = await ref.saveAsync({ format: SaveFormat.JPEG, compress: 1, base64: true });
  if (!saved.base64) throw new Error('Image manipulator returned no pixel data');
  const img = decodeJpeg(base64ToBytes(saved.base64), { useTArray: true, formatAsRGBA: false });
  if (img.width !== size || img.height !== size) {
    throw new Error(`Expected ${size}x${size} image, got ${img.width}x${img.height}`);
  }
  const rgb = new Float32Array(size * size * 3);
  const px = img.data as Uint8Array;
  for (let i = 0; i < rgb.length; i++) rgb[i] = px[i];
  return { rgb, uri: dispSaved.uri, size };
}
