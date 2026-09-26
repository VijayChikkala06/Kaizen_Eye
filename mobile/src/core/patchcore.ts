/**
 * Kaizen Eye - scoring maths (TypeScript port of tools/lab/patchcore_ref.py).
 *
 * Training-free pipeline, same spec as the Python reference (checked against testdata/golden_core.json
 * by scripts/verify-core.ts):
 *   k        = max(minK, floor(ratio * N))       N = frames * gh * gw   (defaults ratio 0.05, minK 256)
 *   coreset  = greedy k-centre over all enrolment patches; start index 0; ties -> lowest index
 *   distance = Euclidean; "nearest" = smallest distance to any bank entry
 *   LOO      = per enrolment frame f: max over its patches of the nearest distance to bank entries not from f
 *   tau      = max_f LOO_f * margin              (the app enrols with margin 1.0 and applies sensitivity at runtime)
 *   score    = max patch distance / (tau * sensitivity);  > 1.0 means REJECT
 *
 * Feature maps are flat Float32Arrays in NHWC order: [gh * gw * dim] (exactly what the .tflite outputs).
 *
 * Speed: Hermes has no JIT, so the loops use exact pruning that never changes the result:
 *   - partial-distance elimination: stop summing a squared distance once it already exceeds the value it
 *     has to beat (all terms are >= 0, so the full sum can only be larger);
 *   - for LOO frame scores only the max over patches is needed, so a patch's nearest-neighbour search stops
 *     as soon as it finds a bank entry closer than the frame's current max.
 */

export interface Profile {
  version: 1;
  gh: number;
  gw: number;
  dim: number;
  /** Memory bank, [k * dim] row-major. */
  bank: Float32Array;
  /** Enrolment frame index each bank row came from, [k]. */
  bankFrame: Int32Array;
  /** Threshold = max LOO score * margin. */
  tau: number;
  margin: number;
  looScores: number[];
  nFrames: number;
  createdAt: number;
}

export interface EnrolOptions {
  ratio?: number;
  minK?: number;
  margin?: number;
  /** Called with a stage label and 0..1 progress; may be used to update the UI. */
  onProgress?: (stage: string, fraction: number) => void;
  /** Awaited now and then so the UI thread can render (pass undefined in Node / tests). */
  yieldFn?: () => Promise<void>;
}

export interface ScoreResult {
  /** Nearest-neighbour distance per patch, [gh * gw]. */
  dmap: Float32Array;
  /** max(dmap) */
  raw: number;
  /** raw / (tau * sensitivity);  > 1.0 = REJECT */
  score: number;
  /** Row / column of the (3x3 smoothed) hottest patch. */
  peak: { row: number; col: number };
}

const YIELD_EVERY_MS = 40;

function makeYielder(yieldFn?: () => Promise<void>) {
  let last = Date.now();
  return async () => {
    if (!yieldFn) return;
    const now = Date.now();
    if (now - last >= YIELD_EVERY_MS) {
      await yieldFn();
      last = Date.now();
    }
  };
}

/** Squared distance between row a of x and row b of y, abandoning once it reaches `limit`. */
function sqdistBounded(x: Float32Array, a: number, y: Float32Array, b: number, dim: number, limit: number): number {
  let acc = 0;
  let t = 0;
  const end8 = dim - (dim % 8);
  while (t < end8) {
    const d0 = x[a + t] - y[b + t];
    const d1 = x[a + t + 1] - y[b + t + 1];
    const d2 = x[a + t + 2] - y[b + t + 2];
    const d3 = x[a + t + 3] - y[b + t + 3];
    const d4 = x[a + t + 4] - y[b + t + 4];
    const d5 = x[a + t + 5] - y[b + t + 5];
    const d6 = x[a + t + 6] - y[b + t + 6];
    const d7 = x[a + t + 7] - y[b + t + 7];
    acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
    if (acc >= limit) return acc;
    t += 8;
  }
  while (t < dim) {
    const d = x[a + t] - y[b + t];
    acc += d * d;
    t++;
  }
  return acc;
}

/** Greedy k-centre coreset. Deterministic: starts at `start`, ties -> lowest index (like np.argmax). */
export async function greedyCoreset(
  x: Float32Array,
  dim: number,
  k: number,
  start = 0,
  onProgress?: (fraction: number) => void,
  yieldFn?: () => Promise<void>,
): Promise<Int32Array> {
  const n = x.length / dim;
  k = Math.min(k, n);
  const sel = new Int32Array(k);
  sel[0] = start;
  const mind = new Float32Array(n);
  const s0 = start * dim;
  for (let i = 0; i < n; i++) mind[i] = sqdistBounded(x, i * dim, x, s0, dim, Infinity);
  const maybeYield = makeYielder(yieldFn);
  for (let s = 1; s < k; s++) {
    let j = 0;
    let best = mind[0];
    for (let i = 1; i < n; i++) {
      if (mind[i] > best) {
        best = mind[i];
        j = i;
      }
    }
    sel[s] = j;
    const jo = j * dim;
    for (let i = 0; i < n; i++) {
      const m = mind[i];
      if (m === 0) continue;
      const d = sqdistBounded(x, i * dim, x, jo, dim, m);
      if (d < m) mind[i] = d;
    }
    if ((s & 15) === 0) {
      onProgress?.(s / k);
      await maybeYield();
    }
  }
  onProgress?.(1);
  return sel;
}

/**
 * Nearest-neighbour squared distance of patch p (row pOff of feats) to the bank rows allowed by `skipFrame`.
 * Stops early once the best found is <= `stopBelow` (used for max-of-min pruning; pass -1 to disable).
 */
function nnSq(
  feats: Float32Array,
  pOff: number,
  bank: Float32Array,
  bankFrame: Int32Array,
  dim: number,
  skipFrame: number,
  stopBelow: number,
): number {
  let best = Infinity;
  const k = bankFrame.length;
  for (let b = 0; b < k; b++) {
    if (bankFrame[b] === skipFrame) continue;
    const d = sqdistBounded(feats, pOff, bank, b * dim, dim, best);
    if (d < best) {
      best = d;
      if (best <= stopBelow) return best;
    }
  }
  return best;
}

/** 3x3 mean filter with edge replication (matches patchcore_ref.smooth3). */
export function smooth3(m: Float32Array, h: number, w: number): Float32Array {
  const out = new Float32Array(h * w);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      let acc = 0;
      for (let dr = -1; dr <= 1; dr++) {
        const rr = Math.min(h - 1, Math.max(0, r + dr));
        for (let dc = -1; dc <= 1; dc++) {
          const cc = Math.min(w - 1, Math.max(0, c + dc));
          acc += m[rr * w + cc];
        }
      }
      out[r * w + c] = acc / 9;
    }
  }
  return out;
}

export function bankSize(nPatches: number, ratio = 0.05, minK = 256): number {
  return Math.max(minK, Math.floor(ratio * nPatches));
}

/** Build the memory bank + threshold from good frames (each [gh*gw*dim]). */
export async function enrol(
  frames: Float32Array[],
  gh: number,
  gw: number,
  dim: number,
  opts: EnrolOptions = {},
): Promise<Profile> {
  const { ratio = 0.05, minK = 256, margin = 1.0, onProgress, yieldFn } = opts;
  const n = frames.length;
  if (n < 2) throw new Error('Need at least 2 good frames to enrol (leave-one-out calibration).');
  const per = gh * gw;
  const x = new Float32Array(n * per * dim);
  frames.forEach((f, i) => {
    if (f.length !== per * dim) throw new Error(`Frame ${i} has ${f.length} values, expected ${per * dim}`);
    x.set(f, i * per * dim);
  });
  const k = bankSize(n * per, ratio, minK);

  onProgress?.('Selecting memory bank (coreset)', 0);
  const sel = await greedyCoreset(x, dim, k, 0, (p) => onProgress?.('Selecting memory bank (coreset)', p), yieldFn);
  const kk = sel.length;
  const bank = new Float32Array(kk * dim);
  const bankFrame = new Int32Array(kk);
  for (let i = 0; i < kk; i++) {
    bank.set(x.subarray(sel[i] * dim, sel[i] * dim + dim), i * dim);
    bankFrame[i] = Math.floor(sel[i] / per);
  }

  // Leave-one-frame-out calibration on the final coreset.
  const maybeYield = makeYielder(yieldFn);
  const held: number[] = [];
  for (let f = 0; f < n; f++) {
    onProgress?.('Calibrating threshold (leave-one-out)', f / n);
    let hasOther = false;
    for (let b = 0; b < kk; b++) if (bankFrame[b] !== f) { hasOther = true; break; }
    if (!hasOther) continue;
    const feats = frames[f];
    let frameMaxSq = -1;
    for (let p = 0; p < per; p++) {
      const d = nnSq(feats, p * dim, bank, bankFrame, dim, f, frameMaxSq);
      if (d > frameMaxSq) frameMaxSq = d;
      if ((p & 63) === 0) await maybeYield();
    }
    held.push(Math.sqrt(frameMaxSq));
  }
  onProgress?.('Calibrating threshold (leave-one-out)', 1);
  const tau = Math.max(...held) * margin;
  return {
    version: 1,
    gh,
    gw,
    dim,
    bank,
    bankFrame,
    tau,
    margin,
    looScores: held,
    nFrames: n,
    createdAt: Date.now(),
  };
}

/** Score one feature map [gh*gw*dim] against a profile. */
export async function score(
  feat: Float32Array,
  prof: Profile,
  sensitivity = 1.0,
  yieldFn?: () => Promise<void>,
): Promise<ScoreResult> {
  const { gh, gw, dim } = prof;
  const per = gh * gw;
  if (feat.length !== per * dim) throw new Error(`Feature map has ${feat.length} values, expected ${per * dim}`);
  const dmap = new Float32Array(per);
  const maybeYield = makeYielder(yieldFn);
  let raw = 0;
  for (let p = 0; p < per; p++) {
    const d = Math.sqrt(nnSq(feat, p * dim, prof.bank, prof.bankFrame, dim, -1, -1));
    dmap[p] = d;
    if (d > raw) raw = d;
    if ((p & 63) === 0) await maybeYield();
  }
  const sm = smooth3(dmap, gh, gw);
  let pk = 0;
  for (let i = 1; i < per; i++) if (sm[i] > sm[pk]) pk = i;
  return {
    dmap,
    raw,
    score: raw / (prof.tau * sensitivity),
    peak: { row: Math.floor(pk / gw), col: pk % gw },
  };
}
