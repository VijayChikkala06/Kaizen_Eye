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
  /**
   * Whole-image check (app extension, not in the Python spec): one L2-normalised mean feature vector per
   * enrolment frame, [nFrames * dim], and its leave-one-out threshold (cosine distance). Catches "wrong part /
   * wrong scene" photos whose individual patches all look familiar (e.g. a background texture seen at enrolment).
   */
  globals?: Float32Array;
  gTau?: number;
  gLoo?: number[];
  /** App: uri of a saved enrolment photo, shown as an alignment guide over the camera. */
  refImage?: string;
  /** Outer patch rows/columns ignored by the frame score (0 = spec). */
  border?: number;
  /** Frame score on the 3x3-smoothed distance map. */
  smooth?: boolean;
  /** Indices (into the photos the user took) of enrolment photos dropped as outliers. */
  droppedFrames?: number[];
}

/** Minimum whole-image threshold, so near-identical enrolment frames do not make the check hair-trigger. */
export const MIN_GLOBAL_TAU = 0.02;

export interface EnrolOptions {
  ratio?: number;
  minK?: number;
  margin?: number;
  /** Called with a stage label and 0..1 progress; may be used to update the UI. */
  onProgress?: (stage: string, fraction: number) => void;
  /** Awaited now and then so the UI thread can render (pass undefined in Node / tests). */
  yieldFn?: () => Promise<void>;
  /** Ignore the outer `border` patch rows/columns in frame scores (0 = spec). */
  border?: number;
  /** Drop enrolment frames whose LOO score is an outlier (median + 3 MAD, max 20%) and re-enrol without them. */
  trimOutliers?: boolean;
  /** Frame score on the 3x3-smoothed distance map (patchcore_ref smooth=True). */
  smooth?: boolean;
}

/**
 * Optional accelerated nearest-neighbour backend (e.g. a TFLite matmul model). Returns, for every patch of `feats`
 * [P*dim], the squared distance to the nearest bank row whose bankFrame != skipFrame (-1 = use all rows).
 */
export type NNBackend = (
  feats: Float32Array,
  bank: Float32Array,
  bankFrame: Int32Array,
  dim: number,
  skipFrame: number,
) => Promise<Float32Array>;

let nnBackend: NNBackend | null = null;
export function setNNBackend(b: NNBackend | null): void {
  nnBackend = b;
}

/** True when patch index p (row-major on a gh x gw grid) is inside the scored area. */
function inside(p: number, gh: number, gw: number, border: number): boolean {
  if (border <= 0) return true;
  const r = Math.floor(p / gw);
  const c = p % gw;
  return r >= border && r < gh - border && c >= border && c < gw - border;
}

export interface ScoreResult {
  /** Nearest-neighbour distance per patch, [gh * gw]. */
  dmap: Float32Array;
  /** max(dmap) */
  raw: number;
  /** max(patchScore, globalScore);  > 1.0 = REJECT */
  score: number;
  /** raw / (tau * sensitivity) - the PatchCore defect score (spec). */
  patchScore: number;
  /** whole-image cosine distance / (gTau * sensitivity); 0 when the profile has no globals. */
  globalScore: number;
  /** Which check decides the verdict. */
  reason: 'ok' | 'defect' | 'different';
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
  // assign[i] = which selected centre (index into sel) is currently nearest to point i.
  const assign = new Int32Array(n);
  // cc[t] = distance from the newest centre to centre sel[t].
  const cc = new Float64Array(k);
  // prune[i] = 2 * sqrt(mind[i]) with a tiny safety margin (see the triangle-inequality test below).
  const prune = new Float64Array(n);
  const s0 = start * dim;
  for (let i = 0; i < n; i++) {
    mind[i] = sqdistBounded(x, i * dim, x, s0, dim, Infinity);
    prune[i] = 2 * Math.sqrt(mind[i]) * (1 + 1e-6) + 1e-12;
  }
  // Farthest point via per-block maxima: only blocks whose points changed are rescanned.
  const B = 128;
  const nb = Math.ceil(n / B);
  const blockMax = new Float32Array(nb);
  const blockArg = new Int32Array(nb);
  const dirty = new Uint8Array(nb).fill(1);
  const maybeYield = makeYielder(yieldFn);
  for (let s = 1; s < k; s++) {
    for (let b = 0; b < nb; b++) {
      if (!dirty[b]) continue;
      const lo = b * B;
      const hi = Math.min(n, lo + B);
      let bm = mind[lo];
      let ba = lo;
      for (let i = lo + 1; i < hi; i++) {
        if (mind[i] > bm) {
          bm = mind[i];
          ba = i;
        }
      }
      blockMax[b] = bm;
      blockArg[b] = ba;
      dirty[b] = 0;
    }
    // Strict '>' over blocks in order + lowest index inside each block = lowest index overall (np.argmax ties).
    let j = blockArg[0];
    let best = blockMax[0];
    for (let b = 1; b < nb; b++) {
      if (blockMax[b] > best) {
        best = blockMax[b];
        j = blockArg[b];
      }
    }
    sel[s] = j;
    const jo = j * dim;
    for (let t = 0; t < s; t++) cc[t] = Math.sqrt(sqdistBounded(x, sel[t] * dim, x, jo, dim, Infinity));
    for (let i = 0; i < n; i++) {
      // Exact pruning (triangle inequality): d(i, new) >= d(new, a_i) - d(i, a_i). If the new centre is at least
      // twice as far from i's current centre a_i as i is, it cannot be closer to i - skip the distance entirely.
      // The safety margin in prune[] keeps float rounding from ever skipping a genuine update.
      if (cc[assign[i]] >= prune[i]) continue;
      const m = mind[i];
      if (m === 0) continue;
      const d = sqdistBounded(x, i * dim, x, jo, dim, m);
      if (d < m) {
        mind[i] = d;
        assign[i] = s;
        prune[i] = 2 * Math.sqrt(mind[i]) * (1 + 1e-6) + 1e-12;
        dirty[(i / B) | 0] = 1;
      }
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

/** L2-normalised mean of the patch vectors of one feature map [per * dim] -> [dim]. */
export function globalDescriptor(feat: Float32Array, per: number, dim: number): Float32Array {
  const g = new Float32Array(dim);
  for (let p = 0; p < per; p++) {
    const o = p * dim;
    for (let t = 0; t < dim; t++) g[t] += feat[o + t];
  }
  let n = 0;
  for (let t = 0; t < dim; t++) n += g[t] * g[t];
  n = Math.sqrt(n) || 1;
  for (let t = 0; t < dim; t++) g[t] /= n;
  return g;
}

/** Cosine distance between unit vector g and row f of the globals matrix. */
function cosDist(g: Float32Array, globals: Float32Array, f: number, dim: number): number {
  let s = 0;
  const o = f * dim;
  for (let t = 0; t < dim; t++) s += g[t] * globals[o + t];
  return 1 - s;
}

/**
 * Frame score (patchcore_ref.frame_score): max of the distance map - optionally 3x3-smoothed first (on the full
 * map) - ignoring the outer `border` patch rows/columns.
 */
export function frameScore(dmap: Float32Array, gh: number, gw: number, smooth = false, border = 0): number {
  const m = smooth ? smooth3(dmap, gh, gw) : dmap;
  let best = -Infinity;
  for (let p = 0; p < gh * gw; p++) if (inside(p, gh, gw, border) && m[p] > best) best = m[p];
  return best;
}

export function bankSize(nPatches: number, ratio = 0.05, minK = 256): number {
  return Math.max(minK, Math.floor(ratio * nPatches));
}

/** Frames whose LOO score is far above the others: > median + 3 * MAD (scaled), at most 20% of the frames. */
export function outlierFrames(loo: number[], maxFraction = 0.2): number[] {
  const n = loo.length;
  if (n < 6) return [];
  const med = (v: number[]) => {
    const s = [...v].sort((a, b) => a - b);
    return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
  };
  const m = med(loo);
  const mad = med(loo.map((v) => Math.abs(v - m))) * 1.4826 || 1e-9;
  return loo
    .map((v, i) => ({ v, i }))
    .filter((o) => o.v > m + 3 * mad)
    .sort((a, b) => b.v - a.v)
    .slice(0, Math.floor(maxFraction * n))
    .map((o) => o.i);
}

/** Build the memory bank + threshold from good frames (each [gh*gw*dim]). */
export async function enrol(
  frames: Float32Array[],
  gh: number,
  gw: number,
  dim: number,
  opts: EnrolOptions = {},
): Promise<Profile> {
  const prof = await enrolOnce(frames, gh, gw, dim, opts);
  if (!opts.trimOutliers) return prof;
  const drop = outlierFrames(prof.looScores);
  if (!drop.length) return { ...prof, droppedFrames: [] };
  const keep = frames.filter((_, i) => !drop.includes(i));
  const trimmed = await enrolOnce(keep, gh, gw, dim, opts);
  return { ...trimmed, droppedFrames: drop.sort((a, b) => a - b) };
}

async function enrolOnce(
  frames: Float32Array[],
  gh: number,
  gw: number,
  dim: number,
  opts: EnrolOptions,
): Promise<Profile> {
  const { ratio = 0.05, minK = 256, margin = 1.0, onProgress, yieldFn, border = 0, smooth = false } = opts;
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
    if (nnBackend || smooth) {
      // Full distance map (needed for smoothing; cheap with an accelerated backend).
      const dm = new Float32Array(per);
      if (nnBackend) {
        const d2 = await nnBackend(feats, bank, bankFrame, dim, f);
        for (let p = 0; p < per; p++) dm[p] = Math.sqrt(Math.max(0, d2[p]));
        await maybeYield();
      } else {
        for (let p = 0; p < per; p++) {
          dm[p] = Math.sqrt(nnSq(feats, p * dim, bank, bankFrame, dim, f, -1));
          if ((p & 63) === 0) await maybeYield();
        }
      }
      held.push(frameScore(dm, gh, gw, smooth, border));
    } else {
      // Spec path: only the max is needed, so prune each patch's search at the frame's running max.
      let frameMaxSq = -1;
      for (let p = 0; p < per; p++) {
        if (!inside(p, gh, gw, border)) continue;
        const d = nnSq(feats, p * dim, bank, bankFrame, dim, f, frameMaxSq);
        if (d > frameMaxSq) frameMaxSq = d;
        if ((p & 63) === 0) await maybeYield();
      }
      held.push(Math.sqrt(Math.max(0, frameMaxSq)));
    }
  }
  onProgress?.('Calibrating threshold (leave-one-out)', 1);
  const tau = Math.max(...held) * margin;

  // Whole-image descriptors + leave-one-out threshold (nearest OTHER enrolment frame).
  const globals = new Float32Array(n * dim);
  frames.forEach((f, i) => globals.set(globalDescriptor(f, per, dim), i * dim));
  const gLoo: number[] = [];
  for (let f = 0; f < n; f++) {
    const g = globals.subarray(f * dim, f * dim + dim);
    let best = Infinity;
    for (let o = 0; o < n; o++) if (o !== f) best = Math.min(best, cosDist(g, globals, o, dim));
    gLoo.push(best);
  }
  const gTau = Math.max(MIN_GLOBAL_TAU, Math.max(...gLoo) * margin);
  return {
    globals,
    gTau,
    gLoo,
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
    border,
    smooth,
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
  const border = prof.border ?? 0;
  const dmap = new Float32Array(per);
  const maybeYield = makeYielder(yieldFn);
  if (nnBackend) {
    const d2 = await nnBackend(feat, prof.bank, prof.bankFrame, dim, -1);
    for (let p = 0; p < per; p++) dmap[p] = Math.sqrt(Math.max(0, d2[p]));
  } else {
    for (let p = 0; p < per; p++) {
      dmap[p] = Math.sqrt(nnSq(feat, p * dim, prof.bank, prof.bankFrame, dim, -1, -1));
      if ((p & 63) === 0) await maybeYield();
    }
  }
  const raw = frameScore(dmap, gh, gw, prof.smooth ?? false, border);
  const sm = smooth3(dmap, gh, gw);
  let pk = -1;
  for (let i = 0; i < per; i++) if (inside(i, gh, gw, border) && (pk < 0 || sm[i] > sm[pk])) pk = i;
  if (pk < 0) pk = 0;

  const patchScore = raw / (prof.tau * sensitivity);
  let globalScore = 0;
  if (prof.globals && prof.gTau) {
    const g = globalDescriptor(feat, per, dim);
    let best = Infinity;
    for (let f = 0; f < prof.globals.length / dim; f++) best = Math.min(best, cosDist(g, prof.globals, f, dim));
    globalScore = best / (prof.gTau * sensitivity);
  }
  const s = Math.max(patchScore, globalScore);
  return {
    dmap,
    raw,
    score: s,
    patchScore,
    globalScore,
    reason: s <= 1.0 ? 'ok' : globalScore > patchScore ? 'different' : 'defect',
    peak: { row: Math.floor(pk / gw), col: pk % gw },
  };
}
