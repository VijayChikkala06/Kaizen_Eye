/**
 * Kaizen Eye - PatchCore nearest-neighbour search on LiteRT, with an exact JavaScript fallback.
 *
 * For each patch vector of an image: the squared distance to the nearest memory-bank row. As a matrix product this is
 * small work for XNNPACK / a GPU / an NPU, so tools/make_knn_model.py builds weight-less .tflite models for it
 * (assets/models/knn_p1600_d128_k{800,1600,2400}.tflite, one per bank-size bucket = 10 / 20 / 30 enrolment photos;
 * inputs feats [1600,128], bankT [128,K], bankNorm [K]; output minD2 [1600]). They run on the delegate accelerator.ts
 * picks for this device (benchmarked once, cached). Maths = tools/lab/patchcore_ref.py sqdist: |f|^2 + |b|^2 - 2 f.b,
 * clamped at 0, in float32 (vs float64: ~6e-5 of the largest distance on real features - irrelevant for scores).
 *
 * API (for core/patchcore.ts)
 *   knnMinSq(feats, bank, bankNormExtra?, dim = 128): Promise<Float32Array>
 *       feats [P*dim] row-major (any P; the model takes 1,600 patches per run, other counts are padded / split),
 *       bank  [K*dim] row-major (any K; the smallest bucket >= K is used, zero rows padded and masked; banks above
 *             2,400 rows are split into chunks and the minima combined).
 *       bankNormExtra [K], optional: added to each bank row's squared distance, >= 0. Put KNN_MASK (1e30) on rows to
 *             exclude (leave-one-out). If every row is excluded the result is ~1e30.
 *       -> [P] squared distances (take Math.sqrt for PatchCore distances).
 *   tfliteNNBackend: NNBackend  (feats, bank, bankFrame, dim, skipFrame) => Promise<Float32Array>
 *       For core/patchcore.ts setNNBackend(tfliteNNBackend): per patch (all P), the squared distance to the nearest
 *       bank row whose bankFrame !== skipFrame (-1 = all rows); excluded rows get KNN_MASK.
 *   prepareBank(bank, dim?)  bank -> padded, transposed model inputs + |b|^2, cached per bank array (WeakMap).
 *       knnMinSq does it lazily; call it after enrolment / profile load to take it off the first inspection.
 *       Do not modify a bank in place afterwards (a cheap fingerprint catches most such changes and re-prepares).
 *   loadKnn() / knnReport()  model + accelerator report (backbone.ts loads it at start-up: BackboneInfo.knn).
 *   runSelfTest()  logs TFLite-vs-brute-force agreement and timings (console.log -> logcat tag ReactNativeJS).
 *
 * Every call does one model run per 1,600 patches per bank chunk (normally exactly one). Runs are serialised per model.
 * Without a usable model (load failure, dim !== 128) knnMinSq falls back to knnMinSqJS: exact search with partial-
 * distance elimination, the same result as the loops in core/patchcore.ts.
 */
import { Asset } from 'expo-asset';
import type { TensorflowModel } from 'react-native-fast-tflite';

import type { NNBackend } from '../core/patchcore';

import {
  type AcceleratedModel,
  type AcceleratedModelSpec,
  type AcceleratorReport,
  loadAccelerated,
  loadWithDecisionOf,
  now,
} from './accelerator';

/** bankNormExtra value that excludes a bank row. */
export const KNN_MASK = 1e30;

const P = 1600;
const D = 128;
const BUCKETS: { K: number; module: number }[] = [
  { K: 800, module: require('../../assets/models/knn_p1600_d128_k800.tflite') },
  { K: 1600, module: require('../../assets/models/knn_p1600_d128_k1600.tflite') },
  { K: 2400, module: require('../../assets/models/knn_p1600_d128_k2400.tflite') },
];
const MAX_K = BUCKETS[BUCKETS.length - 1].K;
const bucketFor = (rows: number) => (BUCKETS.find((b) => b.K >= rows) ?? BUCKETS[BUCKETS.length - 1]).K;

// ------------------------------------------------------------------------------------------------------- models
interface KnnModel {
  K: number;
  accel: AcceleratedModel;
  /** Input index of feats, bankT, bankNorm. */
  order: [number, number, number];
}

/** Map the model's inputs (by name, else by shape) and check them. */
function inputOrder(m: TensorflowModel, K: number): [number, number, number] {
  const ins = m.inputs;
  const same = (s: number[], want: number[]) => s.length === want.length && s.every((v, i) => v === want[i]);
  const find = (name: string, shape: number[]) => {
    let i = ins.findIndex((t) => t.name === name || t.name.endsWith(`_${name}:0`) || t.name.endsWith(`${name}:0`));
    if (i < 0) i = ins.findIndex((t) => same(t.shape, shape));
    if (i < 0 || !same(ins[i].shape, shape) || ins[i].dataType !== 'float32') {
      throw new Error(`k-NN model: no float32 input ${name} [${shape}] (inputs: ${ins.map((t) => `${t.name}[${t.shape}]`).join(', ')})`);
    }
    return i;
  };
  const order: [number, number, number] = [find('feats', [P, D]), find('bankT', [D, K]), find('bankNorm', [K])];
  const out = m.outputs[0];
  if (ins.length !== 3 || new Set(order).size !== 3 || !out || !same(out.shape, [P]) || out.dataType !== 'float32') {
    throw new Error(`k-NN model: unexpected I/O (output ${out ? `[${out.shape}]` : 'missing'})`);
  }
  return order;
}

/** Deterministic benchmark inputs shaped like real features (non-negative, clustered, |f|^2 ~ 2-3). */
function benchInputs(m: TensorflowModel, K: number): ArrayBuffer[] {
  const rnd = lcg(1234 + K);
  const feats = fakeFeatures(rnd, P, D);
  const rows = K - 24; // exercise the padding
  const bank = fakeFeatures(rnd, rows, D);
  const extra = new Float32Array(rows);
  for (let k = 0; k < rows; k += 20) extra[k] = KNN_MASK; // and the leave-one-out mask
  const c = prepareChunk(bank, D, 0, rows, K);
  const norm = withExtra(c, extra);
  const ins: ArrayBuffer[] = [];
  const order = inputOrder(m, K);
  ins[order[0]] = feats.buffer as ArrayBuffer;
  ins[order[1]] = c.bankT;
  ins[order[2]] = norm.buffer as ArrayBuffer;
  return ins;
}

async function specFor(K: number): Promise<AcceleratedModelSpec> {
  const b = BUCKETS.find((x) => x.K === K);
  if (!b) throw new Error(`no k-NN model for K=${K}`);
  // Release builds: bundled assets are raw resources fast-tflite cannot open; expo-asset copies them to a file.
  const asset = Asset.fromModule(b.module);
  await asset.downloadAsync();
  const name = `knn_p${P}_d${D}_k${K}`;
  return {
    name,
    variants: [{ modelId: name, url: asset.localUri ?? asset.uri, hash: asset.hash ?? 'nohash' }],
    inputs: (m) => benchInputs(m, K),
    check: (m) => void inputOrder(m, K),
  };
}

let primary: Promise<KnnModel> | null = null; // the largest bucket: benchmarked, its decision is shared
const others = new Map<number, Promise<KnnModel | null>>();
let lastReport: AcceleratorReport | null = null;

function loadPrimary(): Promise<KnnModel> {
  if (!primary) {
    primary = (async () => {
      const accel = await loadAccelerated(await specFor(MAX_K));
      lastReport = accel.report;
      return { K: MAX_K, accel, order: inputOrder(accel.model, MAX_K) };
    })();
  }
  return primary;
}

function bucketModel(K: number): Promise<KnnModel | null> {
  if (K === MAX_K) return loadPrimary().catch(() => null);
  let p = others.get(K);
  if (!p) {
    p = (async () => {
      const prim = await loadPrimary();
      const accel = await loadWithDecisionOf(await specFor(K), prim.accel);
      return { K, accel, order: inputOrder(accel.model, K) };
    })().catch((e) => {
      console.warn(`[knn] model K=${K} unavailable, using JavaScript:`, e instanceof Error ? e.message : e);
      return null;
    });
    others.set(K, p);
  }
  return p;
}

/** Load the k-NN model (benchmarking delegates on first use) -> its accelerator report. Rejects if unusable. */
export async function loadKnn(): Promise<AcceleratorReport> {
  return (await loadPrimary()).accel.report;
}

/** Accelerator report of the k-NN model, or null if it is not loaded (yet). */
export function knnReport(): AcceleratorReport | null {
  return lastReport;
}

/** Drop the loaded models (e.g. before re-benchmarking); they are reloaded on the next call. */
export function resetKnn(): void {
  const all = [primary, ...others.values()];
  primary = null;
  others.clear();
  lastReport = null;
  for (const p of all) p?.then((m) => m?.accel.dispose()).catch(() => undefined);
}

// -------------------------------------------------------------------------------------------------- bank layout
interface Chunk {
  /** First bank row of this chunk, number of real rows, model bucket. */
  start: number;
  rows: number;
  K: number;
  /** [dim, K] row-major, zero-padded columns. */
  bankT: ArrayBuffer;
  /** [K] |b|^2, KNN_MASK on padded columns. */
  norm: Float32Array;
}
export interface PreparedBank {
  rows: number;
  dim: number;
  chunks: Chunk[];
  fp: number;
}
const prepared = new WeakMap<Float32Array, PreparedBank>();

function prepareChunk(bank: Float32Array, dim: number, start: number, rows: number, K: number): Chunk {
  const t = new Float32Array(dim * K);
  const norm = new Float32Array(K).fill(KNN_MASK);
  for (let r = 0; r < rows; r++) {
    const o = (start + r) * dim;
    let s = 0;
    for (let d = 0; d < dim; d++) {
      const v = bank[o + d];
      t[d * K + r] = v;
      s += v * v;
    }
    norm[r] = s;
  }
  return { start, rows, K, bankT: t.buffer as ArrayBuffer, norm };
}

/** Cheap content fingerprint (length + 128 sampled values) to notice a bank modified in place. */
function fingerprint(a: Float32Array): number {
  const bits = new Int32Array(a.buffer, a.byteOffset, a.length);
  let h = 0x811c9dc5 ^ a.length;
  const step = Math.max(1, Math.floor(a.length / 128));
  for (let i = 0; i < a.length; i += step) h = Math.imul(h ^ bits[i], 16777619);
  return h >>> 0;
}

/** Model inputs for `bank` [K*dim]: cached per bank array. */
export function prepareBank(bank: Float32Array, dim = D): PreparedBank {
  const fp = fingerprint(bank);
  const hit = prepared.get(bank);
  if (hit && hit.dim === dim && hit.fp === fp && hit.rows * dim === bank.length) return hit;
  if (bank.length % dim) throw new Error(`bank length ${bank.length} is not a multiple of dim ${dim}`);
  const rows = bank.length / dim;
  const chunks: Chunk[] = [];
  for (let start = 0; start < rows; start += MAX_K) {
    const n = Math.min(MAX_K, rows - start);
    chunks.push(prepareChunk(bank, dim, start, n, bucketFor(n)));
  }
  const prep: PreparedBank = { rows, dim, chunks, fp };
  prepared.set(bank, prep);
  return prep;
}

function withExtra(c: Chunk, extra: Float32Array): Float32Array {
  const n = new Float32Array(c.norm);
  for (let r = 0; r < c.rows; r++) n[r] += extra[c.start + r];
  return n;
}

// --------------------------------------------------------------------------------------------------- search API
let warned = false;

export async function knnMinSq(
  feats: Float32Array,
  bank: Float32Array,
  bankNormExtra?: Float32Array,
  dim = D,
): Promise<Float32Array> {
  if (feats.length % dim || bank.length % dim) throw new Error('knnMinSq: feats / bank length is not a multiple of dim');
  const rows = bank.length / dim;
  if (bankNormExtra && bankNormExtra.length !== rows) {
    throw new Error(`knnMinSq: bankNormExtra has ${bankNormExtra.length} values, bank has ${rows} rows`);
  }
  if (dim === D && rows > 0 && feats.length > 0) {
    try {
      const prep = prepareBank(bank, dim);
      const models = await Promise.all(prep.chunks.map((c) => bucketModel(c.K)));
      if (models.every((m) => m)) return await runModels(feats, prep, models as KnnModel[], bankNormExtra);
    } catch (e) {
      if (!warned) console.warn('[knn] TFLite k-NN failed, using JavaScript:', e instanceof Error ? e.message : e);
      warned = true;
    }
  }
  return knnMinSqJS(feats, bank, bankNormExtra, dim);
}

async function runModels(
  feats: Float32Array,
  prep: PreparedBank,
  models: KnnModel[],
  extra?: Float32Array,
): Promise<Float32Array> {
  const nq = feats.length / D;
  const out = new Float32Array(nq).fill(Infinity);
  const norms = prep.chunks.map((c) => (extra ? withExtra(c, extra) : c.norm));
  for (let q0 = 0; q0 < nq; q0 += P) {
    const n = Math.min(P, nq - q0);
    let fb: ArrayBuffer;
    if (q0 === 0 && n === P && nq === P && feats.byteOffset === 0 && feats.buffer.byteLength === P * D * 4) {
      fb = feats.buffer as ArrayBuffer; // copied into the interpreter by run()
    } else {
      const blk = new Float32Array(P * D);
      blk.set(feats.subarray(q0 * D, (q0 + n) * D));
      fb = blk.buffer;
    }
    for (let c = 0; c < prep.chunks.length; c++) {
      const m = models[c];
      const ins: ArrayBuffer[] = [];
      ins[m.order[0]] = fb;
      ins[m.order[1]] = prep.chunks[c].bankT;
      ins[m.order[2]] = norms[c].buffer as ArrayBuffer;
      const d = new Float32Array((await m.accel.run(ins))[0]);
      for (let i = 0; i < n; i++) if (d[i] < out[q0 + i]) out[q0 + i] = d[i];
    }
  }
  return out;
}

/**
 * For core/patchcore.ts setNNBackend(): per patch of feats (all of them), the squared distance to the nearest bank row
 * whose bankFrame !== skipFrame (-1 = all rows). The prepared bank (bankT / |b|^2) is cached per bank array.
 */
export const tfliteNNBackend: NNBackend = (feats, bank, bankFrame, dim, skipFrame) => {
  if (skipFrame < 0) return knnMinSq(feats, bank, undefined, dim);
  const mask = new Float32Array(bankFrame.length);
  for (let k = 0; k < mask.length; k++) if (bankFrame[k] === skipFrame) mask[k] = KNN_MASK;
  return knnMinSq(feats, bank, mask, dim);
};

/** Exact JavaScript search with partial-distance elimination (the fallback). Same semantics as knnMinSq. */
export function knnMinSqJS(feats: Float32Array, bank: Float32Array, extra: Float32Array | undefined, dim: number): Float32Array {
  const nq = feats.length / dim;
  const kk = bank.length / dim;
  const out = new Float32Array(nq);
  const end8 = dim - (dim % 8);
  for (let p = 0; p < nq; p++) {
    const a = p * dim;
    let best = Infinity;
    for (let k = 0; k < kk; k++) {
      let acc = extra ? extra[k] : 0;
      if (acc >= best) continue;
      const b = k * dim;
      let t = 0;
      while (t < end8) {
        const d0 = feats[a + t] - bank[b + t];
        const d1 = feats[a + t + 1] - bank[b + t + 1];
        const d2 = feats[a + t + 2] - bank[b + t + 2];
        const d3 = feats[a + t + 3] - bank[b + t + 3];
        const d4 = feats[a + t + 4] - bank[b + t + 4];
        const d5 = feats[a + t + 5] - bank[b + t + 5];
        const d6 = feats[a + t + 6] - bank[b + t + 6];
        const d7 = feats[a + t + 7] - bank[b + t + 7];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        t += 8;
        if (acc >= best) break;
      }
      if (acc >= best) continue;
      while (t < dim) {
        const d = feats[a + t] - bank[b + t];
        acc += d * d;
        t++;
      }
      if (acc < best) best = acc;
    }
    out[p] = best;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------- self-test
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Rows that look like backbone features: non-negative, skewed, from 32 clusters. */
function fakeFeatures(rnd: () => number, rows: number, dim: number): Float32Array {
  const centres = new Float32Array(32 * dim);
  for (let i = 0; i < centres.length; i++) centres[i] = 0.35 * rnd() * rnd();
  const x = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r++) {
    const c = Math.floor(rnd() * 32) * dim;
    for (let d = 0; d < dim; d++) x[r * dim + d] = 0.8 * centres[c + d] + 0.2 * 0.35 * rnd() * rnd() * 2;
  }
  return x;
}

function bruteForce(feats: Float32Array, bank: Float32Array, extra: Float32Array | undefined, dim: number, rowsOf?: number[]) {
  const kk = bank.length / dim;
  const idx = rowsOf ?? Array.from({ length: feats.length / dim }, (_, i) => i);
  return idx.map((p) => {
    let best = Infinity;
    for (let k = 0; k < kk; k++) {
      let s = extra ? extra[k] : 0;
      for (let d = 0; d < dim; d++) {
        const v = feats[p * dim + d] - bank[k * dim + d];
        s += v * v;
      }
      if (s < best) best = s;
    }
    return best;
  });
}

function errorsVs(got: Float32Array, ref: number[], rowsOf?: number[]) {
  const vals = ref.map((r, i) => [got[rowsOf ? rowsOf[i] : i], r]);
  const maxRef = Math.max(...ref.filter((r) => r < 1e29));
  let maxRel = 0;
  let maxAbs = 0;
  for (const [g, r] of vals) {
    const e = Math.abs(g - r);
    maxAbs = Math.max(maxAbs, e);
    maxRel = Math.max(maxRel, e / Math.max(Math.abs(r), 1e-3 * maxRef));
  }
  return { maxRel, maxAbsVsMax: maxAbs / maxRef };
}

export interface SelfTestCase {
  name: string;
  maxRelErr: number;
  maxAbsErrVsMax: number;
  tfliteMs: number;
  jsMs: number | null;
  bruteMs: number;
}

/** Compare knnMinSq (TFLite) with brute-force JavaScript on synthetic data and log agreement + timings. */
export async function runSelfTest(): Promise<{ chosen: string; cases: SelfTestCase[]; pass: boolean }> {
  const report = await loadKnn().catch(() => null);
  const chosen = report?.chosen ?? 'none (JavaScript fallback)';
  const rnd = lcg(42);
  const feats = fakeFeatures(rnd, P, D);
  const cases: SelfTestCase[] = [];
  const time = async <T>(f: () => Promise<T> | T, reps = 1): Promise<[T, number]> => {
    let r!: T;
    const ts: number[] = [];
    for (let i = 0; i < reps; i++) {
      const t0 = now();
      r = await f();
      ts.push(now() - t0);
    }
    ts.sort((a, b) => a - b);
    return [r, ts[Math.floor(ts.length / 2)]];
  };
  // Brute force (the reference) is checked on every `sub`-th patch to keep the test short in Hermes.
  const run = async (name: string, kk: number, masked: boolean, sub: number, withJs = true) => {
    const bank = fakeFeatures(rnd, kk, D);
    const frame = new Int32Array(kk).map((_, k) => k % 20);
    let extra: Float32Array | undefined;
    if (masked) {
      extra = new Float32Array(kk);
      for (let k = 0; k < kk; k++) if (frame[k] === 3) extra[k] = KNN_MASK;
    }
    prepareBank(bank); // as the app would after loading the profile
    const [tfl, tfliteMs] = await time(
      () => (masked ? tfliteNNBackend(feats, bank, frame, D, 3) : tfliteNNBackend(feats, bank, frame, D, -1)),
      5,
    );
    const [js, jsMs] = withJs ? await time(() => knnMinSqJS(feats, bank, extra, D)) : [null, null];
    const rowsOf = Array.from({ length: Math.ceil(P / sub) }, (_, i) => i * sub);
    const [ref, bruteMs] = await time(() => bruteForce(feats, bank, extra, D, rowsOf));
    const e = errorsVs(tfl, ref, rowsOf);
    const ej = js ? errorsVs(js, ref, rowsOf) : null;
    cases.push({ name, maxRelErr: e.maxRel, maxAbsErrVsMax: e.maxAbsVsMax, tfliteMs, jsMs, bruteMs });
    console.log(
      `[knn self-test] ${name}: TFLite(${chosen}) ${tfliteMs.toFixed(1)} ms | ` +
        (jsMs != null ? `JS (pruned, all ${P} patches) ${jsMs.toFixed(0)} ms | ` : '') +
        `brute force (1/${sub} of patches) ${bruteMs.toFixed(0)} ms | TFLite vs brute: max rel err ` +
        `${e.maxRel.toExponential(2)}, max abs err ${e.maxAbsVsMax.toExponential(2)} of max` +
        (ej ? ` | JS vs brute: ${ej.maxRel.toExponential(2)}` : ''),
    );
  };
  await run(`P=${P} K=1600 (20 photos, bucket 1600)`, 1600, false, 4);
  await run(`P=${P} K=1600 leave-one-out (skipFrame 3)`, 1600, true, 4);
  await run(`P=${P} K=400 (5 photos, bucket 800)`, 400, false, 4);
  await run(`P=${P} K=2600 (2 chunks: 2400 + 800)`, 2600, false, 8, false);
  const pass = cases.every((c) => c.maxRelErr < 1e-3);
  console.log(`[knn self-test] ${pass ? 'PASS' : 'FAIL'} (threshold: max rel err < 1e-3)`);
  return { chosen, cases, pass };
}
