/**
 * Kaizen Eye - pick the fastest *accurate* LiteRT delegate (and model variant) for a model on this device, once,
 * and remember it.
 *
 * Candidates (Android), in report order:
 *   'NPU (NNAPI, int8)'  int8 model variant (if the spec has one) on NNAPI
 *   'NPU (NNAPI)'        float model, NNAPI with allow_fp16 (patches/react-native-fast-tflite+3.0.1.patch)
 *   'GPU'                float model, GPU delegate (OpenCL / OpenGL ES), full fp32 precision
 *   'CPU'                float model, XNNPACK on up to 4 threads (the reference and the fallback)
 *   'CPU (int8)'         int8 model on XNNPACK - measured as the baseline NNAPI int8 must beat, so that int8 running
 *                        on the CPU is never reported as "NPU"; selectable only if ALLOW_INT8_ON_CPU is true.
 * The float CPU runs first (accuracy reference). Every candidate is created (failures are caught), run 2 warm-ups +
 * 5 timed runs on the same fixed deterministic input and compared with the float CPU output:
 *   float candidates: all values finite, cosine >= 0.999 and relative L2 error <= 1%;
 *   int8 candidates:  all values finite, mean per-vector (per-patch) cosine >= 0.99.
 * A delegate is used only if it is also at least 10% faster than the CPU running the same model variant - a delegate
 * that silently runs on the CPU (e.g. NNAPI on a phone without an NNAPI accelerator driver) never is. Among the
 * eligible candidates the fastest wins.
 *
 * report.modelId names the chosen model variant (e.g. 'r18_320_f32' / 'r18_320_int8'); features of different
 * variants are not comparable, so store it with a profile and re-enrol when it changes.
 *
 * The decision is cached in <documents>/kaizen_accelerators.json, keyed by model name + model file hashes + app
 * version + Android API level (+ DECISION_VERSION), so the benchmark runs once. forgetAcceleratorDecisions() clears it
 * (backbone.ts rebenchmarkAccelerators()). Crash guard: the candidate being tried is written to that file first; if a
 * delegate takes the app down natively, the next launch records it as crashed and never tries it again.
 *
 * All runs go through runSerial(): fast-tflite's run() is not re-entrant (it copies the inputs into the one
 * interpreter on the JS thread and reuses its output buffers), so overlapping calls on one model would corrupt each
 * other. runSerial queues them per model and returns copies of the outputs.
 */
import { File, Paths } from 'expo-file-system';
import { Platform } from 'react-native';
import { loadTensorflowModel, type TensorflowModel, type TensorflowModelDelegate } from 'react-native-fast-tflite';

import appJson from '../../app.json';

export type AcceleratorName = 'NPU (NNAPI, int8)' | 'NPU (NNAPI)' | 'GPU' | 'CPU' | 'CPU (int8)';

export interface AcceleratorResult {
  name: AcceleratorName;
  /** Model variant this candidate runs. */
  modelId: string;
  /** Created, ran, and matched the float CPU output (per the gate for its variant). */
  ok: boolean;
  /** Median time of the timed runs (ms); null if it did not run. */
  ms: number | null;
  /** Fastest timed run (ms) - used for the "at least 10% faster than the CPU" rule (noise only adds time). */
  minMs: number | null;
  /** Cosine similarity of the whole output with the float CPU output (1 = identical). */
  cosine: number | null;
  /** Mean cosine per output vector (per patch). */
  patchCosine: number | null;
  /** ||out - cpu|| / ||cpu|| (L2). */
  relErr: number | null;
  /** Interpreter creation incl. delegate compilation (ms). */
  loadMs: number | null;
  /** Why it failed or was rejected. */
  error?: string;
  /** Why a working candidate was not chosen, or other information. */
  note?: string;
}

export interface AcceleratorReport {
  /** Model name, e.g. 'backbone_r18_320'. */
  model: string;
  /** The delegate the app uses for this model. */
  chosen: AcceleratorName;
  /** Model variant in use - store it with profiles (features of different variants are not comparable). */
  modelId: string;
  /** One entry per candidate. */
  results: AcceleratorResult[];
  /** true when the decision came from the cache (the results are those of the benchmark that made it). */
  fromCache: boolean;
  /** ISO time of that benchmark. */
  measuredAt: string;
  note?: string;
}

/** Accuracy gates vs the float CPU output. */
export const ACCURACY = { minCosine: 0.999, maxRelErr: 0.01, int8MinPatchCosine: 0.99 };
/** A delegate is used only if it needs at most this fraction of the CPU time for the same model variant. */
export const MAX_TIME_VS_CPU = 0.9;
/** int8 on the CPU is measured (baseline for NNAPI int8) but only selectable if this is true. */
export const ALLOW_INT8_ON_CPU = false;
const WARMUP_RUNS = 2;
const TIMED_RUNS = 5;
/** Bump when the delegate options (the fast-tflite patch) or the selection rules change: invalidates the cache. */
const DECISION_VERSION = 2;
const CACHE_VERSION = 1;

interface Candidate {
  name: AcceleratorName;
  delegates: TensorflowModelDelegate[];
  int8: boolean;
}
const ALL_CANDIDATES: Candidate[] = [
  { name: 'NPU (NNAPI, int8)', delegates: ['nnapi'], int8: true },
  { name: 'NPU (NNAPI)', delegates: ['nnapi'], int8: false },
  { name: 'GPU', delegates: ['android-gpu'], int8: false },
  { name: 'CPU', delegates: [], int8: false },
  { name: 'CPU (int8)', delegates: [], int8: true },
];
const candidateByName = (n: AcceleratorName) => ALL_CANDIDATES.find((c) => c.name === n)!;

export const now = (): number => (globalThis.performance?.now ? globalThis.performance.now() : Date.now());
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const r4 = (x: number) => Math.round(x * 10000) / 10000;

// --------------------------------------------------------------------------------------------- serial model runs
const queues = new WeakMap<object, Promise<unknown>>();

/** model.run(), one call at a time per model; resolves to copies of the output buffers. */
export function runSerial(model: TensorflowModel, inputs: ArrayBuffer[]): Promise<ArrayBuffer[]> {
  const prev = queues.get(model) ?? Promise.resolve();
  const next = prev.then(async () => {
    const outs = await model.run(inputs);
    return outs.map((b) => b.slice(0));
  });
  queues.set(
    model,
    next.catch(() => undefined),
  );
  return next;
}

const BYTES: Record<string, number> = { float32: 4, int32: 4, uint32: 4, float16: 2, int16: 2, uint16: 2, int8: 1, uint8: 1, bool: 1, int64: 8, uint64: 8, float64: 8 };
const inputSizes = new WeakMap<object, number[]>();

/** fast-tflite does not check input sizes (a wrong size silently leaves the tensor unchanged), so check here. */
export function checkInputSizes(model: TensorflowModel, inputs: ArrayBuffer[]): void {
  let sizes = inputSizes.get(model);
  if (!sizes) {
    sizes = model.inputs.map((t) => t.shape.reduce((a, b) => a * b, 1) * (BYTES[t.dataType] ?? 4));
    inputSizes.set(model, sizes);
  }
  if (inputs.length !== sizes.length) throw new Error(`Model expects ${sizes.length} inputs, got ${inputs.length}`);
  inputs.forEach((b, i) => {
    if (b.byteLength !== sizes[i]) throw new Error(`Input ${i} is ${b.byteLength} bytes, the model expects ${sizes[i]}`);
  });
}

/** Dispose a model once its queued runs have finished. */
function disposeWhenIdle(model: TensorflowModel): void {
  (queues.get(model) ?? Promise.resolve()).then(() => {
    try {
      model.dispose();
    } catch {
      // already disposed
    }
  });
}

// ------------------------------------------------------------------------------------------------ decision cache
interface Decision {
  chosen: AcceleratorName;
  modelId: string;
  results: AcceleratorResult[];
  measuredAt: string;
}
interface CacheFile {
  version: number;
  decisions: Record<string, Decision>;
  /** Candidate being tried right now (crash guard). */
  trying?: { key: string; name: AcceleratorName } | null;
  /** Candidates that took the app down, per decision key. */
  crashed?: Record<string, AcceleratorName[]>;
}

const cacheFile = () => new File(Paths.document, 'kaizen_accelerators.json');
let crashChecked = false;

function readCache(): CacheFile {
  let c: CacheFile = { version: CACHE_VERSION, decisions: {} };
  try {
    const f = cacheFile();
    if (f.exists) {
      const parsed = JSON.parse(f.textSync()) as CacheFile;
      if (parsed?.version === CACHE_VERSION && parsed.decisions) c = parsed;
    }
  } catch (e) {
    console.warn('[accelerator] unreadable cache, starting fresh:', errMsg(e));
  }
  if (!crashChecked) {
    crashChecked = true;
    if (c.trying) {
      // The app died while this candidate was being created / run: never try it again for that model.
      const { key, name } = c.trying;
      console.warn(`[accelerator] ${name} crashed the app last time (${key}); it will be skipped`);
      c.crashed = { ...(c.crashed ?? {}), [key]: [...new Set([...(c.crashed?.[key] ?? []), name])] };
      delete c.decisions[key];
      c.trying = null;
      writeCache(c);
    }
  }
  return c;
}

function writeCache(c: CacheFile): void {
  try {
    const f = cacheFile();
    if (f.exists) f.delete();
    f.create();
    f.write(JSON.stringify(c));
  } catch (e) {
    console.warn('[accelerator] could not write cache:', errMsg(e));
  }
}

function updateCache(fn: (c: CacheFile) => void): void {
  const c = readCache();
  fn(c);
  writeCache(c);
}

const setTrying = (key: string, name: AcceleratorName) => updateCache((c) => (c.trying = { key, name }));
const clearTrying = () => updateCache((c) => (c.trying = null));

/** Forget the benchmark decisions (the crash list is kept) so the next load benchmarks again. */
export function forgetAcceleratorDecisions(): void {
  updateCache((c) => {
    c.decisions = {};
    c.trying = null;
  });
}

// --------------------------------------------------------------------------------------------------- measuring
export interface OutputComparison {
  cosine: number;
  patchCosine: number;
  relErr: number;
  finite: boolean;
}

/** Compare with the reference; `vec` = length of one feature vector for the per-vector cosine (0 = whole output). */
export function compareOutputs(out: Float32Array, ref: Float32Array, vec = 0): OutputComparison {
  if (out.length !== ref.length) return { cosine: 0, patchCosine: 0, relErr: Infinity, finite: false };
  let dot = 0;
  let no = 0;
  let nr = 0;
  let diff = 0;
  let finite = true;
  for (let i = 0; i < out.length; i++) {
    const a = out[i];
    const b = ref[i];
    if (!Number.isFinite(a)) finite = false;
    dot += a * b;
    no += a * a;
    nr += b * b;
    diff += (a - b) * (a - b);
  }
  const cos = (d: number, x: number, y: number) => (x === 0 && y === 0 ? 1 : d / Math.sqrt(x * y || 1e-300));
  const cosine = cos(dot, no, nr);
  let patchCosine = cosine;
  if (vec > 0 && out.length % vec === 0) {
    let sum = 0;
    const n = out.length / vec;
    for (let p = 0; p < n; p++) {
      let d = 0;
      let x = 0;
      let y = 0;
      for (let t = p * vec; t < (p + 1) * vec; t++) {
        d += out[t] * ref[t];
        x += out[t] * out[t];
        y += ref[t] * ref[t];
      }
      sum += cos(d, x, y);
    }
    patchCosine = sum / n;
  }
  const relErr = nr === 0 ? Math.sqrt(diff) : Math.sqrt(diff / nr);
  return {
    cosine,
    patchCosine,
    relErr,
    finite: finite && Number.isFinite(cosine) && Number.isFinite(patchCosine) && Number.isFinite(relErr),
  };
}

function accurate(c: OutputComparison, int8: boolean): boolean {
  if (!c.finite) return false;
  return int8
    ? c.patchCosine >= ACCURACY.int8MinPatchCosine
    : c.cosine >= ACCURACY.minCosine && c.relErr <= ACCURACY.maxRelErr;
}

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

async function measure(model: TensorflowModel, inputs: ArrayBuffer[]): Promise<{ ms: number; minMs: number; out: Float32Array }> {
  checkInputSizes(model, inputs);
  let out: ArrayBuffer[] = [];
  for (let i = 0; i < WARMUP_RUNS; i++) out = await runSerial(model, inputs);
  const times: number[] = [];
  for (let i = 0; i < TIMED_RUNS; i++) {
    const t0 = now();
    out = await runSerial(model, inputs);
    times.push(now() - t0);
  }
  return { ms: median(times), minMs: Math.min(...times), out: new Float32Array(out[0]) };
}

// ------------------------------------------------------------------------------------------------------ loading
export interface ModelVariant {
  /** Stable id of this model file, e.g. 'r18_320_f32' (stored with profiles). */
  modelId: string;
  /** Local file uri of the .tflite (expo-asset localUri; release builds cannot load raw resources). */
  url: string;
  /** Content hash of the file (expo-asset `hash`); part of the cache key. */
  hash: string;
  /** Quantised variant: int8 accuracy gate, int8 candidates. */
  int8?: boolean;
}

export interface AcceleratedModelSpec {
  /** Short model name for the report and the cache key, e.g. 'backbone_r18_320'. */
  name: string;
  /** Model files with identical inputs / outputs. The first non-int8 one is the float reference. */
  variants: ModelVariant[];
  /** Deterministic benchmark inputs, in the model's input order, each exactly the tensor's byte size. */
  inputs: (model: TensorflowModel) => ArrayBuffer[];
  /** Throws if the model's inputs / outputs are not what the app expects. */
  check?: (model: TensorflowModel) => void;
  /** Length of one output feature vector (per-patch cosine for the int8 gate); default: the whole output. */
  vectorLength?: number;
}

const floatVariant = (spec: AcceleratedModelSpec) => spec.variants.find((v) => !v.int8) ?? spec.variants[0];
const variantFor = (spec: AcceleratedModelSpec, c: Candidate) =>
  (c.int8 ? spec.variants.find((v) => v.int8) : undefined) ?? floatVariant(spec);

function candidatesFor(spec: AcceleratedModelSpec): Candidate[] {
  const hasInt8 = spec.variants.some((v) => v.int8);
  return ALL_CANDIDATES.filter((c) => (Platform.OS === 'android' || c.delegates.length === 0) && (hasInt8 || !c.int8));
}

/** A loaded model plus the decision. run() is serialised and falls back to the CPU if the accelerator fails. */
export class AcceleratedModel {
  private switching: Promise<void> | null = null;

  constructor(
    readonly spec: AcceleratedModelSpec,
    private readonly key: string,
    public model: TensorflowModel,
    public delegates: TensorflowModelDelegate[],
    public variant: ModelVariant,
    public report: AcceleratorReport,
  ) {}

  get modelId(): string {
    return this.variant.modelId;
  }

  async run(inputs: ArrayBuffer[]): Promise<ArrayBuffer[]> {
    checkInputSizes(this.model, inputs);
    const m = this.model;
    try {
      return await runSerial(m, inputs);
    } catch (e) {
      if (m === this.model && this.delegates.length === 0) throw e;
      if (m === this.model) this.switching ??= this.switchToCpu(m, e);
      await this.switching;
      return runSerial(this.model, inputs);
    }
  }

  /**
   * An accelerator that passed the benchmark failed at run time (e.g. GPU context lost): use the CPU with the SAME
   * model variant for good, so modelId (and the stored profiles) stay valid.
   */
  private async switchToCpu(failed: TensorflowModel, e: unknown): Promise<void> {
    const was = this.report.chosen;
    console.warn(`[accelerator] ${was} failed at run time for ${this.spec.name}; switching to CPU:`, errMsg(e));
    const cpu = await create(this.spec, this.variant, []);
    this.model = cpu;
    this.delegates = [];
    const chosen: AcceleratorName = this.variant.int8 ? 'CPU (int8)' : 'CPU';
    this.report = {
      ...this.report,
      chosen,
      results: this.report.results.map((r) =>
        r.name === was ? { ...r, ok: false, error: `failed at run time: ${errMsg(e)}` } : r,
      ),
    };
    const rep = this.report;
    updateCache((c) => (c.decisions[this.key] = { chosen, modelId: rep.modelId, results: rep.results, measuredAt: rep.measuredAt }));
    disposeWhenIdle(failed);
  }

  /** Free the interpreter (after queued runs). The object must not be used afterwards. */
  dispose(): void {
    disposeWhenIdle(this.model);
  }
}

function decisionKey(spec: AcceleratedModelSpec): string {
  const files = spec.variants.map((v) => `${v.modelId}:${v.hash}`).join('+');
  return `${spec.name}#${files}@${appJson.expo.version}/${Platform.OS}${Platform.Version}/d${DECISION_VERSION}`;
}

async function create(spec: AcceleratedModelSpec, v: ModelVariant, delegates: TensorflowModelDelegate[]): Promise<TensorflowModel> {
  const model = await loadTensorflowModel({ url: v.url }, delegates);
  try {
    spec.check?.(model);
  } catch (e) {
    model.dispose();
    throw e;
  }
  return model;
}

/**
 * Load `spec` with the best delegate / variant for this device: the cached decision if there is one, otherwise
 * benchmark. force = ignore the cache. The model is warmed up (one run) before it is returned.
 */
export async function loadAccelerated(spec: AcceleratedModelSpec, force = false): Promise<AcceleratedModel> {
  const key = decisionKey(spec);
  const cache = readCache();
  const cached = force ? undefined : cache.decisions[key];
  const cands = candidatesFor(spec);
  if (cached && cands.some((c) => c.name === cached.chosen)) {
    const cand = candidateByName(cached.chosen);
    const v = variantFor(spec, cand);
    try {
      if (cand.delegates.length) setTrying(key, cand.name);
      const model = await create(spec, v, cand.delegates);
      await runSerial(model, spec.inputs(model)); // warm-up (and crash-guarded first run)
      if (cand.delegates.length) clearTrying();
      const report: AcceleratorReport = {
        model: spec.name,
        chosen: cand.name,
        modelId: v.modelId,
        results: cached.results,
        fromCache: true,
        measuredAt: cached.measuredAt,
      };
      return new AcceleratedModel(spec, key, model, cand.delegates, v, report);
    } catch (e) {
      if (cand.delegates.length) clearTrying();
      console.warn(`[accelerator] cached choice ${cand.name} for ${spec.name} no longer loads, re-benchmarking:`, errMsg(e));
    }
  }
  return benchmark(spec, key, cands, new Set(cache.crashed?.[key] ?? []));
}

async function benchmark(
  spec: AcceleratedModelSpec,
  key: string,
  cands: Candidate[],
  crashed: Set<AcceleratorName>,
): Promise<AcceleratedModel> {
  const results: AcceleratorResult[] = cands.map((c) => ({
    name: c.name,
    modelId: variantFor(spec, c).modelId,
    ok: false,
    ms: null,
    minMs: null,
    cosine: null,
    patchCosine: null,
    relErr: null,
    loadMs: null,
  }));
  const res = (n: AcceleratorName) => results.find((r) => r.name === n)!;
  const vec = spec.vectorLength ?? 0;

  // Float CPU first: accuracy reference and fallback. If even that cannot run the model, loading fails.
  const fv = floatVariant(spec);
  let t0 = now();
  const cpuModel = await create(spec, fv, []);
  const inputs = spec.inputs(cpuModel);
  const cpu = res('CPU');
  cpu.loadMs = now() - t0;
  const cpuRun = await measure(cpuModel, inputs);
  Object.assign(cpu, { ok: true, ms: cpuRun.ms, minMs: cpuRun.minMs, cosine: 1, patchCosine: 1, relErr: 0 });

  // Then int8 on the CPU (baseline for NNAPI int8), then the delegates.
  const order = [...cands.filter((c) => !c.delegates.length && c.int8), ...cands.filter((c) => c.delegates.length)];
  let best: { cand: Candidate; model: TensorflowModel; ms: number } = { cand: candidateByName('CPU'), model: cpuModel, ms: cpuRun.ms };
  for (const cand of order) {
    const r = res(cand.name);
    if (crashed.has(cand.name)) {
      r.error = 'crashed the app during an earlier attempt - skipped';
      continue;
    }
    let model: TensorflowModel | null = null;
    if (cand.delegates.length) setTrying(key, cand.name);
    try {
      t0 = now();
      model = await create(spec, variantFor(spec, cand), cand.delegates);
      r.loadMs = now() - t0;
      const run = await measure(model, inputs);
      const cmp = compareOutputs(run.out, cpuRun.out, vec);
      Object.assign(r, { ms: run.ms, minMs: run.minMs, cosine: cmp.cosine, patchCosine: cmp.patchCosine, relErr: cmp.relErr });
      r.ok = accurate(cmp, cand.int8);
      // A delegate must beat the CPU running the same variant (int8 CPU time for int8, else float CPU time).
      const cpuInt8 = results.find((x) => x.name === 'CPU (int8)');
      const base = cand.int8 ? (cpuInt8?.minMs != null && !cpuInt8.error ? cpuInt8.minMs : null) : cpuRun.minMs;
      if (!r.ok) {
        r.error = cand.int8
          ? `too far from float CPU (mean patch cosine ${r4(cmp.patchCosine)} < ${ACCURACY.int8MinPatchCosine})`
          : `output differs from CPU (cosine ${r4(cmp.cosine)}, rel. error ${cmp.relErr.toExponential(1)})`;
      } else if (cand.delegates.length && base == null) {
        r.note = 'CPU (int8) baseline unavailable, so acceleration cannot be verified - not used';
      } else if (cand.delegates.length && base != null && run.minMs > MAX_TIME_VS_CPU * base) {
        r.note = `no speed-up over ${cand.int8 ? 'CPU (int8)' : 'CPU'} (best run ${run.minMs.toFixed(1)} vs ${base.toFixed(1)} ms) - no accelerator driver, or ops not supported`;
      } else if (!cand.delegates.length && cand.int8 && !ALLOW_INT8_ON_CPU) {
        r.note = 'baseline for NNAPI int8 (int8 on CPU is not selectable)';
      } else if (run.ms < best.ms) {
        if (best.model !== cpuModel) {
          res(best.cand.name).note = `slower than ${cand.name}`;
          disposeWhenIdle(best.model);
        }
        best = { cand, model, ms: run.ms };
        model = null; // keep it
      } else {
        r.note = `slower than ${best.cand.name}`;
      }
    } catch (e) {
      r.error = errMsg(e);
    } finally {
      if (cand.delegates.length) clearTrying();
      if (model) disposeWhenIdle(model);
    }
  }
  if (best.model !== cpuModel) {
    cpu.note = `${best.cand.name} is faster`;
    disposeWhenIdle(cpuModel);
  }

  const v = variantFor(spec, best.cand);
  const report: AcceleratorReport = {
    model: spec.name,
    chosen: best.cand.name,
    modelId: v.modelId,
    results,
    fromCache: false,
    measuredAt: new Date().toISOString(),
  };
  updateCache((c) => (c.decisions[key] = { chosen: report.chosen, modelId: report.modelId, results, measuredAt: report.measuredAt }));
  console.log(`[accelerator] ${spec.name}: ${describeAccelerator(report)}`);
  return new AcceleratedModel(spec, key, best.model, best.cand.delegates, v, report);
}

/**
 * Load `spec` (float variant) with the delegate another model already chose (same graph, other tensor sizes - e.g.
 * the k-NN buckets). A non-CPU delegate is re-checked against the CPU on the spec's inputs (one run each) first.
 */
export async function loadWithDecisionOf(spec: AcceleratedModelSpec, source: AcceleratedModel): Promise<AcceleratedModel> {
  const key = decisionKey(spec);
  const fv = floatVariant(spec);
  const base: AcceleratorReport = {
    ...source.report,
    model: spec.name,
    modelId: fv.modelId,
    note: `delegate chosen by ${source.report.model}`,
  };
  if (source.delegates.length && !source.variant.int8) {
    const name = source.report.chosen;
    let accel: TensorflowModel | null = null;
    let cpuModel: TensorflowModel | null = null;
    let good = false;
    setTrying(key, name);
    try {
      accel = await create(spec, fv, source.delegates);
      const inputs = spec.inputs(accel);
      const a = new Float32Array((await runSerial(accel, inputs))[0]);
      cpuModel = await create(spec, fv, []);
      const b = new Float32Array((await runSerial(cpuModel, inputs))[0]);
      good = accurate(compareOutputs(a, b, spec.vectorLength ?? 0), false);
      if (!good) console.warn(`[accelerator] ${name} output differs from CPU for ${spec.name}; using CPU`);
    } catch (e) {
      console.warn(`[accelerator] ${name} failed for ${spec.name}; using CPU:`, errMsg(e));
    } finally {
      clearTrying();
    }
    if (good && accel) {
      if (cpuModel) disposeWhenIdle(cpuModel);
      return new AcceleratedModel(spec, key, accel, source.delegates, fv, base);
    }
    if (accel) disposeWhenIdle(accel);
    const model = cpuModel ?? (await create(spec, fv, []));
    return new AcceleratedModel(spec, key, model, [], fv, { ...base, chosen: 'CPU', note: `${name} failed for this model` });
  }
  const model = await create(spec, fv, []);
  await runSerial(model, spec.inputs(model)); // warm-up
  return new AcceleratedModel(spec, key, model, [], fv, { ...base, chosen: 'CPU' });
}

/** One line for logs / UI, e.g. "GPU 9.1 ms (chosen) · NPU (NNAPI): 38.0 ms, no speed-up over CPU ... · CPU 37.5 ms". */
export function describeAccelerator(r: AcceleratorReport): string {
  const parts = r.results.map((x) => {
    const t = x.ms != null ? `${x.ms.toFixed(1)} ms` : '';
    const q = x.patchCosine != null && x.name.includes('int8') ? `, patch cos ${x.patchCosine.toFixed(4)}` : '';
    if (x.name === r.chosen) return `${x.name} ${t}${q} (chosen)`;
    return `${x.name}: ${x.error ? `failed (${x.error})` : `${t}${q}${x.note ? `, ${x.note}` : ''}`}`;
  });
  return `[${r.modelId}] ` + parts.join(' · ') + (r.fromCache ? ' [cached]' : '');
}
