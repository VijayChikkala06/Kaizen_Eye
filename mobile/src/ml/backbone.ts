/**
 * The converted backbone (tools/convert_backbone.py): pretrained ResNet18 patch-feature extractor.
 *   input  : NHWC float32 [1, 320, 320, 3], RGB 0..255 (normalisation is baked into the graph)
 *   output : NHWC float32 [1, 40, 40, 128]  = 1,600 patch vectors
 * Two files with the same I/O: backbone_r18_320.tflite (float32, modelId 'r18_320_f32') and
 * backbone_r18_320_int8.tflite (int8 weights + activations, float32 I/O, modelId 'r18_320_int8'; onnx2tf -oiqt,
 * per-channel, calibrated on 96 generic images; mean per-patch cosine vs float 0.996 on the demo photos).
 *
 * accelerator.ts picks the delegate and variant for this device (NNAPI int8, NNAPI float, GPU, CPU) - benchmarked on
 * first launch against the float CPU output, cached afterwards. info.modelId says which variant embed() uses: store it
 * with a profile and re-enrol if it changes. loadBackbone() also loads the k-NN scoring model (knn.ts) so that both
 * decisions are made at start-up and reported in `info`.
 */
import { Asset } from 'expo-asset';
import type { TensorflowModel } from 'react-native-fast-tflite';

import meta from '../../assets/models/backbone_r18_320.json';
import {
  type AcceleratedModel,
  type AcceleratorReport,
  describeAccelerator,
  forgetAcceleratorDecisions,
  loadAccelerated,
  now,
} from './accelerator';
import { loadKnn, resetKnn, runSelfTest } from './knn';

export const BACKBONE = {
  name: `${meta.backbone} @ ${meta.input_size}px`,
  size: meta.input_size,
  gh: meta.grid_h,
  gw: meta.grid_w,
  dim: meta.dim,
  pretrained: meta.pretrained,
  /** Stable ids of the two model files (see BackboneInfo.modelId). */
  modelIds: { f32: `r18_${meta.input_size}_f32`, int8: `r18_${meta.input_size}_int8` },
};

export interface BackboneInfo {
  inputShape: number[];
  outputShape: number[];
  loadMs: number;
  /**
   * Model variant embed() uses: 'r18_320_f32' or 'r18_320_int8'. Features of different variants are not comparable:
   * store it with the profile at enrolment and ask for re-enrolment when it differs.
   */
  modelId: string;
  /** Delegate that runs the backbone ('NPU (NNAPI, int8)' | 'NPU (NNAPI)' | 'GPU' | 'CPU') + every candidate's numbers. */
  accelerator: AcceleratorReport;
  /** The same for the k-NN scoring model; null if it could not be loaded (scoring then runs in JavaScript). */
  knn: AcceleratorReport | null;
}

interface Loaded {
  model: TensorflowModel;
  info: BackboneInfo;
  accel: AcceleratedModel;
}
let modelPromise: Promise<Loaded> | null = null;

function checkIO(model: TensorflowModel): void {
  const inputShape = model.inputs[0]?.shape ?? [];
  const outputShape = model.outputs[0]?.shape ?? [];
  const wantIn = [1, BACKBONE.size, BACKBONE.size, 3];
  const wantOut = [1, BACKBONE.gh, BACKBONE.gw, BACKBONE.dim];
  if (inputShape.join() !== wantIn.join() || outputShape.join() !== wantOut.join()) {
    throw new Error(
      `Model I/O mismatch: input [${inputShape}] (want [${wantIn}]), output [${outputShape}] (want [${wantOut}])`,
    );
  }
}

/** Fixed synthetic photo-like input (smooth gradients + texture + noise), for benchmarking and accuracy checks. */
export function benchmarkImage(size = BACKBONE.size): Float32Array {
  const a = new Float32Array(size * size * 3);
  let s = 20240926;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      const n = s / 4294967296 - 0.5;
      const i = (y * size + x) * 3;
      const edge = (x > size * 0.3 && x < size * 0.7 && y > size * 0.35 && y < size * 0.65) ? 60 : 0;
      a[i] = 120 + 70 * Math.sin(x / 11 + y / 29) + edge + 25 * n;
      a[i + 1] = 110 + 60 * Math.cos(x / 23 - y / 7) + edge * 0.5 + 25 * n;
      a[i + 2] = 100 + 50 * Math.sin((x * y) / 1500) - edge * 0.3 + 25 * n;
    }
  }
  for (let i = 0; i < a.length; i++) a[i] = Math.min(255, Math.max(0, a[i]));
  return a;
}

export function loadBackbone(): Promise<{ model: TensorflowModel; info: BackboneInfo }> {
  return load();
}

function load(): Promise<Loaded> {
  if (!modelPromise) {
    modelPromise = (async () => {
      const t0 = Date.now();
      // In release builds a bundled asset resolves to an Android raw-resource name, which fast-tflite's loader
      // (java.net.URL) cannot open. expo-asset copies it to a local file first; in dev it downloads from Metro.
      const [f32, int8] = await Promise.all(
        [require('../../assets/models/backbone_r18_320.tflite'), require('../../assets/models/backbone_r18_320_int8.tflite')].map(
          async (m) => {
            const asset = Asset.fromModule(m);
            await asset.downloadAsync();
            return { url: asset.localUri ?? asset.uri, hash: asset.hash ?? 'nohash' };
          },
        ),
      );
      const accel = await loadAccelerated({
        name: `backbone_r18_${BACKBONE.size}`,
        variants: [
          { modelId: BACKBONE.modelIds.f32, ...f32 },
          { modelId: BACKBONE.modelIds.int8, ...int8, int8: true },
        ],
        inputs: () => [benchmarkImage().buffer as ArrayBuffer],
        check: checkIO,
        vectorLength: BACKBONE.dim,
      });
      // The k-NN model is small; load (and on first launch benchmark) it now, after the backbone, so the
      // measurements do not overlap. Failure is not fatal: scoring falls back to JavaScript.
      const knn = await loadKnn().catch((e) => {
        console.warn('k-NN model unavailable, scoring in JavaScript:', e instanceof Error ? e.message : e);
        return null;
      });
      const model = accel.model;
      const info: BackboneInfo = {
        inputShape: model.inputs[0]?.shape ?? [],
        outputShape: model.outputs[0]?.shape ?? [],
        loadMs: Date.now() - t0,
        modelId: accel.modelId,
        accelerator: accel.report,
        knn,
      };
      // TEMP (emulator proof) - remove before hand-over
      setTimeout(() => {
        runBackboneSelfTest()
          .then(() => runSelfTest())
          .catch((e) => console.warn('[self-test] failed:', e));
      }, 1500);
      return { model, info, accel };
    })();
    modelPromise.catch(() => {
      modelPromise = null; // allow a retry
    });
  }
  return modelPromise;
}

/**
 * Forget the cached delegate decisions and benchmark the backbone and the k-NN model again (e.g. from a settings
 * button after an OS / driver update). Resolves with the new info.
 */
export async function rebenchmarkAccelerators(): Promise<BackboneInfo> {
  const old = modelPromise ? await modelPromise.catch(() => null) : null;
  modelPromise = null;
  forgetAcceleratorDecisions();
  resetKnn();
  const { info } = await load();
  old?.accel.dispose(); // after queued runs
  return info;
}

/** rgb: Float32 [size*size*3] RGB 0..255 -> features Float32 [gh*gw*dim]. */
export async function embed(rgb: Float32Array): Promise<Float32Array> {
  const { accel } = await load();
  const input =
    rgb.byteOffset === 0 && rgb.byteLength === rgb.buffer.byteLength
      ? (rgb.buffer as ArrayBuffer)
      : (rgb.buffer.slice(rgb.byteOffset, rgb.byteOffset + rgb.byteLength) as ArrayBuffer);
  const [out] = await accel.run([input]);
  const feats = new Float32Array(out);
  const want = BACKBONE.gh * BACKBONE.gw * BACKBONE.dim;
  if (feats.length !== want) throw new Error(`Backbone returned ${feats.length} values, expected ${want}`);
  return feats;
}

/** Logs the backbone's accelerator decision and embed() timing on the fixed benchmark image (console.log). */
export async function runBackboneSelfTest(runs = 5): Promise<{ modelId: string; chosen: string; embedMs: number }> {
  const { accel, info } = await load();
  const img = benchmarkImage();
  await embed(img);
  const ts: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = now();
    await embed(img);
    ts.push(now() - t0);
  }
  ts.sort((a, b) => a - b);
  const embedMs = ts[Math.floor(ts.length / 2)];
  console.log(`[backbone self-test] ${describeAccelerator(accel.report)}`);
  console.log(
    `[backbone self-test] embed() median of ${runs}: ${embedMs.toFixed(1)} ms on ${accel.report.chosen} ` +
      `(${accel.modelId}); load ${info.loadMs} ms; k-NN: ${info.knn ? describeAccelerator(info.knn) : 'JavaScript'}`,
  );
  return { modelId: accel.modelId, chosen: accel.report.chosen, embedMs };
}
