/**
 * The converted backbone (tools/convert_backbone.py): pretrained ResNet18 patch-feature extractor.
 *   input  : NHWC float32 [1, 256, 256, 3], RGB 0..255 (normalisation is baked into the graph)
 *   output : NHWC float32 [1, 32, 32, 128]  = 1,024 patch vectors
 */
import { Asset } from 'expo-asset';
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite';

import meta from '../../assets/models/backbone_r18_256.json';

export const BACKBONE = {
  name: `${meta.backbone} @ ${meta.input_size}px`,
  size: meta.input_size,
  gh: meta.grid_h,
  gw: meta.grid_w,
  dim: meta.dim,
  pretrained: meta.pretrained,
};

export interface BackboneInfo {
  inputShape: number[];
  outputShape: number[];
  loadMs: number;
}

let modelPromise: Promise<{ model: TensorflowModel; info: BackboneInfo }> | null = null;

export function loadBackbone() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const t0 = Date.now();
      // In release builds a bundled asset resolves to an Android raw-resource name, which fast-tflite's loader
      // (java.net.URL) cannot open. expo-asset copies it to a local file first; in dev it downloads from Metro.
      const asset = Asset.fromModule(require('../../assets/models/backbone_r18_256.tflite'));
      await asset.downloadAsync();
      const url = asset.localUri ?? asset.uri;
      const model = await loadTensorflowModel({ url }, []);
      const inputShape = model.inputs[0]?.shape ?? [];
      const outputShape = model.outputs[0]?.shape ?? [];
      const wantIn = [1, BACKBONE.size, BACKBONE.size, 3];
      const wantOut = [1, BACKBONE.gh, BACKBONE.gw, BACKBONE.dim];
      if (inputShape.join() !== wantIn.join() || outputShape.join() !== wantOut.join()) {
        throw new Error(
          `Model I/O mismatch: input [${inputShape}] (want [${wantIn}]), output [${outputShape}] (want [${wantOut}])`,
        );
      }
      return { model, info: { inputShape, outputShape, loadMs: Date.now() - t0 } };
    })();
    modelPromise.catch(() => {
      modelPromise = null; // allow a retry
    });
  }
  return modelPromise;
}

/** rgb: Float32 [size*size*3] RGB 0..255 -> features Float32 [gh*gw*dim]. */
export async function embed(rgb: Float32Array): Promise<Float32Array> {
  const { model } = await loadBackbone();
  const input = rgb.buffer.slice(rgb.byteOffset, rgb.byteOffset + rgb.byteLength) as ArrayBuffer;
  const [out] = await model.run([input]);
  const feats = new Float32Array(out.slice(0));
  const want = BACKBONE.gh * BACKBONE.gw * BACKBONE.dim;
  if (feats.length !== want) throw new Error(`Backbone returned ${feats.length} values, expected ${want}`);
  return feats;
}
