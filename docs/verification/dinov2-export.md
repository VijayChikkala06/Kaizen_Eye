# Kaizen Eye 2 — laptop model export (ResNet18 ONNX, DINOv2-S/14 @448) and offline A/B tooling

Helper H2b, 2026-09-27. Everything here was produced on the Windows laptop (no phone). Artifacts live in
`tools/dinov2/out/` (git-ignored by the root `out/` rule; hashes in `tools/dinov2/out/SHA256SUMS.txt`), logs in
`tools/dinov2/logs/`, A/B reports in `tools/dinov2/eval/`. Intermediates (raw ONNX, TF SavedModel, onnx2tf log, torch
reference outputs) live outside the repo in `D:\kz-tmp\h2b\dinov2-work\` (override with `KZ_DINOV2_WORK`); they are
not needed at run time. The frozen `mobile/assets/models/` was only read, never written.

## 1. ResNet18 @320 → ONNX (ONNX-Runtime-QNN fallback arm)

Command (under the machine-wide lock, `.venv-convert`):
`tools/convert_backbone.py --backend onnx2tf --backbone resnet18 --size 320 --dim 128 --onnx --out tools/dinov2/out/backbone_r18_320.tflite`
(ImageNet weights from the local torch cache `resnet18-f37072fd.pth`; nothing downloaded.)

| file | bytes | SHA-256 |
|---|---|---|
| `backbone_r18_320.onnx` (opset 17, static `[1,320,320,3]` → `features [1,40,40,128]`) | 11,329,700 | `78f9fd22481ff999d65dc603263180f62116762b083e38e187551cc506200fd3` |
| `backbone_r18_320_sim.onnx` (same, onnxsim-folded: no `Shape`/`Slice` before the `Resize`) | 11,332,422 | `a5be790fdff191f2e7de1f57d93bd91bbfb4f57fcce0bf2f82c94573d8bdfd36` |
| `backbone_r18_320.tflite` (rebuilt by the same run) | 11,335,136 | `e881924d3498291f33a7c5e21c01e4f4489d334a0c5c19151c354c38ca2619ba` |

- The rebuilt TFLite is **byte-identical** to the shipped `mobile/assets/models/backbone_r18_320.tflite`
  (same SHA-256 `e881924d…`), so TFLite-vs-shipped is exact (cosine 1, max |diff| 0).
- ONNX (onnxruntime 1.30 CPU) vs the shipped TFLite (ai_edge_litert 2.2.0), `tools/dinov2/check_r18_onnx.py`:
  random uint8 (seed 0) cosine 1.00000000, max |diff| 6.7e-7 (rel 9.8e-7); demo photos `good_00`/`good_01`
  cosine 1.00000000, max |diff| 2.7e-6 / 3.5e-6 (rel 2.9e-6 / 3.9e-6); mean per-patch cosine 1.00000000.
  Same numbers for `_sim.onnx` (`out/r18_parity.json`, `out/r18_sim_parity.json`).
- ONNX ops: Conv 16, Relu 13, Add 6, AveragePool 2, Transpose 2, Concat 2, MaxPool, Sub, Div, Resize (+ Shape,
  Slice, Constant in the unsimplified file). **Use `_sim.onnx` for QNN** (fully static graph).
- Laptop CPU (4 threads, median of 10, busy machine): ORT ≈ 13–42 ms, TFLite ≈ 27–79 ms.

## 2. DINOv2-S/14 at a fixed 448×448 input

### 2.1 Weights (approved download)
`facebook/dinov2-small` (Apache-2.0) from `https://huggingface.co/facebook/dinov2-small/resolve/main/<file>`,
stored outside the repo in `D:\kz-tmp\dinov2-weights\`:

| file | bytes | SHA-256 |
|---|---|---|
| `model.safetensors` | 88,249,960 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| `config.json` | 547 | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |

(config: hidden 384, 12 layers, 6 heads, patch 14, image 518, LayerNorm eps 1e-6, layerscale, GELU, no SwiGLU,
no register tokens; 223 tensors.)

### 2.2 Model code — `tools/dinov2/dinov2_s14.py`
Self-contained PyTorch module, no transformers/timm/hub. Parses the safetensors format itself (8-byte LE header
length + JSON header + raw LE float32 tensors). Structure = facebookresearch/dinov2 ViT-S/14:
`(x − 255·mean)/(255·std)` (ImageNet, baked in) → Conv 14/14 → `[CLS+pos₀] ++ (tokens + pos₁…)` → 12 pre-norm blocks
(`x += ls1·attn(norm1 x)`, `x += ls2·mlp(norm2 x)`, 6 heads × 64, exact-erf GELU) → final LayerNorm.
- **Position embeddings** are resized once at export from the 37×37 pretrained grid to 32×32 exactly like the
  reference `interpolate_pos_encoding` (float32, `F.interpolate(mode="bicubic", antialias=False,
  scale_factor=(32+0.1)/37)` — the reference's `interpolate_offset = 0.1`) and stored as a constant
  (`cls_pos = cls_token + pos₀`, `patch_pos [1,1024,384]`).
- Export-friendly rewrites: attention scale 0.125 folded into the query weights/bias (a power of two → bit-exact),
  LayerScale folded into `attn.proj`/`mlp.fc2` (fp32-rounding-level change), separate q/k/v Linear layers (all
  tensors ≤ 4-D), CLS taken with a Slice (no Gather).
- Contract: input `image` NHWC float32 `[1,448,448,3]` RGB 0..255; outputs `patch` NHWC `[1,32,32,384]`
  (final-norm patch tokens, **not** L2-normalised — the app normalises per patch, spec §5) and `cls` `[1,384]`.

### 2.3 Independent verification — `tools/dinov2/verify_reference.py` (`out/dinov2_reference_check.json`)
A second implementation in float64 numpy, written differently: its own safetensors reader, its own bicubic resize
(PyTorch cubic-convolution kernel A = −0.75, as two separable weight matrices), im2col patch embedding (no conv),
fused qkv (facebookresearch layout) with the scale applied after the projection, stable softmax, scipy `erf` GELU,
explicit LayerNorm, LayerScale **not** folded.

| check | result |
|---|---|
| bicubic resize is the identity at 37×37 | max |diff| 0 |
| pos-embed resize to 32×32: torch `F.interpolate` vs numpy | max |diff| 1.1e-7 (values up to 0.16) |
| torch module vs numpy @448, random u8 | mean/min per-patch cos 1.000000000 / 1.000000000, CLS cos 1.000000000, max |diff| 4.3e-5 (rel 1.7e-6) |
| torch module vs numpy @448, demo photo | per-patch cos min 0.999999999, CLS 1.000000000, max |diff| 4.4e-4 (rel 1.6e-5) |
| same @518 (no interpolation path), both inputs | per-patch cos min ≥ 0.999999998, CLS 1.000000000 |

Caveat: both implementations encode the same reading of the reference (e.g. `interpolate_offset = 0.1`, no
antialias); a reference-output fixture from the official repo was not available offline (torch.hub not allowed).

### 2.4 Export — `tools/dinov2/export_dinov2.py onnx | tflite | check`
- **ONNX** opset 17, static shapes, TorchScript exporter, onnxsim-simplified (onnxsim merges q/k/v into one MatMul
  + Split). onnxruntime 1.23 / 1.30 vs torch: max |diff| ≤ 6.8e-4 (rel ≤ 2.6e-5), per-patch/CLS cos 1.000000.
- **TFLite** via onnx2tf 1.29.24 / TF 2.19.1 (`.venv-convert`, `-kat image`), **builtin ops only**. onnx2tf wrote
  both `*_float32.tflite` and a float16-weight `*_float16.tflite`.
- **onnx2tf GELU pitfall (fixed):** the first conversion contained 12 `FlexErf` ops (TF Select — not runnable in
  LiteRT). onnx2tf's GeLU fusion (`ops/Div.py`) tests the divisor with `== 1.4142135`, which with numpy 1.26 is
  False for the exporter's 0-d float32 √2 (compared in float64) and True for a 1-D one. `fix_gelu_constants()`
  reshapes that initializer to `[1]` before saving the ONNX (same values). Result: one builtin `GELU` per block.

| file | bytes | SHA-256 |
|---|---|---|
| `dinov2_s14_448.onnx` | 87,728,589 | `5421d92063923b39a7581435db2f81402bcfa2ced2c231ff44e465ea7428d9d4` |
| `dinov2_s14_448_fp32.tflite` | 87,762,856 | `6c44ca64acd23b38bc4e8f813f8792e1baf8c44c1ae2bf825ca15890481dc290` |
| `dinov2_s14_448_fp16w.tflite` | 43,954,424 | `58c86e0b5e1c6ccdaee4fb329b1d6e0e1e8b04577d53c2afd9f9f18d504ec223` |
| `knn_p1024_d384_k2400.tflite` | 1,768 | `41ffa92cd06e3b296048cfb866c358c535d735253795a4bfea2a5afc01159fb8` |
| `dinov2_s14_448.json` (model card) | 4,259 | `5cd391fc30c477b9851ceeee169f79c6374070df60df8fb70c08e4adc2814c73` |
| `dinov2_parity_input_448.rgb` + `_patch.f32` + `_cls.f32` + `_fixture.json` (on-device parity fixture) | 2,176,512 (data) | see `SHA256SUMS.txt` |

**On-device parity fixture** (`tools/dinov2/write_parity_fixture.py`): demo photo `good_00` as raw uint8
`[448,448,3]` plus the PyTorch `patch` `[32,32,384]` and `cls` `[384]` as raw little-endian float32, so the app can
check its LiteRT output against PyTorch (spec §13: cosine ≥ 0.999, rel. L2 ≤ 1 % for fp32 candidates). Laptop fp32
TFLite vs the fixture: cosine 1.00000000, rel. L2 1.3e-5.

**Parity vs torch** (`out/dinov2_parity.json`; 4 fixed inputs: random u8 seed 0, demo photos good_00/good_01, flat
grey; thresholds mean per-patch cosine ≥ 0.99 and CLS cosine ≥ 0.99):

| model | mean per-patch cos (worst input) | min per-patch cos | CLS cos | max |diff| (rel) |
|---|---|---|---|---|
| TFLite fp32 | 1.000000 | 1.000000 | 1.000000 | 7.7e-4 (2.8e-5) |
| TFLite fp16-weight | 0.999998 | 0.999988 | 0.999997 | 5.1e-2 (1.8e-3) |
| ONNX (ORT 1.30) | 1.000000 | 1.000000 | 1.000000 | 6.8e-4 (2.6e-5) |

**TFLite I/O:** input `serving_default_image:0` `[1,448,448,3]` float32. Output index 0 = `patch` `[1,32,32,384]`
(tensor `StatefulPartitionedCall:1`), index 1 = `cls` `[1,384]` (`StatefulPartitionedCall:0`); the
`serving_default` signature names them `patch`/`cls`. **Identify outputs by shape.** Tensor dtypes: float32 +
int32 shape constants (fp16w adds float16 weights); no float64/int64; no dynamic dimensions.

**TFLite op list** (fp32; fp16w is identical plus 177 `DEQUANTIZE`): ADD 111, MUL 76, RESHAPE 51, MEAN 50,
FULLY_CONNECTED 48, TRANSPOSE 48, STRIDED_SLICE 38, SUB 26, SQUARED_DIFFERENCE 25, RSQRT 25, BATCH_MATMUL 24,
SOFTMAX 12, GELU 12, CONV_2D 1, CONCATENATION 1.

**NPU/HTP flags** (heuristic; to be confirmed on the phone in B5):
- none of the usual blockers: no Flex/custom ops, no ERF, no GATHER, no float64/int64, no dynamic shapes, all
  tensors ≤ 4-D;
- `GELU` ×12 — native in QNN HTP (`Gelu`); if a delegate/compiler plugin does not map it, it is decomposed (check
  the per-op placement/profiling);
- LayerNorm is decomposed (MEAN ×2, SUB, SQUARED_DIFFERENCE, RSQRT, MUL, ADD per norm; 25 norms) — lots of small
  elementwise ops; fine on HTP, but a fused LayerNorm would be faster;
- `SOFTMAX` over 1025 tokens on `[1,6,1025,1025]` per block (≈ 6.3 M elements, 25 MB fp32 / 12.6 MB fp16) and the
  two `BATCH_MATMUL`s per block (Q·Kᵀ, A·V) are the memory-bandwidth hot spot — the reason 448 px costs ≈ 5× the
  FLOPs of 224 px (≈ 63 GFLOP per image);
- 48 `TRANSPOSE` (4 per block, 4-D head split/merge) and 36 `STRIDED_SLICE` from the q/k/v Split — cheap but many;
- fp16w only halves the file; activations stay fp32 on CPU/GPU (`DEQUANTIZE` at load).

**Laptop CPU timing** (i-series laptop, ai_edge_litert XNNPACK / ORT, median of 10, busy machine):
TFLite fp32 273 ms @4 threads, 930 ms @1 thread; fp16w 307 / 937 ms; ORT 191 ms @4 threads.

**k-NN graph** `knn_p1024_d384_k2400.tflite` (`tools/make_knn_model.py --P 1024 --D 384 --K 2400
--out-dir tools/dinov2/out --no-realistic`): builtin MUL, SUM, BATCH_MATMUL, ADD, REDUCE_MIN; vs float64 brute
force: max element rel. error 4.9e-6 (full bank), 3.7e-6 (60 % filled), 3.6e-6 (LOO mask); 39.5 ms @1 thread,
13.2 ms @4 threads on the laptop.

Windows route time: ≈ 15 min wall clock (well inside the 60-min cap) — the Colab fallback notebook was not needed.

### 2.5 Reproduce (each heavy step under the machine-wide lock)
```
powershell -NoProfile -Command "& 'native\tools\with-lock.ps1' -- '.venv\Scripts\python.exe' 'tools\dinov2\verify_reference.py'"
powershell -NoProfile -Command "& 'native\tools\with-lock.ps1' -- '.venv-convert\Scripts\python.exe' 'tools\dinov2\export_dinov2.py' onnx"
powershell -NoProfile -Command "& 'native\tools\with-lock.ps1' -- '.venv-convert\Scripts\python.exe' 'tools\dinov2\export_dinov2.py' tflite"
powershell -NoProfile -Command "& 'native\tools\with-lock.ps1' -- '.venv\Scripts\python.exe' 'tools\dinov2\export_dinov2.py' check"
.venv\Scripts\python.exe tools\dinov2\write_manifest.py        # model card + SHA256SUMS.txt
```
Note: `with-lock.ps1` must be called through `-Command "& … -- …"`; with `-File`, PowerShell's binder treats
`--out`/`--size`-style arguments (and a bare `--`) as its own parameters and fails ("parameter name '' is ambiguous").

## 3. What is left for the phone (B5)
- Push `dinov2_s14_448_fp32.tflite` (or fp16w) + `knn_p1024_d384_k2400.tflite` + the parity fixture; run the
  accelerator parity gate (spec §13: cosine ≥ 0.999 fp32 candidates vs CPU, and vs the PyTorch fixture) and time
  NPU/GPU/CPU; check which ops the Qualcomm plugin keeps on
  HTP (GELU, the LayerNorm decomposition, SOFTMAX 1025²).
- Pipeline for DINOv2 Twins: `inputSize 448`, grid 32×32, `dim 384`, `l2NormalizePatches: true`,
  `knnGraphId knn_p1024_d384` (bank ≤ 2400 rows).
- The A/B on real captures (`tools/lab/twin_eval.py` on an export folder, `docs/verification/export-format.md`).

## 4. Offline A/B harness — synthetic smoke run (SYNTHETIC, NOT evidence)

> **Everything in this section comes from PIL-rendered synthetic parts.** It shows that the export format, the
> generator, `twin_eval.py` and both backbone arms run end to end and agree with each other. It says **nothing** about
> accuracy on real parts; the numbers must not go into `claims.csv`.

**Tools.** `tools/lab/make_synthetic_export.py` renders a session (matte sheet; "bracket" = brushed-steel plate with
two holes of different size and a stamped mark; defects: scratch, chipped wedge, marker dot; wrong objects: hex nut,
washer, painted plate, coin, triangle, bolt; look-alike = same outline with a brass cross-hatch surface; ±8° pose,
±4 % scale, ±3 % gain jitter; rotated = 30–180°). Every full-resolution frame goes through `twin_ref`'s mask
pipeline (downscale, sheet fit, segment, main object, sanity, crop square, crop-resize, cov, geometry, theta,
sharpness). The synthetic "phone" = `twin_ref.teach` + `judge` + `calibrate` with the shipped R18 on the exact
(pre-JPEG) 320 px crops. Format: `docs/verification/export-format.md`.
`tools/lab/twin_eval.py` re-embeds the stored 448 px JPEG crops on the laptop and runs 24 variants
(score × bank × rotation × τ rule) through `twin_ref`. Rotation variants rotate the stored crop about its centre
(sheet-colour padding), an approximation of the phone's full-frame re-crop.

**Data** `testdata/synthetic_export/` (6.48 MB; regenerates byte-identically, 320 files): 64 teach frames (3 without
a usable object), 8 negatives, 24 `calib`, 24 held-out `good`, 15 `defect` (5 each), 12 `wrong` (2 per kind),
4 `lookalike`, 10 `rotated`. Phone Twin: 16 keyframes, 1087 bank rows, τ 0.857, τ_id 0.9715 (midpoint with negatives).

**Consistency checks (both runs):** `cov` recomputed from the exported mask RLE + crop square equals the phone's
`cov` in 158/158 crops; geometry features and θ identical (max rel. diff 0). Phone (exact crops) vs laptop (JPEG q95
448 → 320 resample), default variant: verdicts agree 63/65, |s| diff median 0.021 (max 0.088), |sim| diff median
0.0004 (max 0.005).

**Headline — ResNet18** (`tools/dinov2/eval/synthetic_r18/report.md`, 24 variants, 150 s):
- AUROC good-vs-defect = 1.000 and good-vs-wrong = 1.000 for **all 24 variants** → by the stated rule
  (AUROC, then defaults win ties) the **winner is the default `SMOOTHED_MAX/CORESET/NONE/LSO`**. The synthetic set is
  too easy to rank variants by AUROC; only the operating points differ:

| R18 variant (tau rule LSO) | τ | PASS good | PASS rotated | NOT_ENR wrong | NOT_ENR look-alike | DEFECT |
|---|---|---|---|---|---|---|
| default SMOOTHED_MAX/CORESET/NONE | 0.873 | 24/24 = 100 % [86, 100] | 7/10 = 70 % [35, 93] | 12/12 = 100 % [74, 100] | 4/4 | 8/15 = 53 % [27, 79] |
| TOP1_MEAN/TOPK_VIEWS/NONE | 0.880 | 24/24 | 7/10 | 12/12 | 4/4 | 11/15 = 73 % [45, 92] |
| SMOOTHED_MAX/CORESET/CANONICAL | 0.716 | 24/24 | 10/10 = 100 % [69, 100] | 12/12 | 4/4 | 15/15 = 100 % [78, 100] |
| SMOOTHED_MAX/CORESET/AUGMENT4 | 0.864 | 24/24 | 10/10 | 12/12 | 4/4 | 9/15 = 60 % [32, 84] |

  (rates k/n = % [two-sided 95 % Clopper–Pearson]). CALIBRATED equals LSO here: calibration never lowers τ and the
  synthetic calib maximum stays below τ_teach. CANONICAL tightens τ (teach crops no longer spread over ±22°) — the
  mechanism to test first on real captures.

**Same export with DINOv2-S/14 @448** (`tools/dinov2/eval/synthetic_dinov2/report.md`, 340 s, 501 backbone runs):
AUROCs again 1.000 everywhere → default wins. Default operating point: PASS good 24/24, **rotated 10/10 without any
rotation handling**, NOT_ENR wrong 12/12, DEFECT 10/15 = 67 % [38, 88] (R18: 8/15); CANONICAL 12/15. But the four
look-alikes pass the identity gate (sim 0.80–0.83 vs τ_id 0.797) and are rejected as DEFECT (s 1.28–1.36) instead of
NOT_ENROLLED: DINOv2 global similarities spread wider (wrong objects 0.40–0.65), so the §7.3 midpoint lands just below
the look-alikes. **Check this on real look-alikes** before switching backbones (a look-alike is still rejected, but
with the wrong reason and heat map).

## 5. REPLAY regression clip — `testdata/replay_synth/` (SYNTHETIC)

`tools/lab/make_replay_synth.py` (deterministic; regenerates byte-identically): 348 landscape frames 1280×720,
JPEG q90, 30 fps (`timestamps.json` = round(i·1000/30) ms, 0 … 11,567 ms), analysis factor 4, **15.5 MB**.
`script.json` = segments, parameters and the 10 scripted parts; `expected.json` = the verdicts, computed with
`twin_ref` on the **decoded** JPEG frames (what the phone sees) and the shipped R18 TFLite, no borderline voting.

| segment | frames | ms | what |
|---|---|---|---|
| sheet | 0–14 | 0–467 | empty sheet → §2.1 fit (μ = 52, 88, 74; σ = sigmaMin 3) |
| teach | 15–134 | 500–4467 | bracket in the middle, rotating −16° → +16°, drifting ±28/16 px; teach uses every frame → 120 sane, 72 kept, 16 keyframes, bank 1063, τ 0.708, τ_id 0.976 (no negatives) |
| gap | 135–144 | 4500–4800 | empty |
| line | 145–347 | 4833–11567 | 10 parts left → right at 40 px/frame full-res = 10 px/frame analysis, one every 17 frames; fresh tracker, LINE (axis X at 0.5·W, direction ANY, 2 frames) |

Expected: 10 tracks, 10 `LINE` verdicts, 0 unjudged exits — parts 1, 2, 4, 6, 7, 9, 10 good → **PASS**
(s 0.45–0.76); part 3 chipped → **DEFECT** (s 1.40); part 8 large dot → **DEFECT** (s 1.15, `borderline: true`
because sim is within 0.02 of τ_id); part 5 hex nut → **NOT_ENROLLED / IDENTITY** (sim 0.952). `borderline` =
|s − 1| ≤ 0.1 or |sim − τ_id| ≤ 0.02 → any verdict accepted there. The self-test must run with the spec defaults
listed in `script.json` (teach on every frame of the teach window, voting off, sensitivity 1.0).
JPEG decoders may differ by ±1–2 levels between Android and PIL (both libjpeg-turbo); that can move `s`/`sim` a
little, which is what the borderline flag absorbs.

**Provenance / regeneration.** Both synthetic sets were generated with `tools/lab/twin_ref.py` SHA-256
`65986741f17962966b9aa93c7c1a6ac011d75c7d2c28cc9009ddff7c560fbec6` (stamped as `twinRefSha256` in
`testdata/synthetic_export/manifest.json` → `app` and in `testdata/replay_synth/expected.json`). If `twin_ref.py`
changes, regenerate (deterministic, ~1 min each) and re-run the A/B:
```
.venv\Scripts\python.exe tools\lab\make_synthetic_export.py --out testdata\synthetic_export
.venv\Scripts\python.exe tools\lab\make_replay_synth.py --out testdata\replay_synth
.venv\Scripts\python.exe tools\lab\twin_eval.py --export testdata\synthetic_export --backbone r18 --out tools\dinov2\eval\synthetic_r18 --title SYNTHETIC
.venv\Scripts\python.exe tools\lab\twin_eval.py --export testdata\synthetic_export --backbone dinov2 --out tools\dinov2\eval\synthetic_dinov2 --title SYNTHETIC
```

## 6. Open items for the phone
- Implement the export writer exactly as `export-format.md` (mask RLE, crop square, `cov`, geometry, θ, `phone`
  block, manifest `cropSize` 448 + `sheet`), pull a real session and run `twin_eval.py` with `--backbone r18` and
  `--backbone dinov2`; the synthetic folder is the parser test.
- REPLAY self-test: feed `testdata/replay_synth/frames` with `timestamps.json`, follow `script.json`, compare with
  `expected.json` (non-borderline verdicts must match exactly; trigger, trackId and tMs too).
- DINOv2 on the accelerator (§3 above) and the look-alike identity check (§4).
