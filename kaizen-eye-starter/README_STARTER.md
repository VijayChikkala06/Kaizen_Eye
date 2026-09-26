# Kaizen Eye - laptop starter kit

Everything here is **no-training**: a pretrained ImageNet backbone is converted once, and "normal" is learned
at enrolment from ~20 good photos by plain maths (coreset + nearest-neighbour + calibration).

**Verified in a clean sandbox:** Python 3.11, Linux x86_64, torch 2.13, litert-torch 0.9.4, ai-edge-litert 2.2.0.
**Not verified there:** Windows / macOS (use WSL or Colab if pip fails), and the ImageNet weight download
(the sandbox could not reach the weight hosts). So the sandbox runs used `--random-weights`: they prove the
pipeline works, but the *scores* are not what you will see with the real pretrained model.

## 1. Set up (once, with internet)
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r tools/requirements.txt
```
`litert-torch` pins `torch < 2.14`, so install from the requirements file. A plain `pip install torch torchvision`
gives a mismatched pair.

## 2. Convert the backbone (internet needed once, to fetch ImageNet weights)
```
python tools/convert_backbone.py --backbone resnet18 --size 256 --dim 128 --out out/backbone_r18_256.tflite
```
Prints a verification line (`cosine=1.000000` means the .tflite matches PyTorch) and the op list. Copy
`out/backbone_r18_256.tflite` and its `.json` into `app/src/main/assets/`.
Output tensor: NHWC float32 `[1, 32, 32, 128]` = 1,024 patch vectors. Input: NHWC float32 RGB 0..255.

## 3. Mechanics test - no photos needed
```
python tools/lab/synthetic_test.py --model out/backbone_r18_256.tflite
```

## 4. Tune on real phone photos (the important one)
```
photos/good/   20+ photos of good parts (same mount, same light)      -> first 20 enrol, extras = unseen-good check
photos/test/   bad_*.jpg and good_*.jpg
python tools/lab/eval_folder.py --model out/backbone_r18_256.tflite --good photos/good --test photos/test
```
Prints every score, saves heat-map overlays, and sweeps the sensitivity margin to suggest a slider start value.

## 5. Golden vectors for the Kotlin port
```
python tools/lab/make_golden.py > testdata/golden_core.json
```
`testdata/golden_core.json` is already generated. The Kotlin greedy coreset, k-NN and calibration must reproduce
it: coreset indices exactly, floats to ~1e-4 relative. The spec (independently re-implemented with scalar loops
and checked against this file, max difference ~2e-6):
```
k        = max(minK, floor(ratio * N))     N = frames * gridH * gridW   (defaults ratio 0.05, minK 256)
coreset  = greedy k-centre over all enrolment patches; start index 0; ties -> lowest index
distance = Euclidean (sqrt of the summed squares); "nearest" = smallest distance to any bank entry
LOO      = for each enrolment frame f: raw_f = max over its patches of the nearest distance to bank entries whose source frame != f
tau      = max_f raw_f * margin            (golden file uses margin 1.10; the app stores margin 1.0 and multiplies at runtime)
score    = max over patches of nearest distance / (tau * sensitivity);  > 1.0 means REJECT
```

## What v2 changed (measured, random-weight ResNet18 on synthetic parts - re-check on real photos)
* Frame score = max patch distance. AUROC good-vs-defect: max 0.92, 3x3-smoothed max 0.92, 99th percentile 0.87.
* Calibration = leave-one-out on the coreset. 2-fold and 4-fold hold-out give the same false-reject / detection
  curve at ~2x the compute (`--folds 4` still available). Sensitivity (margin) is the real knob: start near 1.10.

## Files
| File | What |
|---|---|
| `tools/convert_backbone.py` | pretrained CNN -> single-output patch-feature `.tflite` (`--onnx` also writes ONNX) |
| `tools/lab/patchcore_ref.py` | the scoring maths in numpy - the spec for Kotlin `:core` |
| `tools/lab/eval_folder.py` | score your own photos, overlays, margin sweep |
| `tools/lab/synthetic_test.py`, `synth_data.py` | synthetic parts + 4 defect types |
| `tools/lab/embedder.py`, `viz.py` | run the .tflite on the laptop; heat-map overlay |
| `tools/lab/make_golden.py` | test vectors for the Kotlin port |
