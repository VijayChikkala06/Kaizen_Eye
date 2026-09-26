# Kaizen Eye

No-training visual inspection on a phone. Photograph ~20 **good** parts, and the app learns what "normal" looks
like; after that it flags any part that looks different (**PASS / REJECT**) and shows a heat map of *where*.

Nothing is trained. A pretrained ImageNet backbone (ResNet18) is converted once into a patch-feature `.tflite`
model; "normal" is learned at enrolment by plain maths: coreset + nearest-neighbour + calibration (PatchCore-style).
Everything runs **on the device** (offline) with LiteRT.

```
 photo ──► centre-crop 256×256 ──► backbone .tflite ──► 32×32 patch vectors (128-d)
                                                           │
   ENROL  (~20 good photos): greedy k-centre coreset ──► memory bank ──► leave-one-out ──► threshold tau
   INSPECT (one photo):      nearest-neighbour distance per patch ──► heat map
                             score = max distance / (tau × sensitivity)      score > 1.0  ⇒  REJECT
```

## Repository layout

| Path | What |
|---|---|
| `mobile/` | The app: Expo SDK 57 / React Native 0.86, TypeScript. Development build (not Expo Go - see below). |
| `mobile/src/core/patchcore.ts` | Scoring maths - TypeScript port of `tools/lab/patchcore_ref.py`, checked against `testdata/golden_core.json`. |
| `mobile/src/ml/` | Backbone runner (`react-native-fast-tflite`) and photo → 256×256 RGB float input. |
| `mobile/assets/models/` | The shipped backbone `backbone_r18_256.tflite` + its `.json` metadata. |
| `tools/` | Laptop Python tooling: backbone conversion, scoring reference, tuning on real photos, golden vectors. |
| `testdata/golden_core.json` | Golden test vectors the TypeScript core must reproduce. |
| `demo_photos/` | Synthetic demo set (`good/` = 20 enrolment photos, `test/` = good + 4 defect types). |

## Why a development build and not Expo Go

Expo Go can only run the native modules bundled inside Expo Go. Kaizen Eye runs a TFLite model on the phone with
`react-native-fast-tflite` (native C++ / LiteRT), so it needs its own build of the app (an APK). It is still a
normal Expo project: `app.json` + config plugins, `npx expo start`, hot reload, EAS - the `android/` folder is
generated (`npx expo prebuild`) and is not committed.

## Run the app

### Option 1 - install the APK
Build it (below) or take a built `app-release.apk`, copy it to an Android phone (Android 7+, arm64) and install it
(allow "install unknown apps"). It is fully offline; the model is inside the APK.

### Option 2 - build the APK on Windows

Prerequisites (one time):
* Node.js 20+ (tested 24) and npm.
* JDK 17+ - use Android Studio's bundled JBR: `C:\Program Files\Android\Android Studio\jbr`.
* Android SDK with platform 36 (Android Studio's SDK Manager). NDK 27.1 and CMake 3.22.1 are installed
  automatically by Gradle on the first build (~1 GB).
* **If your Windows user name contains a space** (e.g. `C:\Users\First Last\...`) the native C++ build fails with
  hundreds of `ld.lld: error: undefined symbol: std::__ndk1::...`. Clang gets called through its 8.3 short name
  (`CLANG_~1.EXE`), which drops it into C mode. Fix: expose the SDK under a path without spaces (no admin needed):
  ```
  mklink /J D:\AndroidSdk "%LOCALAPPDATA%\Android\Sdk"
  ```
  and use `D:\AndroidSdk` below. Delete any `node_modules\**\android\.cxx` folders from a failed build first.

```
cd mobile
npm install
npx expo prebuild -p android
echo sdk.dir=D:/AndroidSdk> android\local.properties
set JAVA_HOME=C:\Program Files\Android\Android Studio\jbr
set ANDROID_HOME=D:\AndroidSdk
cd android
gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a,x86_64
```
The APK is `mobile/android/app/build/outputs/apk/release/app-release.apk` (signed with the debug key - fine for
internal testing, not for the Play Store). `arm64-v8a` covers phones, `x86_64` the Android emulator.

Development loop with hot reload: `npx expo run:android` (debug development build + Metro).
Cloud build instead of a local toolchain: `npx eas-cli@latest build -p android --profile preview` (needs a free
Expo account; profiles are in `mobile/eas.json`).

Checks: `npm run typecheck` and `npm run verify-core` (TypeScript core vs `testdata/golden_core.json`).

## Use the app

1. **Enrol** - photograph ~20 GOOD parts (5-30) with the same mount, distance and light you will inspect with, or
   **Import photos** from the gallery. After the first photo a faint **alignment guide** (that photo) is drawn over
   the camera - line every later shot up with it. Tap **Build profile**: the app selects the memory bank, calibrates
   the threshold (leave-one-out) and stores the profile on the phone.
2. **Inspect** - photograph (or import) a part: **PASS** / **REJECT**, the score (reject when > 1.00), why
   (local defect vs a different part / scene) and a heat map; the ring marks the most anomalous area.
3. **Sensitivity (margin)** on the home screen - the tuning knob, start at ×1.00. Higher = fewer false rejects,
   lower = catches smaller defects. It applies instantly; no re-enrolment needed.

### Capture rules (measured with `tools/lab/handheld_eval.py`)
* **Nothing may enter or leave the frame between photos** - no hands, cables, other objects, changing background.
  This is the one thing that breaks detection: with changing clutter at the edges tau roughly triples and
  detection falls to a few percent, whatever the model. A textured but unchanging desk is fine.
* Keep the **whole part inside the dashed square** with a small margin (the outer 10% of the photo is not scored).
  Filling the frame edge-to-edge is worse, not better.
* Hold still until the image is sharp (blur is the worst remaining factor), keep the lighting constant.
  Framing jitter of about ±4% / ±3° costs almost nothing; ±10% costs about 8 AUROC points.
* **tau is your quality meter**: after enrolment the app shows it. Around 0.4-0.6 is good; above 0.7 the app warns
  that the photos differ too much (usually clutter or blur) and small defects may pass - re-enrol.

Demo without a production part: copy `demo_photos/` to the phone, enrol with the 20 `good/` photos, then inspect
the `test/` photos (`bad_*` should REJECT, `good_*` should PASS).

## Laptop tools (Python 3.10 - 3.12, tested 3.11)

```
py -3.11 -m venv .venv          # Linux/macOS: python3.11 -m venv .venv && source .venv/bin/activate
.venv\Scripts\activate
pip install -r tools/requirements.txt
```
`tools/requirements.txt` pins `torch==2.13.*` / `torchvision==0.28.*` (install them together - a plain
`pip install torch torchvision` gives a mismatched pair). `litert-torch` is skipped on Windows because its
`litert-converter` dependency has no Windows wheels; everything except the conversion step works in this venv.

### Convert the backbone (only needed to change the model)
Linux / macOS / Colab (official LiteRT converter):
```
python tools/convert_backbone.py --backbone resnet18 --size 320 --dim 128 --out out/backbone_r18_320.tflite
```
Windows (torch → ONNX → onnx2tf, in a separate venv because its pins conflict with the main one):
```
py -3.11 -m venv .venv-convert
.venv-convert\Scripts\python -m pip install --no-deps -r tools/requirements-convert-windows.txt
.venv-convert\Scripts\python tools/convert_backbone.py --backend onnx2tf --backbone resnet18 --size 320 --dim 128 --out out/backbone_r18_320.tflite
```
Both print a verification line; `cosine=1.000000` means the `.tflite` matches PyTorch. Copy
`out/backbone_r18_320.tflite` and its `.json` into `mobile/assets/models/` and rebuild the app. Contract:
input NHWC float32 `[1,320,320,3]` RGB 0..255 (normalisation is inside the model), output NHWC float32
`[1,40,40,128]` = 1,600 patch vectors. (The original README model, `--size 256`, gives `[1,32,32,128]`.)

### Mechanics test and tuning on real photos
```
python tools/lab/synthetic_test.py --model out/backbone_r18_320.tflite
python tools/lab/eval_folder.py --model out/backbone_r18_320.tflite --good photos/good --test photos/test
```
`photos/good/`: 20+ good parts (first 20 enrol, extras = unseen-good check). `photos/test/`: `bad_*.jpg` and
`good_*.jpg`. Prints every score, saves heat-map overlays and suggests a sensitivity start value.

### Handheld benchmark (how the app settings were chosen)
```
python tools/lab/handheld_eval.py --model <model.tflite> --regimes careful,casual_nc --variants base,b4+sm
```
Six parts (synthetic board + real-texture objects), phone-like capture (shift / rotation / zoom / light / blur /
noise / JPEG, optional changing clutter) and five defect types (sticker, pen, scratch, missing, stain) at 3/6/10%
of the image side. Results, careful handheld capture, 900 defects / 180 good photos:

| config | AUROC | detected (small / medium / large) | false rejects |
|---|---|---|---|
| README defaults (r18/256, max, sensitivity 1.10) | 0.905 | 55% (37 / 57 / 71) | 0% |
| **app: r18/320 + 3x3 smoothing + 10% border, sensitivity 1.00** | **0.953** | **80% (65 / 84 / 91)** | **0%** |

Not helpful (measured): outlier-frame dropping, percentile calibration, 384 channels at 256 px, L2-normalised
features, MobileNetV3, PaDiM-style per-position normalisation, enrolment augmentation. int8 quantisation kept
AUROC within noise (0.898 vs 0.905) at 3.2x the CPU speed.

### Golden vectors for the TypeScript core
```
python tools/lab/make_golden.py --out testdata/golden_core.json                                  # README spec
python tools/lab/make_golden.py --smooth --border 1 --grid 8 --out testdata/golden_app.json      # app settings
cd mobile && npm run verify-core
```
Use `--out` on Windows: in PowerShell 5.1, `> file.json` writes UTF-16, which breaks JSON parsing.

## Scoring spec (implemented identically in `patchcore_ref.py` and `patchcore.ts`)
```
k        = max(minK, floor(ratio * N))     N = frames * gridH * gridW   (defaults ratio 0.05, minK 256)
coreset  = greedy k-centre over all enrolment patches; start index 0; ties -> lowest index
distance = Euclidean (sqrt of the summed squares); "nearest" = smallest distance to any bank entry
LOO      = for each enrolment frame f: raw_f = max over its patches of the nearest distance to bank entries whose source frame != f
tau      = max_f raw_f * margin            (golden file uses margin 1.10; the app stores margin 1.0 and multiplies at runtime)
score    = max over patches of nearest distance / (tau * sensitivity);  > 1.0 means REJECT
```
Frame score = max patch distance (AUROC good-vs-defect on synthetic parts: max 0.92, 99th percentile 0.87).
Calibration = leave-one-out on the coreset; 2-/4-fold hold-out gave the same curve at ~2x compute.

**App settings** (`mobile/src/config.ts`; `patchcore_ref.enrol(..., smooth=True, border=b)`): the frame score is taken
on the 3x3-smoothed distance map (smoothing applied to the full map), ignoring the outer `b = round(0.1 * grid)`
patch rows/columns (4 of 40 at 320 px) - identically in the LOO calibration and at inspection; default sensitivity
1.00. Under handheld capture smoothing helps (it was neutral on the steady synthetic set). The app also has a
whole-image check (cosine distance of the mean patch vector to the enrolment frames, its own leave-one-out
threshold) that rejects a different part or scene; final score = max(patch score, whole-image score).
