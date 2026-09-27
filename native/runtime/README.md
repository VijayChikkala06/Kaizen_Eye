# `:runtime` — Kaizen Eye 2 model runtime

LiteRT 2.2.0 vision runtime with **honest NPU/GPU/CPU labels** and CPU fallback, an accelerator probe in its own process
with a crash guard, the k-NN graph runner (exact Kotlin fallback), the model locator (SHA-256 verified), and the offline
VLM explainer (LiteRT-LM 0.16.1) in its own process. Android library, namespace `com.kaizeneye.runtime`, minSdk 31.

Specs: `docs/verification/twin-spec.md` §5 (k-NN) and §13 (accelerator honesty). Legacy design ported in spirit:
`mobile/src/ml/{accelerator,knn,backbone}.ts`.

## Wiring (main integrator)
- `settings.gradle.kts`: `include(":runtime")`; `app/build.gradle.kts`: `implementation(project(":runtime"))`.
- The app keeps bundling what the runtime needs: `libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`
  and the QNN v81 libraries in `nativeLibraryDir` (`useLegacyPackaging = true`), `qnn-runtime` (app packaging concern),
  models in `assets/models/` (`noCompress += "tflite"`).
- The runtime's manifest adds **only two services, no permissions**:
  `com.kaizeneye.runtime.ProbeService` (`android:process=":probe"`, `exported=false`) and
  `com.kaizeneye.runtime.VlmService` (`android:process=":vlm"`, `exported=false`).
- Keep `Application.onCreate` empty (it runs in every process).
- `object Runtime` shadows `java.lang.Runtime` for star-importers of `com.kaizeneye.runtime.*`: import it explicitly.

## API (all in `com.kaizeneye.runtime`)
```kotlin
object NpuEnv { fun ensure(context: Context); val socModel: String; val socManufacturer: String; fun npuSocSupported(): Boolean
                val adspLibraryPath: String?; val setenvError: String?; fun processName(): String; fun describe(): String }
enum class Accel { NPU, GPU, CPU }
data class ModelAsset(val id: String, val fileName: String, val sha256: String, val int8: Boolean = false)   // sha256 "" = unpinned
data class BackboneSpec(name, variants, inputSize, gh, gw, dim, l2NormalizePatches)
data class CandidateResult(name, modelId, ok, medianMs, minMs, p95Ms, cosine, patchCosine, relErr, loadMs, error = null, note = null)
data class AcceleratorReport(model, chosen, accel, modelId, badge, speedupVsCpu, results, fromCache, measuredAtMs, cacheKey,
                             forced = false, note = null)
interface FeatureBackbone : AutoCloseable { spec; modelId; sha256; report; fun embed(rgb: ByteArray): FloatArray
                                           val hasCls: Boolean; fun embedWithCls(rgb: ByteArray): Pair<FloatArray, FloatArray?> }
interface KnnEngine : AutoCloseable { val label: String; fun minSq(feats, nq, bank, nb, dim, extra): FloatArray; val report: AcceleratorReport? }
data class LocatedModel(asset, source /* external|imported|bundled */, path, sha256Ok, bytes, actualSha256 = null, note = null)
object ModelLocator { fun locate(context, asset): LocatedModel?; fun importFromUri(context, uri, asset): LocatedModel
                      fun externalDir(context): File?; fun importedDir(context): File; fun assetName(asset): String }
object Runtime { suspend fun loadBackbone(context, spec, forceAccel: Accel? = null, rebenchmark: Boolean = false): FeatureBackbone
                 suspend fun loadKnn(context, P: Int, dim: Int, buckets: List<Int>, forceAccel: Accel? = null): KnnEngine
                 fun forgetDecisions(context); fun forgetCrashes(context) }
object SelfTestProbes { suspend fun backboneParity(context, spec): JSONObject; suspend fun knnAccuracy(context, P, dim, buckets): JSONObject }
object KnownModels { BACKBONE_R18_320; BACKBONE_DINOV2_S14_448; knnGraph(p, d, k); FASTVLM_05B_SM8850; GEMMA4_E2B_GPU; ... }
class VlmService : Service; class ProbeService : Service                       // manifest-instantiated
data class VlmResult(text: String?, backend: String, ms: Long, error: String?)
sealed interface VlmState { Idle; Loading; Ready(backend, initMs, model = "", detail = ""); Failed(message) }
class VlmClient(context) : AutoCloseable { val state: StateFlow<VlmState>; fun bind()
                                           suspend fun explain(imageFile: File, prompt: String, timeoutMs: Long = 4000): VlmResult; fun close() }
```
- `embed(rgb)`: `rgb` = N·N·3 crop bytes (spec §1.4) → float32 0..255 NHWC → `[gh*gw*dim]`, per-patch L2-normalised when
  `spec.l2NormalizePatches`. Serialised; one input FloatArray and one set of TensorBuffers reused. The patch output is the
  model output with exactly `gh*gw*dim` values (names are never used); a CLS output = the one with exactly `dim` values
  (DINOv2 `[1,384]`), returned **raw** by `embedWithCls`. No matching output → clear `IllegalStateException` at load.
- `loadBackbone` throws `IllegalStateException` when no float variant is found/verified or the model cannot run even on
  the CPU. `loadKnn` never throws for missing graphs (exact fallback; `label` says so).
- `KnnEngine.minSq`: twin-spec §5. `extra[k]` added per bank row; `1e30f` excludes; everything excluded or empty bank →
  result `1e30f`. A bank array edited in place between calls is detected (full content hash) and re-prepared.

## Process model
| Process | What runs there | Never |
|---|---|---|
| `:main` (app) | `Runtime`, `FeatureBackbone`, `KnnEngine`, `VlmClient`; LiteRT "plain" env (CPU/GPU); an NPU env only for a chosen/forced NPU candidate that already survived the probe | LiteRT-LM |
| `:probe` | `ProbeService`: benchmarks ONE candidate per request (Messenger IPC), output handed back as a float file in `filesDir/accel/probe/`; ends its process when unbound | LiteRT-LM |
| `:vlm` | `VlmService`: LiteRT-LM `Engine` (FastVLM-0.5B sm8850 → gemma-4-E2B-it-gpu), one explanation at a time; ends its process when unbound | vision LiteRT (`liblitertlm_jni.so` embeds its own LiteRT) |

`NpuEnv.ensure(context)` runs first in each of them (Runtime, ProbeService.onCreate, VlmService.onCreate) and sets
`ADSP_LIBRARY_PATH = "$nativeLibraryDir;/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"`.
If the app calls LiteRT/LiteRT-LM anywhere else, call `NpuEnv.ensure` before.

## Accelerator selection (spec §13, `AccelRules`, JVM-tested)
Candidates = accelerator × model variant. The first usable non-int8 variant is the **reference**; `CPU` on it (4 threads)
is the accuracy reference and the fallback. Every other variant also runs on the CPU as the speed baseline for
accelerators running it (`CPU (int8)`, `CPU (fp32)`; never chosen). GPU (FP32 precision) runs float variants, NPU
(Qualcomm HTP, `SUSTAINED_HIGH_PERFORMANCE`, fixed at load) runs all. R18: `NPU, NPU (int8), GPU, CPU, CPU (int8)`;
DINOv2 (fp16w reference + fp32): `NPU, NPU (fp32), GPU, GPU (fp32), CPU, CPU (fp32)` (fp16w = float gate).
- Probe order: CPU runs first, then GPU, then NPU. Fixed input = port of `benchmarkImage` (`BenchInputs.image`).
  2 warm-ups + 10 timed runs (each = write input + run + read patch output) → median / min / p95.
- Gates vs the reference output: float → finite, cosine ≥ 0.999, rel. L2 ≤ 1 %; int8 → mean per-patch cosine ≥ 0.99.
- Speed: a non-CPU candidate must have min time ≤ 0.9 × the min time of the CPU running the SAME variant, otherwise it is
  reported `"<X> requested - no speed-up (ran on CPU?)"` and not used (`Options(NPU)` silently runs unsupported ops on the
  CPU; a measured speed-up is the only honest NPU evidence). Fastest eligible median wins; ties → CPU.
- Badge: `"NPU 4.1 ms (7.9× CPU)"`, `"NPU int8 3.0 ms (6.7× CPU int8)"`, `"GPU 9.0 ms (2.1× CPU)"`, `"CPU 38 ms"`;
  speed-up = same-variant CPU median / chosen median.
- NPU is attempted only when `Build.SOC_MODEL` starts with `SM8850` (LiteRT's default checker demands exactly `SM8850`;
  ours is passed to `BuiltinNpuAcceleratorProvider`); otherwise the report says `not attempted: SoC '<x>' is not SM8850*`.
- The app process then creates its own `CompiledModel` for the chosen candidate (NPU JIT result reused from the compiler
  cache in `cacheDir`) and warms it up. **Run-time failure** of an accelerator → that instance switches to CPU with the
  same variant for good (`modelId` unchanged), report + cache updated.
- `forceAccel` (A/B toggle): runs the decision's variant on the forced accelerator if that candidate was benchmarked
  without crash/accuracy failure (else the CPU, report says why); badge e.g. `"CPU 38 ms (forced)"`,
  `"NPU requested - no speed-up (ran on CPU?) 36 ms (forced)"`; the cached decision is not changed.
- A decision made while the probe was unavailable is not cached (next start tries again).
- `report.note` carries the device SoC string, skipped/rejected model copies, and per load the in-app create time
  (`app load X ms` — for an NPU candidate, compare with the probe's `loadMs` = cold JIT vs cached JIT).

## Crash guard
Marker `filesDir/accel/trying.json` = `{key, candidate, phase, pid, token, atMs}` written before each candidate, cleared
after. Probe request timeout = 60 s (scaled up for big models: 60 s per 30 MB, e.g. ~170 s for DINOv2 fp32 JIT).
- Probe dies while running a candidate (binderDied / onServiceDisconnected) → exit reason from
  `ApplicationExitInfo`: native crash / crash / ANR / unknown → candidate **crashed** (skipped for this key); killed by the
  system or user (low memory, …) → one strike (second strike = crashed). Timeout → probe killed, candidate crashed.
- Stale marker on the next start (the process that wrote it died): phase `main` (app died creating/warming the chosen
  accelerator) → crash reason ⇒ crashed, unknown (e.g. reboot) ⇒ one strike, external ⇒ nothing; phase `probe` (app died
  while the separate probe ran) → one strike only if nothing is known about the death (reboot loop protection).
- Same policy in `:vlm` (`filesDir/vlm/{trying,strikes}.json`) per (model file, backend, app version, fingerprint, size).
- `Runtime.forgetDecisions` keeps crash records; `Runtime.forgetCrashes` clears them (the probe retries, isolated).

## Decision cache
`filesDir/accel/decisions.json`, key =
`model#variantId:sha256+…@v<versionCode>|<Build.FINGERPRINT>|qnn2.49.0|<native stamp>|d<DECISION_VERSION>`
(native stamp = sizes of `libLiteRtDispatch_Qualcomm.so`, `libLiteRtCompilerPlugin_Qualcomm.so`, `libQnnHtp.so`,
`libQnnHtpV81Skel.so`, so swapping QNN/dispatch builds re-benchmarks). Bump `CacheKeys.DECISION_VERSION` when options or
rules change, `CacheKeys.QNN_VERSION` when the bundled QNN changes.

## Models
`ModelLocator.locate`: `getExternalFilesDir("models")` (adb push) → `filesDir/models` (SAF import) → asset
`models/<file>`. SHA-256 streamed, verdict cached in `filesDir/models/.sha-cache.json` (files: path + size + mtime;
assets: versionCode + install time), cross-process file lock. Mismatch → not usable, `note` says why (a mismatching
external copy does not hide a good bundled one). `importFromUri` hashes while copying (`.part` + rename; blocking — call
off the main thread). Pins: `KnownModels` (R18 f32/int8, DINOv2 fp16w/fp32, k-NN p1600_d128_k{800,1600,2400} and
p1024_d384_k2400, FastVLM sm8850). Push commands:
`adb push FastVLM-0.5B.qualcomm.sm8850.litertlm /sdcard/Android/data/com.kaizeneye.v2/files/models/` (same for DINOv2).

## k-NN
`knn_p{P}_d{D}_k{K}.tflite` (inputs feats [P,D], bankT [D,K], bankNorm [K]; output minD2 [P]; our graphs have no
signature, tensor names `feats/bankT/bankNorm/Identity`). Inputs are identified by **shape** from the flatbuffer
(`TfliteInfo`; the Kotlin API cannot list names), bound by name (fallback: positional buffers), and each graph is
verified against the exact float64 search at load (max |Δd| ≤ 1e-3 · max d). fp32 on the CPU by default (the 1e30 mask
overflows fp16); `forceAccel` may try GPU/NPU (created in the app under the `main` crash-guard phase, verified the same
way). Bank prep (transpose, zero padding, `bankNorm = |b|² + extra`, 1e30 on padding, chunking above the largest bucket,
query padding/splitting at P) is pure Kotlin (`KnnMath`). Any graph failure at run time → exact search for good.

## VLM
`VlmClient.bind()` → `:vlm` starts: `NpuEnv.ensure`; locate `FastVLM-0.5B.qualcomm.sm8850.litertlm` (942,374,912 bytes,
SHA-256 pinned; hashed once, cached) or `gemma-4-E2B-it-gpu.litertlm` (unpinned, digest reported in `Ready.detail`);
backends FastVLM: NPU(nativeLibraryDir) → GPU → CPU; Gemma: GPU → CPU; `visionBackend` = main backend, `maxNumImages = 1`,
`cacheDir = context.cacheDir`. `Ready(backend, initMs)` or `Failed("no VLM model …" | errors)`. `explain(file, prompt)`:
the app writes the crop JPEG into its `cacheDir`; a fresh conversation per request (greedy, ≤ 64 output tokens — if the
engine rejects those settings, output cap only, then library defaults; a simpler level is kept once it answers); one at a
time, ≤ 2 waiting (else `"busy"`); at the deadline → `VlmResult(null, backend, ms, "timeout")` and `cancelProcess()`.
The `:vlm` process death → `Failed("VLM process died")`, the system restarts it while bound. `close()` unbinds → engine
closed, process ends.

## Must be verified on the phone (nothing here ran on a device)
1. `NpuEnv.socModel` / `socManufacturer` real strings (expected `SM8850…`; iQOO 15 may report `SM8850-AC`) and that the
   NPU candidates are attempted.
2. NPU speed-up for R18 f32/int8 (and DINOv2): `SelfTestProbes.backboneParity` → badge, `speedupVsCpu`, cosines; an NPU
   that is not ≥ 1.11× faster than CPU is reported "requested - no speed-up".
3. JIT compile time of the first NPU `CompiledModel.create` (probe `loadMs`) and the cached second create in the app;
   whether the 60 s/scaled probe timeout is enough; whether `libQnnHtpPrepare/Ir/Saver` are needed.
4. Name-based k-NN binding with signature `""` on signature-less graphs (log `bound by name`), `knnAccuracy` pass and ms.
5. Probe IPC end-to-end: `ProbeService` in the merged manifest, `:probe` process visible in `ps`, ends after benchmark.
6. Crash guard: force a probe death (e.g. `adb shell kill -9 <probe pid>` during a benchmark) → candidate skipped next
   start; `ApplicationExitInfo` reasons as expected.
7. VLM: backend actually used (FastVLM NPU vs GPU/CPU), `initMs`, time per explanation (≤ 2 s target), cancel on
   timeout, memory after `close()` (process gone).
8. `ADSP_LIBRARY_PATH` accepted in every process (logcat `KaizenRuntime`), no INTERNET permission added by the merge.
