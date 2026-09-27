package com.kaizeneye.runtime

/*
 * Public API of :runtime (the app is written against these names; members are only ever ADDED, never renamed).
 * Honest-accelerator rules: docs/verification/twin-spec.md §13. k-NN semantics: twin-spec §5. Process model: README.md.
 */

/** The accelerator a model instance runs on. NPU/GPU are only claimed when measured faster than CPU (spec §13). */
enum class Accel { NPU, GPU, CPU }

/**
 * One model file. [fileName] is looked up as getExternalFilesDir("models")/<fileName> (adb push), filesDir/models/<fileName>
 * (SAF import) and the app asset "models/<fileName>" (bundled), in that order. [sha256] is the pinned hex digest (any case);
 * blank = not pinned (the actual digest is still computed and reported). [int8] = quantised variant (int8 accuracy gate).
 */
data class ModelAsset(val id: String, val fileName: String, val sha256: String, val int8: Boolean = false)

/**
 * A patch-feature backbone: NHWC float32 [1, inputSize, inputSize, 3] RGB 0..255 in, [1, gh, gw, dim] out.
 * The first non-int8 variant is the float (CPU accuracy reference) one.
 */
data class BackboneSpec(
    val name: String,
    val variants: List<ModelAsset>,
    val inputSize: Int,
    val gh: Int,
    val gw: Int,
    val dim: Int,
    val l2NormalizePatches: Boolean,
)

/** One benchmarked candidate ("NPU", "NPU (int8)", "GPU", "CPU", "CPU (int8)"). Times in ms; similarity vs the float CPU output. */
data class CandidateResult(
    val name: String,
    val modelId: String,
    val ok: Boolean,
    val medianMs: Double?,
    val minMs: Double?,
    val p95Ms: Double?,
    val cosine: Double?,
    val patchCosine: Double?,
    val relErr: Double?,
    val loadMs: Double?,
    val error: String? = null,
    val note: String? = null,
)

/**
 * The accelerator decision for one model on this device. [chosen] is the candidate name, [badge] the UI text
 * (e.g. "NPU 4.1 ms (7.9× CPU)", "GPU 9.0 ms (2.1× CPU)", "CPU 38 ms"). [speedupVsCpu] = CPU median / chosen median for the
 * same model variant (null when the chosen candidate is a CPU one). [fromCache] = the numbers come from an earlier benchmark
 * (filesDir/accel/decisions.json, key [cacheKey]). [forced] = the caller forced [accel] (A/B toggle); [note] explains
 * anything unusual (forced, run-time fallback, probe unavailable, ...).
 */
data class AcceleratorReport(
    val model: String,
    val chosen: String,
    val accel: Accel,
    val modelId: String,
    val badge: String,
    val speedupVsCpu: Double?,
    val results: List<CandidateResult>,
    val fromCache: Boolean,
    val measuredAtMs: Long,
    val cacheKey: String,
    val forced: Boolean = false,
    val note: String? = null,
)

/** A loaded backbone. Store [modelId] + [sha256] with anything built from its features (features of variants differ). */
interface FeatureBackbone : AutoCloseable {
    val spec: BackboneSpec
    val modelId: String
    val sha256: String
    /** Current decision; changes if the accelerator fails at run time and the instance switches to CPU for good. */
    val report: AcceleratorReport

    /** rgb = N*N*3 crop bytes (twin-spec §1.4) → [gh*gw*dim] floats (per-patch L2-normalised when spec.l2NormalizePatches). Thread-safe (serialised). */
    fun embed(rgb: ByteArray): FloatArray

    /** True when the model also has a CLS output (an output with exactly `dim` values, e.g. DINOv2 [1,384]). */
    val hasCls: Boolean get() = false

    /**
     * Like [embed] (same patch output, same normalisation) plus the CLS vector when [hasCls] (returned RAW - not
     * L2-normalised), else null. Output tensors are identified by element count, never by name.
     */
    fun embedWithCls(rgb: ByteArray): Pair<FloatArray, FloatArray?> = embed(rgb) to null
}

/** k-NN search engine (twin-spec §5). */
interface KnnEngine : AutoCloseable {
    /** Human-readable engine description (graph buckets + accelerator, or the exact CPU fallback). */
    val label: String

    /** Per query row: min squared distance to the bank rows; extra[k] is added per bank row (1e30f excludes it) — twin-spec §5. */
    fun minSq(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?): FloatArray

    /** Timing/accuracy report of the engine (null for engines that do not produce one). */
    val report: AcceleratorReport? get() = null
}

/**
 * Where a model file was found. [source] is "external" | "imported" | "bundled"; [path] is the absolute file path (for
 * "bundled" it is the asset name "models/<file>"). [sha256Ok] = the digest matches the pin (always true for unpinned
 * assets). [actualSha256] = computed digest (lowercase). [note] = why a copy was rejected or anything else worth reporting.
 */
data class LocatedModel(
    val asset: ModelAsset,
    val source: String /* bundled|external|imported */,
    val path: String?,
    val sha256Ok: Boolean,
    val bytes: Long,
    val actualSha256: String? = null,
    val note: String? = null,
)

/** One VLM answer. [text] null on error/timeout; [backend] e.g. "NPU"; [ms] wall time of the request. */
data class VlmResult(val text: String?, val backend: String, val ms: Long, val error: String?)

/** State of the ":vlm" process as seen by [VlmClient]. */
sealed interface VlmState {
    data object Idle : VlmState
    data object Loading : VlmState
    /** [backend] "NPU" | "GPU" | "CPU"; [initMs] engine init time; [model] model file; [detail] e.g. its SHA-256 and skipped candidates. */
    data class Ready(val backend: String, val initMs: Long, val model: String = "", val detail: String = "") : VlmState
    data class Failed(val message: String) : VlmState
}
