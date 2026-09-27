package com.kaizeneye.v2.twin

import com.kaizeneye.core.twin.NegativesStore
import com.kaizeneye.core.twin.PipelineInfo
import com.kaizeneye.core.twin.RotationRule
import com.kaizeneye.core.twin.TwinLoadResult
import com.kaizeneye.core.twin.TwinModel
import com.kaizeneye.core.twin.TwinStore
import com.kaizeneye.v2.AppDirs
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.pipeline.TwinSummary
import java.io.File
import java.util.Locale
import com.kaizeneye.core.mask.MaskParams as SegParams
import com.kaizeneye.core.twin.MaskParams as PipelineMaskParams

/**
 * App-side access to the Twin store (filesDir/twins, twin-spec §8) and the negatives library. Also builds the RUNNING
 * pipeline description from the loaded engines: a stored Twin whose fingerprint differs is refused (plan guardrail 5) and
 * offered for "rebuild from stored crops".
 */
class TwinRepo(dirs: AppDirs) {
    val store = TwinStore(dirs.twins)
    val negatives = NegativesStore(dirs.negatives)

    init {
        try {
            store.cleanTemp()
        } catch (_: Throwable) {
        }
    }

    fun summaries(running: PipelineInfo?): List<TwinSummary> = store.list().map { s ->
        val loadable = running == null || s.fingerprint == running.fingerprint()
        TwinSummary(
            id = s.id, name = s.name, createdAtMs = s.createdAtMs, keyframes = s.keyframes, calibrated = s.calibrated,
            tau = Double.NaN, tauId = Double.NaN,
            headline = if (s.calibrated) "calibrated" else "not calibrated — within-part bound only",
            loadable = loadable,
            problem = if (loadable) null else "built with another pipeline (backbone/model changed) — re-teach or rebuild",
        )
    }

    fun load(id: String, running: PipelineInfo): TwinLoadResult = store.load(id, running)

    fun save(twin: TwinModel, keyframeJpegs: List<ByteArray>?): File = store.save(twin, keyframeJpegs)

    fun delete(id: String) = store.delete(id)

    fun export(id: String, destRoot: File): File = store.exportTo(id, destRoot)

    companion object {
        /** Segmentation parameters used by the fast loop (twin-spec §2 defaults). */
        val SEG = SegParams()

        /** The pipeline description of what is running now (twin-spec §8), from the loaded backbone + k-NN family. */
        fun runningPipeline(r: EngineHolder.State.Ready, seg: SegParams = SEG, accuracy: Boolean = true): PipelineInfo {
            val spec = r.choice.spec
            val id = r.backbone.modelId
            val precision = when {
                id.contains("int8") -> "int8"
                id.contains("fp16") -> "fp16w"
                else -> "fp32"
            }
            return PipelineInfo(
                backboneId = id,
                backboneSha256 = r.backbone.sha256.lowercase(Locale.ROOT),
                inputSize = spec.inputSize,
                gh = spec.gh,
                gw = spec.gw,
                dim = spec.dim,
                precision = precision,
                l2NormalizePatches = spec.l2NormalizePatches,
                preprocessVersion = 1,
                mask = PipelineMaskParams(
                    analysisFactor = seg.analysisFactor, kSigma = seg.kSigma, sigmaMin = seg.sigmaMin, minBlobPx = seg.minBlobPx,
                    borderMargin = seg.borderMargin, cropMargin = seg.cropMargin, coreThreshold = seg.coreThreshold,
                ),
                knnGraphId = r.choice.knnGraphId,
                // Accuracy mode (Twin v2): pose-normalised crops + the FIT gate (docs/verification/accuracy-v2.md). The spec
                // pipeline (rotation NONE, no fit gate) stays available for the self-test regression against the laptop reference.
                rotation = if (accuracy) RotationRule.CANONICAL else RotationRule.NONE,
            )
        }
    }
}
