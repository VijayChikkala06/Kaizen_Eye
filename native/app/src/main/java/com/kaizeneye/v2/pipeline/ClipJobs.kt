package com.kaizeneye.v2.pipeline

import android.os.SystemClock
import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.telemetry.StageTimes
import com.kaizeneye.core.track.Trigger
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.core.twin.PipelineInfo
import com.kaizeneye.core.twin.TwinModel
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraFrame
import com.kaizeneye.v2.camera.ReplayFrameSource
import com.kaizeneye.v2.export.EvalExport
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.twin.TwinRepo
import com.kaizeneye.v2.ui.LineMode
import java.io.File

/**
 * Batch jobs over recorded clips (self-test regression, eval export, "teach from clip"): the SAME session classes as the
 * live app, fed by a non-paced [ReplayFrameSource]. Every clip learns its own empty-sheet model from its first frames, so
 * every clip must start with ~0.5 s of empty sheet.
 */
object ClipJobs {

    class ClipInfo(val sheet: SheetModel, val frameW: Int, val frameH: Int, val frames: Int)

    class TeachClipResult(val built: TeachSession.Built?, val message: String, val sheet: SheetModel?, val frameW: Int, val frameH: Int)

    /** Plays [clip] as fast as possible, calling [onFrame] for every frame on the replay thread; blocks until done. */
    fun play(clip: File, onFrame: (CameraFrame) -> Unit): ReplayFrameSource.Progress {
        val src = ReplayFrameSource(clip, paced = false)
        var result = ReplayFrameSource.Progress(0, 0, true, "not started")
        val lock = Object()
        var done = false
        src.setConsumer { onFrame(it) }
        src.start { p ->
            synchronized(lock) {
                result = p
                done = true
                lock.notifyAll()
            }
        }
        synchronized(lock) { while (!done) lock.wait() }
        return result
    }

    /**
     * Teaches a Twin from a clip: sheet from the first [sheetFrames] frames, then the teach recording = frames with
     * `t ∈ [window.first, window.last]` (default: the next 12 s). Not saved; the caller decides.
     */
    fun teachFromClip(
        clip: File,
        eng: EngineHolder.State.Ready,
        name: String,
        seg: MaskParams = TwinRepo.SEG,
        window: LongRange? = null,
        sheetFrames: Int = PipelineHub.REPLAY_SHEET_FRAMES,
        exporter: EvalExport? = null,
        exportCropSize: Int = 448,
        /** Teach sampling period; 0 = every frame (the laptop reference for the self-test clip uses every frame). */
        sampleEveryMs: Long = 125L,
        /** false = the spec pipeline (rotation NONE, no fit gate): the self-test regression compares with the laptop reference. */
        accuracy: Boolean = true,
        /** Negative-object globals for τ_id (the live teach passes the negatives library; the self-test none). */
        negatives: List<FloatArray> = emptyList(),
    ): TeachClipResult {
        val analysis = FrameAnalysis(seg, StageTimes())
        var frameW = 0
        var frameH = 0
        val pipeline: PipelineInfo = TwinRepo.runningPipeline(eng, seg, accuracy)
        var cap: SheetCapture? = SheetCapture(analysis, sheetFrames)
        var sheet: SheetModel? = null
        var session: TeachSession? = null
        var t0 = Long.MIN_VALUE
        var lastExportT = Long.MIN_VALUE
        val p = play(clip) { f ->
            val c = cap
            if (c != null) {
                if (c.offer(f)) {
                    sheet = c.finish().sheet
                    cap = null
                    frameW = f.width
                    frameH = f.height
                }
                return@play
            }
            val sh = sheet ?: return@play
            if (window != null && f.tMs < window.first) return@play
            if (window != null && f.tMs > window.last) {
                session?.stopRequested = true
                return@play
            }
            if (t0 == Long.MIN_VALUE) t0 = f.tMs
            val ts = session ?: TeachSession(analysis, sh, pipeline, name, SystemClock.elapsedRealtime(),
                durationMs = window?.let { it.last - it.first + 1 } ?: 12_000L, sampleEveryMs = sampleEveryMs).also { session = it }
            if (!ts.finished) {
                val before = ts.framesSeen
                ts.onFrame(f)
                if (exporter != null && ts.framesSeen > before && f.tMs != lastExportT) {
                    lastExportT = f.tMs
                    // Export every analysed teach frame (export-format §5), the SAME frames the teach session sampled.
                    exporter.teach(analysis.teachExport(analysis.analyse(f, sh), pipeline.inputSize, pipeline.gh, pipeline.gw, exportCropSize))
                }
            }
        }
        if (p.error != null) return TeachClipResult(null, "clip error: ${p.error}", sheet, frameW, frameH)
        val ts = session ?: return TeachClipResult(null, "no teach frames in the clip", sheet, frameW, frameH)
        val built = ts.build({ eng.backbone.embed(it) }, KnnAdapter(eng.knn), negatives) { _, _ -> }
        return built.fold(
            { TeachClipResult(it, "taught ${it.twin.keyframeCount} keyframes", sheet, frameW, frameH) },
            { TeachClipResult(null, it.message ?: "teach failed", sheet, frameW, frameH) },
        )
    }

    /**
     * Judges every presentation of a clip against [twin] (sheet from the clip's first frames, then the live-line session).
     * Returns the judged parts in verdict order.
     */
    fun judgeClip(
        g: AppGraph,
        clip: File,
        twin: TwinModel,
        eng: EngineHolder.State.Ready,
        trigger: TriggerConfig,
        lineFraction: Double,
        seg: MaskParams = TwinRepo.SEG,
        fromMs: Long = Long.MIN_VALUE,
        sheetFrames: Int = PipelineHub.REPLAY_SHEET_FRAMES,
        exportCropSize: Int = 0,
        mode: LineMode = LineMode.INSPECT,
        voting: Boolean? = null,
        screenMotion: String? = null,
    ): Pair<List<JudgedPart>, ClipInfo?> {
        val analysis = FrameAnalysis(seg, StageTimes())
        val judged = ArrayList<JudgedPart>()
        var cap: SheetCapture? = SheetCapture(analysis, sheetFrames)
        var line: LineSession? = null
        var info: ClipInfo? = null
        var frames = 0
        val p = play(clip) { f ->
            frames++
            val c = cap
            if (c != null) {
                if (c.offer(f)) {
                    val r = c.finish()
                    cap = null
                    info = ClipInfo(r.sheet, f.width, f.height, 0)
                    line = LineSession(
                        g, twin, mode, "clip:${clip.name}", eng, analysis, r.sheet, trigger, lineFraction, live = false,
                        stages = StageTimes(), sensitivity = twin.thresholds.sensitivity, publish = {},
                        onJudged = { synchronized(judged) { judged += it } }, exportCropSize = exportCropSize,
                        votingOverride = voting, screenMotion = screenMotion,
                    )
                }
                return@play
            }
            if (f.tMs < fromMs) return@play
            line?.onFrame(f)
        }
        line?.drainAndClose()
        if (p.error != null) throw IllegalStateException("clip ${clip.name}: ${p.error}")
        return synchronized(judged) { judged.sortedBy { it.tMs } } to info?.let { ClipInfo(it.sheet, it.frameW, it.frameH, frames) }
    }

    /** Clip label = the file-name prefix before the first '_' (Clips screen naming: `<label>_<stamp>.mp4`). */
    fun labelOf(clip: File): String = clip.name.substringBefore('_').lowercase()

    @Suppress("unused")
    private fun Trigger.short() = name.first()
}
