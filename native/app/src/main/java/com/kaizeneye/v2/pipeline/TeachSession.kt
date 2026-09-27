package com.kaizeneye.v2.pipeline

import android.os.SystemClock
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.math.Stats
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.twin.CertificateText
import com.kaizeneye.core.twin.PipelineInfo
import com.kaizeneye.core.twin.TeachBuilder
import com.kaizeneye.core.twin.TeachFrame
import com.kaizeneye.core.twin.TeachParams
import com.kaizeneye.core.twin.TeachResult
import com.kaizeneye.core.twin.TwinMeta
import com.kaizeneye.core.twin.TwinModel
import com.kaizeneye.v2.camera.CameraFrame
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.util.Images
import java.util.Locale
import kotlin.math.abs
import kotlin.math.hypot

/**
 * Teach (plan "Teach flow", twin-spec §7): records [durationMs] of frames at ~8 fps, keeps a snapshot of every sane frame,
 * then [build] embeds them and runs [TeachBuilder]. The recording half runs on the frame-source thread; [build] on a
 * background thread. Works on camera frames and on replayed clips alike.
 */
class TeachSession(
    private val analysis: FrameAnalysis,
    private val sheet: SheetModel,
    private val pipeline: PipelineInfo,
    val name: String,
    private val tapElapsedMs: Long,
    val durationMs: Long = 12_000L,
    private val sampleEveryMs: Long = 125L,
    private val seg: com.kaizeneye.core.mask.MaskParams = analysis.params,
    /** The circled part (operator drew around it on the preview), read every frame; null = the §2.5 main object. */
    private val circle: () -> RoiFollower? = { null },
) {
    /** Live box + reason of the last analysed frame, for the Teach screen overlay. */
    @Volatile var lastView: InspectUi? = null
        private set
    class Captured(val tMs: Long, val sanity: SanityReason, val snap: CropSnapshot?)

    private val captured = ArrayList<Captured>()
    private var t0 = Long.MIN_VALUE
    private var lastSample = Long.MIN_VALUE
    @Volatile var stopRequested = false
    @Volatile var finished = false
        private set
    private var seen = 0
    private val views = ArrayList<DoubleArray>()
    private val sharpHistory = ArrayList<Double>()
    var lastHint: String? = null
        private set
    var lastSanity: SanityReason = SanityReason.NO_OBJECT
        private set

    val elapsedMs: Long get() = if (t0 == Long.MIN_VALUE) 0 else (lastSample - t0).coerceAtLeast(0)
    val accepted: Int get() = captured.count { it.snap != null }
    val framesSeen: Int get() = seen

    /** Distinct-view proxy for the coverage ring: 16 views (orientation / position / size changes) = 100 %. */
    val coverage: Float get() = (views.size / TARGET_VIEWS.toFloat()).coerceAtMost(1f)

    /** Returns true once recording is complete (time is up or stop was requested after the minimum). */
    fun onFrame(frame: CameraFrame): Boolean {
        if (finished) return true
        if (t0 == Long.MIN_VALUE) t0 = frame.tMs
        val el = frame.tMs - t0
        if (el >= durationMs || (stopRequested && el >= MIN_DURATION_MS)) {
            finished = true
            return true
        }
        if (lastSample != Long.MIN_VALUE && frame.tMs - lastSample < sampleEveryMs) return false
        lastSample = frame.tMs
        seen++
        val a0 = analysis.analyse(frame, sheet)
        val pick = circle()?.pick(a0, seg)
        val a = pick?.analysed ?: a0
        lastView = LiveView.of(a, seg, pick)
        val (sanity, snap) = if (pick != null) analysis.chosenSnapshot(a, pick.obj, pipeline.inputSize, pipeline.gh, pipeline.gw)
        else analysis.mainSnapshot(a, pipeline.inputSize, pipeline.gh, pipeline.gw)
        lastSanity = sanity
        captured += Captured(a.tMs, sanity, snap)
        lastHint = if (pick != null && pick.obj == null) "Keep the part inside your circle" else hint(sanity, snap)
        return false
    }

    private fun hint(sanity: SanityReason, snap: CropSnapshot?): String? {
        if (snap == null) return when (sanity) {
            SanityReason.NO_OBJECT -> "Put the part on the sheet"
            SanityReason.TOUCHES_BORDER -> "Keep the part fully in view"
            SanityReason.TOO_SMALL -> "Move closer / use a bigger part"
            SanityReason.TOO_LARGE -> "Move the phone further away"
            SanityReason.MULTIPLE -> "Only one part at a time"
            SanityReason.NO_CORE -> "Part too small for the grid — move closer"
            SanityReason.OK -> null
        }
        // Coverage: a new "view" = orientation changed by > 15°, or centre moved > 12 % of the size, or area changed > 15 %.
        val c = snap.component
        val size = hypot(c.width.toDouble(), c.height.toDouble())
        val v = doubleArrayOf(snap.theta, c.cx, c.cy, c.area.toDouble(), size)
        val novel = views.none { o ->
            val dTheta = abs(angleDiff(o[0], v[0]))
            dTheta < Math.toRadians(15.0) && hypot(o[1] - v[1], o[2] - v[2]) < 0.12 * o[4] && abs(o[3] - v[3]) < 0.15 * o[3]
        }
        if (novel) views += v
        sharpHistory += snap.sharpness
        val median = if (sharpHistory.size >= 5) Stats.median(sharpHistory.toDoubleArray()) else Double.NaN
        return when {
            !median.isNaN() && snap.sharpness < 0.5 * median -> "Slow down — the part is blurred"
            elapsedMs > 4000 && views.size < 3 -> "Turn the part slowly"
            else -> null
        }
    }

    private fun angleDiff(a: Double, b: Double): Double {
        // Principal axes are 180°-periodic.
        var d = (a - b) % Math.PI
        if (d > Math.PI / 2) d -= Math.PI
        if (d < -Math.PI / 2) d += Math.PI
        return d
    }

    class Built(val twin: TwinModel, val result: TeachResult.Success, val keyframeJpegs: List<ByteArray>, val tapToArmedMs: Long, val lines: List<String>)

    /**
     * Embeds every sane frame and builds the Twin (twin-spec §7). [embed] = the runtime backbone (crop bytes → features),
     * [knn] = the backend used for the leave-segment-out maps (the phone uses the LiteRT graph, like judging does).
     */
    fun build(
        embed: (ByteArray) -> FloatArray,
        knn: com.kaizeneye.core.twin.KnnBackend,
        negatives: List<FloatArray>,
        progress: (String, Float) -> Unit,
    ): Result<Built> = runCatching {
        val sane = captured.filter { it.snap != null }
        val frames = ArrayList<TeachFrame>(captured.size)
        var done = 0
        for (c in captured) {
            val s = c.snap
            if (s == null) {
                frames += TeachFrame(c.tMs, c.sanity, Double.NaN, null, null, null)
                continue
            }
            val f = embed(s.crop)
            frames += TeachFrame(c.tMs, s.sanity, cropSharpness(s), s.cov, s.geometry, pipeline.prepareFeatures(FeatureMap(pipeline.gh, pipeline.gw, pipeline.dim, f)))
            done++
            progress("embedding ${done}/${sane.size}", 0.1f + 0.6f * done / maxOf(1, sane.size))
        }
        progress("keyframes, memory bank, leave-segment-out τ", 0.75f)
        val meta = TwinMeta.create(name)
        val r = TeachBuilder.build(frames, pipeline, meta, TeachParams.forPipeline(pipeline).copy(threads = TEACH_THREADS), negatives, knn)
        when (r) {
            is TeachResult.Failure -> error(failureText(r))
            is TeachResult.Success -> {
                progress("saving", 0.95f)
                val jpegs = r.diagnostics.keyframeFrames.map { fi ->
                    val s = captured[fi].snap ?: error("keyframe $fi has no crop")
                    Images.jpeg(s.crop, s.n, s.n, 95)
                }
                val tw = r.twin
                val th = tw.thresholds
                val lines = listOf(
                    "keyframes ${tw.keyframeCount} from ${r.diagnostics.kept.size} sharp / ${r.diagnostics.accepted.size} usable frames",
                    "bank ${tw.bankRows} rows (coreset of ${r.diagnostics.pooledRows})",
                    String.format(Locale.ROOT, "τ %.3f (1.4 × max LSO over %d segments)", th.tau, r.diagnostics.segmentsUsed),
                    String.format(Locale.ROOT, "τ_id %.3f (%s)", th.tauId, th.tauIdRule),
                    tw.withinPartLine(),
                    "build ${r.diagnostics.timingsMs.values.sum().toInt()} ms",
                )
                Built(tw, r, jpegs, SystemClock.elapsedRealtime() - tapElapsedMs, lines)
            }
        }
    }

    /** Spec §7 step 2: sharpness of an accepted frame = variance of the Laplacian on the grey N×N crop. */
    private fun cropSharpness(s: CropSnapshot): Double {
        val img = com.kaizeneye.core.image.RgbImage(s.n, s.n, s.crop)
        val grey = com.kaizeneye.core.image.ImageOps.grey(img)
        return com.kaizeneye.core.image.ImageOps.laplacianVariance(grey, s.n, s.n, 0, 0, s.n, s.n)
    }

    private fun failureText(f: TeachResult.Failure): String = when (f.reason.name) {
        "NOT_ENOUGH_FRAMES" -> "Not enough sharp frames of the part (${f.diagnostics.accepted.size} usable). Keep the part fully in view, move slowly, and record again."
        "TOO_FEW_KEYFRAMES" -> "The recording shows too few different views. Turn the part slowly while recording."
        else -> "Teach failed (${f.reason}). Record again."
    }

    /** Crop snapshots of the recording (eval export of teach frames). */
    fun capturedFrames(): List<Captured> = captured

    companion object {
        const val TARGET_VIEWS = 16
        /** Parallel teach loops (results are identical for any thread count). */
        const val TEACH_THREADS = 4
        const val MIN_DURATION_MS = 6_000L

        @Suppress("unused")
        fun describeCertificateBefore(tw: TwinModel) = CertificateText.withinPart(tw.segmentsUsed)

        @Suppress("unused")
        private fun EngineHolder.State.Ready.describe() = backbone.report.badge
    }
}
