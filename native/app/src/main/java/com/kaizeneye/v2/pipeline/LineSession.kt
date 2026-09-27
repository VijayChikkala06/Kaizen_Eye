package com.kaizeneye.v2.pipeline

import android.os.SystemClock
import android.util.Log
import com.kaizeneye.core.explain.Facts
import com.kaizeneye.core.explain.FactsLine
import com.kaizeneye.core.explain.GuardDecision
import com.kaizeneye.core.explain.LocationGuard
import com.kaizeneye.core.explain.TemplateExplainer
import com.kaizeneye.core.explain.VlmPrompt
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.math.Binomial
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.model.Verdict
import com.kaizeneye.core.power.Governor
import com.kaizeneye.core.power.IdleDetector
import com.kaizeneye.core.power.VlmMode
import com.kaizeneye.core.telemetry.LatencyStats
import com.kaizeneye.core.telemetry.StageTimes
import com.kaizeneye.core.track.Axis
import com.kaizeneye.core.track.Detections
import com.kaizeneye.core.track.FiredTrigger
import com.kaizeneye.core.track.JudgeQueue
import com.kaizeneye.core.track.LineCounters
import com.kaizeneye.core.track.Tracker
import com.kaizeneye.core.track.TrackState
import com.kaizeneye.core.track.TrackView
import com.kaizeneye.core.track.Trigger
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.core.twin.CalibrationSample
import com.kaizeneye.core.twin.CropInput
import com.kaizeneye.core.twin.RotationRule
import com.kaizeneye.core.twin.Judgement
import com.kaizeneye.core.twin.TwinModel
import com.kaizeneye.core.twin.VerdictEngine
import com.kaizeneye.runtime.VlmState
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraFrame
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.ui.LineMode
import com.kaizeneye.v2.util.Images
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min

/** One judged presentation (for listeners: replay regression, eval export). */
class JudgedPart(
    val trackId: Int,
    val trigger: Trigger?,
    val tMs: Long,
    val judgement: Judgement,
    val snapshot: CropSnapshot?,
    val latencyMs: Double,
)

/**
 * The live line (plan B4) for one Twin: FAST LOOP (per frame, frame-source thread) = analysis → tracker → best-crop
 * snapshots → triggers → overlay; JUDGE WORKER (one thread, FIFO, once per part) = embed → Twin judge (→ borderline vote)
 * → verdict → alarm / reject card / VLM / log / counters / calibration sample. Works identically on camera and replay
 * frames because everything is driven by frame timestamps.
 */
class LineSession(
    private val g: AppGraph,
    val twin: TwinModel,
    val mode: LineMode,
    val source: String,
    private val engines: EngineHolder.State.Ready,
    private val analysis: FrameAnalysis,
    private val sheet: SheetModel,
    private val triggerConfig: TriggerConfig,
    private val lineFraction: Double,
    private val live: Boolean,
    private val stages: StageTimes,
    @Volatile var sensitivity: Double,
    private val publish: (InspectUi) -> Unit,
    private val onJudged: ((JudgedPart) -> Unit)? = null,
    /** > 0: snapshots also capture the eval-export record (mask RLE + crop JPEG at this size). Replay/export only. */
    private val exportCropSize: Int = 0,
    /** Forces borderline voting on/off (replay regression: off, like the laptop reference); null = prefs + governor. */
    private val votingOverride: Boolean? = null,
    /** "H"/"V" = screen motion; the photo-eye's sensor axis is resolved from the first frame's rotation. null = keep the config axis. */
    private val screenMotion: String? = null,
) {
    private val gh = twin.gh
    private val gw = twin.gw
    private val dim = twin.dim
    private val n = twin.pipeline.inputSize
    private val factor = analysis.factor
    /** Accuracy mode: pose-normalised crops (θ and θ + π). Off for the spec pipeline and for the eval export (axis-aligned crops). */
    private val canonical = twin.pipeline.rotation == RotationRule.CANONICAL && exportCropSize == 0
    private val knn = KnnAdapter(engines.knn)
    private val badge = engines.backbone.report.badge

    private var tracker: Tracker? = null
    private var activeConfig: TriggerConfig = triggerConfig
    private val snapshots = HashMap<Long, CropSnapshot>()
    private val marks = ConcurrentHashMap<Int, Mark>()
    private val judgeExec = Executors.newSingleThreadExecutor { r -> Thread(r, "kz-judge").apply { priority = Thread.NORM_PRIORITY + 2 } }
    private val pending = AtomicInteger(0)
    private val counters = LineCounters()
    val latency = LatencyStats()
    private val latencyLog = ArrayList<Double>()
    private val calib = ArrayList<CalibrationSample>()
    private val governor = Governor()
    private val idle = IdleDetector()
    @Volatile private var votingEnabled = votingOverride ?: g.prefs.voting
    @Volatile private var vlmMode = VlmMode.AUTO
    @Volatile private var banner: String? = null
    @Volatile private var governorLine = if (live) "L0 · 30 fps" else "replay (no governor)"
    private var lastGovernorT = Long.MIN_VALUE
    private var rejectNo = 0
    private var consecutiveRejects = 0
    @Volatile private var lastReject: RejectCard? = null
    @Volatile private var lastRejectFacts: Facts? = null
    private var lastPublishNs = 0L
    private var manyBlobFrames = 0
    private var lastFps = 0.0
    private var fpsT0 = -1L
    private var fpsN = 0
    @Volatile private var closed = false
    @Volatile var lastFrameTms: Long = 0
        private set

    // "Circle to inspect": the circled area (only what is inside is analysed) and a pending operator request.
    @Volatile private var zone: Roi? = null
    @Volatile private var circleRequest: Triple<Roi, Long, (CircleResult) -> Unit>? = null
    private var manual: Triple<Long, Long, (CircleResult) -> Unit>? = null // strokeId, deadline tMs, callback
    private var zoneBefore: Roi? = null

    /**
     * The operator circled (or tapped) something on the preview ([roi] in analysis pixels): judge the part inside it now,
     * and from then on analyse only that area. [onResult] says whether a part was found (for the doodle).
     */
    fun circle(roi: Roi, strokeId: Long, onResult: (CircleResult) -> Unit) {
        circleRequest = Triple(roi, strokeId, onResult)
    }

    /** Back to the whole view. */
    fun clearCircle() {
        circleRequest = null
        zone = null
    }

    private class Mark(
        val state: OverlayState, val label: String, val heat: FloatArray?, val sq: com.kaizeneye.core.model.CropSquare?,
        val cxAtSnap: Double, val cyAtSnap: Double,
        /** Angle (radians) the heat grid is rotated by about the square's centre (canonical crops), 0 for axis-aligned ones. */
        val rot: Double = 0.0,
    )

    private class Job(val fired: FiredTrigger, val snaps: List<CropSnapshot>, val arrivalNs: Long, val cxAtSnap: Double, val cyAtSnap: Double)

    // ------------------------------------------------------------------------------------------------ FAST LOOP
    fun onFrame(frame: CameraFrame) {
        if (closed) return
        val arrival = System.nanoTime()
        val a0 = analysis.analyse(frame, sheet)
        lastFrameTms = a0.tMs
        circleRequest?.let { (roi, id, cb) ->
            circleRequest = null
            manual?.let { (oldId, _, oldCb) -> oldCb(CircleResult(oldId, false, "Replaced by the new circle")) }
            zoneBefore = zone
            zone = roi
            manual = Triple(id, a0.tMs + CIRCLE_WAIT_MS, cb)
        }
        val z = zone
        val pick = z?.let { RoiSelect.apply(a0, it, analysis.params) }
        val a = pick?.analysed ?: a0
        val tr = tracker ?: run {
            val axis = screenMotion?.let { sensorAxisFor(it, a.rotation) } ?: triggerConfig.axis
            activeConfig = triggerConfig.copy(axis = axis, position = lineFraction * if (axis == Axis.X) a.w else a.h)
            Tracker(a.w, a.h, activeConfig)
        }.also { tracker = it }
        val t0 = System.nanoTime()
        // Only part-sized blobs are tracked; detections (and assignments) index into [comps].
        // With a circled area, the circled object is the only candidate (everything outside is ignored).
        val comps = if (pick != null) listOfNotNull(pick.obj?.takeIf { it.area >= analysis.params.minAreaFrac * a.w * a.h })
        else a.partCandidates(analysis.params.minAreaFrac)
        val upd = tr.update(a.tMs, Detections.of(comps, a.grey, a.w, a.h, a.index))
        stages.record("track", (System.nanoTime() - t0) / 1e6)
        manual?.let { (id, deadline, cb) -> manualJudge(a, pick, comps, upd, tr, id, deadline, cb, arrival) }

        for (notice in upd.bestCrops) {
            notice.evictedFrameRef?.let { snapshots.remove(key(notice.trackId, it)) }
            val comp = componentOf(notice.trackId, upd.assignments, comps) ?: continue
            snapshots[key(notice.trackId, notice.frameRef)] = analysis.snapshot(a, comp, n, gh, gw, null, exportCropSize, canonical)
        }
        for (fired in upd.triggers) {
            var snaps = fired.frames.mapNotNull { snapshots[key(fired.trackId, it.frameRef)] }
            val view = upd.tracks.firstOrNull { it.id == fired.trackId }
            val first = fired.frames.firstOrNull()
            if (fired.trigger == Trigger.STEADY && view != null && first != null && first.frameRef == a.index &&
                snapshots[key(fired.trackId, first.frameRef)] == null
            ) {
                // STEADY judges the firing (current) frame; snapshot it now if the best-crop buffer did not keep it.
                componentOf(fired.trackId, upd.assignments, comps)?.let { snaps = listOf(analysis.snapshot(a, it, n, gh, gw, null, exportCropSize, canonical)) + snaps }
            }
            marks[fired.trackId] = Mark(OverlayState.JUDGING, "judging…", null, null, 0.0, 0.0)
            pending.incrementAndGet()
            val job = Job(fired, snaps, arrival, snaps.firstOrNull()?.component?.cx ?: view?.cx ?: 0.0, snaps.firstOrNull()?.component?.cy ?: view?.cy ?: 0.0)
            judgeExec.execute { runJudge(job) }
        }
        for (e in upd.exits) {
            counters.onExit(e.judged)
            val prefix = e.trackId.toLong() shl 32
            snapshots.keys.removeIf { it and (0xFFFFFFFFL shl 32) == prefix }
            marks.remove(e.trackId)
        }
        stages.record("frame", (System.nanoTime() - arrival) / 1e6)

        manyBlobFrames = if (z == null && a.components.size >= MANY_BLOBS) manyBlobFrames + 1 else 0
        if (live) {
            // Slow empty-sheet adaptation (twin-spec §2.1, live only): follow gradual lighting drift while nothing is in view.
            if (a0.components.isEmpty()) sheet.adapt(a.small)
            governorTick(a.tMs, a.components.isNotEmpty())
        }
        countFps(a.tMs)
        maybePublish(a, upd.tracks)
    }

    /**
     * Operator trigger: the circled object is judged on this frame (its track is marked judged so LINE/STEADY do not
     * judge it again). Waits up to [CIRCLE_WAIT_MS] for an object to appear inside the circle, else reports failure and
     * restores the previous area.
     */
    private fun manualJudge(
        a: Analysed, pick: RoiPick?, comps: List<com.kaizeneye.core.model.Component>, upd: com.kaizeneye.core.track.FrameUpdate,
        tr: Tracker, strokeId: Long, deadline: Long, cb: (CircleResult) -> Unit, arrival: Long,
    ) {
        val obj = pick?.obj
        val trackId = if (obj != null && comps.isNotEmpty()) upd.assignments.getOrNull(0) else null
        if (obj == null || trackId == null) {
            if (a.tMs > deadline) {
                manual = null
                zone = zoneBefore
                cb(CircleResult(strokeId, false, "Nothing found inside the circle — circle the part again"))
            }
            return
        }
        manual = null
        val z = zone
        if (z != null && z.isTap) zone = Roi.around(obj, 1.8)
        cb(CircleResult(strokeId, true, "Got it — judging the circled part"))
        if (upd.triggers.any { it.trackId == trackId }) return // a trigger fired for it this very frame
        tr.markJudged(trackId)
        val snap = analysis.snapshot(a, obj, n, gh, gw, null, exportCropSize, canonical)
        val fired = FiredTrigger(trackId, Trigger.STEADY, a.tMs, listOf(com.kaizeneye.core.track.BufferedFrame(a.index, 0.0, a.tMs)))
        marks[trackId] = Mark(OverlayState.JUDGING, "judging…", null, null, 0.0, 0.0)
        pending.incrementAndGet()
        val job = Job(fired, listOf(snap), arrival, obj.cx, obj.cy)
        judgeExec.execute { runJudge(job) }
    }

    private fun key(trackId: Int, frameRef: Long): Long = (trackId.toLong() shl 32) or (frameRef and 0xFFFFFFFFL)

    /** The component track [trackId] was assigned this frame (detections are the components in order). */
    private fun componentOf(trackId: Int, assignments: IntArray, comps: List<com.kaizeneye.core.model.Component>) =
        assignments.indexOfFirst { it == trackId }.takeIf { it >= 0 }?.let { comps[it] }

    private fun governorTick(tMs: Long, hasComponent: Boolean) {
        val isIdle = idle.update(tMs, hasComponent)
        if (lastGovernorT != Long.MIN_VALUE && tMs - lastGovernorT < 1000) return
        lastGovernorT = tMs
        val th = g.thermal.latest.value
        val st = governor.update(tMs, th.status, th.headroom, isIdle)
        votingEnabled = votingOverride ?: (st.votingEnabled && g.prefs.voting)
        vlmMode = st.vlmMode
        // Headroom-only L2 (status still below SEVERE) is "near the limit", not "hot": say what was measured.
        banner = st.banner?.let {
            if (th.status < 3) String.format(java.util.Locale.ROOT, "Near the thermal limit (headroom %.2f) — slowed to %d fps, explanations paused", th.headroom, st.fps) else it
        }
        governorLine = "${st.level} · ${st.fps} fps${if (isIdle) " (idle)" else ""} · ${th.statusName}"
        if (st.changed) g.camera.setAnalysisFpsCap(st.fps)
        if (st.levelChanged) g.log.write("governor", JSONObject().put("level", st.level.name).put("fps", st.fps).put("voting", st.votingEnabled).put("vlm", st.vlmMode.name).put("reason", st.reason).put("headroom", th.headroom.toDouble()).put("batteryC", th.batteryTempC).put("status", th.statusName))
    }

    private fun countFps(tMs: Long) {
        if (fpsT0 < 0) fpsT0 = tMs
        fpsN++
        if (tMs - fpsT0 >= 1000) {
            lastFps = (fpsN - 1) * 1000.0 / max(1L, tMs - fpsT0)
            fpsT0 = tMs
            fpsN = 1
        }
    }

    private fun maybePublish(a: Analysed, tracks: List<TrackView>) {
        val now = System.nanoTime()
        if (now - lastPublishNs < 50_000_000L) return
        lastPublishNs = now
        val lat = if (live) DISPLAY_LATENCY_MS else 0.0
        // Only plausible parts are drawn (confirmed, fully inside the view, big enough) plus anything already judged:
        // specks, hands and background edges touching the border are tracked but not shown (they were the "boxes everywhere").
        val minArea = analysis.params.minAreaFrac * a.w * a.h
        val shown = tracks.filter { v -> marks.containsKey(v.id) || (v.state != TrackState.TENTATIVE && !v.touchesBorder && v.area >= minArea) }
        val boxes = shown.map { v ->
            val m = marks[v.id]
            val dx = v.vx * lat
            val dy = v.vy * lat
            val st = m?.state ?: if (v.judged) OverlayState.JUDGING else OverlayState.TRACKING
            val heat = if (m?.heat != null && m.sq != null) {
                // The heat map rides the track: shift the judged crop square by the track's motion since the snapshot.
                val ox = (v.cx + dx - m.cxAtSnap) * factor
                val oy = (v.cy + dy - m.cyAtSnap) * factor
                HeatOverlay(gh, gw, m.heat, (m.sq.x0 + ox).toFloat(), (m.sq.y0 + oy).toFloat(), m.sq.side.toFloat(), m.rot.toFloat())
            } else null
            OverlayBox(
                v.id,
                ((v.minX + dx) * factor).toFloat(), ((v.minY + dy) * factor).toFloat(),
                ((v.maxX + 1 + dx) * factor).toFloat(), ((v.maxY + 1 + dy) * factor).toFloat(),
                st, m?.label, heat,
            )
        }
        val t = tracker
        val lineUi = t?.let { LineOverlay(activeConfig.axis.name, (it.linePosition * factor).toFloat()) }
        publish(
            InspectUi(
                running = true, mode = mode.name, source = source, twinName = twin.name,
                analysisW = a.fullW, analysisH = a.fullH, rotation = a.rotation,
                boxes = boxes, line = if (triggerConfig.lineEnabled) lineUi else null,
                counters = countersUi(a.tMs), fps = lastFps,
                loopMsP95 = stages.stats("frame").p95().let { if (it.isNaN()) 0.0 else it },
                latencyP50 = latency.p50().takeUnless { it.isNaN() }, latencyP95 = latency.p95().takeUnless { it.isNaN() },
                accel = badge, governor = governorLine,
                lineTooFast = JudgeQueue.lineTooFast(pending.get(), triggerConfig.maxQueuedJobs),
                banner = banner, reject = lastReject, calibration = calibrationUi(), sensitivity = sensitivity.toFloat(),
                hint = if (zone != null && comps0Empty(tracks, minArea)) "Put the part inside the circled area (or tap Whole view)." else hintFor(tracks, minArea),
                circled = zone != null,
            ),
        )
    }

    /** What the operator should do next, from what the fast loop sees (plain language, shown over the preview). */
    private fun comps0Empty(tracks: List<TrackView>, minArea: Double) = tracks.none { it.confirmed && it.area >= minArea }

    private fun hintFor(tracks: List<TrackView>, minArea: Double): String? {
        if (manyBlobFrames >= MANY_BLOBS_FRAMES) {
            return "Too many blobs: the background is not plain or the light changed. Clear the sheet and tap RE-LEARN SHEET."
        }
        val parts = tracks.filter { it.confirmed && !it.touchesBorder && it.area >= minArea }
        if (parts.isEmpty()) {
            val edge = tracks.any { it.confirmed && it.touchesBorder && it.area >= minArea }
            return when {
                edge -> "Something touches the edge of the view (a hand, or the part too close to the edge). Parts must be fully inside the view."
                activeConfig.lineEnabled && !activeConfig.steadyEnabled -> "Slide one part at a time across the dashed line."
                activeConfig.steadyEnabled && !activeConfig.lineEnabled -> "Put one part on the sheet and take your hand away — or circle it with your finger."
                else -> "Put one part on the sheet and take your hand away — or circle it with your finger to judge it now."
            }
        }
        if (parts.all { it.judged }) return null
        return when {
            activeConfig.steadyEnabled && activeConfig.lineEnabled -> "Judging when the part crosses the dashed line or stays still for half a second."
            activeConfig.steadyEnabled -> "Keep it still for half a second."
            else -> "Slide it across the dashed line."
        }
    }

    private fun countersUi(tMs: Long) = CountersUi(
        judged = counters.judged, pass = counters.pass, defect = counters.defect, notEnrolled = counters.notEnrolled,
        reframe = counters.reframe, exitedUnjudged = counters.exitedUnjudged, ppm = counters.partsPerMinute(tMs),
    )

    private fun calibrationUi(): CalibrationUi? {
        if (mode != LineMode.CALIBRATE) return null
        val s = synchronized(calib) { calib.toList() }
        val valid = s.count { it.valid }
        return CalibrationUi(
            presented = s.size, valid = valid, target = CALIBRATION_TARGET,
            sanityOk = s.count { it.sanityOk }, identityOk = s.count { it.sanityOk && it.idOk }, geometryOk = s.count { it.sanityOk && it.geoOk },
            alphaPct = if (valid > 0) 100.0 * Binomial.orderStatisticAlpha(valid) else null,
            lastRawOverTau = s.lastOrNull { it.raw != null }?.raw?.let { it / twin.thresholds.tau },
        )
    }

    // ------------------------------------------------------------------------------------------------ JUDGE WORKER
    private fun runJudge(job: Job) {
        try {
            judge(job)
        } catch (t: Throwable) {
            Log.e(TAG, "judge failed", t)
            g.log.write("error", JSONObject().put("where", "judge").put("error", t.toString()))
        } finally {
            pending.decrementAndGet()
        }
    }

    private fun judge(job: Job) {
        val judgeStart = System.nanoTime()
        val snaps = job.snaps
        val s0 = snaps.firstOrNull()
        var embedMs = Double.NaN
        var used = s0
        val canon = s0?.canon
        var judgement: Judgement = when {
            s0 == null -> VerdictEngine.reframe(SanityReason.TOUCHES_BORDER, gh, gw)
            s0.sanity != SanityReason.OK -> VerdictEngine.reframe(s0.sanity, gh, gw)
            canon != null && canonical -> {
                val te = System.nanoTime()
                val a = cropInput(s0.crop, s0.cov)
                val b = cropInput(canon.altCrop, canon.altCov)
                embedMs = (System.nanoTime() - te) / 1e6
                stages.record("embed", embedMs)
                val tj = System.nanoTime()
                val j = twin.judgeCanonical(a, b, s0.geometry, s0.sanity, knn, sensitivity)
                stages.record("knn+score", (System.nanoTime() - tj) / 1e6)
                j
            }
            else -> {
                val te = System.nanoTime()
                val f = engines.backbone.embed(s0.crop)
                embedMs = (System.nanoTime() - te) / 1e6
                stages.record("embed", embedMs)
                val tj = System.nanoTime()
                val j = twin.judge(twin.pipeline.prepareFeatures(FeatureMap(gh, gw, dim, f)), s0.cov, s0.geometry, s0.sanity, knn, sensitivity)
                stages.record("knn+score", (System.nanoTime() - tj) / 1e6)
                j
            }
        }
        val firstLatency = (System.nanoTime() - job.arrivalNs) / 1e6
        var orientation = judgement.orientation
        if (s0 != null && votingEnabled && VerdictEngine.needsVote(judgement) && snaps.size > 1) {
            val extras = snaps.drop(1).filter { it.sanity == SanityReason.OK }.take(2)
            if (extras.isNotEmpty()) {
                if (canonical && extras.all { it.canon != null }) {
                    val pairs = extras.map { cropInput(it.crop, it.cov) to cropInput(it.canon!!.altCrop, it.canon.altCov) }
                    val vote = twin.voteCanonical(judgement, pairs, knn, sensitivity)
                    judgement = vote.judgement
                    used = if (vote.chosen == 0) s0 else extras.getOrNull(vote.chosen - 1) ?: s0
                    // The voted crop's own orientation is not tracked; keep the first crop's (the heat map stays on the primary).
                    if (vote.chosen != 0) orientation = 0
                } else {
                    val inputs = extras.map { CropInput(twin.pipeline.prepareFeatures(FeatureMap(gh, gw, dim, engines.backbone.embed(it.crop))), it.cov) }
                    val vote = twin.vote(judgement, inputs, knn, sensitivity)
                    judgement = vote.judgement
                    used = if (vote.chosen == 0) s0 else extras.getOrNull(vote.chosen - 1) ?: s0
                }
            }
        }
        // Latency claim = the first verdict (voting re-scores are excluded from the latency claim, plan).
        latency.add(firstLatency)
        synchronized(latencyLog) { latencyLog += firstLatency }
        stages.record("trigger→verdict", firstLatency)
        val tMs = job.fired.tMs
        counters.onVerdict(tMs, judgement.verdict)
        val isReject = judgement.verdict == Verdict.DEFECT || judgement.verdict == Verdict.NOT_ENROLLED
        consecutiveRejects = if (isReject) consecutiveRejects + 1 else if (judgement.verdict == Verdict.PASS) 0 else consecutiveRejects
        val label = when (judgement.verdict) {
            Verdict.PASS -> "PASS"
            Verdict.DEFECT -> "REJECT #%03d".format(rejectNo + 1)
            Verdict.NOT_ENROLLED -> "NOT THE PART #%03d".format(rejectNo + 1)
            Verdict.REFRAME -> "REFRAME"
        }
        val heat = judgement.smoothed?.let { sm -> heatValues(sm, judgement.tau * judgement.sensitivity) }
        marks[job.fired.trackId] = Mark(
            stateOf(judgement.verdict), label, if (judgement.verdict == Verdict.DEFECT) heat else null, used?.square, job.cxAtSnap, job.cyAtSnap,
            rot = used?.canon?.angle(orientation) ?: 0.0,
        )

        if (isReject) {
            rejectNo++
            if (live) g.alarm.reject(consecutiveRejects)
            publishReject(judgement, used, rejectNo, orientation)
        }
        if (mode == LineMode.CALIBRATE) {
            val sane = judgement.verdict != Verdict.REFRAME
            // The FIT gate is an identity-type gate: a good part it rejects counts with the identity rejections of the certificate.
            val sample = CalibrationSample(
                sanityOk = sane, idOk = sane && judgement.identityOk && judgement.fitOk, geoOk = sane && judgement.geometryOk,
                raw = if (sane) judgement.raw else null, sim = if (sane) judgement.sim else null, geometry = if (sane) used?.geometry else null,
                fit = if (sane) judgement.fit.takeUnless { it.isNaN() } else null,
            )
            synchronized(calib) { calib += sample }
            if (sample.valid && live) g.alarm.tick()
        }
        g.log.write(
            "verdict",
            JSONObject().put("source", source).put("mode", mode.name).put("twin", twin.id).put("trackId", job.fired.trackId)
                .put("trigger", job.fired.trigger.name).put("tMs", tMs).put("verdict", judgement.verdict.name).put("reason", judgement.reason)
                .put("s", judgement.s.jsonSafe()).put("raw", judgement.raw.jsonSafe()).put("tau", judgement.tau.jsonSafe())
                .put("sensitivity", judgement.sensitivity.jsonSafe()).put("sim", judgement.sim.jsonSafe()).put("tauId", judgement.tauId.jsonSafe())
                .put("areaPct", judgement.areaPct.jsonSafe()).put("peakRow", judgement.peakRow).put("peakCol", judgement.peakCol)
                .put("fit", judgement.fit.jsonSafe()).put("tauFit", judgement.tauFit.jsonSafe()).put("orientation", orientation)
                .put("geometryFailures", judgement.geometryFailures.joinToString()).put("votes", judgement.votes)
                .put("latencyMs", firstLatency).put("embedMs", embedMs.jsonSafe()).put("judgeMs", (System.nanoTime() - judgeStart) / 1e6)
                .put("accel", badge).put("queue", pending.get()),
        )
        onJudged?.invoke(JudgedPart(job.fired.trackId, job.fired.trigger, tMs, judgement, used, firstLatency))
    }

    private fun stateOf(v: Verdict) = when (v) {
        Verdict.PASS -> OverlayState.PASS
        Verdict.DEFECT -> OverlayState.DEFECT
        Verdict.NOT_ENROLLED -> OverlayState.NOT_ENROLLED
        Verdict.REFRAME -> OverlayState.REFRAME
    }

    /** Heat 0..1: 0 at ≤ 0.6·τ, 1 at ≥ 1.2·τ (only the hot end is drawn). NaN (outside S) → 0. */
    private fun heatValues(sm: DoubleArray, tauEff: Double): FloatArray = FloatArray(sm.size) { i ->
        val v = sm[i]
        if (v.isNaN() || tauEff <= 0) 0f else (((v / tauEff) - 0.6) / 0.6).coerceIn(0.0, 1.0).toFloat()
    }

    /** Backbone features of one crop in pipeline space (DINOv2: patch-normalised). */
    private fun cropInput(crop: ByteArray, cov: FloatArray): CropInput =
        CropInput(twin.pipeline.prepareFeatures(FeatureMap(gh, gw, dim, engines.backbone.embed(crop))), cov)

    private fun publishReject(j: Judgement, snap: CropSnapshot?, number: Int, orientation: Int = 0) {
        val facts = Facts(
            partName = twin.name, verdict = j.verdict, reason = j.reason,
            peakRow = j.peakRow.takeIf { it >= 0 }, peakCol = j.peakCol.takeIf { it >= 0 }, gh = gh, gw = gw,
            areaPct = j.areaPct.takeUnless { it.isNaN() }, score = j.s.takeUnless { it.isNaN() }, tau = j.tau.takeUnless { it.isNaN() },
            rejectNumber = number, accelLabel = badge,
        )
        val cropFile = snap?.let {
            try {
                val bytes = if (orientation == 1 && it.canon != null) it.canon.altCrop else it.crop
                Images.writeBoxedCrop(bytes, it.n, max(0, j.peakRow), max(0, j.peakCol), gh, gw, File(g.dirs.vlmCache, "reject_%03d.jpg".format(number % 1000)))
            } catch (t: Throwable) {
                null
            }
        }
        val template = TemplateExplainer.sentence(facts)
        val vlmReady = g.vlm.state.value is VlmState.Ready
        val auto = vlmReady && g.prefs.vlmAuto && vlmMode == VlmMode.AUTO && j.verdict == Verdict.DEFECT && cropFile != null
        val source = when {
            auto -> "offline VLM (working…)"
            vlmReady && vlmMode == VlmMode.ON_TAP -> "tap to explain"
            else -> "template"
        }
        val card = RejectCard(number, "${if (j.verdict == Verdict.DEFECT) "REJECT" else "NOT THE ENROLLED PART"} #%03d".format(number), FactsLine.render(facts), template, source, cropFile, System.currentTimeMillis())
        lastReject = card
        lastRejectFacts = facts
        if (auto) explain(card, facts)
    }

    /** Runs the offline VLM on the boxed crop; the location guard keeps it honest, the template is the fallback. */
    fun explain() {
        val card = lastReject ?: return
        val facts = lastRejectFacts ?: return
        explain(card, facts)
    }

    private fun explain(card: RejectCard, facts: Facts) {
        val crop = card.cropFile ?: return
        g.scope.launch {
            val r = g.vlm.explain(crop, VlmPrompt.build(facts), timeoutMs = 4000)
            val text = r.text?.trim().orEmpty()
            val (sentence, src) = if (text.isNotEmpty()) {
                val guard = LocationGuard.check(text, facts)
                if (guard.decision == GuardDecision.ACCEPT) text to "${r.backend} · ${r.ms} ms"
                else card.sentence to "template (VLM said '${guard.offendingWord}', guard rejected)"
            } else card.sentence to "template (${r.error ?: "no VLM text"})"
            g.log.write("vlm", JSONObject().put("reject", card.number).put("backend", r.backend).put("ms", r.ms).put("text", r.text).put("error", r.error).put("accepted", src.startsWith(r.backend)))
            if (lastReject?.number == card.number) lastReject = card.copy(sentence = sentence, sentenceSource = src)
        }
    }

    // ------------------------------------------------------------------------------------------------ calibration
    fun calibrationSamples(): List<CalibrationSample> = synchronized(calib) { calib.toList() }

    /** Trigger→verdict samples of this session (for the certificate's latency line). */
    fun latencySamples(): DoubleArray = synchronized(latencyLog) { latencyLog.toDoubleArray() }

    /** Waits for queued judge jobs (replay / self-test), then stops the worker. */
    fun drainAndClose(timeoutMs: Long = 60_000) {
        closed = true
        judgeExec.shutdown()
        judgeExec.awaitTermination(timeoutMs, TimeUnit.MILLISECONDS)
    }

    fun close() {
        closed = true
        judgeExec.shutdownNow()
    }

    val judgedCount: Int get() = counters.judged
    val pendingJobs: Int get() = pending.get()

    companion object {
        private const val TAG = "KaizenLine"

        /**
         * Sensor axis whose coordinate changes when a part moves [screenMotion] ("H" = left-right on screen) in a frame
         * displayed with [rotation]: rotated frames (90/270 degrees, the portrait camera) swap the axes.
         */
        fun sensorAxisFor(screenMotion: String, rotation: Int): Axis {
            val swapped = ((rotation % 180) + 180) % 180 == 90
            val horizontal = screenMotion != "V"
            return if (horizontal != swapped) Axis.X else Axis.Y
        }
        const val CALIBRATION_TARGET = 40
        /** How long a circle waits for an object inside it before it reports "nothing found". */
        const val CIRCLE_WAIT_MS = 1500L
        /** Components per frame that mean "the background model is wrong", and for how many consecutive frames. */
        const val MANY_BLOBS = 6
        const val MANY_BLOBS_FRAMES = 30
        /** Camera-to-screen delay used to push boxes forward along the track (plan "overlay extrapolation"); tuned on the phone. */
        const val DISPLAY_LATENCY_MS = 60.0
        private fun Double.jsonSafe(): Any = if (isNaN() || isInfinite()) JSONObject.NULL else this
        @Suppress("unused") private fun clampF(v: Double) = min(1.0, max(0.0, v))
    }
}
