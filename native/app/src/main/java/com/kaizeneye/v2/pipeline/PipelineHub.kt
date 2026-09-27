package com.kaizeneye.v2.pipeline

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.telemetry.StageTimes
import com.kaizeneye.core.track.Axis
import com.kaizeneye.core.track.Direction
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.core.twin.CertificateText
import com.kaizeneye.core.twin.TwinLoadResult
import com.kaizeneye.core.twin.TwinModel
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraFrame
import com.kaizeneye.v2.camera.FrameConsumer
import com.kaizeneye.v2.camera.ReplayFrameSource
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.selftest.SelfTestRunner
import com.kaizeneye.v2.twin.TwinRepo
import com.kaizeneye.v2.ui.LineMode
import com.kaizeneye.v2.util.Images
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.util.Locale

/**
 * Orchestrates sheet capture, teach, inspect / calibrate / replay, negatives and the self-test for the UI. Camera frames
 * arrive on the camera thread through [router] and go to whichever session is active; replay frames go through the same
 * session classes on the replay thread, so REPLAY is the identical pipeline.
 */
class PipelineHub(val g: AppGraph) {
    val stages = StageTimes()
    private val seg = TwinRepo.SEG
    private val camAnalysis = FrameAnalysis(seg, stages)
    val repo by lazy { TwinRepo(g.dirs) }

    private val _sheet = MutableStateFlow(SheetUi())
    val sheet: StateFlow<SheetUi> = _sheet
    private val _twins = MutableStateFlow<List<TwinSummary>>(emptyList())
    val twins: StateFlow<List<TwinSummary>> = _twins
    private val _active = MutableStateFlow<TwinSummary?>(null)
    val active: StateFlow<TwinSummary?> = _active
    private val _teach = MutableStateFlow<TeachUi>(TeachUi.Idle)
    val teach: StateFlow<TeachUi> = _teach
    private val _inspect = MutableStateFlow(InspectUi())
    val inspect: StateFlow<InspectUi> = _inspect
    private val _negatives = MutableStateFlow(NegativesUi(0, 0, null, null))
    val negatives: StateFlow<NegativesUi> = _negatives
    private val _selfTest = MutableStateFlow(SelfTestUi())
    val selfTest: StateFlow<SelfTestUi> = _selfTest

    @Volatile var sheetModel: SheetModel? = null
        private set
    @Volatile var activeTwin: TwinModel? = null
        private set

    private enum class Mode { IDLE, SHEET, PREVIEW, TEACH, LINE, NEGATIVES }

    @Volatile private var mode = Mode.IDLE
    @Volatile private var sheetCapture: SheetCapture? = null
    @Volatile private var afterSheet: (() -> Unit)? = null
    /** Frames skipped after unlocking auto exposure, so AE/AWB/AF settle on the empty sheet before it is learned. */
    @Volatile private var sheetSettle = 0
    /** Inspect/calibrate mode waiting for the LEARN SHEET step. */
    @Volatile private var pendingLineMode: LineMode? = null
    /** Teach screen is open and idle: show the live "what the camera sees" box after the sheet is learned. */
    @Volatile private var previewWanted = false
    private var previewLastMs = Long.MIN_VALUE
    private val _teachView = MutableStateFlow(InspectUi())
    val teachView: StateFlow<InspectUi> = _teachView
    @Volatile private var teachSession: TeachSession? = null
    /** Teach: the circled part (followed while it is turned); null = the biggest object in view is used. */
    @Volatile private var teachCircle: RoiFollower? = null
    /** Teach: a circle drawn but not yet checked against a frame (checked on the next analysed frame). */
    @Volatile private var pendingTeachCircle: Pair<Roi, Long>? = null
    private val _circle = MutableStateFlow<CircleResult?>(null)
    /** Result of the last circle gesture (the doodle turns green / red, then fades). */
    val circle: StateFlow<CircleResult?> = _circle
    @Volatile private var lineSession: LineSession? = null
    @Volatile private var negativesSession: NegativesSession? = null
    @Volatile private var replay: ReplayFrameSource? = null
    @Volatile private var replayPreview: Bitmap? = null

    val selfTestRunner = SelfTestRunner(g)

    init {
        g.scope.launch {
            g.engines.state.collect { st ->
                when (st) {
                    is EngineHolder.State.Ready -> refreshTwins()
                    // The old backbone / k-NN are closed once the new one is up: nothing may keep judging with them.
                    is EngineHolder.State.Loading -> onEngineReloading()
                    else -> Unit
                }
            }
        }
        selfTestRunner.replayStep = { out -> ReplayRegression(g, this).run(out) }
        selfTestRunner.coreStep = { out -> CoreOnDevice(g).run(out) }
    }

    /** Frame entry point for the camera (CameraPreview sets it as the CameraController consumer). */
    val router = FrameConsumer { f -> onCameraFrame(f) }

    private fun onCameraFrame(f: CameraFrame) {
        try {
            when (mode) {
                Mode.SHEET -> sheetCapture?.let { sc ->
                    if (sheetSettle > 0) {
                        sheetSettle--
                        return@let
                    }
                    val done = sc.offer(f)
                    _sheet.value = _sheet.value.copy(progress = sc.progress)
                    if (done && sheetCapture === sc) finishSheet(sc)
                }
                Mode.PREVIEW -> sheetModel?.let { sh ->
                    val pc = pendingTeachCircle
                    if (pc == null && previewLastMs != Long.MIN_VALUE && f.tMs - previewLastMs < 150) return@let
                    previewLastMs = f.tMs
                    val a0 = camAnalysis.analyse(f, sh)
                    if (pc != null) registerTeachCircle(a0, pc)
                    val pick = teachCircle?.pick(a0, seg)
                    _teachView.value = LiveView.of(pick?.analysed ?: a0, seg, pick)
                }
                Mode.TEACH -> teachSession?.let { onTeachFrame(it, f) }
                Mode.LINE -> lineSession?.onFrame(f)
                Mode.NEGATIVES -> negativesSession?.onFrame(f)
                Mode.IDLE -> Unit
            }
        } catch (t: Throwable) {
            Log.e(TAG, "frame handling failed in $mode", t)
        }
    }

    // ------------------------------------------------------------------------------------------------ sheet
    fun captureSheet(then: (() -> Unit)? = null) {
        afterSheet = then
        sheetModel = null
        g.camera.unlockForSheet(g.prefs.cameraPreset)
        sheetSettle = SHEET_SETTLE_FRAMES
        sheetCapture = SheetCapture(camAnalysis, SHEET_FRAMES)
        _sheet.value = SheetUi(SheetUi.State.CAPTURING, "Keep the sheet EMPTY (no part, no hands) for 2 seconds…", null, 0f)
        mode = Mode.SHEET
    }

    /** Forget the sheet (each camera screen learns its own: camera settings are only locked for that session). */
    private fun invalidateSheet() {
        teachCircle = null
        pendingTeachCircle = null
        sheetModel = null
        sheetCapture = null
        _sheet.value = SheetUi()
    }

    fun captureSheet() = captureSheet(null)

    /**
     * Back to "Step 1" on the current camera screen (to change the zoom): forget the sheet and unlock the camera. A new
     * zoom shows a different background, so the sheet must be learned again.
     */
    fun releaseSheet() {
        if (mode == Mode.PREVIEW || mode == Mode.SHEET) mode = Mode.IDLE
        invalidateSheet()
        g.camera.unlockForSheet(g.prefs.cameraPreset)
    }

    private fun finishSheet(sc: SheetCapture) {
        mode = Mode.IDLE
        sheetCapture = null
        val then = afterSheet                                              // taken now: a screen change meanwhile drops it
        afterSheet = null
        val r = try {
            sc.finish()
        } catch (t: Throwable) {
            _sheet.value = SheetUi(SheetUi.State.FAILED, "Sheet capture failed: ${t.message}")
            return
        }
        sheetModel = r.sheet
        // Freeze exposure / white balance / focus now, so parts entering the view cannot change the background.
        if (r.flicker.flicker) {
            val p = g.camera.currentPreset()
            if (p.manualExposure && p.exposureNs < 10_000_000L) {
                // Flicker at a short exposure: switch to the 10 ms preset and learn the sheet AGAIN with it (a sheet learned at
                // 2 ms would be ~5x too dark for 10 ms frames). Only the exposure is persisted, never the AE/AWB/focus locks.
                val safe = p.flickerSafe().copy(aeLock = false, awbLock = false, focusDiopters = null)
                g.prefs.cameraPreset = safe
                sheetModel = null
                _sheet.value = SheetUi(SheetUi.State.CAPTURING, "Flicker detected — switching to a 10 ms exposure and learning the sheet again…", null, 0f)
                g.camera.applyPreset(safe)
                afterSheet = then
                sheetSettle = SHEET_SETTLE_FRAMES
                sheetCapture = SheetCapture(camAnalysis, SHEET_FRAMES)
                mode = Mode.SHEET
                return
            }
        }
        val locked = g.camera.lockForInspection()
        _sheet.value = SheetUi(SheetUi.State.READY, r.describe(), r.flicker.describe() + " · camera locked", 1f)
        if (previewWanted && afterSheet == null && _teach.value is TeachUi.Idle) mode = Mode.PREVIEW
        g.log.write(
            "sheet",
            JSONObject().put("mean", r.sheet.mean.joinToString()).put("sigma", r.sheet.sigma.joinToString())
                .put("banding", r.flicker.bandingLevels).put("drift", r.flicker.temporalBanding).put("pumpingPct", r.flicker.pumpingPct)
                .put("flicker", r.flicker.flicker).put("uniform", r.uniform).put("camera", locked.describe()),
        )
        then?.invoke()
    }

    /**
     * Leaving any camera screen (or opening another one): nothing from the previous screen may keep running on the next
     * screen's frames — a teach recording, a pending sheet capture with its follow-up, a circle. The screen's own session
     * (line / negatives) is stopped by its own stop* call.
     */
    private fun leaveCameraScreen() {
        if (mode == Mode.SHEET || mode == Mode.PREVIEW || mode == Mode.TEACH) mode = Mode.IDLE
        afterSheet = null
        sheetCapture = null
        if (_sheet.value.state == SheetUi.State.CAPTURING) _sheet.value = SheetUi()
        teachSession = null
        pendingTeachCircle = null
        circleCheck = null
        circleBefore = null
        if (_teach.value is TeachUi.Recording) _teach.value = TeachUi.Idle
    }

    /** The engine is being replaced: every session built on the old one stops now (its native buffers are about to close). */
    private fun onEngineReloading() {
        val hadLine = lineSession != null || replay != null
        stopLine()
        stopNegatives()
        fitJob?.cancel()
        if (_teach.value is TeachUi.Recording || _teach.value is TeachUi.Building) {
            _teach.value = TeachUi.Failed("The model was reloaded during the teach — record again.")
        }
        leaveCameraScreen()
        invalidateSheet()
        if (hadLine) _inspect.value = _inspect.value.copy(running = false, needsSheet = true, hint = "The model was reloaded — tap LEARN SHEET to continue.")
    }

    /** Inspect / Negatives screen left: nothing of it may run on the next screen's frames. */
    fun closeCameraScreen() {
        stopLine()
        stopNegatives()
        leaveCameraScreen()
    }

    /** Teach screen opened: learn a fresh sheet on this camera session (Step 1), then show the live detection box. */
    fun openTeach() {
        previewWanted = true
        stopLine()
        stopNegatives()
        leaveCameraScreen()
        if (_teach.value !is TeachUi.Building) {
            _teach.value = TeachUi.Idle
            invalidateSheet()
            _teachView.value = InspectUi()
        }
    }

    fun closeTeach() {
        previewWanted = false
        leaveCameraScreen()
    }

    /** Inspect/calibrate screen opened: nothing runs until the operator clears the sheet and taps LEARN SHEET. */
    fun openInspect(lineMode: LineMode) {
        stopLine()
        stopNegatives()
        leaveCameraScreen()
        pendingLineMode = lineMode
        invalidateSheet()
        val twin = activeTwin
        _inspect.value = InspectUi(
            running = false, mode = lineMode.name, twinName = twin?.name ?: "", needsSheet = true,
            hint = if (twin == null) "No part yet: go back and teach a part first." else null,
            sensitivity = twin?.let { g.prefs.sensitivity(it.id) } ?: 1f,
        )
    }

    /** Inspect: learn the empty sheet (locks the camera), then start the line in the pending mode. */
    fun learnSheetAndStart() {
        val m = lineSession?.mode ?: pendingLineMode ?: LineMode.INSPECT
        stopLine()
        pendingLineMode = m
        _inspect.value = _inspect.value.copy(needsSheet = true, running = false, hint = "Keep the sheet EMPTY for 2 seconds…")
        captureSheet { startLine(m) }
    }

    // ------------------------------------------------------------------------------------------------ twins
    fun refreshTwins() {
        g.scope.launch(Dispatchers.IO) {
            try {
                val eng = g.engines.ready()
                val running = eng?.let { TwinRepo.runningPipeline(it) }
                val list = repo.summaries(running)
                val preferred = g.prefs.activeTwinId
                val id = preferred?.takeIf { p -> list.any { it.id == p && it.loadable } } ?: list.firstOrNull { it.loadable }?.id
                val stale = running != null && activeTwin != null && activeTwin?.fingerprint != running.fingerprint()
                if (stale) activeTwin = null
                if (id != null && running != null && activeTwin?.id != id) {
                    when (val r = repo.load(id, running)) {
                        is TwinLoadResult.Loaded -> setActive(r.twin, persist = false)
                        is TwinLoadResult.FingerprintMismatch -> activeTwin = null
                        is TwinLoadResult.Invalid -> activeTwin = null
                    }
                }
                val act = activeTwin
                _twins.value = list.map { s -> if (act != null && s.id == act.id) summary(act) else s }
                _active.value = act?.let { summary(it) } ?: list.firstOrNull { it.id == id }
            } catch (t: Throwable) {
                Log.e(TAG, "refresh twins failed", t)
            }
        }
    }

    private fun setActive(t: TwinModel, persist: Boolean = true) {
        activeTwin = t.withSensitivity(g.prefs.sensitivity(t.id).toDouble())
        if (persist) g.prefs.activeTwinId = t.id
        _active.value = summary(t)
        refreshFitNegatives()
    }

    @Volatile private var fitJob: kotlinx.coroutines.Job? = null

    /**
     * Re-derives the active Twin's FIT gate against every known different object (other Twins, negatives library) in the
     * background; the Twin used by the next Inspect start carries the result. Cheap: a few k-NN searches per other Twin.
     */
    fun refreshFitNegatives() {
        val t = activeTwin ?: return
        val eng = g.engines.ready() ?: return
        if (t.thresholds.fit == null) return
        fitJob?.cancel()
        var self: kotlinx.coroutines.Job? = null
        self = g.scope.launch(Dispatchers.Default) {
            try {
                val r = FitNegatives.compute(t, repo, KnnAdapter(eng.knn))
                // Apply to whatever the active Twin is NOW (a calibration / slider change meanwhile must not be lost), and
                // only if no newer job has been started since (compute() cannot be cancelled mid-way).
                val cur = activeTwin
                if (cur != null && cur.id == t.id && fitJob === self && isActive) {
                    val next = cur.withFitNegatives(r.fits)
                    activeTwin = next
                    lineSession?.let { if (it.twin.id == next.id) it.twinUpdated(next) }
                    _active.value = summary(next)
                    _twins.value = _twins.value.map { if (it.id == next.id) summary(next) else it }
                    g.log.write("fit", JSONObject().put("twin", t.id).put("manual", r.manual).put("twins", r.twins).put("skipped", r.skipped).put("tauFit", next.thresholds.fit?.tauFit).put("rule", next.thresholds.fit?.rule).put("margin", next.thresholds.fit?.margin))
                }
            } catch (e: Throwable) {
                Log.w(TAG, "fit negatives failed", e)
            }
        }
        fitJob = self
    }

    private fun summary(t: TwinModel): TwinSummary {
        val c = t.certificate
        val alpha = c?.alpha
        val headline = when {
            c != null && alpha != null -> String.format(Locale.ROOT, "Calibrated with %d good parts — false alarms ≤ %.1f %% (%.0f %% confidence)", c.m, 100 * alpha, 100 * c.conf)
            c != null -> "Calibration had no valid part — calibrate again"
            t.thresholds.calibrated -> "Calibration outdated (wrong objects were added) — calibrate again"
            else -> "Not calibrated yet — Calibrate with 40 good parts for a false-alarm bound"
        }
        val fitLine = t.thresholds.fit?.let { f ->
            f.margin?.let { m ->
                if (m <= 0) "⚠ A known wrong object looks like this part — the look-alike gate cannot separate them"
                else "Look-alike protection: ${f.neg.size} other object${if (f.neg.size == 1) "" else "s"} known"
            } ?: "Look-alike protection: teach the look-alike as its own part, or add wrong objects"
        }
        return TwinSummary(t.id, t.name, t.createdAtMs, t.keyframeCount, c != null && alpha != null, t.thresholds.tau, t.thresholds.tauId, headline, true, null, fitLine)
    }

    fun selectTwin(id: String) {
        g.prefs.activeTwinId = id
        activeTwin = null
        refreshTwins()
    }

    fun deleteTwin(id: String) {
        g.scope.launch(Dispatchers.IO) {
            runCatching { repo.delete(id) }.onFailure { Log.e(TAG, "delete failed", it) }
            if (activeTwin?.id == id) {
                activeTwin = null
                g.prefs.activeTwinId = null
                _active.value = null
            }
            refreshTwins()
        }
    }

    fun exportTwin(id: String): File? = try {
        repo.export(id, File(g.dirs.exports, "twins").apply { mkdirs() })
    } catch (t: Throwable) {
        Log.e(TAG, "export failed", t)
        null
    }

    // ------------------------------------------------------------------------------------------------ teach
    fun startTeach(name: String) {
        val eng = g.engines.ready() ?: run { _teach.value = TeachUi.Failed("Model not loaded (${g.engines.badge()})."); return }
        val sheet = sheetModel ?: run { _teach.value = TeachUi.Failed("Learn the empty sheet first."); return }
        val pipeline = TwinRepo.runningPipeline(eng)
        circleCheck = null
        circleBefore = null
        pendingTeachCircle = null
        teachSession = TeachSession(camAnalysis, sheet, pipeline, name, SystemClock.elapsedRealtime(), seg = seg, circle = { teachCircle })
        _teach.value = TeachUi.Recording(0, 12_000, 0, 0, 0f, "Show the part", "…")
        mode = Mode.TEACH
    }

    private fun onTeachFrame(ts: TeachSession, f: CameraFrame) {
        // A circle drawn while recording: switch to it; if nothing is inside on the next sampled frame, go back.
        pendingTeachCircle?.let { (roi, id) ->
            pendingTeachCircle = null
            circleBefore = teachCircle
            teachCircle = RoiFollower(roi)
            circleCheck = id to ts.framesSeen
        }
        val seenBefore = ts.framesSeen
        if (!ts.onFrame(f)) {
            circleCheck?.let { (id, _) ->
                if (ts.framesSeen > seenBefore) {
                    circleCheck = null
                    val found = ts.lastView?.boxes?.isNotEmpty() == true && ts.lastSanity != com.kaizeneye.core.model.SanityReason.NO_OBJECT
                    if (found) {
                        _circle.value = CircleResult(id, true, "Got it — recording only this part")
                    } else {
                        teachCircle = circleBefore
                        _circle.value = CircleResult(id, false, "Nothing found inside the circle — circle the part again")
                    }
                }
            }
            _teach.value = TeachUi.Recording(ts.elapsedMs, ts.durationMs, ts.framesSeen, ts.accepted, ts.coverage, ts.lastHint, ts.lastSanity.name)
            ts.lastView?.let { _teachView.value = it }
            return
        }
        mode = Mode.IDLE
        teachSession = null
        buildTwin(ts)
    }

    /** Builds (and saves + activates) a Twin from a finished recording. Also used by the self-test with replayed frames. */
    fun buildTwin(ts: TeachSession, activate: Boolean = true, onDone: ((TeachSession.Built?) -> Unit)? = null) {
        g.scope.launch(Dispatchers.Default) {
            val eng = g.engines.ready()
            if (eng == null) {
                _teach.value = TeachUi.Failed("Model not loaded."); onDone?.invoke(null); return@launch
            }
            _teach.value = TeachUi.Building("embedding", 0.05f)
            val negatives = try {
                repo.negatives.globals(TwinRepo.runningPipeline(eng).fingerprint())
            } catch (t: Throwable) {
                emptyList()
            }
            val res = ts.build({ eng.backbone.embed(it) }, KnnAdapter(eng.knn), negatives) { stage, p -> _teach.value = TeachUi.Building(stage, p) }
            res.fold(
                onSuccess = { b ->
                    if (activate) {
                        try {
                            repo.save(b.twin, b.keyframeJpegs)
                        } catch (e: Throwable) {
                            Log.e(TAG, "save failed", e)
                            _teach.value = TeachUi.Failed("Could not save the part (${e.message}) — is the storage full?")
                            onDone?.invoke(null)
                            return@fold
                        }
                        setActive(b.twin)
                        refreshTwins()
                    }
                    _teach.value = TeachUi.Armed(summary(b.twin), b.tapToArmedMs, b.lines)
                    g.log.write(
                        "teach",
                        JSONObject().put("twin", b.twin.id).put("name", b.twin.name).put("tapToArmedMs", b.tapToArmedMs)
                            .put("keyframes", b.twin.keyframeCount).put("tau", b.twin.thresholds.tau).put("tauId", b.twin.thresholds.tauId)
                            .put("accepted", b.result.diagnostics.accepted.size).put("kept", b.result.diagnostics.kept.size)
                            .put("timings", JSONObject(b.result.diagnostics.timingsMs.mapValues { it.value })),
                    )
                    onDone?.invoke(b)
                },
                onFailure = { t ->
                    Log.e(TAG, "teach failed", t)
                    _teach.value = TeachUi.Failed(t.message ?: t.javaClass.simpleName)
                    onDone?.invoke(null)
                },
            )
        }
    }

    /**
     * Teach from a recorded clip (Clips → Teach): sheet from its first frames, then the next 12 s — or the "teach" segment
     * of a script.json when the clip has one (the synthetic self-test clip). Saves the Twin and makes it active.
     */
    suspend fun teachFromClip(clip: File): String {
        val eng = g.engines.ensure(com.kaizeneye.v2.ml.Models.choice(g.prefs.backbone)) ?: return "model not loaded (${g.engines.badge()})"
        val window = File(clip, "script.json").takeIf { clip.isDirectory && it.isFile }?.let { f ->
            val seg = JSONObject(f.readText()).optJSONArray("segments")
            (0 until (seg?.length() ?: 0)).map { seg!!.getJSONObject(it) }.firstOrNull { it.optString("name") == "teach" }
                ?.let { it.getLong("startMs")..it.getLong("endMs") }
        }
        val negatives = runCatching { repo.negatives.globals(TwinRepo.runningPipeline(eng).fingerprint()) }.getOrDefault(emptyList())
        val r = try {
            ClipJobs.teachFromClip(clip, eng, clip.nameWithoutExtension.take(24), window = window, sampleEveryMs = if (window != null) 0L else 125L, negatives = negatives)
        } catch (t: Throwable) {
            return "teach from ${clip.name} failed: ${t.message}"
        }
        val b = r.built ?: return "teach from ${clip.name} failed: ${r.message}"
        try {
            repo.save(b.twin, b.keyframeJpegs)
        } catch (t: Throwable) {
            return "could not save the part: ${t.message}"
        }
        setActive(b.twin)
        refreshTwins()
        g.log.write("teach", JSONObject().put("twin", b.twin.id).put("fromClip", clip.name).put("keyframes", b.twin.keyframeCount).put("tau", b.twin.thresholds.tau))
        return "Part '${b.twin.name}' ready: ${b.lines.joinToString(" · ")}"
    }

    fun stopTeach() {
        teachSession?.stopRequested = true
    }

    fun resetTeach() {
        if (mode == Mode.TEACH || mode == Mode.SHEET) mode = Mode.IDLE
        teachSession = null
        _teach.value = TeachUi.Idle
        if (previewWanted && sheetModel != null) mode = Mode.PREVIEW
    }

    private var circleBefore: RoiFollower? = null
    private var circleCheck: Pair<Long, Int>? = null

    /** Checks a circle drawn on the Teach preview against [a0]: keep it only if a part is inside. */
    private fun registerTeachCircle(a0: Analysed, pc: Pair<Roi, Long>) {
        pendingTeachCircle = null
        val f = RoiFollower(pc.first)
        val p = f.pick(a0, seg)
        if (p.obj != null) {
            teachCircle = f
            _circle.value = CircleResult(pc.second, true, "Got it — only this part will be taught")
        } else {
            _circle.value = CircleResult(pc.second, false, "Nothing found inside the circle — circle the part again")
        }
    }

    /**
     * The operator circled (or tapped) something on the preview. [xs]/[ys] are in camera FRAME pixels (the overlay's
     * coordinate system); [strokeId] identifies the doodle that gets the result.
     */
    fun circle(xs: FloatArray, ys: FloatArray, strokeId: Long) {
        val f = camAnalysis.factor.toDouble()
        val ui = if (mode == Mode.LINE) _inspect.value else _teachView.value
        val w = (ui.analysisW / f).toInt()
        val h = (ui.analysisH / f).toInt()
        val roi = if (w > 0 && h > 0) Roi.fromStroke(DoubleArray(xs.size) { xs[it] / f }, DoubleArray(ys.size) { ys[it] / f }, w, h) else null
        if (roi == null) {
            _circle.value = CircleResult(strokeId, false, "Camera not ready yet")
            return
        }
        when (mode) {
            Mode.LINE -> lineSession?.circle(roi, strokeId) { _circle.value = it }
                ?: run { _circle.value = CircleResult(strokeId, false, "Not inspecting yet — learn the sheet first") }
            Mode.PREVIEW, Mode.TEACH -> pendingTeachCircle = roi to strokeId
            Mode.SHEET -> _circle.value = CircleResult(strokeId, false, "Wait for the sheet to be learned")
            else -> _circle.value = CircleResult(strokeId, false, "Learn the sheet first (Step 1), then circle the part")
        }
    }

    /** Forget the circle: the whole view is analysed again. */
    fun clearCircle() {
        teachCircle = null
        pendingTeachCircle = null
        lineSession?.clearCircle()
        if (_teachView.value.circled) _teachView.value = _teachView.value.copy(circled = false)
    }

    // ------------------------------------------------------------------------------------------------ line
    /** Trigger settings from the preferences; the photo-eye axis is resolved per session from the screen motion. */
    fun triggerConfig(): TriggerConfig {
        val m = g.prefs.triggerMode
        return TriggerConfig(
            axis = Axis.X,
            direction = Direction.ANY,
            lineEnabled = m == "LINE" || m == "BOTH",
            steadyEnabled = m == "STEADY" || m == "BOTH",
        )
    }

    fun startLine(mode: LineMode) {
        stopLine()
        val twin = activeTwin ?: return message("No part yet — teach a part first.")
        val eng = g.engines.ready() ?: return message("Model still loading — try again in a moment.")
        if (TwinRepo.runningPipeline(eng).fingerprint() != twin.fingerprint) return message("This part was built with another model — teach it again.")
        val sheet = sheetModel ?: return openInspect(mode)
        pendingLineMode = null
        run {
            val sens = g.prefs.sensitivity(twin.id).toDouble()
            g.log.openSession("${mode.name.lowercase(Locale.ROOT)}_${twin.name}")
            g.log.write("session", JSONObject().put("mode", mode.name).put("twin", twin.id).put("accel", eng.backbone.report.badge).put("preset", g.camera.currentPreset().describe()))
            lineSession = LineSession(
                g, twin, mode, "camera", eng, camAnalysis, sheet, triggerConfig(), g.prefs.lineFraction.toDouble(),
                live = true, stages = stages, sensitivity = sens, publish = { _inspect.value = it }, screenMotion = g.prefs.motion,
            )
            this.mode = Mode.LINE
        }
    }

    fun startReplay(clip: File, paced: Boolean, mode: LineMode) {
        stopLine()
        val twin = activeTwin ?: return message("No part yet — teach a part first (or replay a teach clip).")
        val eng = g.engines.ready() ?: return message("Model still loading — try again in a moment.")
        if (TwinRepo.runningPipeline(eng).fingerprint() != twin.fingerprint) return message("This part was built with another model — teach it again.")
        val analysis = FrameAnalysis(seg, stages)
        val src = ReplayFrameSource(clip, paced)
        var cap: SheetCapture? = SheetCapture(analysis, REPLAY_SHEET_FRAMES)
        var ls: LineSession? = null
        val sens = g.prefs.sensitivity(twin.id).toDouble()
        val publish = { ui: InspectUi ->
            // Rendered from the SAME frame the overlay was computed on (publish runs inside that frame's processing).
            analysis.lastSmall?.let { replayPreview = Images.previewBitmap(it.data, it.width, it.height) }
            val p = src.progress.value
            _inspect.value = ui.copy(source = "replay", replayFrame = replayPreview, replayProgress = if (p.total > 0) p.frames / p.total.toFloat() else null)
        }
        src.setConsumer { f ->
            if (replay !== src) return@setConsumer                       // stopped (or replaced) meanwhile
            val c = cap
            if (c != null) {
                if (c.offer(f)) {
                    val r = c.finish()
                    cap = null
                    val session = LineSession(g, twin, mode, "replay:${clip.name}", eng, analysis, r.sheet, triggerConfig(), g.prefs.lineFraction.toDouble(), live = false, stages = stages, sensitivity = sens, publish = publish, screenMotion = g.prefs.motion)
                    if (replay === src) {
                        ls = session
                        lineSession = session
                    } else {
                        session.close()
                    }
                }
                return@setConsumer
            }
            ls?.onFrame(f)
        }
        replay = src
        g.log.openSession("replay_${clip.nameWithoutExtension}")
        _inspect.value = InspectUi(running = true, mode = mode.name, source = "replay", twinName = twin.name, message = "Replaying ${clip.name} (sheet from the first frames)…")
        src.start { p ->
            val mine = replay === src
            ls?.let { if (mine) it.drainAndClose() else it.close() }
            if (!mine) return@start                                      // a newer session owns the screen now
            _inspect.value = _inspect.value.copy(
                running = false, replayProgress = 1f,
                message = p.error?.let { "Replay error: $it" } ?: "Replay finished: ${p.frames} frames, ${ls?.judgedCount ?: 0} parts judged",
            )
        }
    }

    fun stopLine() {
        replay?.stop()
        replay = null
        if (mode == Mode.LINE) mode = Mode.IDLE
        lineSession?.close()
        lineSession = null
        g.camera.setAnalysisFpsCap(30)
        _inspect.value = _inspect.value.copy(running = false)
    }

    fun setSensitivity(v: Float, persist: Boolean = false) {
        val t = activeTwin ?: return
        if (persist) g.prefs.setSensitivity(t.id, v)
        activeTwin = t.withSensitivity(v.toDouble())
        lineSession?.sensitivity = v.toDouble()
        _inspect.value = _inspect.value.copy(sensitivity = v)
    }

    /** True when the current tolerance voids the calibrated certificate (τ·sensitivity below the calibration maximum). */
    fun certificateVoidAtCurrentTolerance(): Boolean {
        val t = activeTwin ?: return false
        val c = t.certificate ?: return false
        return !c.isValid(t.thresholds.tau, t.thresholds.sensitivity)
    }

    fun finishCalibration() {
        val ls = lineSession ?: return
        if (ls.mode != LineMode.CALIBRATE) return
        val badge = g.engines.badge()
        // Stop feeding frames, let the queued judges finish (the last parts count), then read the samples.
        if (mode == Mode.LINE) mode = Mode.IDLE
        lineSession = null
        g.camera.setAnalysisFpsCap(30)
        _inspect.value = _inspect.value.copy(running = false, message = "Finishing calibration…")
        g.scope.launch(Dispatchers.IO) {
            try {
                ls.drainAndClose(15_000)
                val samples = ls.calibrationSamples()
                val lat = ls.latencySamples()
                val t = activeTwin?.takeIf { it.id == ls.twin.id } ?: ls.twin
                val (calibrated, r) = t.calibrate(samples, latenciesMs = lat, accelerator = badge)
                repo.save(calibrated, null)
                setActive(calibrated)
                g.log.write("calibration", JSONObject().put("twin", t.id).put("n", samples.size).put("m", r.certificate.m).put("tauCal", r.tauCal).put("alpha", r.certificate.alpha))
                refreshTwins()
            } catch (e: Throwable) {
                Log.e(TAG, "calibration failed", e)
                message("Calibration failed: ${e.message}")
            }
        }
    }

    fun explainLast() {
        lineSession?.explain()
    }

    private fun message(m: String) {
        _inspect.value = _inspect.value.copy(message = m, running = false, needsSheet = false, hint = null)
    }

    // ------------------------------------------------------------------------------------------------ negatives
    fun startNegatives() {
        stopNegatives()
        val twin = activeTwin ?: run { _negatives.value = _negatives.value.copy(lastMessage = "No part yet — teach a part first."); return }
        val eng = g.engines.ready() ?: run { _negatives.value = _negatives.value.copy(lastMessage = "Model still loading — try again in a moment."); return }
        if (TwinRepo.runningPipeline(eng).fingerprint() != twin.fingerprint) {
            _negatives.value = _negatives.value.copy(lastMessage = "This part was built with another model — teach it again first.")
            return
        }
        val sheet = sheetModel ?: run {
            _negatives.value = _negatives.value.copy(lastMessage = "Step 1: clear the sheet, then tap LEARN SHEET.")
            return
        }
        run {
            val ns = NegativesSession(camAnalysis, sheet, twin, eng, repo) { ui, updated ->
                _negatives.value = ui
                if (updated != null) setActive(updated)
            }
            negativesSession = ns
            ns.start()
            mode = Mode.NEGATIVES
        }
    }

    /** Negatives screen opened: learn a fresh sheet first. */
    fun openNegatives() {
        stopLine()
        stopNegatives()
        leaveCameraScreen()
        invalidateSheet()
        _negatives.value = NegativesUi(0, 0, "Step 1: clear the sheet, then tap LEARN SHEET.", null)
    }

    fun learnSheetForNegatives() = captureSheet { startNegatives() }

    fun stopNegatives() {
        if (mode == Mode.NEGATIVES) mode = Mode.IDLE
        negativesSession?.close()
        negativesSession = null
    }

    // ------------------------------------------------------------------------------------------------ certificate / telemetry
    fun certificateLines(): List<String> {
        val t = activeTwin ?: return listOf("No part selected.")
        val th = t.thresholds
        val sens = g.prefs.sensitivity(t.id).toDouble()
        val head = String.format(Locale.ROOT, "%s · τ %.3f · τ_id %.3f (%s) · %d keyframes", t.name, th.tau, th.tauId, th.tauIdRule, t.keyframeCount)
        val c = t.certificate
        val body = if (c != null) CertificateText.lines(c)
        else listOf(t.withinPartLine(), "Not calibrated: present ≥ 40 distinct good parts in Calibrate to get the false-alarm bound.")
        val void = if (c != null && !c.isValid(th.tau, sens)) "⚠ Sensitivity ${"%.2f".format(sens)}× is below the calibrated limit — the certificate is void at this setting." else null
        val stale = if (th.calibrated && c == null) "Certificate outdated (negatives changed after calibration) — calibrate again." else null
        return listOfNotNull(head) + body + listOfNotNull(void, stale)
    }

    fun stageSummary(): List<Pair<String, String>> = stages.snapshot().entries.sortedBy { it.key }.map { (k, s) ->
        k to String.format(Locale.ROOT, "%.1f / %.1f ms (n=%d)", s.p50, s.p95, s.count)
    }

    // ------------------------------------------------------------------------------------------------ self-test
    fun runSelfTest(kind: String) {
        if (_selfTest.value.running) return
        _selfTest.value = SelfTestUi(running = true)
        g.scope.launch {
            val steps = LinkedHashMap<String, String>()
            try {
                val (root, file) = selfTestRunner.run(kind) { name, status ->
                    steps[name] = status
                    _selfTest.value = SelfTestUi(running = true, steps = steps.toList())
                }
                _selfTest.value = SelfTestUi(false, steps.toList(), file.absolutePath, root.optBoolean("pass"))
            } catch (t: Throwable) {
                steps["self-test"] = "FAIL ${t.message}"
                _selfTest.value = SelfTestUi(false, steps.toList(), null, false)
            }
        }
    }

    companion object {
        private const val TAG = "KaizenHub"
        const val SHEET_FRAMES = 30
        /** ~0.7 s at 30 fps for auto exposure / white balance / focus to settle on the empty sheet before it is learned. */
        const val SHEET_SETTLE_FRAMES = 20
        const val REPLAY_SHEET_FRAMES = 15
    }
}
