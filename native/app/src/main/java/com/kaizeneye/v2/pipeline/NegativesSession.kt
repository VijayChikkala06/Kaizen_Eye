package com.kaizeneye.v2.pipeline

import android.util.Log
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.track.Tracker
import com.kaizeneye.core.track.Trigger
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.core.twin.Descriptor
import com.kaizeneye.core.twin.PatchSets
import com.kaizeneye.core.twin.TwinModel
import com.kaizeneye.v2.camera.CameraFrame
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.twin.TwinRepo
import com.kaizeneye.v2.util.Images
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Negatives library capture (plan): steady-hold trigger only; each hold of a WRONG object adds its object descriptor to the
 * library (per pipeline fingerprint), then the active Twin's τ_id is re-derived (twin-spec §7.3, midpoint rule).
 */
class NegativesSession(
    private val analysis: FrameAnalysis,
    private val sheet: SheetModel,
    @Volatile private var twin: TwinModel,
    private val engines: EngineHolder.State.Ready,
    private val repo: TwinRepo,
    private val onUpdate: (NegativesUi, TwinModel?) -> Unit,
) {
    private var tracker: Tracker? = null
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "kz-negatives") }
    private var added = 0
    @Volatile private var busy = false

    fun start() = publish(null)

    fun onFrame(frame: CameraFrame) {
        val a = analysis.analyse(frame, sheet)
        val tr = tracker ?: Tracker(a.w, a.h, TriggerConfig(lineEnabled = false, steadyEnabled = true)).also { tracker = it }
        val comps = a.partCandidates(analysis.params.minAreaFrac)
        val upd = tr.update(a.tMs, com.kaizeneye.core.track.Detections.of(comps, a.grey, a.w, a.h, a.index))
        for (fired in upd.triggers) {
            if (fired.trigger != Trigger.STEADY || busy) continue
            val idx = upd.assignments.indexOfFirst { it == fired.trackId }
            if (idx < 0) continue
            val comp = comps[idx]
            val snap = analysis.snapshot(a, comp, twin.pipeline.inputSize, twin.gh, twin.gw)
            if (snap.sanity != SanityReason.OK) {
                publish("Not captured: ${snap.sanity.name.lowercase(Locale.ROOT).replace('_', ' ')} — hold one object fully in view")
                continue
            }
            busy = true
            worker.execute { add(snap) }
        }
    }

    private fun add(snap: CropSnapshot) {
        try {
            val t = twin
            val f = t.pipeline.prepareFeatures(com.kaizeneye.core.model.FeatureMap(t.gh, t.gw, t.dim, engines.backbone.embed(snap.crop))).data
            val sets = PatchSets.fromCov(snap.cov, t.gh, t.gw, t.pipeline.mask.coreThreshold)
            val g = Descriptor.global(f, t.patches, t.dim, sets.coreIdx)
            val global = FloatArray(g.size) { g[it].toFloat() }
            val sim = t.similarity(g)
            repo.negatives.add(t.fingerprint, global, Images.jpeg(snap.crop, snap.n, snap.n, 90))
            added++
            val all = repo.negatives.globals(t.fingerprint)
            val next = t.withNegatives(all)
            repo.save(next, null)
            twin = next
            val warn = if (sim >= t.thresholds.tauId) " (it WOULD have passed the old identity gate)" else ""
            publish(String.format(Locale.ROOT, "Added negative #%d · similarity %.3f%s", added, sim, warn), next)
        } catch (t: Throwable) {
            Log.e("KaizenNegatives", "negative capture failed", t)
            publish("Capture failed: ${t.message}")
        } finally {
            busy = false
        }
    }

    private fun publish(message: String?, updated: TwinModel? = null) {
        val t = twin
        val th = t.thresholds
        val line = String.format(
            Locale.ROOT, "τ_id %.3f (%s)%s", th.tauId, th.tauIdRule,
            th.identityMargin?.let { String.format(Locale.ROOT, " · margin %.3f%s", it, if (it <= 0) " — OVERLAP" else "") } ?: "",
        )
        onUpdate(NegativesUi(repo.negatives.list(t.fingerprint).size, added, message, line), updated)
    }

    fun close() {
        worker.shutdown()
    }
}
