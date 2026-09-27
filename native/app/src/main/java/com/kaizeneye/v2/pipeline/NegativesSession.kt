package com.kaizeneye.v2.pipeline

import android.util.Log
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.track.Tracker
import com.kaizeneye.core.track.Trigger
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.core.twin.Descriptor
import com.kaizeneye.core.twin.NegativeMap
import com.kaizeneye.core.twin.RotationRule
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
            if (fired.trigger != Trigger.STEADY) continue
            if (busy) {
                publish("Still saving the previous object — lift it and hold it again")
                continue
            }
            val idx = upd.assignments.indexOfFirst { it == fired.trackId }
            if (idx < 0) continue
            val comp = comps[idx]
            val snap = analysis.snapshot(a, comp, twin.pipeline.inputSize, twin.gh, twin.gw, canonical = twin.pipeline.rotation == RotationRule.CANONICAL)
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
            val entry = repo.negatives.add(t.fingerprint, global, Images.jpeg(snap.crop, snap.n, snap.n, 90))
            // FIT gate: keep the feature maps of both orientations (canonical crops) so the negative can be scored against any Twin.
            val maps = ArrayList<NegativeMap>()
            maps += NegativeMap(f.copyOf(), snap.cov)
            snap.canon?.let { c ->
                val f1 = t.pipeline.prepareFeatures(com.kaizeneye.core.model.FeatureMap(t.gh, t.gw, t.dim, engines.backbone.embed(c.altCrop))).data
                maps += NegativeMap(f1, c.altCov)
            }
            repo.negatives.addMaps(entry.id, maps, t.patches, t.dim)
            added++
            val all = repo.negatives.globals(t.fingerprint)
            var next = t.withNegatives(all)
            val fitBefore = next.thresholds.fit?.tauFit
            if (next.thresholds.fit != null) next = next.withFitNegatives(FitNegatives.compute(next, repo, KnnAdapter(engines.knn)).fits)
            repo.save(next, null)
            twin = next
            val warn = if (sim >= t.thresholds.tauId) " — it looked like your part before; now it is rejected" else ""
            publish("Added wrong object #$added ✓$warn", next)
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
        val fit = th.fit?.let { f ->
            String.format(
                Locale.ROOT, " · look-alike gate %.3f (%s)%s", f.tauFit, f.rule,
                f.margin?.let { m -> String.format(Locale.ROOT, ", margin %+.3f%s", m, if (m <= 0) " — OVERLAP" else "") } ?: "",
            )
        } ?: ""
        onUpdate(NegativesUi(repo.negatives.list(t.fingerprint).size, added, message, line + fit), updated)
    }

    fun close() {
        worker.shutdown()
    }
}
