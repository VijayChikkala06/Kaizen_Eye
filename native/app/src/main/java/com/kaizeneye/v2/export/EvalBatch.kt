package com.kaizeneye.v2.export

import android.os.Build
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.core.twin.Descriptor
import com.kaizeneye.core.twin.PatchSets
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.BuildConfig
import com.kaizeneye.v2.camera.ReplayFrameSource
import com.kaizeneye.v2.ml.Models
import com.kaizeneye.v2.pipeline.ClipJobs
import com.kaizeneye.v2.ui.LineMode
import org.json.JSONObject
import java.io.File

/**
 * "Export eval set" (plan "A/B (offline, twin_eval.py on crops/masks exported from the phone)"): replays the labelled clips
 * recorded in the Clips screen through the identical pipeline and writes an export folder
 * (docs/verification/export-format.md) under getExternalFilesDir("exports"):
 *  1. teach: the newest `teach_*` clip → a fresh Twin; every analysed teach frame exported;
 *  2. negatives: every `negatives_*` clip, steady-hold captures → negatives/ (and τ_id of the fresh Twin);
 *  3. parts: every `good|defect|wrong|rotated|lookalike|calib_*` clip judged with the app's trigger settings → parts/.
 * Pull it with `adb pull /sdcard/Android/data/com.kaizeneye.v2/files/exports` and run tools/lab/twin_eval.py.
 */
class EvalBatch(private val g: AppGraph) {

    fun run(progress: (String) -> Unit): File {
        val clips = (g.dirs.clips.listFiles()?.toList() ?: emptyList()).filter { ReplayFrameSource.isClip(it) }.sortedBy { it.lastModified() }
        val teachClip = clips.lastOrNull { ClipJobs.labelOf(it) == "teach" } ?: error("Record a 'teach' clip first (Clips → label teach).")
        val eng = kotlinx.coroutines.runBlocking { g.engines.ensure(Models.choice(g.prefs.backbone)) } ?: error("Model not loaded (${g.engines.badge()}).")
        val exporter = EvalExport(g.dirs.exports.apply { mkdirs() }, cropSize = 448)

        progress("teach from ${teachClip.name}…")
        val tr = ClipJobs.teachFromClip(teachClip, eng, "eval-${teachClip.nameWithoutExtension}", exporter = exporter, exportCropSize = exporter.cropSize, accuracy = false)
        var twin = tr.built?.twin ?: error("teach failed: ${tr.message}")
        val sheet = tr.sheet ?: error("no sheet in the teach clip")

        val negGlobals = ArrayList<FloatArray>()
        for (clip in clips.filter { ClipJobs.labelOf(it) == "negatives" }) {
            progress("negatives from ${clip.name}…")
            val steady = TriggerConfig(lineEnabled = false, steadyEnabled = true)
            val (parts, _) = ClipJobs.judgeClip(g, clip, twin, eng, steady, 0.5, exportCropSize = exporter.cropSize)
            for (p in parts) {
                val s = p.snapshot ?: continue
                s.export?.let { exporter.negative(it) }
                if (s.sanity.name != "OK") continue
                val f = twin.pipeline.prepareFeatures(com.kaizeneye.core.model.FeatureMap(twin.gh, twin.gw, twin.dim, eng.backbone.embed(s.crop))).data
                val sets = PatchSets.fromCov(s.cov, twin.gh, twin.gw, twin.pipeline.mask.coreThreshold)
                if (sets.isEmpty) continue
                val gq = Descriptor.global(f, twin.patches, twin.dim, sets.coreIdx)
                negGlobals += FloatArray(gq.size) { gq[it].toFloat() }
            }
        }
        if (negGlobals.isNotEmpty()) twin = twin.withNegatives(negGlobals)

        val labels = setOf("good", "defect", "wrong", "rotated", "lookalike", "calib")
        val trigger = g.hub.triggerConfig()
        var nParts = 0
        for (clip in clips.filter { ClipJobs.labelOf(it) in labels }) {
            progress("judging ${clip.name}…")
            val label = ClipJobs.labelOf(clip)
            val (parts, _) = ClipJobs.judgeClip(g, clip, twin, eng, trigger, g.prefs.lineFraction.toDouble(), exportCropSize = exporter.cropSize, mode = LineMode.INSPECT, screenMotion = g.prefs.motion)
            val clipName = EvalExport.clipName(clip.nameWithoutExtension.replace('_', '-'))
            for (p in parts) {
                val e = p.snapshot?.export ?: continue
                exporter.part(clipName, label, p.trackId, p.trigger?.name ?: "LINE", e, p.judgement, p.latencyMs)
                nParts++
            }
        }
        progress("writing manifest…")
        val app = JSONObject().put("versionName", BuildConfig.VERSION_NAME).put("device", "${Build.MANUFACTURER} ${Build.MODEL}").put("socModel", Build.SOC_MODEL)
            .put("backbone", eng.backbone.report.badge)
        val dir = exporter.finish(twin, sheet, tr.frameW, tr.frameH, app)
        progress("done: ${dir.name} (${exporter.counts.first} teach frames, $nParts parts, ${exporter.counts.third} negatives)")
        return dir
    }
}
