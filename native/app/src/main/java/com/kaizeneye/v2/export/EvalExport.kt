package com.kaizeneye.v2.export

import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.twin.Judgement
import com.kaizeneye.core.twin.TwinModel
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Writes one evaluation export folder (docs/verification/export-format.md, schema 1) for tools/lab/twin_eval.py:
 * `manifest.json`, `teach/NNNN.{jpg,json}`, `parts/<clip>_<trackId>.{jpg,json}`, `negatives/NNNN.{jpg,json}`.
 * Written into `<name>.tmp/` and renamed on [finish] (the laptop ignores *.tmp).
 */
class EvalExport(private val root: File, val cropSize: Int = 448) {
    private val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
    private val dir = File(root, "export_$stamp.tmp")
    private val teachDir = File(dir, "teach")
    private val partsDir = File(dir, "parts")
    private val negDir = File(dir, "negatives")
    private var teachN = 0
    private var partsN = 0
    private var negN = 0

    init {
        listOf(teachDir, partsDir, negDir).forEach { it.mkdirs() }
    }

    fun teach(e: ExportFrame) {
        val base = "%04d".format(Locale.ROOT, teachN)
        write(teachDir, base, e, e.cropInfo().put("index", teachN))
        teachN++
    }

    fun negative(e: ExportFrame) {
        val base = "%04d".format(Locale.ROOT, negN)
        write(negDir, base, e, e.cropInfo().put("index", negN))
        negN++
    }

    fun part(clip: String, label: String, trackId: Int, trigger: String, e: ExportFrame, j: Judgement, latencyMs: Double) {
        val c = clipName(clip)
        val base = "${c}_%04d".format(Locale.ROOT, trackId)
        val reason: Any = when {
            j.reason == null -> JSONObject.NULL
            else -> j.reason!!.uppercase(Locale.ROOT)
        }
        val phone = JSONObject().put("verdict", j.verdict.name).put("reason", reason)
            .put("raw", j.raw.safe()).put("s", j.s.safe()).put("sim", j.sim.safe()).put("a", j.anomalousFraction.safe())
            .put("peak", if (j.peakRow >= 0) JSONArray(listOf(j.peakRow, j.peakCol)) else JSONObject.NULL)
            .put("ms", latencyMs.safe())
        write(partsDir, base, e, e.cropInfo().put("clip", c).put("trackId", trackId).put("label", label).put("trigger", trigger).put("phone", phone))
        partsN++
    }

    private fun write(folder: File, base: String, e: ExportFrame, json: JSONObject) {
        e.jpeg?.let { File(folder, "$base.jpg").writeBytes(it) }
        File(folder, "$base.json").writeText(json.toString())
    }

    /** Writes the manifest and renames the folder to `<twinId>_<stamp>` (export-format §1). */
    fun finish(twin: TwinModel, sheet: SheetModel, frameW: Int, frameH: Int, app: JSONObject): File {
        val finalDir = File(root, "${twin.id}_$stamp")
        val th = twin.thresholds
        val pipelineJson = JSONObject(twin.pipeline.canonicalJson())
        val m = JSONObject()
            .put("schema", 1).put("createdAtMs", System.currentTimeMillis()).put("twinId", twin.id).put("twinName", twin.name)
            .put("pipeline", pipelineJson).put("fingerprint", twin.fingerprint)
            .put("analysisFactor", twin.pipeline.mask.analysisFactor).put("inputSize", twin.pipeline.inputSize).put("cropSize", cropSize)
            .put("frameW", frameW).put("frameH", frameH)
            .put("sheet", JSONObject().put("mu", JSONArray(sheet.mean.toList())).put("sigma", JSONArray(sheet.sigma.toList())))
            .put(
                "twin",
                JSONObject().put("tau", th.tau).put("tauTeach", th.tauTeach).put("tauId", th.tauId).put("sensitivity", th.sensitivity)
                    .put("coverageCut", th.coverageCut).put("calibrated", th.calibrated),
            )
            .put("counts", JSONObject().put("teach", teachN).put("parts", partsN).put("negatives", negN))
            .put("app", app)
        File(dir, "manifest.json").writeText(m.toString(2))
        if (!dir.renameTo(finalDir)) error("could not rename ${dir.name}")
        return finalDir
    }

    val counts: Triple<Int, Int, Int> get() = Triple(teachN, partsN, negN)

    companion object {
        /** export-format §1: `[A-Za-z0-9-]{1,32}`, no underscore. */
        fun clipName(s: String): String = s.replace(Regex("[^A-Za-z0-9-]"), "-").trim('-').take(32).ifEmpty { "clip" }

        private fun Double.safe(): Any = if (isNaN() || isInfinite()) JSONObject.NULL else this
    }
}
