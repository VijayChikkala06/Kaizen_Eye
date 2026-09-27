package com.kaizeneye.v2.pipeline

import android.content.Context
import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.model.Verdict
import com.kaizeneye.core.replay.ReplayComparator
import com.kaizeneye.core.replay.VerdictRecord
import com.kaizeneye.core.track.Axis
import com.kaizeneye.core.track.Direction
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.ml.Models
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

/**
 * Self-test REPLAY regression (plan "Verification — Device"): the bundled synthetic clip (tools/lab/make_replay_synth.py →
 * assets/selftest/replay_synth: JPEG frames + timestamps + expected.json computed on the laptop by the numpy reference with
 * the same ResNet18 TFLite) is taught and judged on the phone through the identical pipeline; the verdict list must match
 * (entries marked borderline accept any verdict). Deterministic: frame timestamps only, no pacing, no governor.
 */
class ReplayRegression(private val g: AppGraph, @Suppress("unused") private val hub: PipelineHub) {

    suspend fun run(out: JSONObject): String {
        val dir = extract(g.context) ?: return "SKIP synthetic replay clip not bundled (assets/selftest/replay_synth)"
        val expected = JSONObject(File(dir, "expected.json").readText())
        val script = File(dir, "script.json").takeIf { it.isFile }?.let { JSONObject(it.readText()) } ?: JSONObject()
        val factor = expected.optInt("analysisFactor", script.optInt("analysisFactor", 4))
        val seg = MaskParams(analysisFactor = factor)
        // The expected verdicts were computed with ResNet18; with any other backbone (e.g. the emulator's DEBUG Kotlin
        // features) only the mechanics are compared: same parts triggered at the same times, verdict classes free.
        val current = g.engines.ready()?.choice
        val mechanicsOnly = current != null && current.id != Models.R18_CHOICE.id
        // Only an already loaded engine is used (loading another one here would close the engine the app is using).
        val eng = g.engines.ready() ?: return "SKIP model still loading (${g.engines.badge()}) — run the self-test again"
        val teach = expected.getJSONObject("teach")
        val window = teach.getLong("startMs")..teach.getLong("endMs")
        // The laptop reference teaches from EVERY frame of the window (script.json) — sample every frame here too.
        val tr = ClipJobs.teachFromClip(dir, eng, "selftest", seg, window, sheetFrames = expected.optInt("sheetFrames", 15), sampleEveryMs = 0L, accuracy = false)
        out.put("teach", tr.message)
        val twin = tr.built?.twin ?: return "FAIL teach from the synthetic clip: ${tr.message}"
        out.put("tau", twin.thresholds.tau).put("tauId", twin.thresholds.tauId).put("keyframes", twin.keyframeCount)
        expected.optJSONObject("twin")?.let { ref ->
            // The laptop's numbers for the same clip (differences come from JPEG decoding, float32 graph vs float64, int8/NPU).
            out.put("reference", JSONObject().put("tau", ref.optDouble("tau")).put("tauId", ref.optDouble("tauId")).put("keyframes", ref.optInt("keyframes")).put("bankRows", ref.optInt("bankRows")))
        }

        val trig = expected.optJSONObject("trigger") ?: script.optJSONObject("trigger") ?: JSONObject()
        val config = TriggerConfig(
            axis = if (trig.optString("axis", "X") == "Y") Axis.Y else Axis.X,
            direction = runCatching { Direction.valueOf(trig.optString("direction", "ANY")) }.getOrDefault(Direction.ANY),
            lineEnabled = trig.optBoolean("line", true),
            steadyEnabled = trig.optBoolean("steady", true),
        )
        val fraction = trig.optDouble("fraction", 0.5)
        val lineStart = expected.optJSONObject("line")?.optLong("startMs", Long.MIN_VALUE) ?: Long.MIN_VALUE
        val (parts, _) = ClipJobs.judgeClip(
            g, dir, twin, eng, config, fraction, seg, fromMs = lineStart, sheetFrames = expected.optInt("sheetFrames", 15), voting = false,
        )

        val exp = expected.getJSONArray("verdicts")
        val expRecords = ArrayList<VerdictRecord>()
        val borderline = HashSet<Int>()
        for (i in 0 until exp.length()) {
            val o = exp.getJSONObject(i)
            expRecords += VerdictRecord(o.getLong("tMs"), o.optInt("trackId"), Verdict.valueOf(o.getString("verdict")), o.optString("reason").takeIf { it.isNotBlank() && it != "null" }?.lowercase(Locale.ROOT))
            if (o.optBoolean("borderline", false) || mechanicsOnly) borderline += i
        }
        val actual = parts.map { VerdictRecord(it.tMs, it.trackId, it.judgement.verdict, it.judgement.reason?.lowercase(Locale.ROOT)) }
        val report = ReplayComparator.compare(expRecords, actual, timeToleranceMs = expected.optLong("timeToleranceMs", 150L), borderlineAllowed = borderline)
        out.put("expected", expRecords.size).put("actual", actual.size).put("report", report.toString())
        out.put("actualVerdicts", JSONArray(parts.map { JSONObject().put("tMs", it.tMs).put("trackId", it.trackId).put("verdict", it.judgement.verdict.name).put("reason", it.judgement.reason).put("s", it.judgement.s.takeUnless { v -> v.isNaN() }).put("sim", it.judgement.sim.takeUnless { v -> v.isNaN() }).put("latencyMs", it.latencyMs) }))
        out.put("mechanicsOnly", mechanicsOnly)
        if (mechanicsOnly) return if (report.pass) "PASS (mechanics only, backbone ${current?.label}): ${report.matched.size}/${expRecords.size} parts triggered at the expected times; verdict classes not compared"
        else "FAIL (mechanics) $report"
        return if (report.pass) "PASS ${report.matched.size}/${expRecords.size} verdicts match (${report.borderlineFlips.size} borderline flips allowed)"
        else "FAIL $report"
    }

    companion object {
        private const val ASSET_DIR = "selftest/replay_synth"

        /** Copies the bundled clip to filesDir (ReplayFrameSource reads files); re-copies when the asset set changes. */
        fun extract(ctx: Context): File? {
            val names = try {
                ctx.assets.list(ASSET_DIR)?.toList() ?: emptyList()
            } catch (t: Throwable) {
                emptyList()
            }
            if ("expected.json" !in names || "timestamps.json" !in names) return null
            val dst = File(ctx.filesDir, ASSET_DIR)
            val marker = File(dst, ".complete")
            val frameNames = ctx.assets.list("$ASSET_DIR/frames")?.sorted() ?: emptyList()
            val stamp = "${names.size}:${frameNames.size}:${frameNames.lastOrNull()}"
            if (marker.isFile && marker.readText() == stamp) return dst
            dst.deleteRecursively()
            File(dst, "frames").mkdirs()
            for (n in names) if (n != "frames") ctx.assets.open("$ASSET_DIR/$n").use { i -> File(dst, n).outputStream().use { i.copyTo(it) } }
            for (n in frameNames) ctx.assets.open("$ASSET_DIR/frames/$n").use { i -> File(dst, "frames/$n").outputStream().use { i.copyTo(it) } }
            marker.writeText(stamp)
            return dst
        }
    }
}
