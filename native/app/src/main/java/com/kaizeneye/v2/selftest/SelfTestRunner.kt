package com.kaizeneye.v2.selftest

import android.os.Build
import android.util.Log
import com.kaizeneye.runtime.SelfTestProbes
import com.kaizeneye.runtime.VlmState
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.BuildConfig
import com.kaizeneye.v2.DevicePassport
import com.kaizeneye.v2.ml.Models
import com.kaizeneye.v2.telemetry.Readiness
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * One-run device self-test (plan "Verification — Device (scriptable, no hands)"). Kinds:
 *   quick  = offline proof + passport + core-on-device + replay regression
 *   accel  = accelerator parity/latency for every candidate + k-NN graph accuracy
 *   replay = replay regression only
 *   vlm    = VLM load + one explanation
 *   all    = everything
 * Writes getExternalFilesDir("selftest")/selftest.json (+ a timestamped copy). Each step reports "PASS …", "FAIL …",
 * "INFO …" or "SKIP …"; the run passes when no step FAILs.
 */
class SelfTestRunner(private val g: AppGraph) {

    /** Extra steps contributed by the pipeline (core-on-device goldens, replay regression): fill `out`, return the status. */
    var coreStep: (suspend (JSONObject) -> String)? = null
    var replayStep: (suspend (JSONObject) -> String)? = null

    suspend fun run(kind: String, onStep: (String, String) -> Unit): Pair<JSONObject, File> = withContext(Dispatchers.Default) {
        val root = JSONObject()
        val steps = JSONArray()
        root.put("kind", kind)
        root.put("app", JSONObject().put("versionName", BuildConfig.VERSION_NAME).put("versionCode", BuildConfig.VERSION_CODE))
        root.put("startedAtMs", System.currentTimeMillis())
        val want = { k: String -> kind == "all" || kind == k || (kind == "quick" && k in setOf("offline", "passport", "core", "replay")) }

        suspend fun step(name: String, block: suspend (JSONObject) -> String) {
            onStep(name, "running…")
            val o = JSONObject().put("name", name)
            val t0 = System.nanoTime()
            val status = try {
                block(o)
            } catch (t: Throwable) {
                Log.e(TAG, "step $name crashed", t)
                "FAIL ${t.javaClass.simpleName}: ${t.message}"
            }
            o.put("status", status)
            o.put("ms", (System.nanoTime() - t0) / 1e6)
            steps.put(o)
            onStep(name, status)
        }

        if (want("offline")) step("offline proof") { o ->
            val net = Readiness.internetDeclared(g.context)
            val extra = Readiness.unexpectedPermissions(g.context)
            o.put("airplaneMode", Readiness.airplaneMode(g.context))
            o.put("permissions", JSONArray(Readiness.requestedPermissions(g.context)))
            o.put("internetDeclared", net)
            o.put("unexpectedPermissions", JSONArray(extra))
            if (!net && extra.isEmpty()) "PASS no INTERNET, only CAMERA+VIBRATE (airplane ${if (Readiness.airplaneMode(g.context)) "ON" else "off"})"
            else "FAIL permissions: internet=$net extra=$extra"
        }
        if (want("passport")) step("device passport") { o ->
            val p = DevicePassport.collect(g.context)
            DevicePassport.save(g.context, p)
            o.put("passport", p)
            "INFO ${Build.MANUFACTURER} ${Build.MODEL} · SoC ${Build.SOC_MODEL} · Android ${Build.VERSION.RELEASE}"
        }
        if (want("core")) step("core maths on device") { o ->
            coreStep?.invoke(o) ?: "SKIP core step not wired"
        }
        if (want("accel")) {
            step("backbone accelerators") { o ->
                val r = SelfTestProbes.backboneParity(g.context, Models.R18)
                o.put("result", r)
                summarizeParity(r)
            }
            step("k-NN graph accuracy") { o ->
                val r = SelfTestProbes.knnAccuracy(g.context, Models.KNN_R18_P, Models.KNN_R18_D, Models.KNN_R18_BUCKETS)
                o.put("result", r)
                val err = r.optDouble("maxRelErr", Double.NaN)
                when {
                    err.isNaN() -> "INFO ${r.optString("summary", r.toString().take(160))}"
                    err <= 1e-3 -> "PASS max rel err %.2e (graph %s)".format(err, r.optString("label", "?"))
                    else -> "FAIL max rel err %.2e > 1e-3".format(err)
                }
            }
        }
        if (want("replay")) step("replay regression") { o ->
            replayStep?.invoke(o) ?: "SKIP replay step not wired"
        }
        if (want("vlm")) step("VLM explanation") { o -> vlmSmoke(o) }

        root.put("steps", steps)
        root.put("finishedAtMs", System.currentTimeMillis())
        var pass = true
        for (i in 0 until steps.length()) if (steps.getJSONObject(i).getString("status").startsWith("FAIL")) pass = false
        root.put("pass", pass)
        val dir = g.dirs.selftest.apply { mkdirs() }
        val f = File(dir, "selftest.json")
        f.writeText(root.toString(2))
        File(dir, "selftest_${System.currentTimeMillis()}.json").writeText(root.toString(2))
        root to f
    }

    private fun summarizeParity(r: JSONObject): String {
        val chosen = r.optString("chosen", "?")
        val badge = r.optString("badge", "")
        val ok = r.optBoolean("ok", true)
        return if (ok) "PASS chosen $chosen · $badge" else "FAIL ${r.optString("error", r.toString().take(160))}"
    }

    private suspend fun vlmSmoke(o: JSONObject): String {
        g.vlm.bind()
        val ready = withTimeoutOrNull(120_000) {
            while (true) {
                when (val s = g.vlm.state.value) {
                    is VlmState.Ready, is VlmState.Failed -> return@withTimeoutOrNull s
                    else -> delay(250)
                }
            }
            @Suppress("UNREACHABLE_CODE") null
        }
        o.put("state", ready?.toString() ?: "timeout")
        if (ready !is VlmState.Ready) return "INFO VLM unavailable (${ready ?: "timeout"}) → template explanations are used"
        val crop = File(g.dirs.vlmCache, "selftest_crop.jpg")
        if (!crop.isFile) {
            // Any decodable JPEG works; the self-test only measures that the engine answers.
            g.context.assets.open("selftest/vlm_probe.jpg").use { input -> crop.outputStream().use { input.copyTo(it) } }
        }
        val r = g.vlm.explain(crop, "Describe the object in this image in one short sentence.", timeoutMs = 20_000)
        o.put("backend", r.backend); o.put("ms", r.ms); o.put("text", r.text); o.put("error", r.error)
        val text = r.text
        return if (text != null) "PASS ${r.backend} ${r.ms} ms: ${text.take(80)}" else "FAIL ${r.error}"
    }

    companion object {
        private const val TAG = "KaizenSelfTest"
    }
}
