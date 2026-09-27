package com.kaizeneye.runtime

import android.app.ActivityManager
import android.content.Context
import android.os.Process
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/**
 * Accelerator decisions and crash-guard state of the APP process (the probe never writes here; single writer):
 *  - filesDir/accel/decisions.json : {"version":1, "decisions":{key: report}, "strikes":{key:{candidate label: n}}}
 *  - filesDir/accel/trying.json    : the crash-guard marker {key, candidate, phase, pid, token, atMs} while a candidate is tried.
 * On first use in a process, a marker left by a dead process is judged with its ApplicationExitInfo (CrashGuard policy).
 */
internal class DecisionStore private constructor(private val context: Context, private val dir: File) {
    private val decisionsFile = File(dir, "decisions.json")
    private val tryingFile = File(dir, "trying.json")
    private var root: JSONObject = load()

    companion object {
        const val VERSION = 1
        /** Identifies this process instance in markers (pids are reused). */
        val PROCESS_TOKEN: String = UUID.randomUUID().toString()

        @Volatile private var instance: DecisionStore? = null

        fun get(context: Context): DecisionStore {
            instance?.let { return it }
            synchronized(this) {
                instance?.let { return it }
                val app = context.applicationContext ?: context
                val s = DecisionStore(app, File(app.filesDir, "accel"))
                s.judgeStaleMarker()
                instance = s
                return s
            }
        }

        /** Exit reason (ApplicationExitInfo.REASON_*) of our process [pid] that died at/after [notBeforeMs]; null if unknown. */
        fun exitReason(context: Context, pid: Int, notBeforeMs: Long): Int? = try {
            val am = context.getSystemService(ActivityManager::class.java)
            am?.getHistoricalProcessExitReasons(context.packageName, pid, 8)
                ?.firstOrNull { it.pid == pid && it.timestamp >= notBeforeMs - 10_000 }?.reason
        } catch (t: Throwable) {
            RtLog.w("exit-reason lookup failed: ${t.brief()}")
            null
        }
    }

    private fun load(): JSONObject = try {
        if (decisionsFile.isFile) {
            val o = JSONObject(decisionsFile.readText())
            if (o.optInt("version") == VERSION) o else fresh()
        } else {
            fresh()
        }
    } catch (t: Throwable) {
        RtLog.w("unreadable ${decisionsFile.name}, starting fresh: ${t.brief()}")
        fresh()
    }

    private fun fresh() = JSONObject().put("version", VERSION).put("decisions", JSONObject()).put("strikes", JSONObject())

    private fun save() {
        try {
            writeTextAtomic(decisionsFile, root.toString())
        } catch (t: Throwable) {
            RtLog.w("cannot write ${decisionsFile.name}: ${t.brief()}")
        }
    }

    private fun decisions(): JSONObject = root.optJSONObject("decisions") ?: JSONObject().also { root.put("decisions", it) }
    private fun strikesRoot(): JSONObject = root.optJSONObject("strikes") ?: JSONObject().also { root.put("strikes", it) }

    // ------------------------------------------------------------------------------------------------- decisions
    @Synchronized
    fun decision(key: String): AcceleratorReport? = try {
        decisions().optJSONObject(key)?.let { ReportJson.fromJson(it) }
    } catch (t: Throwable) {
        RtLog.w("bad cached decision for $key: ${t.brief()}")
        null
    }

    @Synchronized
    fun saveDecision(key: String, report: AcceleratorReport) {
        decisions().put(key, ReportJson.toJson(report.copy(fromCache = false, forced = false)))
        save()
    }

    @Synchronized
    fun forgetDecisions() {
        root.put("decisions", JSONObject())
        save()
        clearTrying()
    }

    @Synchronized
    fun forgetCrashes() {
        root.put("strikes", JSONObject())
        save()
    }

    // --------------------------------------------------------------------------------------------------- strikes
    @Synchronized
    fun strikes(key: String, label: String): Int = strikesRoot().optJSONObject(key)?.optInt(label, 0) ?: 0

    @Synchronized
    fun setStrikes(key: String, label: String, n: Int) {
        val k = strikesRoot().optJSONObject(key) ?: JSONObject().also { strikesRoot().put(key, it) }
        k.put(label, n)
        save()
    }

    fun isCrashed(key: String, label: String): Boolean = CrashGuard.isCrashed(strikes(key, label))

    // ---------------------------------------------------------------------------------------------------- marker
    @Synchronized
    fun writeTrying(key: String, label: String, phase: String) {
        val m = JSONObject()
            .put("key", key).put("candidate", label).put("phase", phase)
            .put("pid", Process.myPid()).put("token", PROCESS_TOKEN).put("atMs", System.currentTimeMillis())
        try {
            writeTextAtomic(tryingFile, m.toString())
        } catch (t: Throwable) {
            RtLog.w("cannot write crash-guard marker: ${t.brief()}")
        }
    }

    @Synchronized
    fun clearTrying() {
        try {
            if (tryingFile.exists()) tryingFile.delete()
        } catch (_: Throwable) {
        }
    }

    private fun readTrying(): TryingMarker? = try {
        if (!tryingFile.isFile) null else JSONObject(tryingFile.readText()).let {
            TryingMarker(
                it.getString("key"), it.getString("candidate"), it.optString("phase", CrashGuard.Phase.PROBE),
                it.optInt("pid", 0), it.optString("token", ""), it.optLong("atMs", 0L),
            )
        }
    } catch (t: Throwable) {
        RtLog.w("unreadable crash-guard marker, ignoring: ${t.brief()}")
        null
    }

    /** A marker from an earlier app process: that process died while a candidate was being tried. */
    private fun judgeStaleMarker() {
        val m = readTrying() ?: return
        if (m.token == PROCESS_TOKEN) return
        val reason = exitReason(context, m.pid, m.atMs)
        val death = CrashGuard.classify(reason)
        val before = strikes(m.key, m.candidate)
        val after = CrashGuard.strikesAfterStaleMarker(before, m.phase, death)
        if (after != before) setStrikes(m.key, m.candidate, after)
        RtLog.w(
            "crash guard: app process ${m.pid} died (${CrashGuard.reasonName(reason)}) while trying ${m.candidate} " +
                "(phase ${m.phase}); strikes $before -> $after${if (CrashGuard.isCrashed(after)) " - skipped from now on" else ""}",
        )
        clearTrying()
    }
}

/** AcceleratorReport <-> JSON (org.json; non-finite numbers are stored as null). */
internal object ReportJson {
    fun toJson(r: AcceleratorReport): JSONObject = JSONObject()
        .put("model", r.model).put("chosen", r.chosen).put("accel", r.accel.name).put("modelId", r.modelId)
        .put("badge", r.badge).putNum("speedupVsCpu", r.speedupVsCpu)
        .put("results", JSONArray().apply { r.results.forEach { put(candToJson(it)) } })
        .put("fromCache", r.fromCache).put("measuredAtMs", r.measuredAtMs).put("cacheKey", r.cacheKey)
        .put("forced", r.forced).put("note", r.note ?: JSONObject.NULL)

    fun fromJson(o: JSONObject): AcceleratorReport {
        val arr = o.optJSONArray("results") ?: JSONArray()
        return AcceleratorReport(
            model = o.getString("model"),
            chosen = o.getString("chosen"),
            accel = Accel.valueOf(o.getString("accel")),
            modelId = o.getString("modelId"),
            badge = o.optString("badge", ""),
            speedupVsCpu = o.num("speedupVsCpu"),
            results = List(arr.length()) { candFromJson(arr.getJSONObject(it)) },
            fromCache = o.optBoolean("fromCache", false),
            measuredAtMs = o.optLong("measuredAtMs", 0L),
            cacheKey = o.optString("cacheKey", ""),
            forced = o.optBoolean("forced", false),
            note = o.str("note"),
        )
    }

    fun candToJson(c: CandidateResult): JSONObject = JSONObject()
        .put("name", c.name).put("modelId", c.modelId).put("ok", c.ok)
        .putNum("medianMs", c.medianMs).putNum("minMs", c.minMs).putNum("p95Ms", c.p95Ms)
        .putNum("cosine", c.cosine).putNum("patchCosine", c.patchCosine).putNum("relErr", c.relErr).putNum("loadMs", c.loadMs)
        .put("error", c.error ?: JSONObject.NULL).put("note", c.note ?: JSONObject.NULL)

    fun candFromJson(o: JSONObject) = CandidateResult(
        name = o.getString("name"),
        modelId = o.optString("modelId", ""),
        ok = o.optBoolean("ok", false),
        medianMs = o.num("medianMs"),
        minMs = o.num("minMs"),
        p95Ms = o.num("p95Ms"),
        cosine = o.num("cosine"),
        patchCosine = o.num("patchCosine"),
        relErr = o.num("relErr"),
        loadMs = o.num("loadMs"),
        error = o.str("error"),
        note = o.str("note"),
    )

    private fun JSONObject.putNum(k: String, v: Double?): JSONObject = put(k, if (v == null || !v.isFinite()) JSONObject.NULL else v)

    private fun JSONObject.num(k: String): Double? = if (!has(k) || isNull(k)) null else optDouble(k).takeIf { !it.isNaN() }

    private fun JSONObject.str(k: String): String? = if (!has(k) || isNull(k)) null else optString(k)
}
