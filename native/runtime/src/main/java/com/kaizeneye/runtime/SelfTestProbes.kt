package com.kaizeneye.runtime

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max

/**
 * Device self-tests for the scripted verification (`adb shell am start ... --es selftest all` in the app writes their JSON
 * to selftest.json). Never throw: failures come back as {"pass": false, "error": ...}.
 */
object SelfTestProbes {
    /**
     * Loads [spec] as the app would (cached decision / benchmark) and compares the chosen instance with a fresh float CPU
     * run of the reference variant in this process, on the fixed benchmark image: cosine, relative L2 error, mean
     * per-patch cosine, the gate verdict for the chosen variant, median/p95 timings, measured speed-up; plus an embed()
     * sanity check on the rounded image bytes.
     */
    suspend fun backboneParity(context: Context, spec: BackboneSpec): JSONObject = withContext(Dispatchers.IO) {
        val o = JSONObject().put("test", "backboneParity").put("model", spec.name).put("device", deviceJson())
        var loaded: FeatureBackbone? = null
        try {
            val bb = Runtime.loadBackbone(context, spec)
            loaded = bb
            o.put("report", ReportJson.toJson(bb.report))
            o.put("summary", AccelRules.describe(bb.report))
            o.put("modelId", bb.modelId).put("sha256", bb.sha256).put("hasCls", bb.hasCls)
            val impl = bb as BackboneImpl
            val img = BenchInputs.image(spec.inputSize)
            val chosenOut = impl.runFloat(img)
            val chosenTimes = DoubleArray(AccelRules.TIMED_RUNS) {
                val t = nowMs()
                impl.runFloat(img)
                nowMs() - t
            }
            val ref = spec.variants.filter { !it.int8 }
                .firstNotNullOfOrNull { v -> ModelLocator.locate(context, v)?.takeIf { it.sha256Ok } }
                ?: throw IllegalStateException("no usable float variant for the CPU reference")
            val cpu = BackboneBench.measure(
                context, ref.source, requireNotNull(ref.path), Accel.CPU, spec.inputSize, spec.gh * spec.gw * spec.dim, spec.dim,
                AccelRules.WARMUP_RUNS, AccelRules.TIMED_RUNS,
            )
            val int8 = spec.variants.firstOrNull { it.id == bb.modelId }?.int8 == true
            val cmp = AccelRules.compare(chosenOut, cpu.patches, spec.dim)
            val gate = AccelRules.accurate(cmp, int8)
            o.put(
                "parity",
                JSONObject().put("reference", "${ref.asset.id} on CPU (${LiteRtModels.CPU_THREADS} threads, this process)")
                    .put("cosine", num(cmp.cosine)).put("relErr", num(cmp.relErr)).put("patchCosine", num(cmp.patchCosine))
                    .put("finite", cmp.finite).put("gate", if (int8) "int8: patch cosine >= ${AccelRules.INT8_MIN_PATCH_COSINE}" else "float: cosine >= ${AccelRules.MIN_COSINE}, relErr <= ${AccelRules.MAX_REL_ERR}")
                    .put("gatePass", gate),
            )
            val tc = AccelRules.timing(chosenTimes)
            val tr = AccelRules.timing(cpu.timesMs)
            o.put(
                "timing",
                JSONObject().put("chosen", timingJson(tc)).put("cpuReference", timingJson(tr))
                    .put("speedupVsCpuReference", if (tc != null && tr != null && tc.medianMs > 0) num(tr.medianMs / tc.medianMs) else JSONObject.NULL)
                    .put("accel", impl.accel.name),
            )
            val e = bb.embed(BenchInputs.imageBytes(spec.inputSize))
            val finite = e.all { it.isFinite() }
            o.put("embed", JSONObject().put("values", e.size).put("expected", spec.gh * spec.gw * spec.dim).put("finite", finite))
            o.put("pass", gate && finite && e.size == spec.gh * spec.gw * spec.dim)
        } catch (t: Throwable) {
            RtLog.e("self-test backboneParity(${spec.name}) failed: ${t.brief()}", t)
            o.put("pass", false).put("error", t.brief())
        } finally {
            try { loaded?.close() } catch (_: Throwable) { }
        }
        o
    }

    /**
     * Loads the k-NN engine as the app would and compares it with the exact float64 search (every 4th query) on synthetic
     * feature-like data: a bank that fills a middle bucket, the same with a leave-one-out mask, a small bank (bucket
     * padding), a bank above the largest bucket (2 chunks) and P+37 queries (query split + padding). twin-spec §5 criterion:
     * max |Δd| <= 1e-3 * max(d). Timings: engine median of 5, exact search (the fallback) once.
     */
    suspend fun knnAccuracy(context: Context, P: Int, dim: Int, buckets: List<Int>): JSONObject = withContext(Dispatchers.IO) {
        val o = JSONObject().put("test", "knnAccuracy").put("P", P).put("dim", dim)
            .put("buckets", JSONArray(buckets)).put("device", deviceJson())
        var loaded: KnnEngine? = null
        try {
            val eng = Runtime.loadKnn(context, P, dim, buckets)
            loaded = eng
            o.put("label", eng.label)
            eng.report?.let { o.put("report", ReportJson.toJson(it)) }
            val ks = buckets.filter { it > 0 }.distinct().sorted()
            val kMin = ks.first()
            val kMax = ks.last()
            val kMid = ks[ks.size / 2]
            class Case(val name: String, val rows: Int, val masked: Boolean, val nq: Int)
            val cases = listOf(
                Case("bank $kMid rows (bucket $kMid)", kMid, false, P),
                Case("bank $kMid rows, leave-one-out mask", kMid, true, P),
                Case("bank ${max(1, kMin / 2)} rows (bucket $kMin, padded)", max(1, kMin / 2), false, P),
                Case("bank ${kMax + 200} rows (2 chunks)", kMax + 200, false, P),
                Case("${P + 37} queries (split + padding), masked", kMid, true, P + 37),
            )
            val rnd = BenchInputs.Lcg(42)
            val feats = BenchInputs.fakeFeatures(rnd, P + 37, dim)
            val arr = JSONArray()
            var allPass = true
            val threads = KnnLoader.exactThreads()
            for ((ci, c) in cases.withIndex()) {
                val bank = BenchInputs.fakeFeatures(rnd, c.rows, dim)
                val extra = if (c.masked) FloatArray(c.rows) { if (it % 20 == 3) KnnMath.KNN_MASK else 0f } else null
                val got = eng.minSq(feats, c.nq, bank, c.rows, dim, extra)
                val times = DoubleArray(5) {
                    val t = nowMs()
                    eng.minSq(feats, c.nq, bank, c.rows, dim, extra)
                    nowMs() - t
                }
                val sub = IntArray((c.nq + 3) / 4) { it * 4 }
                val subFeats = FloatArray(sub.size * dim)
                for ((j, q) in sub.withIndex()) System.arraycopy(feats, q * dim, subFeats, j * dim, dim)
                val ref = KnnMath.exactMinSq(subFeats, sub.size, bank, c.rows, dim, extra, threads)
                val a = KnnMath.agreement(FloatArray(sub.size) { got[sub[it]] }, ref)
                val exactMs = if (ci == 0) {
                    val t = nowMs()
                    KnnMath.exactMinSq(feats, c.nq, bank, c.rows, dim, extra, threads)
                    nowMs() - t
                } else {
                    null
                }
                allPass = allPass && a.pass
                arr.put(
                    JSONObject().put("case", c.name).put("rows", c.rows).put("queries", c.nq)
                        .put("engineMedianMs", num(AccelRules.timing(times)?.medianMs))
                        .put("exactAllQueriesMs", if (exactMs == null) JSONObject.NULL else num(exactMs))
                        .put("maxAbsDdOverMaxD", num(a.maxAbsDdOverMaxD)).put("maxRelErrD2", num(a.maxRelErrD2))
                        .put("checkedQueries", a.rows).put("pass", a.pass),
                )
            }
            o.put("cases", arr).put("pass", allPass)
        } catch (t: Throwable) {
            RtLog.e("self-test knnAccuracy failed: ${t.brief()}", t)
            o.put("pass", false).put("error", t.brief())
        } finally {
            try { loaded?.close() } catch (_: Throwable) { }
        }
        o
    }

    private fun num(v: Double?): Any = if (v == null || !v.isFinite()) JSONObject.NULL else v

    private fun timingJson(t: AccelRules.Timing?): Any =
        if (t == null) JSONObject.NULL else JSONObject().put("medianMs", t.medianMs).put("minMs", t.minMs).put("p95Ms", t.p95Ms)

    private fun deviceJson(): JSONObject = JSONObject()
        .put("socManufacturer", NpuEnv.socManufacturer).put("socModel", NpuEnv.socModel)
        .put("npuAllowed", NpuEnv.npuSocSupported()).put("adspLibraryPath", NpuEnv.adspLibraryPath ?: JSONObject.NULL)
        .put("process", NpuEnv.processName()).put("fingerprint", AppInfo.fingerprint()).put("sdk", Build.VERSION.SDK_INT)
}
