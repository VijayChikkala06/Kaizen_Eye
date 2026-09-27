package com.kaizeneye.runtime

import android.content.Context
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.max
import kotlin.math.min

/** One k-NN graph bucket: feats [p*d], bankT [d*k], bankNorm [k] -> minD2 [p]. Not re-entrant. */
internal interface KnnGraph : AutoCloseable {
    val p: Int
    val d: Int
    val k: Int
    fun run(featsBlock: FloatArray, bankT: FloatArray, bankNorm: FloatArray): FloatArray
}

/** Pure checks of a graph against the exact search on fixed data (KnnMath.agreement, twin-spec §5 tolerance). */
internal object KnnVerify {
    /** Deterministic problem for bucket K: P fake feature rows vs K-24 bank rows (padding exercised), every 20th row masked. */
    class Problem(val feats: FloatArray, val bank: FloatArray, val rows: Int, val extra: FloatArray)

    fun problem(p: Int, d: Int, k: Int): Problem {
        val rnd = BenchInputs.Lcg(1234 + k)
        val feats = BenchInputs.fakeFeatures(rnd, p, d)
        val rows = max(1, k - 24)
        val bank = BenchInputs.fakeFeatures(rnd, rows, d)
        val extra = FloatArray(rows)
        var i = 0
        while (i < rows) {
            extra[i] = KnnMath.KNN_MASK
            i += 20
        }
        return Problem(feats, bank, rows, extra)
    }

    /** Graph vs exact float64 on every [stride]-th query row. */
    fun check(g: KnnGraph, stride: Int = 16): KnnMath.Agreement {
        val pr = problem(g.p, g.d, g.k)
        val prep = KnnMath.prepareBank(pr.bank, pr.rows, g.d, intArrayOf(g.k))
        val got = KnnMath.graphMinSq(pr.feats, g.p, g.d, prep, pr.extra, g.p) { _, fb, bt, bn -> g.run(fb, bt, bn) }
        val sub = IntArray((g.p + stride - 1) / stride) { it * stride }
        val subFeats = FloatArray(sub.size * g.d)
        for ((j, q) in sub.withIndex()) System.arraycopy(pr.feats, q * g.d, subFeats, j * g.d, g.d)
        val ref = KnnMath.exactMinSq(subFeats, sub.size, pr.bank, pr.rows, g.d, pr.extra)
        return KnnMath.agreement(FloatArray(sub.size) { got[sub[it]] }, ref)
    }

    /** [warmup] + [runs] timed graph runs (write inputs + run + read output) on the fixed problem. */
    fun time(g: KnnGraph, warmup: Int = AccelRules.WARMUP_RUNS, runs: Int = AccelRules.TIMED_RUNS): DoubleArray {
        val pr = problem(g.p, g.d, g.k)
        val prep = KnnMath.prepareBank(pr.bank, pr.rows, g.d, intArrayOf(g.k))
        val c = prep.chunks[0]
        val norm = FloatArray(c.k).also { KnnMath.fillBankNorm(c, pr.extra, it) }
        repeat(warmup) { g.run(pr.feats, c.bankT, norm) }
        return DoubleArray(runs) {
            val t = nowMs()
            g.run(pr.feats, c.bankT, norm)
            nowMs() - t
        }
    }
}

/**
 * A knn_p{P}_d{D}_k{K}.tflite graph on LiteRT. Inputs are identified by SHAPE from the flatbuffer (TfliteInfo): feats [P,D],
 * bankT [D,K], bankNorm [K], output [P]. Binding: by input name (signature names, or tensor names when the graph has no
 * signature - the case for our graphs), falling back to the positional buffer lists; each binding must reproduce the
 * exact search (KnnVerify) before it is used.
 */
internal class LiteRtKnnGraph private constructor(
    private val model: CompiledModel,
    override val p: Int,
    override val d: Int,
    override val k: Int,
    private val featsBuf: TensorBuffer,
    private val bankTBuf: TensorBuffer,
    private val normBuf: TensorBuffer,
    private val outBuf: TensorBuffer,
    private val allBuffers: List<TensorBuffer>,
    private val runner: () -> Unit,
    val binding: String,
    val accel: Accel,
) : KnnGraph {
    var agreement: KnnMath.Agreement? = null
        private set
    var loadMs: Double = 0.0
        private set

    override fun run(featsBlock: FloatArray, bankT: FloatArray, bankNorm: FloatArray): FloatArray {
        featsBuf.writeFloat(featsBlock)
        bankTBuf.writeFloat(bankT)
        normBuf.writeFloat(bankNorm)
        runner()
        return outBuf.readFloat()
    }

    private fun closeBuffers() = LiteRtModels.closeQuietly(allBuffers)

    override fun close() {
        closeBuffers()
        LiteRtModels.closeQuietly(model)
    }

    companion object {
        fun create(ctx: Context, loc: LocatedModel, p: Int, d: Int, k: Int, accel: Accel): LiteRtKnnGraph {
            val info = TfliteInfo.parse(LiteRtModels.readBytes(ctx, loc))
            val tf = info.inputWithShape(p, d)
            val tb = info.inputWithShape(d, k)
            val tn = info.inputWithShape(k)
            val to = info.outputWithShape(p)
            check(tf != null && tb != null && tn != null && to != null && setOf(tf.index, tb.index, tn.index).size == 3) {
                "${loc.asset.fileName}: expected inputs [$p,$d] [$d,$k] [$k] and output [$p]; the model has $info"
            }
            check(listOf(tf, tb, tn, to).all { it.type == TfliteInfo.FLOAT32 }) { "${loc.asset.fileName}: k-NN graph must be float32 ($info)" }
            val t0 = nowMs()
            val model = LiteRtModels.create(ctx, loc, accel)
            val loadMs = nowMs() - t0
            var last: Throwable? = null
            for (how in listOf("name", "index")) {
                var g: LiteRtKnnGraph? = null
                try {
                    g = if (how == "name") byName(model, info, tf, tb, tn, to, p, d, k, accel) else byIndex(model, info, tf, tb, tn, to, p, d, k, accel)
                    val a = KnnVerify.check(g)
                    if (a.pass) {
                        g.agreement = a
                        g.loadMs = loadMs
                        RtLog.i("${loc.asset.fileName} on $accel: bound by $how, max|Δd|/max d = ${a.maxAbsDdOverMaxD}")
                        return g
                    }
                    last = IllegalStateException("binding by $how disagrees with the exact search (max|Δd|/max d = ${a.maxAbsDdOverMaxD})")
                    g.closeBuffers()
                } catch (t: Throwable) {
                    last = t
                    g?.closeBuffers()
                }
            }
            LiteRtModels.closeQuietly(model)
            throw IllegalStateException("${loc.asset.fileName} unusable on $accel: ${last?.brief()}", last)
        }

        private fun byName(
            model: CompiledModel, info: TfliteInfo,
            tf: TfliteInfo.Tensor, tb: TfliteInfo.Tensor, tn: TfliteInfo.Tensor, to: TfliteInfo.Tensor,
            p: Int, d: Int, k: Int, accel: Accel,
        ): LiteRtKnnGraph {
            val sig = info.signatures.firstOrNull { it.subgraph == 0 }?.key ?: ""
            val fName = info.signatureInputName(tf)
            val bName = info.signatureInputName(tb)
            val nName = info.signatureInputName(tn)
            val oName = info.signatureOutputName(to)
            val made = ArrayList<TensorBuffer>()
            try {
                val fb = model.createInputBuffer(inputName = fName, signature = sig).also { made += it }
                val bb = model.createInputBuffer(inputName = bName, signature = sig).also { made += it }
                val nb = model.createInputBuffer(inputName = nName, signature = sig).also { made += it }
                val ob = model.createOutputBuffer(outputName = oName, signature = sig).also { made += it }
                val ins = mapOf(fName to fb, bName to bb, nName to nb)
                val outs = mapOf(oName to ob)
                return LiteRtKnnGraph(model, p, d, k, fb, bb, nb, ob, made, { model.run(ins, outs, sig) }, "name", accel)
            } catch (t: Throwable) {
                LiteRtModels.closeQuietly(made)
                throw t
            }
        }

        private fun byIndex(
            model: CompiledModel, info: TfliteInfo,
            tf: TfliteInfo.Tensor, tb: TfliteInfo.Tensor, tn: TfliteInfo.Tensor, to: TfliteInfo.Tensor,
            p: Int, d: Int, k: Int, accel: Accel,
        ): LiteRtKnnGraph {
            val sig = info.signatures.firstOrNull { it.subgraph == 0 }
            val inOrder = sig?.inputs?.map { it.second } ?: info.inputs.map { it.index }
            val outOrder = sig?.outputs?.map { it.second } ?: info.outputs.map { it.index }
            val ins = model.createInputBuffers()
            val outs = try {
                model.createOutputBuffers()
            } catch (t: Throwable) {
                LiteRtModels.closeQuietly(ins)
                throw t
            }
            val all = ins + outs
            try {
                check(ins.size == inOrder.size && outs.size == outOrder.size) {
                    "buffer counts ${ins.size}/${outs.size} differ from the model's ${inOrder.size}/${outOrder.size}"
                }
                val fi = inOrder.indexOf(tf.index)
                val bi = inOrder.indexOf(tb.index)
                val ni = inOrder.indexOf(tn.index)
                val oi = outOrder.indexOf(to.index)
                check(fi >= 0 && bi >= 0 && ni >= 0 && oi >= 0) { "cannot map tensors to buffer positions" }
                return LiteRtKnnGraph(model, p, d, k, ins[fi], ins[bi], ins[ni], outs[oi], all, { model.run(ins, outs) }, "index", accel)
            } catch (t: Throwable) {
                LiteRtModels.closeQuietly(all)
                throw t
            }
        }
    }
}

/**
 * KnnEngine: LiteRT graphs (bucketed, chunked, padded - KnnMath) when available, else the exact Kotlin search. A graph
 * failure at run time switches the engine to the exact search for good (logged, label + report updated). Serialised.
 */
internal class KnnEngineImpl(
    graphs: Map<Int, KnnGraph>,
    private val p: Int,
    private val d: Int,
    label: String,
    report: AcceleratorReport?,
    private val exactThreads: Int,
) : KnnEngine {
    private val lock = Any()
    private var graphs: Map<Int, KnnGraph> = graphs
    private val buckets: IntArray = graphs.keys.sorted().toIntArray()
    private var disabled: String? = null
    private var closed = false
    private var cachedBank: FloatArray? = null
    private var cachedPrep: KnnMath.PreparedBank? = null
    private var warnedDim = false

    @Volatile private var labelNow: String = label
    @Volatile private var reportNow: AcceleratorReport? = report

    override val label: String get() = labelNow
    override val report: AcceleratorReport? get() = reportNow

    /** True while the LiteRT graphs are in use. */
    val usingGraphs: Boolean get() = synchronized(lock) { graphs.isNotEmpty() && disabled == null }

    override fun minSq(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?): FloatArray =
        synchronized(lock) {
            check(!closed) { "k-NN engine is closed" }
            KnnMath.validate(feats, nq, bank, nb, dim, extra)
            if (nq == 0) return FloatArray(0)
            if (nb == 0) return FloatArray(nq) { KnnMath.KNN_MASK }
            // A NaN / infinite feature (a glitched accelerator frame) must not disable the graphs for good: answer NaN.
            for (i in 0 until nq * dim) if (!feats[i].isFinite()) return FloatArray(nq) { Float.NaN }
            if (extra != null) for (v in extra) if (v.isNaN()) return FloatArray(nq) { Float.NaN }
            if (graphs.isNotEmpty() && disabled == null) {
                if (dim == d) {
                    try {
                        val prep = prepared(bank, nb, dim)
                        return KnnMath.graphMinSq(feats, nq, dim, prep, extra, p) { k, fb, bt, bn -> graphs.getValue(k).run(fb, bt, bn) }
                    } catch (t: Throwable) {
                        disableGraphsLocked(t)
                    }
                } else if (!warnedDim) {
                    warnedDim = true
                    RtLog.w("k-NN: dim $dim != graph dim $d, using the exact search for these calls")
                }
            }
            KnnMath.exactMinSq(feats, nq, bank, nb, dim, extra, exactThreads)
        }

    private fun prepared(bank: FloatArray, nb: Int, dim: Int): KnnMath.PreparedBank {
        val c = cachedPrep
        if (c != null && cachedBank === bank && c.nb == nb && c.dim == dim && c.fingerprint == KnnMath.fingerprint(bank, nb * dim)) return c
        val prep = KnnMath.prepareBank(bank, nb, dim, buckets)
        cachedBank = bank
        cachedPrep = prep
        return prep
    }

    private fun disableGraphsLocked(t: Throwable) {
        disabled = t.brief()
        RtLog.e("k-NN graph failed at run time (${t.brief()}); exact CPU search from now on", t)
        graphs.values.forEach { g -> try { g.close() } catch (_: Throwable) { } }
        graphs = emptyMap()
        labelNow = "k-NN exact CPU search (Kotlin, float64, $exactThreads threads) - graph failed at run time: ${t.brief()}"
        reportNow = reportNow?.copy(
            chosen = "CPU exact", accel = Accel.CPU, badge = "CPU exact (graph failed)", speedupVsCpu = null,
            note = joinNotes(reportNow?.note, "graph failed at run time (${t.brief()}); exact search for good"),
        )
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            graphs.values.forEach { g -> try { g.close() } catch (_: Throwable) { } }
            graphs = emptyMap()
            cachedBank = null
            cachedPrep = null
        }
    }
}

/** Runtime.loadKnn. */
internal object KnnLoader {
    fun exactThreads(): Int = min(4, java.lang.Runtime.getRuntime().availableProcessors()).coerceAtLeast(1)

    fun load(ctx: Context, p: Int, dim: Int, buckets: List<Int>, forceAccel: Accel?): KnnEngine {
        NpuEnv.ensure(ctx)
        require(p > 0 && dim > 0) { "bad k-NN graph size P=$p D=$dim" }
        val ks = buckets.filter { it > 0 }.distinct().sorted()
        require(ks.isNotEmpty()) { "no k-NN buckets" }
        val name = "knn_p${p}_d$dim"
        val threads = exactThreads()
        val notes = ArrayList<String>()
        val located = LinkedHashMap<Int, LocatedModel>()
        for (k in ks) {
            val a = KnownModels.knnGraph(p, dim, k)
            val l = try { ModelLocator.locate(ctx, a) } catch (t: Throwable) { null }
            when {
                l == null -> notes += "k=$k: ${a.fileName} not found"
                !l.sha256Ok -> notes += "k=$k: ${l.note}"
                else -> located[k] = l
            }
        }
        val key = CacheKeys.decisionKey(
            name, located.map { (k, l) -> "k$k" to (l.actualSha256 ?: l.asset.sha256) },
            AppInfo.versionCode(ctx), AppInfo.fingerprint(), AppInfo.nativeStamp(ctx),
        )
        val cpuGraphs = LinkedHashMap<Int, KnnGraph>()
        for ((k, l) in located) {
            try {
                cpuGraphs[k] = LiteRtKnnGraph.create(ctx, l, p, dim, k, Accel.CPU)
            } catch (t: Throwable) {
                notes += "k=$k graph unusable: ${t.brief()}"
                RtLog.w("k-NN k=$k: ${t.brief()}")
            }
        }
        if (cpuGraphs.isEmpty()) return exactEngine(name, key, threads, notes)

        val kMax = cpuGraphs.keys.max()
        val big = cpuGraphs.getValue(kMax) as LiteRtKnnGraph
        val cpuT = try { AccelRules.timing(KnnVerify.time(big)) } catch (t: Throwable) { null }
        val results = ArrayList<CandidateResult>()
        results += CandidateResult(
            name = "CPU", modelId = "${name}_k$kMax", ok = true,
            medianMs = cpuT?.medianMs, minMs = cpuT?.minMs, p95Ms = cpuT?.p95Ms,
            cosine = null, patchCosine = null, relErr = big.agreement?.maxAbsDdOverMaxD, loadMs = big.loadMs,
            note = "fp32 graph (bound by ${big.binding}); relErr = max|Δd|/max d vs exact float64 (twin-spec §5: <= 1e-3)",
        )
        var graphs: Map<Int, KnnGraph> = cpuGraphs
        var accel = Accel.CPU
        var badge = AccelRules.badge(Candidate(Accel.CPU, KnownModels.knnGraph(p, dim, kMax), null), cpuT?.medianMs, null)
        var speedup: Double? = null
        var forcedNote: String? = null

        if (forceAccel != null && forceAccel != Accel.CPU) {
            val label = forceAccel.name
            when {
                forceAccel == Accel.NPU && !NpuEnv.npuSocSupported() -> forcedNote = "forced NPU refused: SoC '${NpuEnv.socModel}' is not SM8850*"
                DecisionStore.get(ctx).isCrashed(key, label) -> forcedNote = "forced $label refused: crashed during an earlier attempt"
                else -> {
                    // Forced accelerator for the graphs (spec default is fp32 CPU: KNN_MASK overflows fp16). Created in the app
                    // under the crash-guard marker ("main" phase), verified against the exact search like the CPU graphs.
                    val store = DecisionStore.get(ctx)
                    val acc = LinkedHashMap<Int, KnnGraph>()
                    var err: Throwable? = null
                    store.writeTrying(key, label, CrashGuard.Phase.MAIN)
                    try {
                        for (k in cpuGraphs.keys) acc[k] = LiteRtKnnGraph.create(ctx, located.getValue(k), p, dim, k, forceAccel)
                    } catch (t: Throwable) {
                        err = t
                    }
                    store.clearTrying()
                    if (err == null) {
                        val t = try { AccelRules.timing(KnnVerify.time(acc.getValue(kMax))) } catch (x: Throwable) { null }
                        val cand = Candidate(forceAccel, KnownModels.knnGraph(p, dim, kMax), null)
                        val r = CandidateResult(
                            name = label, modelId = "${name}_k$kMax", ok = true,
                            medianMs = t?.medianMs, minMs = t?.minMs, p95Ms = t?.p95Ms,
                            cosine = null, patchCosine = null, relErr = (acc.getValue(kMax) as LiteRtKnnGraph).agreement?.maxAbsDdOverMaxD,
                            loadMs = (acc.getValue(kMax) as LiteRtKnnGraph).loadMs,
                        )
                        results.add(0, r)
                        badge = AccelRules.forcedBadge(cand, r, results.last())
                        speedup = if (t != null && cpuT != null) cpuT.medianMs / t.medianMs else null
                        cpuGraphs.values.forEach { g -> try { g.close() } catch (_: Throwable) { } }
                        graphs = acc
                        accel = forceAccel
                    } else {
                        acc.values.forEach { g -> try { g.close() } catch (_: Throwable) { } }
                        results.add(0, CandidateResult(label, "${name}_k$kMax", false, null, null, null, null, null, null, null, error = err.brief()))
                        forcedNote = "forced $label failed (${err.brief()}); using the CPU graphs"
                    }
                }
            }
        }
        val report = AcceleratorReport(
            model = name,
            chosen = accel.name,
            accel = accel,
            modelId = "${name}_k${graphs.keys.joinToString("+")}",
            badge = badge,
            speedupVsCpu = speedup,
            results = results,
            fromCache = false,
            measuredAtMs = System.currentTimeMillis(),
            cacheKey = key,
            forced = forceAccel != null,
            note = joinNotes(*(notes + listOfNotNull(forcedNote)).toTypedArray()),
        )
        val label = "k-NN LiteRT $accel fp32 graphs P=$p D=$dim K=${graphs.keys.joinToString(",")} (exact CPU fallback)"
        RtLog.i("$label: ${AccelRules.describe(report)}")
        return KnnEngineImpl(graphs, p, dim, label, report, threads)
    }

    private fun exactEngine(name: String, key: String, threads: Int, notes: List<String>): KnnEngine {
        val label = "k-NN exact CPU search (Kotlin, float64, $threads threads) - graphs unavailable: ${notes.joinToString("; ")}"
        RtLog.w(label)
        val report = AcceleratorReport(
            model = name, chosen = "CPU exact", accel = Accel.CPU, modelId = "$name-exact", badge = "CPU exact",
            speedupVsCpu = null, results = emptyList(), fromCache = false, measuredAtMs = System.currentTimeMillis(),
            cacheKey = key, note = joinNotes(*notes.toTypedArray()),
        )
        return KnnEngineImpl(emptyMap(), 0, 0, label, report, threads)
    }
}
