package com.kaizeneye.runtime

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/*
 * k-NN maths (twin-spec §5), pure Kotlin (JVM-tested in KnnMathTest).
 *
 * minSq(p) = min over bank rows b of ( extra[b] + ||f_p - b||^2 ). extra[b] = KNN_MASK (1e30) excludes a row; if every row
 * is excluded (or the bank is empty) the result is >= 1e30.
 *
 * LiteRT graph knn_p{P}_d{D}_k{K}.tflite (tools/make_knn_model.py): inputs feats [P,D], bankT [D,K], bankNorm [K], output
 * minD2 [P] = relu( min_k(bankNorm_k - 2 f.b_k) + |f|^2 ), float32. Driving it:
 *   - bank rows are split into chunks of at most the largest bucket K; each chunk uses the smallest bucket K >= its rows,
 *     transposed into bankT with zero columns for padding;
 *   - bankNorm = |b|^2 (float64, rounded once) + extra, and KNN_MASK on padding columns;
 *   - queries are processed in blocks of P rows, zero-padded (outputs of padding rows are ignored);
 *   - the result is the minimum over chunks.
 * The exact CPU fallback accumulates in float64 in index order with partial-distance elimination (same minimum).
 */
internal object KnnMath {
    /** bankNorm / extra value that excludes a bank row (overflows fp16: graphs must run fp32). */
    const val KNN_MASK = 1e30f

    fun validate(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?) {
        require(dim > 0) { "dim must be > 0 (got $dim)" }
        require(nq >= 0 && nb >= 0) { "negative row count (nq=$nq, nb=$nb)" }
        require(feats.size >= nq * dim) { "feats has ${feats.size} values, need nq*dim = ${nq * dim}" }
        require(bank.size >= nb * dim) { "bank has ${bank.size} values, need nb*dim = ${nb * dim}" }
        require(extra == null || extra.size >= nb) { "extra has ${extra?.size} values, bank has $nb rows" }
    }

    // ------------------------------------------------------------------------------------------------ exact search
    @Volatile private var pool: ExecutorService? = null

    private fun pool(threads: Int): ExecutorService {
        pool?.let { return it }
        synchronized(this) {
            pool?.let { return it }
            val p = Executors.newFixedThreadPool(threads) { r -> Thread(r, "kz-knn-exact").apply { isDaemon = true } }
            pool = p
            return p
        }
    }

    /** Exact search (float64 accumulation in index order, partial-distance elimination). [threads] > 1 splits the queries. */
    fun exactMinSq(
        feats: FloatArray,
        nq: Int,
        bank: FloatArray,
        nb: Int,
        dim: Int,
        extra: FloatArray?,
        threads: Int = 1,
    ): FloatArray {
        validate(feats, nq, bank, nb, dim, extra)
        val out = FloatArray(nq)
        if (nq == 0) return out
        if (nb == 0) {
            out.fill(KNN_MASK)
            return out
        }
        val t = threads.coerceIn(1, 8)
        if (t == 1 || nq < 32) {
            exactRange(feats, bank, nb, dim, extra, 0, nq, out)
            return out
        }
        val ex = pool(t)
        val step = (nq + t - 1) / t
        val jobs = ArrayList<Future<*>>()
        var q0 = 0
        while (q0 < nq) {
            val a = q0
            val b = min(nq, q0 + step)
            jobs += ex.submit { exactRange(feats, bank, nb, dim, extra, a, b, out) }
            q0 = b
        }
        for (j in jobs) j.get()
        return out
    }

    private fun exactRange(feats: FloatArray, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?, q0: Int, q1: Int, out: FloatArray) {
        for (p in q0 until q1) {
            val a = p * dim
            var best = Double.POSITIVE_INFINITY
            for (k in 0 until nb) {
                var acc = if (extra != null) extra[k].toDouble() else 0.0
                if (acc >= best) continue
                val b = k * dim
                var t = 0
                while (t < dim) {
                    val end = min(dim, t + 8)
                    while (t < end) {
                        val d = feats[a + t].toDouble() - bank[b + t].toDouble()
                        acc += d * d
                        t++
                    }
                    if (acc >= best) break
                }
                if (acc < best) best = acc
            }
            out[p] = if (best > KNN_MASK) KNN_MASK else best.toFloat()
        }
    }

    // ------------------------------------------------------------------------------------------------ bank layout
    /** Bank rows [start, start+rows) transposed into [dim, k] (zero-padded columns) + their |b|^2 (float64). */
    class Chunk(val start: Int, val rows: Int, val k: Int, val bankT: FloatArray, val norm: DoubleArray)

    class PreparedBank(val nb: Int, val dim: Int, val chunks: List<Chunk>, val fingerprint: Long)

    /** Smallest bucket >= rows (the largest bucket if none is big enough). [buckets] ascending. */
    fun bucketFor(rows: Int, buckets: IntArray): Int = buckets.firstOrNull { it >= rows } ?: buckets.last()

    fun prepareBank(bank: FloatArray, nb: Int, dim: Int, buckets: IntArray): PreparedBank {
        require(buckets.isNotEmpty()) { "no k-NN buckets" }
        require(buckets[0] > 0 && (1 until buckets.size).all { buckets[it - 1] < buckets[it] }) {
            "buckets must be positive and ascending: ${buckets.toList()}"
        }
        require(bank.size >= nb * dim) { "bank has ${bank.size} values, need ${nb * dim}" }
        val maxK = buckets.last()
        val chunks = ArrayList<Chunk>()
        var start = 0
        while (start < nb) {
            val rows = min(maxK, nb - start)
            chunks += chunk(bank, dim, start, rows, bucketFor(rows, buckets))
            start += rows
        }
        return PreparedBank(nb, dim, chunks, fingerprint(bank, nb * dim))
    }

    private fun chunk(bank: FloatArray, dim: Int, start: Int, rows: Int, k: Int): Chunk {
        val t = FloatArray(dim * k)
        val norm = DoubleArray(rows)
        for (r in 0 until rows) {
            val o = (start + r) * dim
            var s = 0.0
            for (d in 0 until dim) {
                val v = bank[o + d]
                t[d * k + r] = v
                s += v.toDouble() * v.toDouble()
            }
            norm[r] = s
        }
        return Chunk(start, rows, k, t, norm)
    }

    /** bankNorm input for one chunk: |b|^2 + extra (one float rounding, capped at KNN_MASK), KNN_MASK on padding. */
    fun fillBankNorm(c: Chunk, extra: FloatArray?, out: FloatArray) {
        require(out.size >= c.k)
        for (r in 0 until c.rows) {
            val v = c.norm[r] + (extra?.get(c.start + r)?.toDouble() ?: 0.0)
            out[r] = if (v >= KNN_MASK.toDouble()) KNN_MASK else v.toFloat()
        }
        for (r in c.rows until c.k) out[r] = KNN_MASK
    }

    /** Copy query rows [q0, q0+P) into [out] (P*dim), zero-padding past nq. Returns the number of real rows. */
    fun fillQueryBlock(feats: FloatArray, nq: Int, dim: Int, q0: Int, p: Int, out: FloatArray): Int {
        val n = min(p, nq - q0)
        System.arraycopy(feats, q0 * dim, out, 0, n * dim)
        if (n < p) java.util.Arrays.fill(out, n * dim, p * dim, 0f)
        return n
    }

    /** One graph invocation: bucket [k], feats block [P*D], bankT [D*k], bankNorm [k] → minD2 [P]. */
    fun interface GraphRunner {
        fun run(k: Int, featsBlock: FloatArray, bankT: FloatArray, bankNorm: FloatArray): FloatArray
    }

    /** Drive the graph for any nq / nb: query blocks x bank chunks, minima combined. Throws on NaN or short outputs. */
    fun graphMinSq(
        feats: FloatArray,
        nq: Int,
        dim: Int,
        prep: PreparedBank,
        extra: FloatArray?,
        p: Int,
        runner: GraphRunner,
    ): FloatArray {
        require(prep.dim == dim) { "prepared bank has dim ${prep.dim}, queries $dim" }
        require(feats.size >= nq * dim) { "feats has ${feats.size} values, need ${nq * dim}" }
        require(extra == null || extra.size >= prep.nb) { "extra has ${extra?.size} values, bank has ${prep.nb} rows" }
        val out = FloatArray(nq)
        if (nq == 0) return out
        if (prep.nb == 0 || prep.chunks.isEmpty()) {
            out.fill(KNN_MASK)
            return out
        }
        out.fill(Float.POSITIVE_INFINITY)
        val norms = prep.chunks.map { c -> FloatArray(c.k).also { fillBankNorm(c, extra, it) } }
        val block = FloatArray(p * dim)
        var q0 = 0
        while (q0 < nq) {
            val n = fillQueryBlock(feats, nq, dim, q0, p, block)
            for (ci in prep.chunks.indices) {
                val c = prep.chunks[ci]
                val d = runner.run(c.k, block, c.bankT, norms[ci])
                check(d.size >= n) { "k-NN graph returned ${d.size} values, expected $p" }
                for (i in 0 until n) {
                    val v = d[i]
                    check(!v.isNaN()) { "k-NN graph returned NaN" }
                    if (v < out[q0 + i]) out[q0 + i] = v
                }
            }
            q0 += p
        }
        for (i in 0 until nq) if (out[i] > KNN_MASK) out[i] = KNN_MASK
        return out
    }

    /**
     * Float32 re-implementation of the graph's arithmetic (MUL by -2, matmul, ADD bankNorm, REDUCE_MIN, SUM f^2, ADD, RELU),
     * used by the JVM tests as a stand-in for the LiteRT graph.
     */
    fun graphEmulation(featsBlock: FloatArray, p: Int, dim: Int, bankT: FloatArray, k: Int, bankNorm: FloatArray): FloatArray {
        val out = FloatArray(p)
        for (i in 0 until p) {
            val a = i * dim
            var fnorm = 0f
            for (t in 0 until dim) fnorm += featsBlock[a + t] * featsBlock[a + t]
            var near = Float.POSITIVE_INFINITY
            for (j in 0 until k) {
                var cross = 0f
                for (t in 0 until dim) cross += (featsBlock[a + t] * -2f) * bankT[t * k + j]
                val v = cross + bankNorm[j]
                if (v < near) near = v
            }
            out[i] = max(0f, near + fnorm)
        }
        return out
    }

    /**
     * Content hash of the first [n] values (all of them: ~0.3 ms for a 2400x128 bank, far below one graph run), so a bank
     * array modified in place between calls is re-prepared instead of silently using stale graph inputs.
     */
    fun fingerprint(a: FloatArray, n: Int): Long {
        var h = -0x340d631b7bdddcdbL xor n.toLong()
        for (i in 0 until n) h = (h xor a[i].toRawBits().toLong()) * 0x100000001b3L
        return h
    }

    // ------------------------------------------------------------------------------------------------- agreement
    /**
     * twin-spec §5 agreement of [got] with the exact [ref] (both squared distances) over [rows] (all when null):
     * maxAbsDdOverMaxD = max |sqrt(got) - sqrt(ref)| / max sqrt(ref) (rows excluded in both, >= 1e29, are skipped;
     * excluded in only one of them counts as infinite error); maxRelErrD2 = legacy knn.ts metric on d^2
     * (|g - r| / max(|r|, 1e-3 max r)). Pass iff maxAbsDdOverMaxD <= 1e-3.
     */
    data class Agreement(val maxAbsDdOverMaxD: Double, val maxRelErrD2: Double, val rows: Int) {
        val pass: Boolean get() = maxAbsDdOverMaxD <= 1e-3
    }

    fun agreement(got: FloatArray, ref: FloatArray, rows: IntArray? = null): Agreement {
        val idx = rows ?: IntArray(ref.size) { it }
        var maxD = 0.0
        var maxR = 0.0
        for (i in idx) {
            val r = ref[i].toDouble()
            if (r < 1e29) {
                maxD = max(maxD, sqrt(max(0.0, r)))
                maxR = max(maxR, r)
            }
        }
        var maxDd = 0.0
        var maxRel = 0.0
        for (i in idx) {
            val g = got[i].toDouble()
            val r = ref[i].toDouble()
            val gm = g >= 1e29 || g.isInfinite()
            val rm = r >= 1e29
            if (gm && rm) continue
            if (gm != rm || g.isNaN()) {
                maxDd = Double.POSITIVE_INFINITY
                maxRel = Double.POSITIVE_INFINITY
                continue
            }
            maxDd = max(maxDd, abs(sqrt(max(0.0, g)) - sqrt(max(0.0, r))))
            maxRel = max(maxRel, abs(g - r) / max(abs(r), 1e-3 * maxR).coerceAtLeast(1e-30))
        }
        val norm = if (maxD > 0) maxDd / maxD else if (maxDd == 0.0) 0.0 else Double.POSITIVE_INFINITY
        return Agreement(norm, maxRel, idx.size)
    }
}
