package com.kaizeneye.core.twin

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Added to a bank row's squared distance to exclude it (spec §5; the LiteRT graph's `bankNorm` semantics).
 * If every row is excluded the result is >= 1e30.
 */
const val KNN_MASK: Float = 1e30f

/**
 * Nearest-neighbour backend (spec §5). The CPU default is [Knn.CPU]; the Android runtime can plug in the LiteRT matmul
 * graphs (`knn_p{P}_d{D}_k{K}.tflite`), which must agree with the exact search to `max |Δd| ≤ 1e-3 · max(d)`.
 */
fun interface KnnBackend {
    /**
     * For each of the first [nq] rows of [feats] ([nq, dim] row-major):
     * `out[i] = min over the nb rows b of bank of ||feats_i − bank_b||² + extra[b]` ([extra] null = all zeros).
     * Rows with `extra = KNN_MASK` are excluded; when every row is excluded (or [nb] = 0) the result is >= 1e30
     * (or +inf). [bank] is `[nb, dim]` row-major. Implementations may cache prepared banks by array identity.
     */
    fun minSq(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?, out: DoubleArray)
}

/** Exact k-NN (spec §5): float64 accumulation in index order, result-preserving partial-distance elimination. */
object Knn {

    /** The exact CPU backend. */
    val CPU: KnnBackend = KnnBackend { feats, nq, bank, nb, dim, extra, out -> minSq(feats, nq, bank, nb, dim, extra, out) }

    /** `d = sqrt(max(0, minSq))` (spec §5). */
    fun distance(minSq: Double): Double = sqrt(max(0.0, minSq))

    /**
     * Exact search: see [KnnBackend.minSq]. Each squared distance is `Σ_t (f_t − b_t)²` accumulated in float64 in index
     * order, plus `extra[b]`. A candidate is abandoned as soon as its partial sum can no longer beat the best found so
     * far (all terms are >= 0 and float64 rounding is monotone), so the result equals the full search bit for bit.
     * The previous query's nearest row is tried first (neighbouring patches are similar), which tightens the bound early
     * without changing the minimum.
     */
    fun minSq(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?, out: DoubleArray) {
        require(dim > 0) { "dim must be > 0" }
        require(nq >= 0 && feats.size >= nq * dim) { "feats has ${feats.size} floats, need ${nq * dim}" }
        require(nb >= 0 && bank.size >= nb * dim) { "bank has ${bank.size} floats, need ${nb * dim}" }
        require(extra == null || extra.size >= nb) { "extra has ${extra?.size} values, need $nb" }
        require(out.size >= nq) { "out has ${out.size} values, need $nq" }
        var hint = 0
        for (q in 0 until nq) {
            val qo = q * dim
            var best = Double.POSITIVE_INFINITY
            var bestRow = -1
            if (nb > 0) {
                best = sqDist(feats, qo, bank, hint * dim, dim, extraOf(extra, hint), Double.POSITIVE_INFINITY)
                bestRow = hint
            }
            if (extra == null) {
                for (b in 0 until nb) {
                    if (b == hint) continue
                    val d = sqDist(feats, qo, bank, b * dim, dim, 0.0, best)
                    if (d < best) {
                        best = d
                        bestRow = b
                    }
                }
            } else {
                for (b in 0 until nb) {
                    if (b == hint) continue
                    val e = extra[b].toDouble()
                    if (e >= best) continue          // the full sum can only be larger
                    val d = sqDist(feats, qo, bank, b * dim, dim, e, best)
                    if (d < best) {
                        best = d
                        bestRow = b
                    }
                }
            }
            out[q] = best
            if (bestRow >= 0) hint = bestRow
        }
    }

    private fun extraOf(extra: FloatArray?, b: Int): Double = extra?.get(b)?.toDouble() ?: 0.0

    /**
     * `e + Σ_t (x_t − y_t)²` over `dim` values (float64, index order). Returns early with a value >= [limit] as soon as
     * `e + partial >= limit` (checked every 8 terms).
     */
    fun sqDist(x: FloatArray, xo: Int, y: FloatArray, yo: Int, dim: Int, e: Double, limit: Double): Double {
        var acc = 0.0
        var t = 0
        val end8 = dim - (dim and 7)
        while (t < end8) {
            val a = xo + t
            val b = yo + t
            val d0 = x[a].toDouble() - y[b].toDouble()
            acc += d0 * d0
            val d1 = x[a + 1].toDouble() - y[b + 1].toDouble()
            acc += d1 * d1
            val d2 = x[a + 2].toDouble() - y[b + 2].toDouble()
            acc += d2 * d2
            val d3 = x[a + 3].toDouble() - y[b + 3].toDouble()
            acc += d3 * d3
            val d4 = x[a + 4].toDouble() - y[b + 4].toDouble()
            acc += d4 * d4
            val d5 = x[a + 5].toDouble() - y[b + 5].toDouble()
            acc += d5 * d5
            val d6 = x[a + 6].toDouble() - y[b + 6].toDouble()
            acc += d6 * d6
            val d7 = x[a + 7].toDouble() - y[b + 7].toDouble()
            acc += d7 * d7
            t += 8
            if (acc + e >= limit) return acc + e
        }
        while (t < dim) {
            val d = x[xo + t].toDouble() - y[yo + t].toDouble()
            acc += d * d
            t++
        }
        return acc + e
    }

    /** Brute-force reference (no elimination); used by tests and as documentation of the maths. */
    fun minSqBrute(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?, out: DoubleArray) {
        for (q in 0 until nq) {
            var best = Double.POSITIVE_INFINITY
            for (b in 0 until nb) {
                var acc = 0.0
                for (t in 0 until dim) {
                    val d = feats[q * dim + t].toDouble() - bank[b * dim + t].toDouble()
                    acc += d * d
                }
                val v = acc + extraOf(extra, b)
                if (v < best) best = v
            }
            out[q] = best
        }
    }
}
