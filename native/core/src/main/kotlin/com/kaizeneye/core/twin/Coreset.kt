package com.kaizeneye.core.twin

import kotlin.math.min
import kotlin.math.sqrt

/** Greedy k-center coreset of the Twin bank (spec §7 step 6). */
object Coreset {

    private const val BLOCK = 128

    /**
     * Greedy k-center over the [n] rows of [x] (`[n, dim]` float32, row-major): start at row [start], then repeatedly add
     * the row with the largest float64 squared Euclidean distance to its nearest selected row (ties → lowest index).
     * Returns `min(k, n)` row indices in selection order.
     *
     * Exact speed-ups that never change the result (every distance that can win is computed in full, in index order):
     *  - triangle-inequality pruning: row i (nearest centre a, distance r) cannot move closer to a new centre j when
     *    `d(j, a) >= 2r` (a relative safety margin of 1e-6 covers float64 rounding); since j is the farthest row,
     *    `d(j, a) >= sqrt(mind[j])` for every centre, so most rows are skipped before `d(j, a)` is even computed;
     *  - partial-distance elimination against the row's current min-distance;
     *  - per-block maxima, rescanning only blocks whose rows changed;
     *  - optional [threads]: the row updates of one step run in parallel over whole blocks (same result for any count).
     */
    fun greedy(x: FloatArray, n: Int, dim: Int, k: Int, start: Int = 0, threads: Int = 1): IntArray {
        require(dim > 0 && n >= 0 && x.size >= n * dim) { "x has ${x.size} floats, need ${n * dim}" }
        val kk = min(k, n)
        if (kk <= 0) return IntArray(0)
        require(start in 0 until n) { "start $start out of range" }
        val sel = IntArray(kk)
        sel[0] = start
        val mind = DoubleArray(n)
        val assign = IntArray(n)
        val prune = DoubleArray(n)
        val s0 = start * dim
        for (i in 0 until n) {
            mind[i] = Knn.sqDist(x, i * dim, x, s0, dim, 0.0, Double.POSITIVE_INFINITY)
            prune[i] = pruneBound(mind[i])
        }
        val nb = (n + BLOCK - 1) / BLOCK
        val blockMax = DoubleArray(nb)
        val blockArg = IntArray(nb)
        val dirty = BooleanArray(nb) { true }
        // Parallel parts are contiguous runs of whole blocks; each part owns its rows, its blocks' dirty flags and its
        // own cache of centre distances, so the result does not depend on the number of threads.
        val parts = minOf(maxOf(1, threads), nb)
        val partLo = IntArray(parts) { (it.toLong() * nb / parts).toInt() * BLOCK }
        val partHi = IntArray(parts) { minOf(n, ((it + 1).toLong() * nb / parts).toInt() * BLOCK) }
        val cc = Array(parts) { DoubleArray(kk) }
        val ccStamp = Array(parts) { IntArray(kk) { -1 } }
        Workers(parts).use { workers ->
            for (s in 1 until kk) {
                for (b in 0 until nb) {
                    if (!dirty[b]) continue
                    val lo = b * BLOCK
                    val hi = min(n, lo + BLOCK)
                    var bm = mind[lo]
                    var ba = lo
                    for (i in lo + 1 until hi) {
                        if (mind[i] > bm) {
                            bm = mind[i]
                            ba = i
                        }
                    }
                    blockMax[b] = bm
                    blockArg[b] = ba
                    dirty[b] = false
                }
                // Strict '>' over blocks in order + lowest index inside each block = lowest index overall.
                var j = blockArg[0]
                var best = blockMax[0]
                for (b in 1 until nb) {
                    if (blockMax[b] > best) {
                        best = blockMax[b]
                        j = blockArg[b]
                    }
                }
                sel[s] = j
                val jo = j * dim
                // j is the farthest row, so every centre is at least R = sqrt(mind[j]) from it: rows with 2·r_i <= R are
                // skipped without computing their centre's distance (the triangle test below would skip them anyway).
                val skipBelow = sqrt(best) * (1.0 - 1e-9)
                val step = s
                workers.run(parts) { part ->
                    val ccp = cc[part]
                    val stp = ccStamp[part]
                    for (i in partLo[part] until partHi[part]) {
                        val pr = prune[i]
                        if (pr <= skipBelow) continue
                        val m = mind[i]
                        if (m == 0.0) continue
                        val a = assign[i]
                        if (stp[a] != step) {
                            ccp[a] = sqrt(Knn.sqDist(x, sel[a] * dim, x, jo, dim, 0.0, Double.POSITIVE_INFINITY))
                            stp[a] = step
                        }
                        if (ccp[a] >= pr) continue
                        val d = Knn.sqDist(x, i * dim, x, jo, dim, 0.0, m)
                        if (d < m) {
                            mind[i] = d
                            assign[i] = step
                            prune[i] = pruneBound(d)
                            dirty[i / BLOCK] = true
                        }
                    }
                }
            }
        }
        return sel
    }

    private fun pruneBound(minSq: Double): Double = 2.0 * sqrt(minSq) * (1.0 + 1e-6) + 1e-12

    /** Unpruned reference implementation (tests only; O(n·k·dim)). */
    fun greedyBrute(x: FloatArray, n: Int, dim: Int, k: Int, start: Int = 0): IntArray {
        val kk = min(k, n)
        if (kk <= 0) return IntArray(0)
        val sel = IntArray(kk)
        sel[0] = start
        val mind = DoubleArray(n) { Knn.sqDist(x, it * dim, x, start * dim, dim, 0.0, Double.POSITIVE_INFINITY) }
        for (s in 1 until kk) {
            var j = 0
            for (i in 1 until n) if (mind[i] > mind[j]) j = i
            sel[s] = j
            for (i in 0 until n) {
                val d = Knn.sqDist(x, i * dim, x, j * dim, dim, 0.0, Double.POSITIVE_INFINITY)
                if (d < mind[i]) mind[i] = d
            }
        }
        return sel
    }
}
