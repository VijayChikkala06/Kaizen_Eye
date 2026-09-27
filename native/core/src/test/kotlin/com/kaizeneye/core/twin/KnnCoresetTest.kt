package com.kaizeneye.core.twin

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class KnnCoresetTest {

    private fun clustered(n: Int, dim: Int, rnd: Random, clusters: Int = 10, spread: Double = 0.3): FloatArray {
        val centres = Array(clusters) { FloatArray(dim) { (rnd.nextGaussian() * 2).toFloat() } }
        val x = FloatArray(n * dim)
        for (i in 0 until n) {
            val c = centres[rnd.nextInt(clusters)]
            for (t in 0 until dim) x[i * dim + t] = c[t] + (rnd.nextGaussian() * spread).toFloat()
        }
        return x
    }

    @Test
    fun exactSearchEqualsBruteForceBitForBit() {
        val rnd = Random(5)
        for (trial in 0 until 12) {
            val dim = intArrayOf(8, 13, 64, 128)[trial % 4]
            val nq = 1 + rnd.nextInt(90)
            val nb = 1 + rnd.nextInt(300)
            val q = clustered(nq, dim, rnd)
            val b = clustered(nb, dim, rnd)
            val extra = if (trial % 2 == 0) null else FloatArray(nb) { if (rnd.nextInt(3) == 0) KNN_MASK else 0f }
            val got = DoubleArray(nq)
            val want = DoubleArray(nq)
            Knn.minSq(q, nq, b, nb, dim, extra, got)
            Knn.minSqBrute(q, nq, b, nb, dim, extra, want)
            assertArrayEquals(want, got, 0.0)
            Knn.CPU.minSq(q, nq, b, nb, dim, extra, got)
            assertArrayEquals(want, got, 0.0)
        }
    }

    @Test
    fun maskedRowsAndEdgeCases() {
        val dim = 4
        val q = floatArrayOf(0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f)
        val b = floatArrayOf(0f, 0f, 0f, 1f, 5f, 5f, 5f, 5f)
        val out = DoubleArray(2)
        Knn.minSq(q, 2, b, 2, dim, floatArrayOf(0f, KNN_MASK), out)
        assertEquals(1.0, out[0], 0.0)
        assertEquals(3.0, out[1], 0.0)             // (1,1,1,0) away from the only allowed row
        Knn.minSq(q, 2, b, 2, dim, floatArrayOf(KNN_MASK, KNN_MASK), out)
        assertTrue(out[0] >= 1e30 && out[1] >= 1e30)
        Knn.minSq(q, 2, b, 0, dim, null, out)
        assertTrue(out[0].isInfinite())
        assertEquals(0.0, Knn.distance(-1e-9), 0.0)
        assertEquals(2.0, Knn.distance(4.0), 0.0)
    }

    @Test
    fun prunedCoresetEqualsBruteForce() {
        val rnd = Random(9)
        for (trial in 0 until 8) {
            val dim = intArrayOf(8, 13, 32, 128)[trial % 4]
            val n = 300 + rnd.nextInt(900)
            val x = clustered(n, dim, rnd, clusters = 6 + rnd.nextInt(10))
            for (i in 0 until 15) x.copyInto(x, (n - 1 - i) * dim, i * dim, i * dim + dim)   // exact duplicates
            val k = 20 + rnd.nextInt(200)
            val start = if (trial % 3 == 0) rnd.nextInt(n) else 0
            val want = Coreset.greedyBrute(x, n, dim, k, start)
            assertArrayEquals(want, Coreset.greedy(x, n, dim, k, start))
            for (threads in intArrayOf(2, 3, 8)) assertArrayEquals(want, Coreset.greedy(x, n, dim, k, start, threads))
        }
        // k > n returns every row once
        val x = clustered(10, 4, rnd)
        assertEquals(10, Coreset.greedy(x, 10, 4, 50).toSet().size)
    }

    @Test
    fun coresetTiesGoToTheLowestIndex() {
        // Points on a line: 0, 1, 2, 3 and duplicates of 3 → first pick (from 0) is the first 3 (index 3), then 1 or 2
        // are at distance 1 or 2 ... check against brute force with exact ties.
        val x = floatArrayOf(0f, 1f, 2f, 3f, 3f, 1.5f, 1.5f)
        assertArrayEquals(intArrayOf(0, 3, 5, 1, 2, 4, 6).copyOf(4), Coreset.greedy(x, 7, 1, 4))
        assertArrayEquals(Coreset.greedyBrute(x, 7, 1, 7), Coreset.greedy(x, 7, 1, 7))
    }

    /** Timing smoke test: the teach bank size of spec §7 (28,000 pooled rows × 128 → 2,400 centres). */
    @Test
    fun coresetTimingSmoke() {
        val rnd = Random(1)
        val n = 28_000
        val dim = 128
        val x = clustered(n, dim, rnd, clusters = 400, spread = 0.5)
        val t0 = System.nanoTime()
        val sel = Coreset.greedy(x, n, dim, 2400)
        val ms = (System.nanoTime() - t0) / 1e6
        val t1 = System.nanoTime()
        val sel4 = Coreset.greedy(x, n, dim, 2400, threads = 4)
        val ms4 = (System.nanoTime() - t1) / 1e6
        println("[timing] Coreset.greedy 28000 x 128 -> 2400: %.0f ms (1 thread), %.0f ms (4 threads)".format(ms, ms4))
        assertEquals(2400, sel.toSet().size)
        assertArrayEquals(sel, sel4)
    }
}
