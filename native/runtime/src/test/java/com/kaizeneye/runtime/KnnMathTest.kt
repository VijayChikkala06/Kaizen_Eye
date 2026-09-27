package com.kaizeneye.runtime

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.max

class KnnMathTest {
    private val mask = KnnMath.KNN_MASK

    private fun rand(seed: Long, n: Int, scale: Float = 1f): FloatArray {
        val r = Random(seed)
        return FloatArray(n) { (r.nextFloat() * scale) }
    }

    /** Plain float64 brute force in index order (the spec definition, no pruning). */
    private fun brute(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?): FloatArray =
        FloatArray(nq) { p ->
            var best = Double.POSITIVE_INFINITY
            for (k in 0 until nb) {
                var s = extra?.get(k)?.toDouble() ?: 0.0
                for (t in 0 until dim) {
                    val d = feats[p * dim + t].toDouble() - bank[k * dim + t].toDouble()
                    s += d * d
                }
                if (s < best) best = s
            }
            if (nb == 0) mask else if (best > mask) mask else best.toFloat()
        }

    @Test
    fun exactSearchEqualsBruteForceWithMasks() {
        val dim = 24
        val nq = 50
        val nb = 300
        val feats = rand(1, nq * dim)
        val bank = rand(2, nb * dim)
        val extra = FloatArray(nb) { if (it % 7 == 3) mask else if (it % 11 == 0) 0.25f else 0f }
        assertArrayEquals(brute(feats, nq, bank, nb, dim, null), KnnMath.exactMinSq(feats, nq, bank, nb, dim, null), 0f)
        assertArrayEquals(brute(feats, nq, bank, nb, dim, extra), KnnMath.exactMinSq(feats, nq, bank, nb, dim, extra), 0f)
    }

    @Test
    fun exactSearchIsIdenticalWithThreads() {
        val dim = 16
        val nq = 333
        val nb = 200
        val feats = rand(3, nq * dim)
        val bank = rand(4, nb * dim)
        val extra = FloatArray(nb) { if (it % 5 == 0) mask else 0f }
        assertArrayEquals(
            KnnMath.exactMinSq(feats, nq, bank, nb, dim, extra, threads = 1),
            KnnMath.exactMinSq(feats, nq, bank, nb, dim, extra, threads = 4),
            0f,
        )
    }

    @Test
    fun edgeCases() {
        val dim = 4
        val feats = rand(5, 3 * dim)
        val bank = rand(6, 5 * dim)
        assertEquals(0, KnnMath.exactMinSq(feats, 0, bank, 5, dim, null).size)
        assertTrue(KnnMath.exactMinSq(feats, 3, bank, 0, dim, null).all { it == mask })
        val all = KnnMath.exactMinSq(feats, 3, bank, 5, dim, FloatArray(5) { mask })
        assertTrue(all.all { it >= 1e30f })
        // a query equal to a bank row has distance 0
        val q = bank.copyOfRange(2 * dim, 3 * dim)
        assertEquals(0f, KnnMath.exactMinSq(q, 1, bank, 5, dim, null)[0], 0f)
        try {
            KnnMath.exactMinSq(feats, 4, bank, 5, dim, null)
            fail("short feats must be rejected")
        } catch (_: IllegalArgumentException) {
        }
        try {
            KnnMath.exactMinSq(feats, 3, bank, 5, dim, FloatArray(4))
            fail("short extra must be rejected")
        } catch (_: IllegalArgumentException) {
        }
    }

    @Test
    fun bankPreparationChunksPadsAndTransposes() {
        val dim = 3
        val nb = 300
        val bank = rand(7, nb * dim)
        val prep = KnnMath.prepareBank(bank, nb, dim, intArrayOf(40, 80, 120))
        assertEquals(listOf(0, 120, 240), prep.chunks.map { it.start })
        assertEquals(listOf(120, 120, 60), prep.chunks.map { it.rows })
        assertEquals(listOf(120, 120, 80), prep.chunks.map { it.k })
        val c = prep.chunks[2]
        for (r in 0 until c.rows) for (d in 0 until dim) assertEquals(bank[(c.start + r) * dim + d], c.bankT[d * c.k + r], 0f)
        for (r in c.rows until c.k) for (d in 0 until dim) assertEquals(0f, c.bankT[d * c.k + r], 0f)
        val extra = FloatArray(nb) { if (it == 250) mask else 0.5f }
        val norm = FloatArray(c.k)
        KnnMath.fillBankNorm(c, extra, norm)
        val row = 245
        var n2 = 0.0
        for (d in 0 until dim) n2 += bank[row * dim + d].toDouble() * bank[row * dim + d]
        assertEquals((n2 + 0.5).toFloat(), norm[row - 240], 0f)
        assertEquals(mask, norm[250 - 240], 0f)               // excluded
        for (r in c.rows until c.k) assertEquals(mask, norm[r], 0f)   // padding
        assertEquals(40, KnnMath.bucketFor(1, intArrayOf(40, 80, 120)))
        assertEquals(80, KnnMath.bucketFor(41, intArrayOf(40, 80, 120)))
        assertEquals(120, KnnMath.bucketFor(999, intArrayOf(40, 80, 120)))
    }

    @Test
    fun queryBlocksArePaddedWithZeros() {
        val dim = 2
        val feats = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val block = FloatArray(4 * dim) { 9f }
        assertEquals(2, KnnMath.fillQueryBlock(feats, 3, dim, 1, 4, block))
        assertArrayEquals(floatArrayOf(3f, 4f, 5f, 6f, 0f, 0f, 0f, 0f), block, 0f)
    }

    private fun emulated(p: Int, dim: Int) = KnnMath.GraphRunner { k, fb, bt, bn -> KnnMath.graphEmulation(fb, p, dim, bt, k, bn) }

    @Test
    fun graphPathAgreesWithExactSearch() {
        val p = 64
        val dim = 16
        val buckets = intArrayOf(40, 80, 120)
        val rnd = BenchInputs.Lcg(99)
        for ((nq, nb) in listOf(64 to 30, 64 to 80, 10 to 100, 200 to 300, 129 to 241, 1 to 1)) {
            val feats = BenchInputs.fakeFeatures(rnd, nq, dim)
            val bank = BenchInputs.fakeFeatures(rnd, nb, dim)
            for (masked in listOf(false, true)) {
                val extra = if (masked) FloatArray(nb) { if (it % 20 == 3) mask else 0f } else null
                val prep = KnnMath.prepareBank(bank, nb, dim, buckets)
                val got = KnnMath.graphMinSq(feats, nq, dim, prep, extra, p, emulated(p, dim))
                val ref = KnnMath.exactMinSq(feats, nq, bank, nb, dim, extra)
                val a = KnnMath.agreement(got, ref)
                assertTrue("nq=$nq nb=$nb masked=$masked: $a", a.pass)
                assertTrue("nq=$nq nb=$nb masked=$masked: $a", a.maxRelErrD2 < 1e-3)
            }
        }
    }

    @Test
    fun graphPathWithEverythingMaskedStaysAtTheSentinel() {
        val p = 8
        val dim = 4
        val feats = rand(8, 5 * dim)
        val bank = rand(9, 6 * dim)
        val prep = KnnMath.prepareBank(bank, 6, dim, intArrayOf(8))
        val got = KnnMath.graphMinSq(feats, 5, dim, prep, FloatArray(6) { mask }, p, emulated(p, dim))
        assertTrue(got.contentToString(), got.all { it >= 1e30f && it <= mask })
        val ref = KnnMath.exactMinSq(feats, 5, bank, 6, dim, FloatArray(6) { mask })
        assertTrue(KnnMath.agreement(got, ref).pass)
    }

    @Test
    fun graphPathRejectsNaNAndShortOutputs() {
        val dim = 4
        val feats = rand(10, 4 * dim)
        val bank = rand(11, 4 * dim)
        val prep = KnnMath.prepareBank(bank, 4, dim, intArrayOf(4))
        try {
            KnnMath.graphMinSq(feats, 4, dim, prep, null, 4) { _, _, _, _ -> floatArrayOf(1f, Float.NaN, 1f, 1f) }
            fail("NaN must be rejected")
        } catch (_: IllegalStateException) {
        }
        try {
            KnnMath.graphMinSq(feats, 4, dim, prep, null, 4) { _, _, _, _ -> floatArrayOf(1f) }
            fail("short output must be rejected")
        } catch (_: IllegalStateException) {
        }
    }

    @Test
    fun agreementMetric() {
        val ref = floatArrayOf(4f, 1f, 1e30f, 0f)
        assertEquals(0.0, KnnMath.agreement(ref.copyOf(), ref).maxAbsDdOverMaxD, 0.0)
        val off = floatArrayOf(4.004f, 1f, 1e30f, 0f)        // d: 2.001 vs 2 -> 1e-3 * 2 / max d 2
        assertEquals(0.0005, KnnMath.agreement(off, ref).maxAbsDdOverMaxD, 1e-5)
        val unmasked = floatArrayOf(4f, 1f, 3f, 0f)           // excluded in the reference only
        assertFalse(KnnMath.agreement(unmasked, ref).pass)
        val sub = KnnMath.agreement(off, ref, intArrayOf(1, 3))
        assertEquals(0.0, sub.maxAbsDdOverMaxD, 0.0)
        assertEquals(2, sub.rows)
    }

    @Test
    fun fingerprintSeesAnyInPlaceChange() {
        val a = rand(12, 1000)
        val h = KnnMath.fingerprint(a, a.size)
        val b = a.copyOf()
        b[517] = b[517] + 1e-3f
        assertNotEquals(h, KnnMath.fingerprint(b, b.size))
        assertEquals(h, KnnMath.fingerprint(a.copyOf(), a.size))
    }

    @Test
    fun emulationMatchesTheGraphFormula() {
        // relu(min_k(bankNorm_k - 2 f.b_k) + |f|^2) on a hand-checkable case
        val f = floatArrayOf(1f, 2f)                          // |f|^2 = 5
        val bankT = floatArrayOf(1f, 0f, /* d0 */ 2f, 0f)     // rows b0 = (1,2), b1 = (0,0) ; k = 2
        val norm = floatArrayOf(5f, 0f)
        val out = KnnMath.graphEmulation(f, 1, 2, bankT, 2, norm)
        assertEquals(0f, out[0], 0f)                           // nearest is b0 itself
        val out2 = KnnMath.graphEmulation(f, 1, 2, bankT, 2, floatArrayOf(mask, 0f))
        assertEquals(5f, out2[0], 0f)                          // b0 excluded: distance to the origin
        assertTrue(max(0f, abs(out2[0] - 5f)) == 0f)
    }
}
