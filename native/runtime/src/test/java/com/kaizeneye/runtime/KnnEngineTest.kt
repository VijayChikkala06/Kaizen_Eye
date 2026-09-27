package com.kaizeneye.runtime

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** KnnEngineImpl (bank prep + graph driving + exact fallback) with a float32 stand-in for the LiteRT graph. */
class KnnEngineTest {
    private val mask = KnnMath.KNN_MASK

    private class FakeGraph(override val p: Int, override val d: Int, override val k: Int, val failAfter: Int = Int.MAX_VALUE) : KnnGraph {
        var runs = 0
        var closed = false

        override fun run(featsBlock: FloatArray, bankT: FloatArray, bankNorm: FloatArray): FloatArray {
            check(featsBlock.size == p * d && bankT.size == d * k && bankNorm.size == k) { "wrong input sizes" }
            if (++runs > failAfter) throw IllegalStateException("accelerator lost")
            return KnnMath.graphEmulation(featsBlock, p, d, bankT, k, bankNorm)
        }

        override fun close() {
            closed = true
        }
    }

    /** A graph wired wrongly (bankNorm ignored, i.e. masks/padding lost): the verification must reject it. */
    private class NoNormGraph(override val p: Int, override val d: Int, override val k: Int) : KnnGraph {
        override fun run(featsBlock: FloatArray, bankT: FloatArray, bankNorm: FloatArray): FloatArray =
            KnnMath.graphEmulation(featsBlock, p, d, bankT, k, FloatArray(k))
        override fun close() {}
    }

    private fun engine(p: Int, d: Int, ks: List<Int>, failAfter: Int = Int.MAX_VALUE): Pair<KnnEngineImpl, List<FakeGraph>> {
        val gs = ks.map { FakeGraph(p, d, it, failAfter) }
        return KnnEngineImpl(gs.associateBy { it.k }, p, d, "test", null, 2) to gs
    }

    @Test
    fun engineMatchesExactSearchWithMasksChunksAndPadding() {
        val p = 32
        val d = 12
        val (eng, graphs) = engine(p, d, listOf(20, 40, 60))
        val rnd = BenchInputs.Lcg(7)
        for ((nq, nb) in listOf(32 to 15, 5 to 40, 70 to 59, 33 to 61, 100 to 175)) {
            val feats = BenchInputs.fakeFeatures(rnd, nq, d)
            val bank = BenchInputs.fakeFeatures(rnd, nb, d)
            val extra = FloatArray(nb) { if (it % 20 == 3) mask else 0f }
            for (e in listOf(null, extra)) {
                val got = eng.minSq(feats, nq, bank, nb, d, e)
                val ref = KnnMath.exactMinSq(feats, nq, bank, nb, d, e)
                val a = KnnMath.agreement(got, ref)
                assertTrue("nq=$nq nb=$nb extra=${e != null}: $a", a.pass)
            }
        }
        assertTrue(eng.usingGraphs)
        assertTrue(graphs.sumOf { it.runs } > 0)
        eng.close()
        assertTrue(graphs.all { it.closed })
    }

    @Test
    fun graphFailureSwitchesToExactForGood() {
        val p = 16
        val d = 8
        val (eng, _) = engine(p, d, listOf(32), failAfter = 1)
        val rnd = BenchInputs.Lcg(8)
        val feats = BenchInputs.fakeFeatures(rnd, 40, d)             // 3 query blocks -> the 2nd graph run fails
        val bank = BenchInputs.fakeFeatures(rnd, 30, d)
        val got = eng.minSq(feats, 40, bank, 30, d, null)
        assertArrayEquals(KnnMath.exactMinSq(feats, 40, bank, 30, d, null), got, 0f)   // exact result after the failure
        assertFalse(eng.usingGraphs)
        assertTrue(eng.label, eng.label.startsWith("k-NN exact CPU search"))
        val again = eng.minSq(feats, 40, bank, 30, d, null)
        assertArrayEquals(got, again, 0f)
    }

    @Test
    fun otherDimensionsAndEmptyInputsUseTheExactSearch() {
        val (eng, graphs) = engine(8, 4, listOf(16))
        val feats = BenchInputs.fakeFeatures(BenchInputs.Lcg(9), 5, 6)
        val bank = BenchInputs.fakeFeatures(BenchInputs.Lcg(10), 7, 6)
        assertArrayEquals(KnnMath.exactMinSq(feats, 5, bank, 7, 6, null), eng.minSq(feats, 5, bank, 7, 6, null), 0f)
        assertEquals(0, graphs[0].runs)
        assertEquals(0, eng.minSq(feats, 0, bank, 7, 6, null).size)
        assertTrue(eng.minSq(feats, 5, bank, 0, 6, null).all { it == mask })
    }

    @Test
    fun bankModifiedInPlaceIsPreparedAgain() {
        val p = 8
        val d = 4
        val (eng, _) = engine(p, d, listOf(16))
        val rnd = BenchInputs.Lcg(11)
        val feats = BenchInputs.fakeFeatures(rnd, 8, d)
        val bank = BenchInputs.fakeFeatures(rnd, 10, d)
        val first = eng.minSq(feats, 8, bank, 10, d, null)
        // move bank row 7 onto query 3: its distance must drop to ~0
        System.arraycopy(feats, 3 * d, bank, 7 * d, d)
        val second = eng.minSq(feats, 8, bank, 10, d, null)
        assertTrue(second[3] < 1e-6f)
        assertTrue(first[3] > 1e-4f)
    }

    @Test
    fun verificationAcceptsACorrectGraphAndRejectsAMiswiredOne() {
        val ok = KnnVerify.check(FakeGraph(64, 16, 80))
        assertTrue(ok.toString(), ok.pass)
        val bad = KnnVerify.check(NoNormGraph(64, 16, 80))
        assertFalse(bad.toString(), bad.pass)
        val times = KnnVerify.time(FakeGraph(16, 4, 8), warmup = 1, runs = 3)
        assertEquals(3, times.size)
    }
}
