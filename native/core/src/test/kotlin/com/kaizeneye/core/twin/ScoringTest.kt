package com.kaizeneye.core.twin

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class ScoringTest {

    private fun grid(vararg rows: String): BooleanArray =
        rows.joinToString("").map { it == '#' }.toBooleanArray()

    @Test
    fun patchSetsDilateOnTheGridWithTheBorderOutside() {
        val cov = FloatArray(25).also { it[0] = 0.5f; it[12] = 1f; it[13] = 0.4375f }
        val s = PatchSets.fromCov(cov, 5, 5)
        assertArrayEquals(intArrayOf(0, 12), s.coreIdx)
        assertArrayEquals(
            grid("##...", "##...", ".....", ".....", ".....").let { a ->
                val b = grid(".....", ".###.", ".###.", ".###.", ".....")
                BooleanArray(25) { a[it] || b[it] }
            },
            s.s,
        )
        assertEquals(25, s.bIdx.size)             // radius 2 around (0,0) and (2,2) covers the 5x5 grid
        assertTrue(PatchSets.fromCov(FloatArray(25), 5, 5).isEmpty)
    }

    @Test
    fun maskedSmoothingNeverLeaksBackground() {
        // 3x3 grid, S = left two columns; background values are huge.
        val s = grid("##.", "##.", "##.")
        val d = doubleArrayOf(1.0, 2.0, 1e9, 3.0, 4.0, 1e9, 5.0, 6.0, 1e9)
        val sm = Scoring.smoothMasked(d, 3, 3, s)
        assertEquals((1 + 2 + 3 + 4) / 4.0, sm[0], 0.0)
        assertEquals((1 + 2 + 3 + 4 + 5 + 6) / 6.0, sm[4], 0.0)
        assertTrue(sm[2].isNaN())
    }

    @Test
    fun smoothedMaxPeakAndAnomalousFraction() {
        // 4x4, core = centre 2x2, S = whole grid (dilation); one hot patch at (1,1).
        val core = grid("....", ".##.", ".##.", "....")
        val sets = PatchSets.fromCore(core, 4, 4)
        val d = DoubleArray(16) { 1.0 }
        d[5] = 10.0
        val r = Scoring.score(d, sets, tau = 2.0, sensitivity = 1.5)
        // The corner (0,0) averages 4 values (3·1 + 10)/4; (1,1) averages 9: (8 + 10)/9 = 2.
        assertEquals((3 + 10) / 4.0, r.raw, 1e-15)
        assertEquals(0, r.peak)
        assertEquals(r.raw / 3.0, r.s, 1e-15)
        assertEquals(2.0, r.smoothed[5], 1e-15)
        assertEquals(2.5, r.smoothed[1], 1e-15)    // (0,1): 6 neighbours incl. the hot one
        assertEquals(0.0, r.anomalousFraction, 0.0)  // every core patch sm = 2.0 <= 3.0
        assertEquals(4, r.coreCount)

        // 1x4 strip, all core: d = 0 0 0 12 → sm = 0, 0, 4, 6.
        val strip = PatchSets.fromCore(BooleanArray(4) { true }, 1, 4)
        val ds = doubleArrayOf(0.0, 0.0, 0.0, 12.0)
        val a = Scoring.score(ds, strip, tau = 2.0, sensitivity = 1.5)   // limit 3
        assertArrayEquals(doubleArrayOf(0.0, 0.0, 4.0, 6.0), a.smoothed, 1e-15)
        assertEquals(6.0, a.raw, 0.0)
        assertEquals(2.0, a.s, 0.0)
        assertEquals(3, a.peak)
        assertEquals(0.5, a.anomalousFraction, 0.0)
        assertEquals(50.0, a.areaPct, 0.0)
        assertEquals(2, a.anomalousCount)
        val b = Scoring.score(ds, strip, tau = 2.0, sensitivity = 2.0)   // limit 4: sm = 4 is not anomalous
        assertEquals(0.25, b.anomalousFraction, 0.0)
    }

    @Test
    fun smEqualToTheLimitIsNotAnomalousAndTiesPickTheLowestPeak() {
        val core = grid("###", "###", "###")
        val sets = PatchSets.fromCore(core, 3, 3)
        val d = DoubleArray(9) { 2.0 }
        val r = Scoring.score(d, sets, tau = 2.0, sensitivity = 1.0)
        assertEquals(1.0, r.s, 0.0)
        assertEquals(0.0, r.anomalousFraction, 0.0)
        assertEquals(0, r.peak)
    }

    @Test
    fun top1MeanTakesTheLargestWithTiesByIndex() {
        val sets = PatchSets.fromCore(BooleanArray(400) { true }, 20, 20)
        val d = DoubleArray(400) { 1.0 }
        d[7] = 9.0; d[100] = 5.0; d[200] = 5.0; d[300] = 5.0
        // n = ceil(0.01 * 400) = 4 → 9, 5, 5, 5
        val r = Scoring.score(d, sets, 1.0, 1.0, ScoreRule.TOP1_MEAN)
        assertEquals(4, r.topN)
        assertEquals((9.0 + 15.0) / 4, r.raw, 0.0)
        assertEquals(1, Scoring.topN(50))
        assertEquals(2, Scoring.topN(101))
        assertEquals(3.0, Scoring.topMean(doubleArrayOf(1.0, 3.0, 3.0), intArrayOf(0, 1, 2), 0.01), 0.0)
    }

    @Test
    fun distancesOutsideTheScoreSetNeverChangeTheScore() {
        val rnd = Random(21)
        repeat(50) {
            val gh = 6 + rnd.nextInt(6)
            val gw = 6 + rnd.nextInt(6)
            val cov = FloatArray(gh * gw) { if (rnd.nextInt(4) == 0) 1f else 0f }
            val sets = PatchSets.fromCov(cov, gh, gw)
            if (sets.isEmpty) return@repeat
            val d = DoubleArray(gh * gw) { rnd.nextDouble() * 3 }
            for (rule in ScoreRule.values()) {
                val a = Scoring.score(d, sets, 1.3, 1.1, rule)
                val d2 = d.copyOf()
                for (p in d2.indices) if (!sets.s[p]) d2[p] = rnd.nextDouble() * 100
                val b = Scoring.score(d2, sets, 1.3, 1.1, rule)
                assertEquals(a.raw, b.raw, 0.0)
                assertEquals(a.peak, b.peak)
                assertEquals(a.anomalousFraction, b.anomalousFraction, 0.0)
                for (p in sets.sIdx) assertEquals(a.smoothed[p], b.smoothed[p], 0.0)
            }
        }
    }

    @Test
    fun lowerRawWinsForTheFlipChallenger() {
        val sets = PatchSets.fromCore(BooleanArray(9) { true }, 3, 3)
        val a = Scoring.score(DoubleArray(9) { 2.0 }, sets, 1.0)
        val b = Scoring.score(DoubleArray(9) { 1.0 }, sets, 1.0)
        assertTrue(Scoring.lowerRaw(a, b) === b)
        assertTrue(Scoring.lowerRaw(b, a) === b)
        assertTrue(Scoring.lowerRaw(a, a) === a)
        assertFalse(Scoring.lowerRaw(a, b) === a)
    }
}
