package com.kaizeneye.core.vision

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.CropSquare
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CropPatchTest {

    private fun comp(minX: Int, minY: Int, maxX: Int, maxY: Int) =
        Component(1, 1, minX, minY, maxX, maxY, (minX + maxX + 1) / 2.0, (minY + maxY + 1) / 2.0, false)

    @Test
    fun cropSquareIsTheMarginedLongSideAroundTheBboxCentre() {
        // bbox 10..29 x 20..29 (analysis) -> full res 40..120 x 80..120, centre (80, 100), side 80*1.2 = 96
        val sq = CropMath.square(comp(10, 20, 29, 29), 4, 1280, 720, 0.10)
        assertEquals(96.0, sq.side, 1e-12)
        assertEquals(32.0, sq.x0, 1e-12)
        assertEquals(52.0, sq.y0, 1e-12)
    }

    @Test
    fun cropSquareIsClampedIntoTheFrameAndCappedAtItsShortSide() {
        val left = CropMath.square(comp(0, 40, 9, 59), 4, 1280, 720, 0.10) // bbox 0..40 x 160..240 -> side 96
        assertEquals(0.0, left.x0, 0.0)
        assertEquals(96.0, left.side, 1e-12)
        val bottomRight = CropMath.square(comp(300, 170, 319, 179), 4, 1280, 720, 0.10)
        assertEquals(1280.0 - bottomRight.side, bottomRight.x0, 1e-12)
        assertEquals(720.0 - bottomRight.side, bottomRight.y0, 1e-12)
        val huge = CropMath.square(comp(0, 0, 319, 179), 4, 1280, 720, 0.10) // 1280*1.2 capped at 720
        assertEquals(720.0, huge.side, 0.0)
        assertEquals(0.0, huge.y0, 0.0)
        assertEquals(640.0 - 360.0, huge.x0, 0.0)
        val a = CropMath.toAnalysis(CropSquare(520.0, 260.0, 240.0), 4)
        assertEquals(CropSquare(130.0, 65.0, 60.0), a)
        assertTrue(CropMath.contains(a, 130.0, 65.0))
        assertFalse(CropMath.contains(a, 190.0, 100.0))
    }

    @Test
    fun coverageCountsSixteenSamplesPerCell() {
        // 8x8 analysis map, factor 1, a 4x8 block on the left half; crop = whole map, 2x2 grid -> cells of 4x4 pixels
        val w = 8
        val h = 8
        val labels = IntArray(w * h) { if (it % w < 4) 5 else 0 }
        val cov = PatchCoverage.coverage(labels, w, h, 5, CropSquare(0.0, 0.0, 8.0), 1, 2, 2)
        assertArrayEquals(floatArrayOf(1f, 0f, 1f, 0f), cov, 0f)
        // shift the crop by 2 px: each left cell sees 2 of its 4 sample columns on the block -> 0.5
        val shifted = PatchCoverage.coverage(labels, w, h, 5, CropSquare(2.0, 0.0, 8.0), 1, 2, 2)
        assertArrayEquals(floatArrayOf(0.5f, 0f, 0.5f, 0f), shifted, 0f)
        // samples outside the analysis image count as 0
        val outside = PatchCoverage.coverage(labels, w, h, 5, CropSquare(-4.0, 0.0, 8.0), 1, 2, 2)
        assertArrayEquals(floatArrayOf(0f, 1f, 0f, 1f), outside, 0f)
        // with factor 4 the same map is sampled in full-res coordinates
        val f4 = PatchCoverage.coverage(labels, w, h, 5, CropSquare(0.0, 0.0, 32.0), 4, 2, 2)
        assertArrayEquals(floatArrayOf(1f, 0f, 1f, 0f), f4, 0f)
        // a different label is not counted
        assertArrayEquals(floatArrayOf(0f, 0f, 0f, 0f), PatchCoverage.coverage(labels, w, h, 6, CropSquare(0.0, 0.0, 8.0), 1, 2, 2), 0f)
    }

    @Test
    fun coreScoreAndBankSetsAreDilationsWithTheGridBorderOutside() {
        val gh = 6
        val gw = 6
        val cov = FloatArray(gh * gw)
        cov[1 * gw + 1] = 0.5f // exactly at the threshold -> core
        cov[4 * gw + 5] = 0.4375f // below
        val sets = PatchCoverage.sets(cov, gh, gw, 0.5)
        assertEquals(1, sets.coreCount)
        assertTrue(sets.core[1 * gw + 1])
        assertEquals(9, sets.sCount)
        for (r in 0..2) for (c in 0..2) assertTrue(sets.s[r * gw + c])
        assertEquals(16, sets.bCount) // 5x5 clipped at the top/left border -> 4x4
        for (r in 0..3) for (c in 0..3) assertTrue(sets.b[r * gw + c])
        assertFalse(sets.isCoreEmpty)
        assertTrue(PatchCoverage.sets(FloatArray(gh * gw), gh, gw, 0.5).isCoreEmpty)
    }
}
