package com.kaizeneye.core.mask

import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.math.Stats
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.util.Random

class ByteHistogramTest {

    private fun hist(values: IntArray): IntArray = IntArray(256).also { h -> values.forEach { h[it]++ } }

    @Test
    fun handCasesIncludingEvenCounts() {
        assertEquals(7.0, ByteHistogram.median(hist(intArrayOf(7))), 0.0)
        assertEquals(1.5, ByteHistogram.median(hist(intArrayOf(1, 2))), 0.0)
        assertEquals(0.5, ByteHistogram.mad(hist(intArrayOf(1, 2)), 1.5), 0.0)
        // [1,2,3,4]: median 2.5, deviations [1.5, .5, .5, 1.5] -> MAD 1.0
        assertEquals(2.5, ByteHistogram.median(hist(intArrayOf(4, 1, 3, 2))), 0.0)
        assertEquals(1.0, ByteHistogram.mad(hist(intArrayOf(4, 1, 3, 2)), 2.5), 0.0)
        // [0, 0, 255, 255]: median 127.5, MAD 127.5
        assertEquals(127.5, ByteHistogram.median(hist(intArrayOf(0, 255, 0, 255))), 0.0)
        assertEquals(127.5, ByteHistogram.mad(hist(intArrayOf(0, 255, 0, 255)), 127.5), 0.0)
    }

    @Test
    fun histogramMedianAndMadEqualTheSortDefinitionsOnRandomData() {
        val rnd = Random(1234)
        val scratch = IntArray(ByteHistogram.DEVIATION_BINS)
        repeat(400) { trial ->
            val n = 1 + rnd.nextInt(if (trial % 3 == 0) 8 else 300)
            val span = 1 + rnd.nextInt(256) // narrow spans produce many duplicates
            val lo = rnd.nextInt(257 - span)
            val v = IntArray(n) { lo + rnd.nextInt(span) }
            val d = DoubleArray(n) { v[it].toDouble() }
            val h = hist(v)
            val med = ByteHistogram.median(h)
            assertEquals("median n=$n", Stats.median(d), med, 0.0)
            assertEquals("mad n=$n", Stats.mad(d), ByteHistogram.mad(h, med, scratch), 0.0)
        }
    }

    @Test
    fun sheetFitUsesExactMedianMadOverAllFramesAndTheRoi() {
        val rnd = Random(99)
        val frames = List(3) { RgbImage(10, 6).also { rnd.nextBytes(it.data) } }
        val roi = intArrayOf(2, 1, 9, 5)
        val model = SheetModel.fit(frames, roi, sigmaMin = 3.0)
        for (c in 0 until 3) {
            val vals = ArrayList<Double>()
            for (f in frames) for (y in 1 until 5) for (x in 2 until 9) vals += f[x, y, c].toDouble()
            val arr = vals.toDoubleArray()
            assertEquals(Stats.median(arr), model.mean[c], 0.0)
            val mad = Stats.mad(arr)
            assertNotNull(model.mad)
            assertEquals(mad, model.mad!![c], 0.0)
            assertEquals(maxOf(1.4826 * mad, 3.0), model.sigma[c], 0.0)
        }
        assertArrayEquals(roi, model.roi)
    }

    @Test
    fun sigmaFloorAppliesToAFlatSheet() {
        val flat = RgbImage(8, 8).also { it.fill(120, 121, 122) }
        val model = SheetModel.fit(listOf(flat, flat))
        assertArrayEquals(doubleArrayOf(120.0, 121.0, 122.0), model.mean, 0.0)
        assertArrayEquals(doubleArrayOf(3.0, 3.0, 3.0), model.sigma, 0.0)
        assertArrayEquals(doubleArrayOf(0.0, 0.0, 0.0), model.mad!!, 0.0)
    }

    @Test
    fun adaptMovesTheMeanTowardsTheFrameMedianByBeta() {
        val model = SheetModel(doubleArrayOf(100.0, 100.0, 100.0), doubleArrayOf(5.0, 5.0, 5.0))
        val frame = RgbImage(4, 4).also { it.fill(150, 100, 50) }
        model.adapt(frame, beta = 0.02)
        assertArrayEquals(doubleArrayOf(0.98 * 100 + 0.02 * 150, 100.0, 0.98 * 100 + 0.02 * 50), model.mean, 1e-12)
        assertArrayEquals(doubleArrayOf(5.0, 5.0, 5.0), model.sigma, 0.0)
        // the ROI of the model is respected: a frame whose ROI is (150, 100, 50) but whose outside is black
        val roiModel = SheetModel(doubleArrayOf(100.0, 100.0, 100.0), doubleArrayOf(5.0, 5.0, 5.0), intArrayOf(1, 1, 3, 3))
        val framed = RgbImage(4, 4).also { it.fill(0, 0, 0) }
        for (y in 1 until 3) for (x in 1 until 3) framed.setRgb(x, y, 150, 100, 50)
        roiModel.adapt(framed, beta = 0.5)
        assertArrayEquals(doubleArrayOf(125.0, 100.0, 75.0), roiModel.mean, 0.0)
    }
}
