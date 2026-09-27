package com.kaizeneye.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Reference values computed with the legacy JavaScript (mobile/src/ml/backbone.ts, knn.ts) under node 24. */
class BenchInputsTest {
    @Test
    fun benchmarkImageMatchesTheJavaScriptGenerator() {
        val img = BenchInputs.image(320)
        assertEquals(307200, img.size)
        var sum = 0.0
        for (v in img) sum += v
        assertEquals(34975519.533, sum, 0.05)
        val expected = mapOf(
            0 to 123.90166473388672, 1 to 173.90167236328125, 2 to 103.90166473388672, 3 to 125.7837905883789,
            1000 to 159.2750244140625, 99999 to 177.9164276123047, 154080 to 244.03665161132812,
        )
        for ((i, v) in expected) assertEquals("index $i", v, img[i].toDouble(), 1e-4)
        assertTrue(img.all { it in 0f..255f })
        val small = BenchInputs.image(32)
        assertEquals(398069.7756, small.sumOf { it.toDouble() }, 0.01)
        assertEquals(169.37220764160156, small[4].toDouble(), 1e-4)
        assertEquals(99.42891693115234, small[5].toDouble(), 1e-4)
    }

    @Test
    fun imageBytesAreTheRoundedImage() {
        val f = BenchInputs.image(16)
        val b = BenchInputs.imageBytes(16)
        for (i in f.indices) assertEquals(Math.floor(f[i] + 0.5).toInt().coerceIn(0, 255), b[i].toInt() and 0xff)
    }

    @Test
    fun lcgAndFakeFeaturesMatchTheJavaScript() {
        val r = BenchInputs.Lcg(1234 + 2400)
        assertEquals(0.6444334930274636, r.next(), 0.0)
        assertEquals(0.8960495116189122, r.next(), 0.0)
        val f = BenchInputs.fakeFeatures(BenchInputs.Lcg(42), 10, 8)
        assertEquals(80, f.size)
        assertEquals(9.276854, f.sumOf { it.toDouble() }, 1e-5)
        assertEquals(0.13558000326156616, f[0].toDouble(), 0.0)
        assertEquals(0.11459100246429443, f[1].toDouble(), 0.0)
        assertEquals(0.08829758316278458, f[2].toDouble(), 0.0)
        assertEquals(0.14027293026447296, f[3].toDouble(), 0.0)
        assertEquals(0.08742368221282959, f[79].toDouble(), 0.0)
        assertTrue(f.all { it >= 0f })
    }
}
