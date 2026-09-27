package com.kaizeneye.core.image

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.floor

class ImageOpsTest {

    private fun randomRgba(w: Int, h: Int, stride: Int, seed: Long): RgbaFrame {
        val rnd = Random(seed)
        val f = RgbaFrame.allocate(w, h, stride)
        rnd.nextBytes(f.data) // padding bytes are junk on purpose
        return f
    }

    private fun randomRgb(w: Int, h: Int, seed: Long): RgbImage {
        val img = RgbImage(w, h)
        Random(seed).nextBytes(img.data)
        return img
    }

    /** floor((Σ + f²/2) / f²) with f²/2 as a real number (§1.1), computed naively in float64. */
    private fun naiveDownscale(src: RgbaFrame, f: Int): RgbImage {
        val out = RgbImage(src.width / f, src.height / f)
        for (y in 0 until out.height) for (x in 0 until out.width) for (c in 0 until 3) {
            var s = 0
            for (i in 0 until f) for (j in 0 until f) s += src[f * x + j, f * y + i, c]
            out[x, y, c] = floor((s + f * f / 2.0) / (f * f)).toInt()
        }
        return out
    }

    @Test
    fun downscaleMatchesNaiveBoxAverageForEvenAndOddFactors() {
        for ((f, seed) in listOf(4 to 1L, 2 to 2L, 3 to 3L, 1 to 4L)) {
            val src = randomRgba(12 * f, 6 * f, 12 * f * 4 + 20, seed) // stride padded by 20 junk bytes
            val dst = RgbImage(src.width / f, src.height / f)
            ImageOps.downscaleRgba(src, f, dst)
            assertArrayEquals("factor $f", naiveDownscale(src, f).data, dst.data)
        }
    }

    @Test
    fun framesCopyFromACameraPlaneBuffer() {
        val stride = 3 * 4 + 4
        val bytes = ByteArray(stride * 1 + 3 * 4) { it.toByte() } // last row without padding, as camera planes often are
        val plane = java.nio.ByteBuffer.allocateDirect(bytes.size).put(bytes)
        plane.position(5) // a consumer may have moved it; the copy starts at 0 and leaves it alone
        val f = RgbaFrame.allocate(3, 2, stride)
        assertEquals(bytes.size, f.copyFrom(plane))
        assertEquals(5, plane.position())
        assertEquals(stride + 4 + 1, f[1, 1, 1]) // byte index y*stride + x*4 + c
    }

    @Test
    fun downscaleRoundsHalfUp() {
        val src = RgbaFrame.allocate(4, 4)
        // 8 pixels of 1 and 8 of 0 in R → Σ = 8 → floor((8 + 8) / 16) = 1 (a mean of exactly 0.5 rounds up)
        for (y in 0 until 4) for (x in 0 until 4) src[x, y, 0] = if (y < 2) 1 else 0
        // 7 pixels of 1 in G → Σ = 7 → 0
        for (i in 0 until 7) src[i % 4, i / 4, 1] = 1
        val dst = RgbImage(1, 1)
        ImageOps.downscaleRgba(src, 4, dst)
        assertEquals(1, dst[0, 0, 0])
        assertEquals(0, dst[0, 0, 1])
        assertEquals(0, dst[0, 0, 2])
    }

    @Test
    fun greyIsTheExactWeightedSum() {
        val img = randomRgb(9, 7, 11L)
        val g = ImageOps.grey(img)
        for (y in 0 until 7) for (x in 0 until 9) {
            val expected = 0.299 * img[x, y, 0] + 0.587 * img[x, y, 1] + 0.114 * img[x, y, 2]
            assertEquals(expected, g[y * 9 + x], 0.0)
        }
    }

    @Test
    fun laplacianVarianceOfAnImpulseIsComputedByHand() {
        val w = 5
        val h = 5
        val g = DoubleArray(w * h)
        g[2 * w + 2] = 1.0
        // interior 3x3: L = -4 at the centre, +1 at its 4 neighbours, 0 at the corners -> mean 0, var = 20/9
        assertEquals(20.0 / 9.0, ImageOps.laplacianVariance(g, w, h, 0, 0, w, h), 1e-15)
        assertEquals(9, ImageOps.sharpnessPixelCount(w, h, 0, 0, w, h))
        // rectangles reaching outside the image are clamped
        assertEquals(20.0 / 9.0, ImageOps.laplacianVariance(g, w, h, -10, -3, 99, 77), 1e-15)
        // fewer than 9 qualifying pixels -> 0
        assertEquals(6, ImageOps.sharpnessPixelCount(w, h, 1, 1, 3, 4))
        assertEquals(0.0, ImageOps.laplacianVariance(g, w, h, 1, 1, 3, 4), 0.0)
        // constant image -> 0
        assertEquals(0.0, ImageOps.laplacianVariance(DoubleArray(w * h) { 7.5 }, w, h, 0, 0, w, h), 0.0)
        // empty rectangle
        assertEquals(0, ImageOps.sharpnessPixelCount(w, h, 3, 3, 2, 2))
        assertEquals(0.0, ImageOps.laplacianVariance(g, w, h, 3, 3, 2, 2), 0.0)
    }

    @Test
    fun laplacianVarianceMatchesATwoPassReference() {
        val img = randomRgb(31, 17, 5L)
        val g = ImageOps.grey(img)
        val ls = ArrayList<Double>()
        for (y in 3 until 12) for (x in 2 until 20) {
            ls += g[(y - 1) * 31 + x] + g[(y + 1) * 31 + x] + g[y * 31 + x - 1] + g[y * 31 + x + 1] - 4.0 * g[y * 31 + x]
        }
        val mean = ls.sum() / ls.size
        val v = ls.sumOf { (it - mean) * (it - mean) } / ls.size
        assertEquals(v, ImageOps.laplacianVariance(g, 31, 17, 2, 3, 20, 12), 1e-9 * v)
    }

    @Test
    fun bilinearSampleHitsPixelCentresAndInterpolates() {
        val img = RgbImage(3, 2)
        img.setRgb(0, 0, 10, 20, 30)
        img.setRgb(1, 0, 11, 40, 50)
        img.setRgb(2, 0, 12, 60, 70)
        img.setRgb(0, 1, 100, 0, 0)
        img.setRgb(1, 1, 101, 0, 0)
        img.setRgb(2, 1, 102, 0, 255)
        val out = DoubleArray(3)
        ImageOps.sample(img, 1.5, 0.5, out) // centre of pixel (1, 0)
        assertArrayEquals(doubleArrayOf(11.0, 40.0, 50.0), out, 0.0)
        ImageOps.sample(img, 1.0, 0.5, out) // halfway between (0,0) and (1,0)
        assertArrayEquals(doubleArrayOf(10.5, 30.0, 40.0), out, 0.0)
        ImageOps.sample(img, 1.0, 1.0, out) // centre of the 2x2 block (0..1, 0..1)
        assertArrayEquals(doubleArrayOf((10 + 11 + 100 + 101) / 4.0, 15.0, 20.0), out, 1e-12)
        ImageOps.sample(img, -7.0, 99.0, out) // clamped to the bottom-left pixel
        assertArrayEquals(doubleArrayOf(100.0, 0.0, 0.0), out, 0.0)
        ImageOps.sample(img, 3.0, 2.0, out) // clamped to the bottom-right pixel
        assertArrayEquals(doubleArrayOf(102.0, 0.0, 255.0), out, 0.0)
    }

    @Test
    fun cropResizeOfTheWholeImageAtNativeSizeIsTheIdentity() {
        val img = randomRgb(16, 16, 7L)
        val out = ByteArray(16 * 16 * 3)
        ImageOps.cropResize(img, 0.0, 0.0, 16.0, 16, out)
        assertArrayEquals(img.data, out)
        val f = DoubleArray(16 * 16 * 3)
        ImageOps.cropResizeFloat(img, 0.0, 0.0, 16.0, 16, f)
        for (i in f.indices) assertEquals((img.data[i].toInt() and 0xFF).toDouble(), f[i], 0.0)
    }

    @Test
    fun cropBytesRoundHalfUp() {
        val img = RgbImage(2, 1)
        img.setRgb(0, 0, 10, 0, 254)
        img.setRgb(1, 0, 11, 1, 255)
        val out = ByteArray(3)
        val f = DoubleArray(3)
        ImageOps.cropResizeFloat(img, 0.5, 0.0, 1.0, 1, f) // u = 1.0 -> halfway between the two pixels
        assertArrayEquals(doubleArrayOf(10.5, 0.5, 254.5), f, 0.0)
        ImageOps.cropResize(img, 0.5, 0.0, 1.0, 1, out)
        assertEquals(11, out[0].toInt() and 0xFF)
        assertEquals(1, out[1].toInt() and 0xFF)
        assertEquals(255, out[2].toInt() and 0xFF)
        assertEquals(0, ImageOps.roundToByte(-3.2))
        assertEquals(0, ImageOps.roundToByte(0.49999999))
        assertEquals(255, ImageOps.roundToByte(300.0))
        assertEquals(128, ImageOps.roundToByte(127.5))
    }

    @Test
    fun rgbaAndRgbSourcesGiveTheSameCrop() {
        val frame = randomRgba(40, 30, 40 * 4 + 12, 9L)
        val rgb = RgbImage(40, 30)
        for (y in 0 until 30) for (x in 0 until 40) rgb.setRgb(x, y, frame[x, y, 0], frame[x, y, 1], frame[x, y, 2])
        val a = ByteArray(24 * 24 * 3)
        val b = ByteArray(24 * 24 * 3)
        ImageOps.cropResize(frame, 3.3, -2.25, 31.7, 24, a)
        ImageOps.cropResize(rgb, 3.3, -2.25, 31.7, 24, b)
        assertArrayEquals(b, a)
        val ra = ByteArray(24 * 24 * 3)
        val rb = ByteArray(24 * 24 * 3)
        ImageOps.cropResizeRotated(frame, 20.2, 14.9, 25.0, 0.7, 24, ra)
        ImageOps.cropResizeRotated(rgb, 20.2, 14.9, 25.0, 0.7, 24, rb)
        assertArrayEquals(rb, ra)
    }

    @Test
    fun rotatedCropAtZeroAngleEqualsTheAxisAlignedCrop() {
        val img = randomRgb(20, 20, 13L)
        val n = 4
        val axis = DoubleArray(n * n * 3)
        val rot = DoubleArray(n * n * 3)
        ImageOps.cropResizeFloat(img, 2.0, 5.0, 8.0, n, axis) // dyadic geometry: every coordinate exact
        ImageOps.cropResizeRotatedFloat(img, 6.0, 9.0, 8.0, 0.0, n, rot)
        assertArrayEquals(axis, rot, 0.0)
    }

    @Test
    fun rotatedCropByNinetyDegreesEqualsRot90() {
        val img = randomRgb(8, 8, 17L)
        val out = ByteArray(8 * 8 * 3)
        ImageOps.cropResizeRotated(img, 4.0, 4.0, 8.0, Math.PI / 2, 8, out)
        assertArrayEquals(ImageOps.rotate90(img, 1).data, out)
    }

    @Test
    fun rotate90FollowsNumpyRot90() {
        // 3 wide x 2 high, R channel = 1..6 row-major:  [[1 2 3], [4 5 6]]
        val img = RgbImage(3, 2)
        for (i in 0 until 6) img.setRgb(i % 3, i / 3, i + 1, 0, 0)
        fun red(m: RgbImage) = (0 until m.width * m.height).map { m.data[it * 3].toInt() }
        val r1 = ImageOps.rotate90(img, 1) // numpy.rot90 -> [[3 6], [2 5], [1 4]]
        assertEquals(2, r1.width)
        assertEquals(3, r1.height)
        assertEquals(listOf(3, 6, 2, 5, 1, 4), red(r1))
        assertEquals(listOf(6, 5, 4, 3, 2, 1), red(ImageOps.rotate90(img, 2)))
        assertEquals(listOf(4, 1, 5, 2, 6, 3), red(ImageOps.rotate90(img, 3)))
        assertEquals(listOf(4, 1, 5, 2, 6, 3), red(ImageOps.rotate90(img, -1)))
        assertEquals(red(img), red(ImageOps.rotate90(img, 4)))
    }

    // ---------------------------------------------------------------------------------------------------------------
    // JVM timing smoke tests (the phone budget is a few ms; these print the laptop numbers and only fail if absurd)

    private inline fun medianMs(runs: Int, warmup: Int, block: () -> Unit): Double {
        repeat(warmup) { block() }
        val t = DoubleArray(runs)
        for (i in 0 until runs) {
            val t0 = System.nanoTime()
            block()
            t[i] = (System.nanoTime() - t0) / 1e6
        }
        t.sort()
        return t[runs / 2]
    }

    @Test
    fun timingDownscale1280x720() {
        val src = randomRgba(1280, 720, 1280 * 4, 21L)
        val dst = RgbImage(320, 180)
        val ms = medianMs(runs = 50, warmup = 100) { ImageOps.downscaleRgba(src, 4, dst) }
        println("[timing] downscaleRgba 1280x720 -> 320x180: %.3f ms (JVM median)".format(ms))
        assertTrue("downscale took $ms ms", ms < 40.0)
    }

    @Test
    fun timingCropResize1280x720To320() {
        val src = randomRgba(1280, 720, 1280 * 4, 22L)
        val out = ByteArray(320 * 320 * 3)
        val ms = medianMs(runs = 40, warmup = 60) { ImageOps.cropResize(src, 400.3, 150.7, 402.5, 320, out) }
        println("[timing] cropResize RGBA 1280x720 -> 320x320: %.3f ms (JVM median)".format(ms))
        val ms448 = medianMs(runs = 20, warmup = 20) { ImageOps.cropResize(src, 400.3, 150.7, 402.5, 448, ByteArray(448 * 448 * 3)) }
        println("[timing] cropResize RGBA 1280x720 -> 448x448: %.3f ms (JVM median)".format(ms448))
        val msRot = medianMs(runs = 20, warmup = 20) { ImageOps.cropResizeRotated(src, 600.0, 350.0, 402.5, 0.6, 320, out) }
        println("[timing] cropResizeRotated RGBA -> 320x320: %.3f ms (JVM median)".format(msRot))
        assertTrue("cropResize took $ms ms", ms < 60.0)
    }

    @Test
    fun timingGreyAndSharpness320x180() {
        val img = randomRgb(320, 180, 23L)
        val g = DoubleArray(320 * 180)
        val ms = medianMs(runs = 50, warmup = 200) {
            ImageOps.grey(img, g)
            ImageOps.laplacianVariance(g, 320, 180, 100, 50, 180, 130)
        }
        println("[timing] grey 320x180 + sharpness 80x80: %.3f ms (JVM median)".format(ms))
        assertTrue(ms < 40.0)
    }
}
