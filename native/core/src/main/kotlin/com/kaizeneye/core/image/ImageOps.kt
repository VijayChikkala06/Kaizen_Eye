package com.kaizeneye.core.image

import kotlin.math.floor

/**
 * Pixel operations of spec §1 (docs/verification/twin-spec.md). All float maths is float64; every loop runs over primitive
 * arrays with no per-pixel allocation, so these are safe on the 30 fps analyzer thread.
 *
 * Crops ([cropResize], [cropResizeRotated]) allocate only a few `N`-sized index/weight tables per call (once per judged
 * part, not per frame).
 */
object ImageOps {

    private val GREY_R = DoubleArray(256) { 0.299 * it }
    private val GREY_G = DoubleArray(256) { 0.587 * it }
    private val GREY_B = DoubleArray(256) { 0.114 * it }

    // ---------------------------------------------------------------------------------------------------------------
    // §1.1 analysis downscale

    /**
     * Spec §1.1: `f×f` box average of an RGBA frame into [dst] (RGB, `W/f × H/f`), alpha ignored, integer maths:
     * `out = floor((Σ + f²/2) / f²)`. Zero allocation. Requires `W` and `H` divisible by [factor].
     */
    fun downscaleRgba(src: RgbaFrame, factor: Int, dst: RgbImage) {
        require(factor >= 1) { "factor must be >= 1, got $factor" }
        require(src.width % factor == 0 && src.height % factor == 0) {
            "frame ${src.width}x${src.height} is not divisible by $factor"
        }
        require(dst.width == src.width / factor && dst.height == src.height / factor) {
            "dst is ${dst.width}x${dst.height}, expected ${src.width / factor}x${src.height / factor}"
        }
        if (factor == 4) downscale4(src, dst) else downscaleGeneric(src, factor, dst)
    }

    private fun downscaleGeneric(src: RgbaFrame, f: Int, dst: RgbImage) {
        val sd = src.data
        val stride = src.rowStride
        val dd = dst.data
        val area = f * f
        val half = area / 2 // for odd f, floor((S + f²/2)/f²) == floor((S + (f² div 2))/f²) because S is an integer
        val ow = dst.width
        val oh = dst.height
        val blockStep = f * RgbaFrame.PIXEL_STRIDE
        var o = 0
        for (oy in 0 until oh) {
            val rowBase = oy * f * stride
            var block = rowBase
            for (ox in 0 until ow) {
                var r = 0
                var g = 0
                var b = 0
                var p = block
                for (i in 0 until f) {
                    var q = p
                    for (j in 0 until f) {
                        r += sd[q].toInt() and 0xFF
                        g += sd[q + 1].toInt() and 0xFF
                        b += sd[q + 2].toInt() and 0xFF
                        q += RgbaFrame.PIXEL_STRIDE
                    }
                    p += stride
                }
                dd[o] = ((r + half) / area).toByte()
                dd[o + 1] = ((g + half) / area).toByte()
                dd[o + 2] = ((b + half) / area).toByte()
                o += 3
                block += blockStep
            }
        }
    }

    @Suppress("NOTHING_TO_INLINE")
    private inline fun sum4(d: ByteArray, p: Int): Int =
        (d[p].toInt() and 0xFF) + (d[p + 4].toInt() and 0xFF) + (d[p + 8].toInt() and 0xFF) + (d[p + 12].toInt() and 0xFF)

    /** The default factor 4 (1280×720 → 320×180), unrolled: 16 pixels per output pixel, (Σ + 8) >> 4. */
    private fun downscale4(src: RgbaFrame, dst: RgbImage) {
        val sd = src.data
        val stride = src.rowStride
        val dd = dst.data
        val ow = dst.width
        val oh = dst.height
        var o = 0
        for (oy in 0 until oh) {
            var p0 = oy * 4 * stride
            var p1 = p0 + stride
            var p2 = p1 + stride
            var p3 = p2 + stride
            for (ox in 0 until ow) {
                val r = sum4(sd, p0) + sum4(sd, p1) + sum4(sd, p2) + sum4(sd, p3)
                val g = sum4(sd, p0 + 1) + sum4(sd, p1 + 1) + sum4(sd, p2 + 1) + sum4(sd, p3 + 1)
                val b = sum4(sd, p0 + 2) + sum4(sd, p1 + 2) + sum4(sd, p2 + 2) + sum4(sd, p3 + 2)
                dd[o] = ((r + 8) shr 4).toByte()
                dd[o + 1] = ((g + 8) shr 4).toByte()
                dd[o + 2] = ((b + 8) shr 4).toByte()
                o += 3
                p0 += 16
                p1 += 16
                p2 += 16
                p3 += 16
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // §1.2 grey, §1.3 sharpness

    /** Spec §1.2 grey image `0.299 R + 0.587 G + 0.114 B` (float64), row-major `[H*W]`. Allocates the result. */
    fun grey(img: RgbImage): DoubleArray = DoubleArray(img.width * img.height).also { grey(img, it) }

    /** Spec §1.2 into a preallocated [dst] of at least `W*H` values. Zero allocation. */
    fun grey(img: RgbImage, dst: DoubleArray) {
        val n = img.width * img.height
        require(dst.size >= n) { "dst has ${dst.size} values, need $n" }
        val d = img.data
        var p = 0
        for (i in 0 until n) {
            dst[i] = GREY_R[d[p].toInt() and 0xFF] + GREY_G[d[p + 1].toInt() and 0xFF] + GREY_B[d[p + 2].toInt() and 0xFF]
            p += 3
        }
    }

    /**
     * Spec §1.3 sharpness: population variance of the 4-neighbour Laplacian
     * `G[y−1][x] + G[y+1][x] + G[y][x−1] + G[y][x+1] − 4·G[y][x]` over the pixels with `max(x0,1) ≤ x < min(x1, W−1)` and
     * `max(y0,1) ≤ y < min(y1, H−1)` (rectangle `[x0, x1) × [y0, y1)` clamped to the image). `0.0` if fewer than 9
     * pixels qualify. Two passes (mean, then squared deviations), float64 in row-major order. Zero allocation.
     */
    fun laplacianVariance(grey: DoubleArray, w: Int, h: Int, x0: Int, y0: Int, x1: Int, y1: Int): Double {
        require(grey.size >= w * h) { "grey has ${grey.size} values, need ${w * h}" }
        val xs = maxOf(x0, 1)
        val xe = minOf(x1, w - 1)
        val ys = maxOf(y0, 1)
        val ye = minOf(y1, h - 1)
        if (xe <= xs || ye <= ys) return 0.0
        val n = (xe - xs).toLong() * (ye - ys)
        if (n < MIN_SHARPNESS_PIXELS) return 0.0
        var sum = 0.0
        for (y in ys until ye) {
            var i = y * w + xs
            for (x in xs until xe) {
                sum += grey[i - w] + grey[i + w] + grey[i - 1] + grey[i + 1] - 4.0 * grey[i]
                i++
            }
        }
        val mean = sum / n
        var ss = 0.0
        for (y in ys until ye) {
            var i = y * w + xs
            for (x in xs until xe) {
                val d = grey[i - w] + grey[i + w] + grey[i - 1] + grey[i + 1] - 4.0 * grey[i] - mean
                ss += d * d
                i++
            }
        }
        return ss / n
    }

    /** §1.3: fewer qualifying pixels than this → sharpness 0. */
    const val MIN_SHARPNESS_PIXELS = 9

    /** Number of pixels that qualify for §1.3 in the rectangle `[x0, x1) × [y0, y1)` of a `w×h` image. */
    fun sharpnessPixelCount(w: Int, h: Int, x0: Int, y0: Int, x1: Int, y1: Int): Int {
        val nx = minOf(x1, w - 1) - maxOf(x0, 1)
        val ny = minOf(y1, h - 1) - maxOf(y0, 1)
        return if (nx <= 0 || ny <= 0) 0 else nx * ny
    }

    // ---------------------------------------------------------------------------------------------------------------
    // §1.4 bilinear sample, crop-resize, rotated crop

    /** Spec §0 rounding of a pixel value: `floor(v + 0.5)` clamped to `[0, 255]`. */
    fun roundToByte(v: Double): Int = when {
        v <= 0.0 -> 0
        v >= 254.5 -> 255
        else -> (v + 0.5).toInt() // v + 0.5 > 0, so truncation == floor
    }

    /** Spec §1.4 `sample(I, u, v)` at continuous point ([u], [v]) → [out] (3 channels, float64). */
    fun sample(img: RgbImage, u: Double, v: Double, out: DoubleArray) =
        sampleAt(img.data, img.width, img.height, img.width * 3, 3, u, v, out)

    /** Spec §1.4 `sample(I, u, v)` on an RGBA frame (alpha ignored) → [out] (3 channels, float64). */
    fun sample(frame: RgbaFrame, u: Double, v: Double, out: DoubleArray) =
        sampleAt(frame.data, frame.width, frame.height, frame.rowStride, RgbaFrame.PIXEL_STRIDE, u, v, out)

    private fun sampleAt(data: ByteArray, w: Int, h: Int, rowStride: Int, ps: Int, u: Double, v: Double, out: DoubleArray) {
        val x = u - 0.5
        val y = v - 0.5
        val xf = floor(x)
        val yf = floor(y)
        val fx = x - xf
        val fy = y - yf
        val xi = floorToIndex(xf, w)
        val yi = floorToIndex(yf, h)
        val c0 = clampIndex(xi, w) * ps
        val c1 = clampIndex(xi + 1, w) * ps
        val r0 = clampIndex(yi, h) * rowStride
        val r1 = clampIndex(yi + 1, h) * rowStride
        val gx = 1.0 - fx
        val gy = 1.0 - fy
        val w00 = gx * gy
        val w01 = fx * gy
        val w10 = gx * fy
        val w11 = fx * fy
        for (c in 0 until 3) {
            out[c] = w00 * (data[r0 + c0 + c].toInt() and 0xFF) + w01 * (data[r0 + c1 + c].toInt() and 0xFF) +
                w10 * (data[r1 + c0 + c].toInt() and 0xFF) + w11 * (data[r1 + c1 + c].toInt() and 0xFF)
        }
    }

    /**
     * Spec §1.4 `cropResize(I, x0, y0, side, N)` from a full-resolution RGBA frame: output pixel `(i, j)` =
     * `sample(I, x0 + (j+0.5)·side/N, y0 + (i+0.5)·side/N)`, written to [dst] (`N·N·3` bytes) as the **crop bytes**
     * `clamp(round(v))`. Column/row taps and weights are precomputed once per call; no per-pixel allocation.
     */
    fun cropResize(src: RgbaFrame, x0: Double, y0: Double, side: Double, n: Int, dst: ByteArray) =
        crop(src.data, src.width, src.height, src.rowStride, RgbaFrame.PIXEL_STRIDE, x0, y0, side, n, dst, null)

    /** [cropResize] from an RGB image (e.g. a stored crop or a test image). */
    fun cropResize(src: RgbImage, x0: Double, y0: Double, side: Double, n: Int, dst: ByteArray) =
        crop(src.data, src.width, src.height, src.width * 3, 3, x0, y0, side, n, dst, null)

    /** [cropResize] returning the unrounded float64 samples (`N·N·3`), for tests against the golden floats. */
    fun cropResizeFloat(src: RgbaFrame, x0: Double, y0: Double, side: Double, n: Int, dst: DoubleArray) =
        crop(src.data, src.width, src.height, src.rowStride, RgbaFrame.PIXEL_STRIDE, x0, y0, side, n, null, dst)

    /** [cropResize] returning the unrounded float64 samples (`N·N·3`), for tests against the golden floats. */
    fun cropResizeFloat(src: RgbImage, x0: Double, y0: Double, side: Double, n: Int, dst: DoubleArray) =
        crop(src.data, src.width, src.height, src.width * 3, 3, x0, y0, side, n, null, dst)

    private fun crop(
        data: ByteArray, w: Int, h: Int, rowStride: Int, ps: Int,
        x0: Double, y0: Double, side: Double, n: Int,
        dstBytes: ByteArray?, dstFloat: DoubleArray?,
    ) {
        require(n > 0) { "N must be > 0" }
        require(side > 0.0) { "side must be > 0, got $side" }
        dstBytes?.let { require(it.size >= n * n * 3) { "dst has ${it.size} bytes, need ${n * n * 3}" } }
        dstFloat?.let { require(it.size >= n * n * 3) { "dst has ${it.size} values, need ${n * n * 3}" } }
        val cx0 = IntArray(n)
        val cx1 = IntArray(n)
        val cfx = DoubleArray(n)
        val ry0 = IntArray(n)
        val ry1 = IntArray(n)
        val rfy = DoubleArray(n)
        taps(x0, side, n, w, ps, cx0, cx1, cfx)
        taps(y0, side, n, h, rowStride, ry0, ry1, rfy)
        var o = 0
        for (i in 0 until n) {
            val fy = rfy[i]
            val gy = 1.0 - fy
            val r0 = ry0[i]
            val r1 = ry1[i]
            for (j in 0 until n) {
                val fx = cfx[j]
                val gx = 1.0 - fx
                val w00 = gx * gy
                val w01 = fx * gy
                val w10 = gx * fy
                val w11 = fx * fy
                val p00 = r0 + cx0[j]
                val p01 = r0 + cx1[j]
                val p10 = r1 + cx0[j]
                val p11 = r1 + cx1[j]
                val vr = w00 * (data[p00].toInt() and 0xFF) + w01 * (data[p01].toInt() and 0xFF) +
                    w10 * (data[p10].toInt() and 0xFF) + w11 * (data[p11].toInt() and 0xFF)
                val vg = w00 * (data[p00 + 1].toInt() and 0xFF) + w01 * (data[p01 + 1].toInt() and 0xFF) +
                    w10 * (data[p10 + 1].toInt() and 0xFF) + w11 * (data[p11 + 1].toInt() and 0xFF)
                val vb = w00 * (data[p00 + 2].toInt() and 0xFF) + w01 * (data[p01 + 2].toInt() and 0xFF) +
                    w10 * (data[p10 + 2].toInt() and 0xFF) + w11 * (data[p11 + 2].toInt() and 0xFF)
                if (dstBytes != null) {
                    dstBytes[o] = roundToByte(vr).toByte()
                    dstBytes[o + 1] = roundToByte(vg).toByte()
                    dstBytes[o + 2] = roundToByte(vb).toByte()
                } else {
                    dstFloat!![o] = vr
                    dstFloat[o + 1] = vg
                    dstFloat[o + 2] = vb
                }
                o += 3
            }
        }
    }

    /** One axis of the separable crop: sample k at `origin + (k+0.5)·side/n` → clamped tap offsets and weight. */
    private fun taps(origin: Double, side: Double, n: Int, size: Int, mul: Int, i0: IntArray, i1: IntArray, frac: DoubleArray) {
        for (k in 0 until n) {
            val u = origin + (k + 0.5) * side / n
            val x = u - 0.5
            val xf = floor(x)
            frac[k] = x - xf
            val xi = floorToIndex(xf, size)
            i0[k] = clampIndex(xi, size) * mul
            i1[k] = clampIndex(xi + 1, size) * mul
        }
    }

    /**
     * Spec §1.4 `cropResizeRotated(I, cx, cy, side, θ, N)` (rotation challenger §6.5): output pixel `(i, j)` samples at
     * `(cx + a·cosθ − b·sinθ, cy + a·sinθ + b·cosθ)`, `a = (j+0.5)·side/N − side/2`, `b = (i+0.5)·side/N − side/2`;
     * rounded crop bytes into [dst].
     */
    fun cropResizeRotated(src: RgbaFrame, cx: Double, cy: Double, side: Double, theta: Double, n: Int, dst: ByteArray) =
        cropRotated(src.data, src.width, src.height, src.rowStride, RgbaFrame.PIXEL_STRIDE, cx, cy, side, theta, n, dst, null)

    /** [cropResizeRotated] from an RGB image. */
    fun cropResizeRotated(src: RgbImage, cx: Double, cy: Double, side: Double, theta: Double, n: Int, dst: ByteArray) =
        cropRotated(src.data, src.width, src.height, src.width * 3, 3, cx, cy, side, theta, n, dst, null)

    /** [cropResizeRotated] returning the unrounded float64 samples. */
    fun cropResizeRotatedFloat(src: RgbaFrame, cx: Double, cy: Double, side: Double, theta: Double, n: Int, dst: DoubleArray) =
        cropRotated(src.data, src.width, src.height, src.rowStride, RgbaFrame.PIXEL_STRIDE, cx, cy, side, theta, n, null, dst)

    /** [cropResizeRotated] returning the unrounded float64 samples. */
    fun cropResizeRotatedFloat(src: RgbImage, cx: Double, cy: Double, side: Double, theta: Double, n: Int, dst: DoubleArray) =
        cropRotated(src.data, src.width, src.height, src.width * 3, 3, cx, cy, side, theta, n, null, dst)

    private fun cropRotated(
        data: ByteArray, w: Int, h: Int, rowStride: Int, ps: Int,
        cx: Double, cy: Double, side: Double, theta: Double, n: Int,
        dstBytes: ByteArray?, dstFloat: DoubleArray?,
    ) {
        require(n > 0) { "N must be > 0" }
        require(side > 0.0) { "side must be > 0, got $side" }
        dstBytes?.let { require(it.size >= n * n * 3) { "dst has ${it.size} bytes, need ${n * n * 3}" } }
        dstFloat?.let { require(it.size >= n * n * 3) { "dst has ${it.size} values, need ${n * n * 3}" } }
        val c = StrictMath.cos(theta)
        val s = StrictMath.sin(theta)
        val off = DoubleArray(n) { (it + 0.5) * side / n - side / 2 }
        var o = 0
        for (i in 0 until n) {
            val b = off[i]
            for (j in 0 until n) {
                val a = off[j]
                val x = cx + a * c - b * s - 0.5
                val y = cy + a * s + b * c - 0.5
                val xf = floor(x)
                val yf = floor(y)
                val fx = x - xf
                val fy = y - yf
                val xi = floorToIndex(xf, w)
                val yi = floorToIndex(yf, h)
                val c0 = clampIndex(xi, w) * ps
                val c1 = clampIndex(xi + 1, w) * ps
                val r0 = clampIndex(yi, h) * rowStride
                val r1 = clampIndex(yi + 1, h) * rowStride
                val gx = 1.0 - fx
                val gy = 1.0 - fy
                val w00 = gx * gy
                val w01 = fx * gy
                val w10 = gx * fy
                val w11 = fx * fy
                for (ch in 0 until 3) {
                    val v = w00 * (data[r0 + c0 + ch].toInt() and 0xFF) + w01 * (data[r0 + c1 + ch].toInt() and 0xFF) +
                        w10 * (data[r1 + c0 + ch].toInt() and 0xFF) + w11 * (data[r1 + c1 + ch].toInt() and 0xFF)
                    if (dstBytes != null) dstBytes[o + ch] = roundToByte(v).toByte() else dstFloat!![o + ch] = v
                }
                o += 3
            }
        }
    }

    // NOTE: `x = u − 0.5` is computed as `(cx + a·cosθ − b·sinθ) − 0.5`, i.e. u first, then the half-pixel shift, exactly as
    // written in §1.4 (u, then x = u − 0.5).

    /** A floored coordinate as an Int that cannot overflow when 1 is added (values far outside are clamped anyway). */
    private fun floorToIndex(xf: Double, size: Int): Int = when {
        xf < -1.0 -> -1
        xf > size.toDouble() -> size
        else -> xf.toInt()
    }

    private fun clampIndex(i: Int, size: Int): Int = if (i < 0) 0 else if (i >= size) size - 1 else i

    // ---------------------------------------------------------------------------------------------------------------
    // rotations (AUGMENT4 keyframes, §6.5)

    /**
     * Rotates an image by `k·90°` exactly like `numpy.rot90(img, k)` on an `[H, W, 3]` array: positive [k] turns
     * counter-clockwise as displayed (row 0 at the top). `k = 1`: `out[i][j] = in[j][W−1−i]` (output is `H×W` wide×high
     * swapped); `k = 2`: `out[i][j] = in[H−1−i][W−1−j]`; `k = 3`: `out[i][j] = in[H−1−j][i]`. Any integer [k] (mod 4).
     */
    fun rotate90(img: RgbImage, k: Int): RgbImage {
        val w = img.width
        val h = img.height
        val src = img.data
        return when (((k % 4) + 4) % 4) {
            0 -> img.copy()
            1 -> RgbImage(h, w).also { out ->
                val d = out.data
                var o = 0
                for (i in 0 until w) for (j in 0 until h) { copyPixel(src, (j * w + (w - 1 - i)) * 3, d, o); o += 3 }
            }
            2 -> RgbImage(w, h).also { out ->
                val d = out.data
                var o = 0
                for (i in 0 until h) for (j in 0 until w) { copyPixel(src, ((h - 1 - i) * w + (w - 1 - j)) * 3, d, o); o += 3 }
            }
            else -> RgbImage(h, w).also { out ->
                val d = out.data
                var o = 0
                for (i in 0 until w) for (j in 0 until h) { copyPixel(src, ((h - 1 - j) * w + i) * 3, d, o); o += 3 }
            }
        }
    }

    private fun copyPixel(src: ByteArray, s: Int, dst: ByteArray, d: Int) {
        dst[d] = src[s]
        dst[d + 1] = src[s + 1]
        dst[d + 2] = src[s + 2]
    }
}
