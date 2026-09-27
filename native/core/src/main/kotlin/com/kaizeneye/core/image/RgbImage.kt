package com.kaizeneye.core.image

/**
 * A packed 8-bit RGB image `[height, width, 3]`, row-major (spec §0): channel `c` of pixel `(x, y)` is
 * `data[(y * width + x) * 3 + c]`. Used for the analysis image (§1.1) and for backbone crops (§1.4, `N×N×3` crop bytes).
 */
class RgbImage(val width: Int, val height: Int, val data: ByteArray = ByteArray(width * height * 3)) {
    init {
        require(width > 0 && height > 0) { "image must be non-empty, got ${width}x$height" }
        require(data.size == width * height * 3) { "data has ${data.size} bytes, expected ${width * height * 3}" }
    }

    /** Channel [c] of pixel ([x], [y]) as 0..255. For tests and tools, not hot loops. */
    operator fun get(x: Int, y: Int, c: Int): Int = data[(y * width + x) * 3 + c].toInt() and 0xFF

    /** Sets channel [c] of pixel ([x], [y]) to [v] (0..255). */
    operator fun set(x: Int, y: Int, c: Int, v: Int) {
        data[(y * width + x) * 3 + c] = v.toByte()
    }

    /** Sets all three channels of pixel ([x], [y]). */
    fun setRgb(x: Int, y: Int, r: Int, g: Int, b: Int) {
        val p = (y * width + x) * 3
        data[p] = r.toByte()
        data[p + 1] = g.toByte()
        data[p + 2] = b.toByte()
    }

    /** Fills the whole image with one colour. */
    fun fill(r: Int, g: Int, b: Int) {
        var p = 0
        while (p < data.size) {
            data[p] = r.toByte()
            data[p + 1] = g.toByte()
            data[p + 2] = b.toByte()
            p += 3
        }
    }

    fun copy(): RgbImage = RgbImage(width, height, data.copyOf())
}
