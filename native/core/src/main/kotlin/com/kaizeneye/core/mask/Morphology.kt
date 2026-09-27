package com.kaizeneye.core.mask

/**
 * Binary morphology of spec §2.3 on 0/1 byte masks (`[h, w]` row-major): 3×3 square structuring element, neighbours
 * outside the image are **ignored** (erosion: a pixel stays 1 iff all its in-image 3×3 neighbours are 1; dilation: it
 * becomes 1 iff any in-image neighbour is 1 — OpenCV's default border handling).
 *
 * The 3×3 square is separable and the in-image neighbourhood is a product of the in-image row and column ranges, so each
 * operation is an exact horizontal pass into `tmp` followed by a vertical pass. Zero allocation; `src`, `dst` and the
 * temporaries must be distinct arrays of at least `w*h` bytes.
 */
object Morphology {

    fun erode(src: ByteArray, dst: ByteArray, w: Int, h: Int, tmp: ByteArray) {
        horizontal(src, tmp, w, h, erode = true)
        vertical(tmp, dst, w, h, erode = true)
    }

    fun dilate(src: ByteArray, dst: ByteArray, w: Int, h: Int, tmp: ByteArray) {
        horizontal(src, tmp, w, h, erode = false)
        vertical(tmp, dst, w, h, erode = false)
    }

    /** Opening = erode then dilate (removes specks). Uses [tmp] and [tmp2] as scratch. */
    fun open(src: ByteArray, dst: ByteArray, w: Int, h: Int, tmp: ByteArray, tmp2: ByteArray) {
        erode(src, tmp2, w, h, tmp)
        dilate(tmp2, dst, w, h, tmp)
    }

    /** Closing = dilate then erode (fills pinholes). Uses [tmp] and [tmp2] as scratch. */
    fun close(src: ByteArray, dst: ByteArray, w: Int, h: Int, tmp: ByteArray, tmp2: ByteArray) {
        dilate(src, tmp2, w, h, tmp)
        erode(tmp2, dst, w, h, tmp)
    }

    private fun horizontal(src: ByteArray, dst: ByteArray, w: Int, h: Int, erode: Boolean) {
        for (y in 0 until h) {
            val row = y * w
            if (w == 1) {
                dst[row] = src[row]
                continue
            }
            val last = row + w - 1
            if (erode) {
                dst[row] = (src[row].toInt() and src[row + 1].toInt()).toByte()
                for (i in row + 1 until last) {
                    dst[i] = (src[i - 1].toInt() and src[i].toInt() and src[i + 1].toInt()).toByte()
                }
                dst[last] = (src[last - 1].toInt() and src[last].toInt()).toByte()
            } else {
                dst[row] = (src[row].toInt() or src[row + 1].toInt()).toByte()
                for (i in row + 1 until last) {
                    dst[i] = (src[i - 1].toInt() or src[i].toInt() or src[i + 1].toInt()).toByte()
                }
                dst[last] = (src[last - 1].toInt() or src[last].toInt()).toByte()
            }
        }
    }

    private fun vertical(src: ByteArray, dst: ByteArray, w: Int, h: Int, erode: Boolean) {
        val n = w * h
        if (h == 1) {
            System.arraycopy(src, 0, dst, 0, n)
            return
        }
        val lastRow = n - w
        if (erode) {
            for (i in 0 until w) dst[i] = (src[i].toInt() and src[i + w].toInt()).toByte()
            for (i in w until lastRow) dst[i] = (src[i - w].toInt() and src[i].toInt() and src[i + w].toInt()).toByte()
            for (i in lastRow until n) dst[i] = (src[i - w].toInt() and src[i].toInt()).toByte()
        } else {
            for (i in 0 until w) dst[i] = (src[i].toInt() or src[i + w].toInt()).toByte()
            for (i in w until lastRow) dst[i] = (src[i - w].toInt() or src[i].toInt() or src[i + w].toInt()).toByte()
            for (i in lastRow until n) dst[i] = (src[i - w].toInt() or src[i].toInt()).toByte()
        }
    }
}
