package com.kaizeneye.core.image

/**
 * A copy of one CameraX `RGBA_8888` plane (spec §1.1): [height] rows of [rowStride] bytes, pixel stride 4 (R, G, B, A).
 * Only the first `width * 4` bytes of a row are pixels; the rest of the row is padding and is never read.
 *
 * The analyzer thread keeps a small pool of these and copies each camera plane into one, so the image proxy can be closed
 * immediately; [data] is therefore reused frame after frame — hold on to the frame only while it is in the pool.
 */
class RgbaFrame(val width: Int, val height: Int, val rowStride: Int, val data: ByteArray) {
    init {
        require(width > 0 && height > 0) { "frame must be non-empty, got ${width}x$height" }
        require(rowStride >= width * PIXEL_STRIDE) { "rowStride $rowStride < width*4 = ${width * PIXEL_STRIDE}" }
        require(data.size >= rowStride * (height - 1) + width * PIXEL_STRIDE) {
            "data has ${data.size} bytes, need at least ${rowStride * (height - 1) + width * PIXEL_STRIDE}"
        }
    }

    /** Channel [c] (0 = R, 1 = G, 2 = B, 3 = A) of pixel ([x], [y]) as 0..255. For tests and tools, not hot loops. */
    operator fun get(x: Int, y: Int, c: Int): Int = data[y * rowStride + x * PIXEL_STRIDE + c].toInt() and 0xFF

    /** Sets channel [c] of pixel ([x], [y]) to [v] (0..255). For tests and tools. */
    operator fun set(x: Int, y: Int, c: Int, v: Int) {
        data[y * rowStride + x * PIXEL_STRIDE + c] = v.toByte()
    }

    /**
     * Bulk-copies a camera plane (e.g. CameraX `ImageProxy.planes[0].buffer`, whose row stride must equal [rowStride])
     * into [data], from the buffer's start; the buffer's own position is not changed. One memcpy, no allocation besides
     * the buffer view. Returns the number of bytes copied.
     */
    fun copyFrom(plane: java.nio.ByteBuffer): Int {
        val src = plane.duplicate()
        (src as java.nio.Buffer).rewind() // Buffer, not the JDK 9+ covariant ByteBuffer override older ART lacks
        val n = minOf(src.remaining(), data.size)
        src.get(data, 0, n)
        return n
    }

    companion object {
        const val PIXEL_STRIDE = 4

        /** A zero-filled frame with the given (possibly padded) row stride. */
        fun allocate(width: Int, height: Int, rowStride: Int = width * PIXEL_STRIDE): RgbaFrame =
            RgbaFrame(width, height, rowStride, ByteArray(rowStride * (height - 1) + width * PIXEL_STRIDE))
    }
}
