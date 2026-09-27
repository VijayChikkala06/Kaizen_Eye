package com.kaizeneye.v2.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.io.ByteArrayOutputStream
import java.io.File

/** Conversions between packed RGB bytes (what the pipeline and the backbone use) and Android bitmaps / JPEG. */
object Images {

    fun rgbToBitmap(rgb: ByteArray, w: Int, h: Int): Bitmap {
        val px = IntArray(w * h)
        var j = 0
        for (i in 0 until w * h) {
            val r = rgb[j].toInt() and 0xFF
            val g = rgb[j + 1].toInt() and 0xFF
            val b = rgb[j + 2].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            j += 3
        }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    fun bitmapToRgb(bmp: Bitmap): ByteArray {
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        val out = ByteArray(w * h * 3)
        var j = 0
        for (p in px) {
            out[j] = (p shr 16 and 0xFF).toByte()
            out[j + 1] = (p shr 8 and 0xFF).toByte()
            out[j + 2] = (p and 0xFF).toByte()
            j += 3
        }
        return out
    }

    fun jpeg(rgb: ByteArray, w: Int, h: Int, quality: Int = 95): ByteArray {
        val bmp = rgbToBitmap(rgb, w, h)
        val bos = ByteArrayOutputStream(w * h / 2)
        bmp.compress(Bitmap.CompressFormat.JPEG, quality, bos)
        bmp.recycle()
        return bos.toByteArray()
    }

    /** Decodes a JPEG into packed RGB (exact size check against [w]×[h] when given). */
    fun decodeJpegRgb(bytes: ByteArray, w: Int? = null, h: Int? = null): ByteArray {
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 })
            ?: error("not a decodable image")
        if (w != null && h != null) require(bmp.width == w && bmp.height == h) { "image is ${bmp.width}x${bmp.height}, expected ${w}x$h" }
        return bitmapToRgb(bmp).also { bmp.recycle() }
    }

    fun decodeFileRgb(f: File): Pair<ByteArray, Pair<Int, Int>> {
        val bmp = BitmapFactory.decodeFile(f.absolutePath, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 })
            ?: error("cannot decode ${f.name}")
        return bitmapToRgb(bmp) to (bmp.width to bmp.height).also { bmp.recycle() }
    }

    /**
     * The crop the VLM sees: the part crop with the anomaly cell boxed (plan: "boxed crop + facts line"), written as a JPEG
     * into the cache dir shared with the :vlm process.
     */
    fun writeBoxedCrop(rgb: ByteArray, n: Int, peakRow: Int, peakCol: Int, gh: Int, gw: Int, out: File, boxCells: Int = 3): File {
        val bmp = rgbToBitmap(rgb, n, n).copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(bmp)
        val cw = n.toFloat() / gw
        val ch = n.toFloat() / gh
        val half = boxCells / 2f
        val p = Paint().apply { style = Paint.Style.STROKE; strokeWidth = n / 100f + 2f; color = Color.RED; isAntiAlias = true }
        c.drawRect((peakCol + 0.5f - half) * cw, (peakRow + 0.5f - half) * ch, (peakCol + 0.5f + half) * cw, (peakRow + 0.5f + half) * ch, p)
        out.parentFile?.mkdirs()
        out.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        bmp.recycle()
        return out
    }

    /** Small RGB → Bitmap preview for replay mode (the analysis image, e.g. 320×180). */
    fun previewBitmap(rgb: ByteArray, w: Int, h: Int): Bitmap = rgbToBitmap(rgb, w, h)
}
