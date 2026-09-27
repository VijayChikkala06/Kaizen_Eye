package com.kaizeneye.v2.export

import com.kaizeneye.core.image.ImageOps
import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.CropSquare
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import org.json.JSONArray
import org.json.JSONObject

/** The chosen component's pixels over its bbox as alternating 0/1 run lengths starting with a 0-run (export-format §3). */
class MaskRle(val x: Int, val y: Int, val w: Int, val h: Int, val rle: IntArray) {
    fun toJson(): JSONObject = JSONObject().put("x", x).put("y", y).put("w", w).put("h", h).put("rle", JSONArray(rle.toList()))

    companion object {
        fun of(labels: IntArray, imgW: Int, c: Component): MaskRle {
            val runs = ArrayList<Int>()
            var cur = 0 // 0-run first
            var len = 0
            for (yy in c.minY..c.maxY) {
                val row = yy * imgW
                for (xx in c.minX..c.maxX) {
                    val v = if (labels[row + xx] == c.label) 1 else 0
                    if (v == cur) len++ else {
                        runs += len
                        cur = v
                        len = 1
                    }
                }
            }
            runs += len
            return MaskRle(c.minX, c.minY, c.width, c.height, runs.toIntArray())
        }
    }
}

/**
 * One exported crop (docs/verification/export-format.md "CROPINFO" + its JPEG). [jpeg] is the crop at the export
 * `cropSize` (recommended 448, so both backbones can be evaluated without up-sampling).
 */
class ExportFrame(
    val tMs: Long,
    val sanity: SanityReason,
    val sharpness: Double?,
    val crop: CropSquare?,
    val mask: MaskRle?,
    val cov: FloatArray?,
    val geometry: GeometryFeatures?,
    val theta: Double?,
    val jpeg: ByteArray?,
) {
    fun cropInfo(): JSONObject {
        val o = JSONObject().put("tMs", tMs).put("sanity", sanity.name)
        o.put("sharpness", sharpness?.takeUnless { it.isNaN() } ?: JSONObject.NULL)
        o.put("crop", crop?.let { JSONObject().put("x0", it.x0).put("y0", it.y0).put("side", it.side) } ?: JSONObject.NULL)
        o.put("mask", mask?.toJson() ?: JSONObject.NULL)
        o.put("cov", cov?.let { c -> JSONArray().also { a -> c.forEach { a.put(it.toDouble()) } } } ?: JSONObject.NULL)
        o.put(
            "geometry",
            geometry?.let { JSONObject().put("area", it.area.toInt()).put("fill", it.fill).put("aspect", it.aspect).put("hu1", it.hu1).put("solidity", it.solidity) }
                ?: JSONObject.NULL,
        )
        o.put("theta", theta ?: JSONObject.NULL)
        return o
    }

    companion object {
        /** §1.3 sharpness on the grey N×N backbone crop, full rectangle. */
        fun cropSharpness(crop: ByteArray, n: Int): Double {
            val grey = ImageOps.grey(com.kaizeneye.core.image.RgbImage(n, n, crop))
            return ImageOps.laplacianVariance(grey, n, n, 0, 0, n, n)
        }
    }
}
