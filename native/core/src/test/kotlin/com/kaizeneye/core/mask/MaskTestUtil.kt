package com.kaizeneye.core.mask

import com.kaizeneye.core.image.RgbImage
import java.util.ArrayDeque

/** Hand-made grids and slow, obviously-correct reference implementations for the mask tests. */
object MaskTestUtil {

    /** Rows of '#' (1) and '.' (0) → a 0/1 mask; returns (mask, w, h). */
    fun grid(vararg rows: String): Triple<ByteArray, Int, Int> {
        val h = rows.size
        val w = rows[0].length
        require(rows.all { it.length == w })
        val m = ByteArray(w * h)
        for (y in 0 until h) for (x in 0 until w) m[y * w + x] = if (rows[y][x] == '#') 1 else 0
        return Triple(m, w, h)
    }

    /** Direct (non-separable) 3×3 erosion/dilation with out-of-image neighbours ignored (§2.3). */
    fun naiveMorph(src: ByteArray, w: Int, h: Int, erode: Boolean): ByteArray {
        val out = ByteArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var all = true
            var any = false
            for (dy in -1..1) for (dx in -1..1) {
                val xx = x + dx
                val yy = y + dy
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
                if (src[yy * w + xx].toInt() == 1) any = true else all = false
            }
            out[y * w + x] = if (if (erode) all else any) 1 else 0
        }
        return out
    }

    /** BFS 8-connected labelling in first-pixel row-major order with speck removal and renumbering (§2.4). */
    fun naiveLabels(mask: ByteArray, w: Int, h: Int, minBlob: Int): Pair<IntArray, List<Int>> {
        val lab = IntArray(w * h)
        var next = 0
        val areas = ArrayList<Int>()
        val q = ArrayDeque<Int>()
        for (i in 0 until w * h) {
            if (mask[i].toInt() == 0 || lab[i] != 0) continue
            next++
            var area = 0
            lab[i] = next
            q.add(i)
            while (q.isNotEmpty()) {
                val p = q.poll()
                area++
                val px = p % w
                val py = p / w
                for (dy in -1..1) for (dx in -1..1) {
                    val xx = px + dx
                    val yy = py + dy
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
                    val j = yy * w + xx
                    if (mask[j].toInt() == 1 && lab[j] == 0) {
                        lab[j] = next
                        q.add(j)
                    }
                }
            }
            areas += area
        }
        val remap = IntArray(next + 1)
        var k = 0
        for (l in 1..next) remap[l] = if (areas[l - 1] >= minBlob) ++k else 0
        for (i in lab.indices) lab[i] = remap[lab[i]]
        return lab to areas
    }

    /** A w×h image filled with the sheet colour. */
    fun sheet(w: Int, h: Int, r: Int = 200, g: Int = 190, b: Int = 180): RgbImage = RgbImage(w, h).also { it.fill(r, g, b) }

    fun fillRect(img: RgbImage, x0: Int, y0: Int, x1: Int, y1: Int, r: Int, g: Int, b: Int) {
        for (y in y0 until y1) for (x in x0 until x1) img.setRgb(x, y, r, g, b)
    }

    fun fillDisc(img: RgbImage, cx: Double, cy: Double, radius: Double, r: Int, g: Int, b: Int) {
        for (y in 0 until img.height) for (x in 0 until img.width) {
            val dx = x + 0.5 - cx
            val dy = y + 0.5 - cy
            if (dx * dx + dy * dy <= radius * radius) img.setRgb(x, y, r, g, b)
        }
    }
}
