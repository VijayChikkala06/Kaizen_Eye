package com.kaizeneye.core.mask

import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.model.Component

/**
 * The analysis-resolution foreground segmentation of spec §2.2–§2.4 for one fixed frame size: colour threshold against
 * the [SheetModel] → open → close → 8-connected components (labels in first-pixel row-major order, components smaller
 * than `minBlobPx` removed, the rest renumbered `1, 2, …`).
 *
 * Every buffer is allocated once in the constructor; [segment] allocates nothing but its small result list (one
 * [Component] per kept blob). Not thread-safe: one instance per analyzer thread. The intermediate masks ([fgRaw],
 * [afterOpen], [afterClose]; 0/1 bytes) and the final [labels] stay valid until the next call.
 */
class Segmenter(val width: Int, val height: Int, val params: MaskParams = MaskParams()) {

    private val n = width * height

    /** §2.2 raw foreground (`d² > kSigma²`), 0/1. */
    val fgRaw = ByteArray(n)

    /** After the §2.3 opening, 0/1. */
    val afterOpen = ByteArray(n)

    /** After the §2.3 closing (the final mask), 0/1. */
    val afterClose = ByteArray(n)

    /** §2.4 final label map (0 = background, kept components `1..count`). */
    val labels = IntArray(n)

    /** Number of kept components of the last [segment] / [label] call. */
    var componentCount = 0
        private set

    private val tmp = ByteArray(n)
    private val tmp2 = ByteArray(n)
    private val lut = DoubleArray(3 * 256)

    // In one row a new provisional label needs a background W neighbour, so there are at most ceil(w/2) per row.
    private val maxLabels = ((width + 1) / 2) * height + 1
    private val parent = IntArray(maxLabels + 1)
    private val compact = IntArray(maxLabels + 1)
    private val finalOf = IntArray(maxLabels + 1)
    private val area = IntArray(maxLabels + 1)
    private val minX = IntArray(maxLabels + 1)
    private val minY = IntArray(maxLabels + 1)
    private val maxX = IntArray(maxLabels + 1)
    private val maxY = IntArray(maxLabels + 1)
    private val sumX = LongArray(maxLabels + 1)
    private val sumY = LongArray(maxLabels + 1)

    init {
        require(width > 0 && height > 0) { "segmenter needs a non-empty frame, got ${width}x$height" }
    }

    /** Full §2.2–§2.4 pipeline on an analysis image. */
    fun segment(img: RgbImage, sheet: SheetModel): List<Component> {
        foreground(img, sheet)
        Morphology.open(fgRaw, afterOpen, width, height, tmp, tmp2)
        Morphology.close(afterOpen, afterClose, width, height, tmp, tmp2)
        return label(afterClose)
    }

    /**
     * Spec §2.2 into [fgRaw]: `d²(x, y) = Σ_c ((I_c − μ_c)/σ_c)²` (float64, channels in order), foreground iff
     * `d² > kSigma²`. The per-channel terms come from a 3×256 table built with exactly that expression, so the sum is
     * bit-identical to the direct formula.
     */
    fun foreground(img: RgbImage, sheet: SheetModel): ByteArray {
        require(img.width == width && img.height == height) {
            "image is ${img.width}x${img.height}, segmenter is ${width}x$height"
        }
        buildLut(sheet)
        val k2 = params.kSigma * params.kSigma
        val d = img.data
        var p = 0
        for (i in 0 until n) {
            val d2 = lut[d[p].toInt() and 0xFF] + lut[256 + (d[p + 1].toInt() and 0xFF)] + lut[512 + (d[p + 2].toInt() and 0xFF)]
            fgRaw[i] = if (d2 > k2) 1 else 0
            p += 3
        }
        return fgRaw
    }

    /** The §2.2 `d²` of every pixel into [dst] (`w·h` values; diagnostics and golden tests). */
    fun distanceSquared(img: RgbImage, sheet: SheetModel, dst: DoubleArray) {
        require(img.width == width && img.height == height) {
            "image is ${img.width}x${img.height}, segmenter is ${width}x$height"
        }
        require(dst.size >= n) { "dst has ${dst.size} values, need $n" }
        buildLut(sheet)
        val d = img.data
        var p = 0
        for (i in 0 until n) {
            dst[i] = lut[d[p].toInt() and 0xFF] + lut[256 + (d[p + 1].toInt() and 0xFF)] + lut[512 + (d[p + 2].toInt() and 0xFF)]
            p += 3
        }
    }

    private fun buildLut(sheet: SheetModel) {
        for (c in 0 until 3) {
            val mu = sheet.mean[c]
            val sd = sheet.sigma[c]
            val base = c * 256
            for (v in 0 until 256) {
                val z = (v - mu) / sd
                lut[base + v] = z * z
            }
        }
    }

    /** Number of 8-connected components of the last [label] call before the `minBlobPx` removal. */
    var rawComponentCount = 0
        private set

    /** Areas of the last [label] call's components before the `minBlobPx` removal, in first-pixel order (a copy). */
    fun rawAreas(): IntArray = IntArray(rawComponentCount) { area[it + 1] }

    /**
     * Spec §2.4 on a final 0/1 [mask]: 8-connected components into [labels], labels assigned in the order of each
     * component's first pixel in a row-major scan, components with `area < minBlobPx` removed and the rest renumbered in
     * the same order. Returns the kept components by label.
     *
     * Two-pass union–find with the smaller root always kept: the first pixel of a component (row-major) always opens a
     * new provisional label (its W/NW/N/NE neighbours come earlier, so they cannot belong to it), every later label of
     * the component is larger, hence each root is its component's first-pixel label and ranking roots by value gives
     * exactly the first-pixel order.
     */
    fun label(mask: ByteArray): List<Component> {
        require(mask.size >= n) { "mask has ${mask.size} values, need $n" }
        val w = width
        val h = height
        val lab = labels
        var next = 1
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val i = row + x
                if (mask[i].toInt() == 0) {
                    lab[i] = 0
                    continue
                }
                var l = 0
                if (x > 0) l = lab[i - 1]
                if (y > 0) {
                    val up = i - w
                    if (x > 0) l = merge(l, lab[up - 1])
                    l = merge(l, lab[up])
                    if (x < w - 1) l = merge(l, lab[up + 1])
                }
                if (l == 0) {
                    l = next
                    parent[next] = next
                    next++
                }
                lab[i] = l
            }
        }

        // Provisional → compact labels 1..count, in root (= first-pixel) order.
        var count = 0
        for (p in 1 until next) {
            val r = find(p)
            if (r == p) {
                count++
                compact[p] = count
                area[count] = 0
                minX[count] = Int.MAX_VALUE
                minY[count] = Int.MAX_VALUE
                maxX[count] = -1
                maxY[count] = -1
                sumX[count] = 0L
                sumY[count] = 0L
            } else {
                compact[p] = compact[r] // r < p: already assigned
            }
        }

        // Stats per compact label.
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val i = row + x
                val p = lab[i]
                if (p == 0) continue
                val c = compact[p]
                lab[i] = c
                area[c]++
                if (x < minX[c]) minX[c] = x
                if (x > maxX[c]) maxX[c] = x
                if (y < minY[c]) minY[c] = y
                maxY[c] = y // rows are scanned in order
                sumX[c] += x.toLong()
                sumY[c] += y.toLong()
            }
        }

        rawComponentCount = count

        // Remove small blobs and renumber in the same order.
        var kept = 0
        for (c in 1..count) finalOf[c] = if (area[c] >= params.minBlobPx) ++kept else 0
        if (kept != count) {
            for (i in 0 until n) {
                val c = lab[i]
                if (c != 0) lab[i] = finalOf[c]
            }
        }
        componentCount = kept

        val m = params.borderMargin
        val out = ArrayList<Component>(kept)
        for (c in 1..count) {
            val id = finalOf[c]
            if (id == 0) continue
            val a = area[c]
            out.add(
                Component(
                    label = id,
                    area = a,
                    minX = minX[c],
                    minY = minY[c],
                    maxX = maxX[c],
                    maxY = maxY[c],
                    cx = sumX[c].toDouble() / a + 0.5,
                    cy = sumY[c].toDouble() / a + 0.5,
                    touchesBorder = minX[c] < m || minY[c] < m || maxX[c] >= w - m || maxY[c] >= h - m,
                ),
            )
        }
        return out
    }

    /** Current label [l] (0 = none yet) joined with neighbour label [a] (0 = background); returns a label of the union. */
    private fun merge(l: Int, a: Int): Int {
        if (a == 0) return l
        if (l == 0 || l == a) return a
        val ra = find(a)
        val rl = find(l)
        return when {
            ra < rl -> { parent[rl] = ra; ra }
            rl < ra -> { parent[ra] = rl; rl }
            else -> ra
        }
    }

    private fun find(x: Int): Int {
        var v = x
        while (parent[v] != v) {
            parent[v] = parent[parent[v]] // path halving
            v = parent[v]
        }
        return v
    }
}
