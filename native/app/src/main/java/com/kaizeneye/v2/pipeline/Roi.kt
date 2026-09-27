package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.model.Component
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * "Circle to select": an outline the operator drew with a finger over the preview, in ANALYSIS pixels (the downscaled
 * frame the segmenter works on). A closed polygon; a tap becomes a small circle around the tapped point.
 */
class Roi(val xs: DoubleArray, val ys: DoubleArray, val isTap: Boolean = false) {
    init {
        require(xs.size == ys.size && xs.size >= 3) { "an ROI needs at least 3 points" }
    }

    val cx: Double = xs.average()
    val cy: Double = ys.average()
    val minX: Double = xs.min()
    val maxX: Double = xs.max()
    val minY: Double = ys.min()
    val maxY: Double = ys.max()

    /** Point-in-polygon (even-odd) after growing the polygon by [grow] about its centre. */
    fun contains(x: Double, y: Double, grow: Double = 1.0): Boolean {
        val px = cx + (x - cx) / grow
        val py = cy + (y - cy) / grow
        var inside = false
        var j = xs.size - 1
        for (i in xs.indices) {
            val yi = ys[i]
            val yj = ys[j]
            if ((yi > py) != (yj > py)) {
                val xAt = xs[i] + (py - yi) * (xs[j] - xs[i]) / (yj - yi)
                if (px < xAt) inside = !inside
            }
            j = i
        }
        return inside
    }

    fun translated(dx: Double, dy: Double): Roi =
        Roi(DoubleArray(xs.size) { xs[it] + dx }, DoubleArray(ys.size) { ys[it] + dy }, isTap)

    companion object {
        /**
         * The operator's stroke (analysis pixels) as an ROI of a `w×h` frame. A stroke whose extent is under 4 % of the
         * frame diagonal is a tap: a circle of 6 % of the shorter side around its centre (the part under the finger is
         * found through the label map, see [RoiSelect]). Null when there are no points.
         */
        fun fromStroke(px: DoubleArray, py: DoubleArray, w: Int, h: Int): Roi? {
            if (px.isEmpty()) return null
            val ext = hypot(px.max() - px.min(), py.max() - py.min())
            if (px.size < 3 || ext < 0.04 * hypot(w.toDouble(), h.toDouble())) {
                val r = 0.06 * min(w, h)
                val x0 = px.average()
                val y0 = py.average()
                val k = 16
                return Roi(
                    DoubleArray(k) { x0 + r * kotlin.math.cos(2 * Math.PI * it / k) },
                    DoubleArray(k) { y0 + r * kotlin.math.sin(2 * Math.PI * it / k) },
                    isTap = true,
                )
            }
            return Roi(px, py)
        }

        /** An axis-aligned rectangle ROI around a component's bbox grown by [grow] (used to follow a tapped part). */
        fun around(c: Component, grow: Double): Roi {
            val hw = c.width * grow / 2
            val hh = c.height * grow / 2
            val x0 = (c.minX + c.maxX + 1) / 2.0
            val y0 = (c.minY + c.maxY + 1) / 2.0
            return Roi(doubleArrayOf(x0 - hw, x0 + hw, x0 + hw, x0 - hw), doubleArrayOf(y0 - hh, y0 - hh, y0 + hh, y0 + hh))
        }
    }
}

/** What the circle found in one frame: the frame with the ROI applied, and the selected object (null = nothing inside). */
class RoiPick(val analysed: Analysed, val obj: Component?)

/**
 * Applies an [Roi] to an analysed frame: every foreground component whose centroid lies inside the (slightly grown)
 * outline — or that sits under the centre of a tap — belongs to the circled object. Pieces of ONE part that the empty-sheet
 * mask split apart (glare, holes, colours close to the sheet) are merged back into a single component: their pixels are
 * relabelled to the biggest piece in the label map, so crop, patch coverage and shape features see the whole part.
 * Specks far from the biggest piece are dropped. Components outside the outline stay in the list only for the
 * "another object next to the part" (MULTIPLE) check; they are never selected.
 */
object RoiSelect {
    /** Outline growth for the "inside" test (hand-drawn circles are often a little tight). */
    const val GROW = 1.15
    /** Pieces smaller than this fraction of the biggest piece are dust, not part of the part. */
    private const val MIN_PIECE_FRAC = 0.03
    /** Pieces are merged only if the gap to the object is at most this fraction of the object's longer side. */
    private const val PIECE_GAP = 0.35
    /** Whole view: a candidate larger than this fraction of the biggest one is another part, never a piece of it. */
    private const val WHOLE_VIEW_MAX_PIECE_FRAC = 0.5

    /**
     * Whole-view counterpart of the circle merge: pieces of ONE part that the mask split (glare, holes) are joined onto the
     * biggest candidate with the same gap rule, so a part taught with a circle (merged) judges the same without one.
     * Candidates far from the biggest one stay separate parts. Returns [candidates] unchanged when nothing merges.
     */
    fun mergeSplitParts(a: Analysed, candidates: List<Component>): Pair<Analysed, List<Component>> {
        if (candidates.size < 2) return a to candidates
        val main = candidates.maxWith(compareBy<Component> { it.area }.thenByDescending { it.label })
        // Only clearly smaller pieces are joined: a second part of similar size stays a part of its own.
        val (pieces, rest) = gather(main, candidates.filter { it !== main && it.area <= WHOLE_VIEW_MAX_PIECE_FRAC * main.area })
        if (pieces.size == 1) return a to candidates
        val merged = merge(a, main, pieces)
        val gone = pieces.map { it.label }.toSet()
        val frame = a.withComponents(listOf(merged) + a.components.filter { it.label !in gone })
        return frame to listOf(merged) + rest + candidates.filter { it !== main && it.area > WHOLE_VIEW_MAX_PIECE_FRAC * main.area }
    }

    /** Grows [main] by the pieces (from [others]) within reach; returns the pieces (main first) and the leftovers. */
    private fun gather(main: Component, others: List<Component>): Pair<List<Component>, List<Component>> {
        val pieces = arrayListOf(main)
        val rest = others.filter { it.area >= MIN_PIECE_FRAC * main.area }.toMutableList()
        val far = others.filter { it.area < MIN_PIECE_FRAC * main.area }.toMutableList()
        var ox0 = main.minX; var oy0 = main.minY; var ox1 = main.maxX; var oy1 = main.maxY
        var grew = true
        while (grew) {
            grew = false
            val reach = PIECE_GAP * max(ox1 - ox0 + 1, oy1 - oy0 + 1)
            val iter = rest.iterator()
            while (iter.hasNext()) {
                val c = iter.next()
                val gapX = max(0, max(c.minX - ox1, ox0 - c.maxX))
                val gapY = max(0, max(c.minY - oy1, oy0 - c.maxY))
                if (gapX <= reach && gapY <= reach) {
                    pieces += c
                    iter.remove()
                    ox0 = min(ox0, c.minX); oy0 = min(oy0, c.minY); ox1 = max(ox1, c.maxX); oy1 = max(oy1, c.maxY)
                    grew = true
                }
            }
        }
        return pieces to (rest + far)
    }

    fun apply(a: Analysed, roi: Roi, params: MaskParams): RoiPick {
        val tapLabel = if (roi.isTap) labelAt(a, roi.cx, roi.cy) else 0
        val inside = ArrayList<Component>()
        val outside = ArrayList<Component>()
        for (c in a.components) {
            val sel = roi.contains(c.cx, c.cy, GROW) || (tapLabel != 0 && c.label == tapLabel)
            if (sel && c.area >= params.minBlobPx) inside += c else if (!sel) outside += c
        }
        if (inside.isEmpty()) return RoiPick(a.withComponents(outside), null)
        val main = inside.maxWith(compareBy<Component> { it.area }.thenByDescending { it.label })
        // Grow the object from the biggest piece: add every piece whose bbox gap to the object so far is small compared
        // with the object's size (repeat until nothing changes, so a part split in three is joined too).
        val pieces = arrayListOf(main)
        val rest = inside.filter { it !== main && it.area >= MIN_PIECE_FRAC * main.area }.toMutableList()
        var ox0 = main.minX; var oy0 = main.minY; var ox1 = main.maxX; var oy1 = main.maxY
        var grew = true
        while (grew) {
            grew = false
            val reach = PIECE_GAP * max(ox1 - ox0 + 1, oy1 - oy0 + 1)
            val iter = rest.iterator()
            while (iter.hasNext()) {
                val c = iter.next()
                val gapX = max(0, max(c.minX - ox1, ox0 - c.maxX))
                val gapY = max(0, max(c.minY - oy1, oy0 - c.maxY))
                if (gapX <= reach && gapY <= reach) {
                    pieces += c
                    iter.remove()
                    ox0 = min(ox0, c.minX); oy0 = min(oy0, c.minY); ox1 = max(ox1, c.maxX); oy1 = max(oy1, c.maxY)
                    grew = true
                }
            }
        }
        val obj = if (pieces.size == 1) main else merge(a, main, pieces)
        return RoiPick(a.withComponents(listOf(obj) + outside), obj)
    }

    private fun labelAt(a: Analysed, x: Double, y: Double): Int {
        val xi = x.toInt().coerceIn(0, a.w - 1)
        val yi = y.toInt().coerceIn(0, a.h - 1)
        return a.labels[yi * a.w + xi]
    }

    /** Relabels every piece's pixels to [main]'s label and returns the union component. */
    private fun merge(a: Analysed, main: Component, pieces: List<Component>): Component {
        var area = 0
        var sx = 0.0
        var sy = 0.0
        var x0 = Int.MAX_VALUE
        var y0 = Int.MAX_VALUE
        var x1 = Int.MIN_VALUE
        var y1 = Int.MIN_VALUE
        var border = false
        for (p in pieces) {
            if (p !== main) {
                for (y in p.minY..p.maxY) {
                    val row = y * a.w
                    for (x in p.minX..p.maxX) if (a.labels[row + x] == p.label) a.labels[row + x] = main.label
                }
            }
            area += p.area
            sx += p.cx * p.area
            sy += p.cy * p.area
            x0 = min(x0, p.minX); y0 = min(y0, p.minY); x1 = max(x1, p.maxX); y1 = max(y1, p.maxY)
            border = border || p.touchesBorder
        }
        return Component(main.label, area, x0, y0, x1, y1, sx / area, sy / area, border)
    }
}

/**
 * Teach: keeps the circled part selected while the operator turns and shifts it. After every frame in which the part was
 * found (fully in view), the outline moves with the part's centre; a tap is replaced by a box around the tapped part.
 * Used from the frame thread only.
 */
class RoiFollower(initial: Roi) {
    @Volatile var roi: Roi = initial
        private set
    private var lastCx = Double.NaN
    private var lastCy = Double.NaN

    fun pick(a: Analysed, params: MaskParams): RoiPick {
        val p = RoiSelect.apply(a, roi, params)
        val o = p.obj
        if (o != null && !o.touchesBorder) {
            if (roi.isTap) {
                roi = Roi.around(o, 1.5)
            } else if (!lastCx.isNaN()) {
                roi = roi.translated(o.cx - lastCx, o.cy - lastCy)
            }
            lastCx = o.cx
            lastCy = o.cy
        }
        return p
    }
}

/** Outcome of one circle gesture, for the doodle (green = part found and registered, red = nothing inside). */
data class CircleResult(val strokeId: Long, val ok: Boolean, val message: String)
