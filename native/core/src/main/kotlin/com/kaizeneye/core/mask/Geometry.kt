package com.kaizeneye.core.mask

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.GeometryFeatures
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Shape features of a component (spec §2.7), computed from the label map over the component's pixels with integer
 * coordinates `(x, y)`: `hu1`, covariance `aspect`, oriented-box `fill`, convex-hull `solidity` and the raw `area`.
 * Runs once per judged part / teach frame; allocates only row-sized scratch arrays.
 */
object Geometry {

    /** Spec §2.7 features of [component] (label [Component.label] in [labels], a `w×h` map). */
    fun features(labels: IntArray, w: Int, h: Int, component: Component): GeometryFeatures {
        val s = Shape.of(labels, w, h, component)
        return GeometryFeatures(
            area = s.area,
            fill = s.fill(),
            aspect = s.aspect(),
            hu1 = s.hu1(),
            solidity = s.area / s.hullArea(),
        )
    }

    /**
     * Spec §2.7 principal angle `θ = 0.5 · atan2(2·μ11, μ20 − μ02)` (radians, image axes: x right, y down). Used by the
     * CANONICAL rotation challenger (§6.5).
     */
    fun principalAngle(labels: IntArray, w: Int, h: Int, component: Component): Double =
        Shape.of(labels, w, h, component).theta()

    /**
     * Shoelace area of the convex hull of the pixel-corner points `(x, y), (x+1, y), (x, y+1), (x+1, y+1)` of all the
     * component's pixels (Andrew's monotone chain, collinear points dropped). Exact (integer arithmetic).
     */
    fun hullArea(labels: IntArray, w: Int, h: Int, component: Component): Double =
        Shape.of(labels, w, h, component).hullArea()

    /** Moments and per-row extents of one component; every feature is derived from these. */
    private class Shape(
        val area: Double,
        val mu20: Double,
        val mu02: Double,
        val mu11: Double,
        val rowY: IntArray,
        val rowMin: IntArray,
        val rowMax: IntArray,
    ) {
        fun hu1(): Double = (mu20 + mu02) / (area * area)

        /** `sqrt(λ2/λ1)` of the covariance `[[μ20, μ11], [μ11, μ02]] / A` (1.0 if λ1 == 0). */
        fun aspect(): Double {
            val a = mu20 / area
            val b = mu11 / area
            val c = mu02 / area
            val half = (a + c) / 2
            val hd = (a - c) / 2
            val d = sqrt(hd * hd + b * b)
            val l1 = half + d
            val l2 = maxOf(0.0, half - d)
            return if (l1 == 0.0) 1.0 else sqrt(l2 / l1)
        }

        fun theta(): Double = 0.5 * StrictMath.atan2(2 * mu11, mu20 - mu02)

        /**
         * `A / (wu · wv)` with `wu = max(u·p) − min(u·p) + 1` along `u = (cosθ, sinθ)` and `wv` along
         * `v = (−sinθ, cosθ)` over the pixel coordinates. The projections are monotone in x on a row, so each row's
         * extreme pixels give the exact same max/min as all its pixels.
         */
        fun fill(): Double {
            val t = theta()
            val ct = StrictMath.cos(t)
            val st = StrictMath.sin(t)
            var uMin = Double.POSITIVE_INFINITY
            var uMax = Double.NEGATIVE_INFINITY
            var vMin = Double.POSITIVE_INFINITY
            var vMax = Double.NEGATIVE_INFINITY
            for (r in rowMin.indices) {
                val y = rowY[r].toDouble()
                for (k in 0 until 2) {
                    val x = (if (k == 0) rowMin[r] else rowMax[r]).toDouble()
                    val u = x * ct + y * st
                    val v = -x * st + y * ct
                    if (u < uMin) uMin = u
                    if (u > uMax) uMax = u
                    if (v < vMin) vMin = v
                    if (v > vMax) vMax = v
                }
            }
            val wu = uMax - uMin + 1
            val wv = vMax - vMin + 1
            return area / (wu * wv)
        }

        /** Convex hull of the pixel corners; only each row's leftmost and rightmost pixel corners can be hull points. */
        fun hullArea(): Double {
            val rows = rowMin.size
            val keys = LongArray(rows * 4)
            var m = 0
            for (r in 0 until rows) {
                val y = rowY[r]
                val xl = rowMin[r]
                val xr = rowMax[r] + 1
                keys[m++] = key(xl, y)
                keys[m++] = key(xl, y + 1)
                keys[m++] = key(xr, y)
                keys[m++] = key(xr, y + 1)
            }
            keys.sort()
            val px = IntArray(m)
            val py = IntArray(m)
            var u = 0
            for (i in 0 until m) {
                if (i > 0 && keys[i] == keys[i - 1]) continue
                px[u] = (keys[i] ushr 32).toInt()
                py[u] = (keys[i] and 0xFFFFFFFFL).toInt()
                u++
            }
            // Andrew's monotone chain: lower hull left→right, upper hull right→left; cross <= 0 pops (drops collinear).
            val hx = IntArray(2 * u + 1)
            val hy = IntArray(2 * u + 1)
            var k = 0
            for (i in 0 until u) {
                while (k >= 2 && cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], px[i], py[i]) <= 0L) k--
                hx[k] = px[i]
                hy[k] = py[i]
                k++
            }
            val lowerSize = k + 1
            for (i in u - 2 downTo 0) {
                while (k >= lowerSize && cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], px[i], py[i]) <= 0L) k--
                hx[k] = px[i]
                hy[k] = py[i]
                k++
            }
            val count = k - 1 // the last point repeats the first
            var twice = 0L
            for (i in 0 until count) {
                val j = if (i + 1 == count) 0 else i + 1
                twice += hx[i].toLong() * hy[j] - hx[j].toLong() * hy[i]
            }
            return abs(twice) / 2.0
        }

        private fun key(x: Int, y: Int): Long = (x.toLong() shl 32) or y.toLong()

        private fun cross(ox: Int, oy: Int, ax: Int, ay: Int, bx: Int, by: Int): Long =
            (ax - ox).toLong() * (by - oy) - (ay - oy).toLong() * (bx - ox)

        companion object {
            fun of(labels: IntArray, w: Int, h: Int, c: Component): Shape {
                require(labels.size >= w * h) { "label map has ${labels.size} values, need ${w * h}" }
                require(c.minX >= 0 && c.minY >= 0 && c.maxX < w && c.maxY < h && c.minX <= c.maxX && c.minY <= c.maxY) {
                    "component bbox outside the ${w}x$h map: $c"
                }
                val lab = c.label
                val rows = c.maxY - c.minY + 1
                val rowMin = IntArray(rows) { Int.MAX_VALUE }
                val rowMax = IntArray(rows) { -1 }
                var n = 0L
                var sx = 0L
                var sy = 0L
                for (y in c.minY..c.maxY) {
                    var i = y * w + c.minX
                    val r = y - c.minY
                    for (x in c.minX..c.maxX) {
                        if (labels[i] == lab) {
                            n++
                            sx += x
                            sy += y
                            if (x < rowMin[r]) rowMin[r] = x
                            rowMax[r] = x
                        }
                        i++
                    }
                }
                require(n > 0) { "component ${c.label} has no pixels in the label map" }
                val a = n.toDouble()
                val xm = sx / a
                val ym = sy / a
                var m20 = 0.0
                var m02 = 0.0
                var m11 = 0.0
                for (y in c.minY..c.maxY) {
                    var i = y * w + c.minX
                    val dy = y - ym
                    for (x in c.minX..c.maxX) {
                        if (labels[i] == lab) {
                            val dx = x - xm
                            m20 += dx * dx
                            m02 += dy * dy
                            m11 += dx * dy
                        }
                        i++
                    }
                }
                // Keep only rows that hold pixels (every bbox row does for an 8-connected component).
                var used = 0
                for (r in 0 until rows) if (rowMax[r] >= 0) used++
                val ys = IntArray(used)
                val mins = IntArray(used)
                val maxs = IntArray(used)
                var k = 0
                for (r in 0 until rows) {
                    if (rowMax[r] < 0) continue
                    ys[k] = c.minY + r
                    mins[k] = rowMin[r]
                    maxs[k] = rowMax[r]
                    k++
                }
                return Shape(a, m20, m02, m11, ys, mins, maxs)
            }
        }
    }
}
