package com.kaizeneye.core.vision

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.CropSquare
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * The square of a CANONICAL (pose-normalised) crop (twin-spec §6.5, refined): centre ([cx], [cy]) and [side] in full-resolution
 * pixels, rotated by [theta] (the crop's horizontal axis points along `(cos θ, sin θ)` of the frame).
 *
 * The spec rotates the axis-aligned crop square about its own centre with the same side. That clips the ends of a long part
 * that lies diagonally (the axis-aligned bounding box of a needle at 45° is only 0.71 of its length) and leaves the part
 * off-centre. Here the square is fitted to the part's extent IN the rotated frame, so the part is always centred and always
 * inside the crop, whatever its angle.
 */
data class CanonicalSquare(val cx: Double, val cy: Double, val side: Double, val theta: Double) {
    /** The unrotated square with the same centre and side (overlay anchor: rotate it by θ about its centre). */
    val asCropSquare: CropSquare get() = CropSquare(cx - side / 2, cy - side / 2, side)
}

object CanonicalCrop {

    /**
     * Fits the canonical square of [component] at principal angle [theta]: `side = (1 + 2·margin) · max(extent along the
     * axis, extent across it)` measured on the component's pixels in the rotated frame, centred on that extent.
     */
    fun square(
        labels: IntArray, w: Int, h: Int, component: Component, theta: Double, factor: Int, margin: Double,
    ): CanonicalSquare {
        val f = factor.toDouble()
        val cx0 = f * (component.minX + component.maxX + 1) / 2.0
        val cy0 = f * (component.minY + component.maxY + 1) / 2.0
        val c = cos(theta)
        val s = sin(theta)
        var aMin = Double.POSITIVE_INFINITY
        var aMax = Double.NEGATIVE_INFINITY
        var bMin = Double.POSITIVE_INFINITY
        var bMax = Double.NEGATIVE_INFINITY
        for (y in component.minY..component.maxY) {
            val row = y * w
            for (x in component.minX..component.maxX) {
                if (labels[row + x] != component.label) continue
                // the four pixel corners, so the extent is the true outline (not the pixel centres)
                for (dy in 0..1) for (dx in 0..1) {
                    val px = f * (x + dx) - cx0
                    val py = f * (y + dy) - cy0
                    val a = px * c + py * s
                    val b = -px * s + py * c
                    if (a < aMin) aMin = a
                    if (a > aMax) aMax = a
                    if (b < bMin) bMin = b
                    if (b > bMax) bMax = b
                }
            }
        }
        if (aMin > aMax) return CanonicalSquare(cx0, cy0, f * max(component.width, component.height) * (1 + 2 * margin), theta)
        val aMid = (aMin + aMax) / 2
        val bMid = (bMin + bMax) / 2
        val side = max(aMax - aMin, bMax - bMin) * (1 + 2 * margin)
        return CanonicalSquare(cx0 + aMid * c - bMid * s, cy0 + aMid * s + bMid * c, max(side, 2 * f), theta)
    }

    /**
     * Patch coverage (spec §4: 4×4 sample points per cell) of a canonical crop viewed at angle [theta] (the square's own θ
     * or θ + π): the sample points are pushed through the rotation of `cropResizeRotated` (`u = cx + a·cosθ − b·sinθ`,
     * `v = cy + a·sinθ + b·cosθ`, `a` / `b` = the point's offsets from the crop centre). Identical to
     * [PatchCoverage.coverage] for θ = 0 and the square of the axis-aligned crop.
     */
    fun coverage(
        labels: IntArray, w: Int, h: Int, label: Int, sq: CanonicalSquare, theta: Double, factor: Int, gh: Int, gw: Int,
        dst: FloatArray = FloatArray(gh * gw),
    ): FloatArray {
        require(labels.size >= w * h) { "label map has ${labels.size} values, need ${w * h}" }
        require(gh > 0 && gw > 0 && dst.size >= gh * gw) { "bad grid ${gh}x$gw or dst size ${dst.size}" }
        val n = PatchCoverage.SAMPLES
        val f = factor.toDouble()
        val c = cos(theta)
        val s = sin(theta)
        val perCell = (n * n).toFloat()
        for (r in 0 until gh) {
            for (col in 0 until gw) {
                var hits = 0
                for (i in 0 until n) {
                    val b = (r + (i + 0.5) / n) * sq.side / gh - sq.side / 2
                    for (j in 0 until n) {
                        val a = (col + (j + 0.5) / n) * sq.side / gw - sq.side / 2
                        val u = sq.cx + a * c - b * s
                        val v = sq.cy + a * s + b * c
                        val ax = floor(u / f)
                        val ay = floor(v / f)
                        if (ax < 0.0 || ay < 0.0 || ax >= w || ay >= h) continue
                        if (labels[ay.toInt() * w + ax.toInt()] == label) hits++
                    }
                }
                dst[r * gw + col] = hits / perCell
            }
        }
        return dst
    }

    /** The four corners (full-res frame pixels) of grid cell (`row`, `col`) of a canonical crop at angle [theta], for the overlay. */
    fun cellCorners(sq: CanonicalSquare, theta: Double, gh: Int, gw: Int, row: Int, col: Int): DoubleArray {
        val c = cos(theta)
        val s = sin(theta)
        val out = DoubleArray(8)
        var k = 0
        for ((rr, cc) in listOf(row to col, row to col + 1, row + 1 to col + 1, row + 1 to col)) {
            val a = cc * sq.side / gw - sq.side / 2
            val b = rr * sq.side / gh - sq.side / 2
            out[k++] = sq.cx + a * c - b * s
            out[k++] = sq.cy + a * s + b * c
        }
        return out
    }

    /** Smallest angle difference helper for tests: `θ` mod π folded to `(−π/2, π/2]`. */
    fun foldPi(theta: Double): Double {
        var t = theta % Math.PI
        if (t > Math.PI / 2) t -= Math.PI
        if (t <= -Math.PI / 2) t += Math.PI
        return min(t, Math.PI / 2)
    }
}
