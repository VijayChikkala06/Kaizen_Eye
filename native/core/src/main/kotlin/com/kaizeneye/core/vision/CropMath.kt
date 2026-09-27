package com.kaizeneye.core.vision

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.CropSquare

/** The square crop around a component (spec §3), all float64. */
object CropMath {

    /**
     * Spec §3 crop square in full-resolution continuous coordinates for [component] (analysis coordinates, factor
     * [factor]): bbox `bx0 = f·minX`, `by0 = f·minY`, `bx1 = f·(maxX+1)`, `by1 = f·(maxY+1)`; centre = bbox centre;
     * `side = max(bx1−bx0, by1−by0) · (1 + 2·margin)`, then `min(side, fullW, fullH)`;
     * `x0 = clamp(cx − side/2, 0, fullW − side)`, `y0 = clamp(cy − side/2, 0, fullH − side)`.
     */
    fun square(component: Component, factor: Int, fullW: Int, fullH: Int, margin: Double): CropSquare {
        require(factor >= 1) { "factor must be >= 1" }
        require(fullW > 0 && fullH > 0) { "full frame must be non-empty" }
        val f = factor.toDouble()
        val bx0 = f * component.minX
        val by0 = f * component.minY
        val bx1 = f * (component.maxX + 1)
        val by1 = f * (component.maxY + 1)
        val cx = (bx0 + bx1) / 2
        val cy = (by0 + by1) / 2
        var side = maxOf(bx1 - bx0, by1 - by0) * (1 + 2 * margin)
        side = minOf(side, fullW.toDouble(), fullH.toDouble())
        val x0 = clamp(cx - side / 2, 0.0, fullW - side)
        val y0 = clamp(cy - side / 2, 0.0, fullH - side)
        return CropSquare(x0, y0, side)
    }

    /** The same square in analysis coordinates (every value divided by [factor]), as used by `MULTIPLE` (§2.6). */
    fun toAnalysis(square: CropSquare, factor: Int): CropSquare {
        val f = factor.toDouble()
        return CropSquare(square.x0 / f, square.y0 / f, square.side / f)
    }

    /** Spec §3 "inside": `x0 ≤ x < x0 + side` and `y0 ≤ y < y0 + side`. */
    fun contains(square: CropSquare, x: Double, y: Double): Boolean =
        square.x0 <= x && x < square.x0 + square.side && square.y0 <= y && y < square.y0 + square.side

    /** `min(max(v, lo), hi)` (numpy.clip); `lo <= hi` holds for every call above because `side <= fullW, fullH`. */
    private fun clamp(v: Double, lo: Double, hi: Double): Double = minOf(maxOf(v, lo), hi)
}
