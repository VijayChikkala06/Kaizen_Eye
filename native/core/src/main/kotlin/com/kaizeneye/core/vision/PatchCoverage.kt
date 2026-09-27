package com.kaizeneye.core.vision

import com.kaizeneye.core.model.CropSquare
import kotlin.math.floor

/**
 * The patch sets of one crop (spec §4) on a `gh × gw` backbone grid, row-major (`p = r·gw + c`):
 * [core] = `cov ≥ coreThreshold`, [s] = `dilate(core, 1)` (score set), [b] = `dilate(core, 2)` (bank set).
 * An empty core means the presentation is `NO_CORE` (REFRAME).
 */
class PatchSets(val gh: Int, val gw: Int, val core: BooleanArray, val s: BooleanArray, val b: BooleanArray) {
    val coreCount: Int = core.count { it }
    val sCount: Int = s.count { it }
    val bCount: Int = b.count { it }

    /** §4: `core` empty → `NO_CORE`. */
    val isCoreEmpty: Boolean get() = coreCount == 0
}

/** Patch-grid object coverage (spec §4). */
object PatchCoverage {

    /** Samples per cell side (4×4 = 16 samples per cell). */
    const val SAMPLES = 4

    /**
     * Spec §4 `cov[r][c]`: the fraction of the 4×4 sample points of cell `(r, c)` that land on component [label] of the
     * `w×h` analysis [labels] map. Sample `(i, j)` → full-res `u = x0 + (c + (j+0.5)/4)·side/gw`,
     * `v = y0 + (r + (i+0.5)/4)·side/gh` → analysis pixel `(floor(u/f), floor(v/f))`; points outside the analysis image
     * count as 0. Returns `gh·gw` values (multiples of 1/16).
     */
    fun coverage(
        labels: IntArray, w: Int, h: Int, label: Int, crop: CropSquare, factor: Int, gh: Int, gw: Int,
    ): FloatArray = FloatArray(gh * gw).also { coverage(labels, w, h, label, crop, factor, gh, gw, it) }

    /** [coverage] into a preallocated [dst] of `gh·gw` floats. */
    fun coverage(
        labels: IntArray, w: Int, h: Int, label: Int, crop: CropSquare, factor: Int, gh: Int, gw: Int, dst: FloatArray,
    ) {
        require(labels.size >= w * h) { "label map has ${labels.size} values, need ${w * h}" }
        require(gh > 0 && gw > 0 && dst.size >= gh * gw) { "bad grid ${gh}x$gw or dst size ${dst.size}" }
        val f = factor.toDouble()
        val col = IntArray(gw * SAMPLES)
        val row = IntArray(gh * SAMPLES)
        for (c in 0 until gw) for (j in 0 until SAMPLES) {
            val u = crop.x0 + (c + (j + 0.5) / SAMPLES) * crop.side / gw
            col[c * SAMPLES + j] = analysisIndex(u, f, w)
        }
        for (r in 0 until gh) for (i in 0 until SAMPLES) {
            val v = crop.y0 + (r + (i + 0.5) / SAMPLES) * crop.side / gh
            row[r * SAMPLES + i] = analysisIndex(v, f, h)
        }
        val perCell = (SAMPLES * SAMPLES).toFloat()
        for (r in 0 until gh) {
            for (c in 0 until gw) {
                var hits = 0
                for (i in 0 until SAMPLES) {
                    val ay = row[r * SAMPLES + i]
                    if (ay < 0) continue
                    val base = ay * w
                    for (j in 0 until SAMPLES) {
                        val ax = col[c * SAMPLES + j]
                        if (ax >= 0 && labels[base + ax] == label) hits++
                    }
                }
                dst[r * gw + c] = hits / perCell
            }
        }
    }

    /** `floor(coord / f)` as an index into `[0, size)`, or −1 when the point is outside the analysis image. */
    private fun analysisIndex(coord: Double, f: Double, size: Int): Int {
        val a = floor(coord / f)
        return if (a < 0.0 || a >= size) -1 else a.toInt()
    }

    /**
     * Spec §4 sets from [cov]: `core = {p : cov[p] ≥ coreThr}`, `S = dilate(core, 1)` (3×3), `B = dilate(core, 2)` (5×5);
     * the grid border counts as outside.
     */
    fun sets(cov: FloatArray, gh: Int, gw: Int, coreThr: Double = 0.5): PatchSets {
        require(cov.size >= gh * gw) { "cov has ${cov.size} values, need ${gh * gw}" }
        val core = BooleanArray(gh * gw) { cov[it].toDouble() >= coreThr }
        return PatchSets(gh, gw, core, dilate(core, gh, gw, 1), dilate(core, gh, gw, 2))
    }

    /** Square `(2·radius+1)²` dilation of a patch set on the grid (out-of-grid neighbours ignored). */
    fun dilate(set: BooleanArray, gh: Int, gw: Int, radius: Int): BooleanArray {
        val out = BooleanArray(gh * gw)
        for (r in 0 until gh) for (c in 0 until gw) {
            if (!set[r * gw + c]) continue
            for (rr in maxOf(0, r - radius)..minOf(gh - 1, r + radius)) {
                for (cc in maxOf(0, c - radius)..minOf(gw - 1, c + radius)) out[rr * gw + cc] = true
            }
        }
        return out
    }
}
