package com.kaizeneye.core.twin

/**
 * Patch-grid object sets of one crop (spec §4): `core = {p : cov[p] >= coreThreshold}`, `S = dilate(core, 1)` (score set,
 * 3×3) and `B = dilate(core, 2)` (bank set, 5×5); the grid border counts as outside. Index arrays are ascending (row-major).
 * `cov` itself is computed from the component mask by the mask package.
 */
class PatchSets private constructor(
    val gh: Int,
    val gw: Int,
    val core: BooleanArray,
    val s: BooleanArray,
    val b: BooleanArray,
) {
    val coreIdx: IntArray = indicesOf(core)
    val sIdx: IntArray = indicesOf(s)
    val bIdx: IntArray = indicesOf(b)

    val patches: Int get() = gh * gw

    /** Empty core = `NO_CORE` (spec §2.6 item 6, §4). */
    val isEmpty: Boolean get() = coreIdx.isEmpty()

    companion object {
        /** Default radii of spec §4. */
        const val S_RADIUS = 1
        const val B_RADIUS = 2

        /** Builds the sets from a coverage grid `cov[gh·gw]` (spec §4). */
        fun fromCov(
            cov: FloatArray,
            gh: Int,
            gw: Int,
            coreThreshold: Double = 0.5,
            sRadius: Int = S_RADIUS,
            bRadius: Int = B_RADIUS,
        ): PatchSets {
            require(gh > 0 && gw > 0) { "empty grid" }
            require(cov.size == gh * gw) { "cov has ${cov.size} values, expected ${gh * gw}" }
            val core = BooleanArray(gh * gw) { cov[it].toDouble() >= coreThreshold }
            return fromCore(core, gh, gw, sRadius, bRadius)
        }

        /** Builds the sets from an explicit core mask. */
        fun fromCore(core: BooleanArray, gh: Int, gw: Int, sRadius: Int = S_RADIUS, bRadius: Int = B_RADIUS): PatchSets {
            require(core.size == gh * gw) { "core has ${core.size} values, expected ${gh * gw}" }
            return PatchSets(gh, gw, core.copyOf(), dilate(core, gh, gw, sRadius), dilate(core, gh, gw, bRadius))
        }

        /** Square dilation by radius [r] on the grid ((2r+1)×(2r+1) structuring element; outside the grid = empty). */
        fun dilate(mask: BooleanArray, gh: Int, gw: Int, r: Int): BooleanArray {
            require(r >= 0) { "radius must be >= 0" }
            val out = BooleanArray(gh * gw)
            for (row in 0 until gh) {
                for (col in 0 until gw) {
                    if (!mask[row * gw + col]) continue
                    val r0 = maxOf(0, row - r)
                    val r1 = minOf(gh - 1, row + r)
                    val c0 = maxOf(0, col - r)
                    val c1 = minOf(gw - 1, col + r)
                    for (rr in r0..r1) for (cc in c0..c1) out[rr * gw + cc] = true
                }
            }
            return out
        }

        private fun indicesOf(m: BooleanArray): IntArray {
            var n = 0
            for (v in m) if (v) n++
            val out = IntArray(n)
            var j = 0
            for (i in m.indices) if (m[i]) out[j++] = i
            return out
        }
    }
}
