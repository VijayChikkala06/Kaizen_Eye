package com.kaizeneye.core.twin

import kotlin.math.sqrt

/** Global object descriptor and patch normalisation (spec §0, §5). */
object Descriptor {

    /**
     * Global object descriptor `g = L2normalise(Σ_{p ∈ core} f_p)` (spec §5), all patches when [core] is null or empty.
     * Sums in float64 in index order; the result stays float64 (it is rounded through binary16 only when stored, §7.6).
     */
    fun global(features: FloatArray, patches: Int, dim: Int, core: BooleanArray?): DoubleArray {
        require(features.size >= patches * dim) { "features has ${features.size} floats, need ${patches * dim}" }
        val useAll = core == null || core.none { it }
        val g = DoubleArray(dim)
        for (p in 0 until patches) {
            if (!useAll && !core!![p]) continue
            val o = p * dim
            for (t in 0 until dim) g[t] += features[o + t].toDouble()
        }
        return l2NormalizeInPlace(g)
    }

    /** [global] of a set of patches given by ascending indices (all patches when [idx] is empty). */
    fun global(features: FloatArray, patches: Int, dim: Int, idx: IntArray): DoubleArray {
        if (idx.isEmpty()) return global(features, patches, dim, null as BooleanArray?)
        val g = DoubleArray(dim)
        for (p in idx) {
            val o = p * dim
            for (t in 0 until dim) g[t] += features[o + t].toDouble()
        }
        return l2NormalizeInPlace(g)
    }

    /** L2 normalise in place (spec §0): `v / ||v||`, unchanged when `||v|| == 0`. */
    fun l2NormalizeInPlace(v: DoubleArray): DoubleArray {
        var n = 0.0
        for (x in v) n += x * x
        n = sqrt(n)
        if (n == 0.0) return v
        for (i in v.indices) v[i] = v[i] / n
        return v
    }

    /**
     * Per-patch L2 normalisation of a DINOv2 feature map, in place (spec §5: before anything else). The norm is
     * accumulated in float64; each value is `(v / ||v||)` rounded to float32. All-zero patches are left unchanged.
     */
    fun l2NormalizePatches(data: FloatArray, patches: Int, dim: Int): FloatArray {
        require(data.size >= patches * dim) { "data has ${data.size} floats, need ${patches * dim}" }
        for (p in 0 until patches) {
            val o = p * dim
            var n = 0.0
            for (t in 0 until dim) {
                val x = data[o + t].toDouble()
                n += x * x
            }
            n = sqrt(n)
            if (n == 0.0) continue
            for (t in 0 until dim) data[o + t] = (data[o + t].toDouble() / n).toFloat()
        }
        return data
    }

    /** Float64 dot product of [a] with row [row] of [m] (`[rows, a.size]`). */
    fun dot(a: DoubleArray, m: FloatArray, row: Int): Double {
        val o = row * a.size
        var s = 0.0
        for (t in a.indices) s += a[t] * m[o + t].toDouble()
        return s
    }

    /** Float64 dot product of rows [i] of [a] and [j] of [b] (both `[·, dim]`). */
    fun dot(a: FloatArray, i: Int, b: FloatArray, j: Int, dim: Int): Double {
        val ao = i * dim
        val bo = j * dim
        var s = 0.0
        for (t in 0 until dim) s += a[ao + t].toDouble() * b[bo + t].toDouble()
        return s
    }
}
