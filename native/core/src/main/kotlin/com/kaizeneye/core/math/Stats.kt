package com.kaizeneye.core.math

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sqrt

/** Order statistics exactly as spec §0 (numpy's default "linear" percentile, population std). Shared by all core packages. */
object Stats {

    /** Percentile q in [0, 100] of the values (numpy.percentile, method "linear"). */
    fun percentile(values: DoubleArray, q: Double): Double {
        require(values.isNotEmpty()) { "percentile of an empty array" }
        require(q in 0.0..100.0) { "q must be in [0, 100], got $q" }
        val s = values.copyOf().also { it.sort() }
        return percentileOfSorted(s, q)
    }

    fun percentile(values: List<Double>, q: Double): Double = percentile(values.toDoubleArray(), q)

    /** Same as [percentile] for an already ascending-sorted array (no copy). */
    fun percentileOfSorted(sorted: DoubleArray, q: Double): Double {
        val n = sorted.size
        val h = (n - 1) * q / 100.0
        val lo = floor(h).toInt()
        val hi = min(lo + 1, n - 1)
        return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
    }

    fun median(values: DoubleArray): Double = percentile(values, 50.0)

    /** Raw median absolute deviation (multiply by 1.4826 for a robust sigma). */
    fun mad(values: DoubleArray): Double {
        val m = median(values)
        return median(DoubleArray(values.size) { abs(values[it] - m) })
    }

    fun mean(values: DoubleArray): Double {
        require(values.isNotEmpty()) { "mean of an empty array" }
        var s = 0.0
        for (v in values) s += v
        return s / values.size
    }

    /** Population standard deviation (ddof = 0). */
    fun std(values: DoubleArray): Double {
        val m = mean(values)
        var s = 0.0
        for (v in values) s += (v - m) * (v - m)
        return sqrt(s / values.size)
    }
}
