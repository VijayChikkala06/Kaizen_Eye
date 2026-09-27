package com.kaizeneye.core.math

/** Evaluation helper of spec §14. */
object Auroc {

    /**
     * AUROC of scores [pos] (should be high) vs [neg]: Mann–Whitney `U / (|pos|·|neg|)`, ties counted 0.5 (spec §14).
     * Exact (half-integer counts in float64). NaN when either side is empty. Values must not be NaN.
     */
    fun auroc(pos: DoubleArray, neg: DoubleArray): Double {
        if (pos.isEmpty() || neg.isEmpty()) return Double.NaN
        require(pos.none { it.isNaN() } && neg.none { it.isNaN() }) { "NaN score" }
        val sorted = neg.copyOf().also { it.sort() }
        var u = 0.0
        for (p in pos) {
            val less = lowerBound(sorted, p)          // #neg < p
            val lessOrEqual = upperBound(sorted, p)   // #neg <= p
            u += less + 0.5 * (lessOrEqual - less)
        }
        return u / (pos.size.toDouble() * neg.size.toDouble())
    }

    fun auroc(pos: List<Double>, neg: List<Double>): Double = auroc(pos.toDoubleArray(), neg.toDoubleArray())

    /** First index with a[i] >= v. */
    private fun lowerBound(a: DoubleArray, v: Double): Int {
        var lo = 0
        var hi = a.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (a[mid] < v) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** First index with a[i] > v. */
    private fun upperBound(a: DoubleArray, v: Double): Int {
        var lo = 0
        var hi = a.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (a[mid] <= v) lo = mid + 1 else hi = mid
        }
        return lo
    }
}
