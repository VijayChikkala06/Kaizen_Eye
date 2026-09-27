package com.kaizeneye.core.mask

/**
 * Exact median / MAD of 8-bit values from a 256-bin histogram (spec §2.1). The results are bit-identical to the
 * sort-based definitions of §0 (`Stats.median`, `Stats.mad`: numpy "linear" percentile), including even counts, where
 * the median is the mean of the two middle values (a `.5` value). Deviations from a `.5` median are half-integers, so
 * they are binned doubled (`|2v − 2·median|`, 511 bins) — every intermediate stays exact in float64.
 */
object ByteHistogram {

    /** Bins needed for the doubled deviations `|2v − 2m|` of 8-bit values (0..510). */
    const val DEVIATION_BINS = 511

    /** Number of values counted in [hist]. */
    fun total(hist: IntArray): Long {
        var n = 0L
        for (c in hist) n += c
        return n
    }

    /** Median (§0 `P_50`) of the values counted in [hist] (256 bins). */
    fun median(hist: IntArray): Double {
        val n = total(hist)
        require(n > 0) { "median of an empty histogram" }
        return p50(hist, n, 1.0)
    }

    /**
     * Raw MAD (§0: median of `|x − median(x)|`) of the values counted in [hist], given their exact [median].
     * [scratch] must hold at least [DEVIATION_BINS] ints; it is overwritten (pass one to stay allocation-free).
     */
    fun mad(hist: IntArray, median: Double, scratch: IntArray = IntArray(DEVIATION_BINS)): Double {
        require(scratch.size >= DEVIATION_BINS) { "scratch needs $DEVIATION_BINS bins" }
        val m2 = median * 2.0
        val med2 = m2.toInt()
        require(med2.toDouble() == m2) { "median $median is not a multiple of 0.5" }
        scratch.fill(0, 0, DEVIATION_BINS)
        var n = 0L
        for (v in hist.indices) {
            val c = hist[v]
            if (c == 0) continue
            val d2 = if (2 * v >= med2) 2 * v - med2 else med2 - 2 * v
            scratch[d2] += c
            n += c
        }
        require(n > 0) { "MAD of an empty histogram" }
        return p50(scratch, n, 0.5)
    }

    /**
     * §0 percentile q = 50 over the values `bin · scale` counted in [hist]: `h = (n−1)·50/100`, `lo = floor(h)`,
     * `hi = min(lo+1, n−1)`, `s[lo] + (h−lo)·(s[hi]−s[lo])` — the same expression as `Stats.percentileOfSorted`.
     */
    private fun p50(hist: IntArray, n: Long, scale: Double): Double {
        val h = (n - 1) * 50.0 / 100.0
        val lo = kotlin.math.floor(h).toLong()
        val hi = minOf(lo + 1, n - 1)
        val sLo = valueAtRank(hist, lo) * scale
        val sHi = valueAtRank(hist, hi) * scale
        return sLo + (h - lo) * (sHi - sLo)
    }

    /** The value (bin index) at 0-based position [rank] of the ascending sort of the counted values. */
    fun valueAtRank(hist: IntArray, rank: Long): Int {
        var cum = 0L
        for (b in hist.indices) {
            cum += hist[b]
            if (cum > rank) return b
        }
        throw IllegalArgumentException("rank $rank >= count $cum")
    }
}
