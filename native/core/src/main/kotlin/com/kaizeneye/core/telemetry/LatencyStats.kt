package com.kaizeneye.core.telemetry

import com.kaizeneye.core.math.Stats

/** Summary of a latency ring buffer: sample [count] and §0/§14 statistics in ms (NaN when empty). */
data class LatencySummary(val count: Int, val mean: Double, val p50: Double, val p95: Double, val max: Double) {
    override fun toString(): String =
        if (count == 0) "n=0" else String.format(java.util.Locale.ROOT, "n=%d mean=%.2f p50=%.2f p95=%.2f max=%.2f ms", count, mean, p50, p95, max)
}

/**
 * Latency statistics over a ring buffer of the last [capacity] samples (spec §14: `N = 512`, p50/p95 by the §0
 * percentile). [add] is allocation-free; the summaries copy and sort the (at most [capacity]) samples. Thread-safe.
 */
class LatencyStats(val capacity: Int = DEFAULT_CAPACITY) {
    init {
        require(capacity > 0) { "capacity must be > 0" }
    }

    private val buf = DoubleArray(capacity)
    private var head = 0
    private var size = 0

    /** Total samples ever added (the buffer holds the last [capacity] of them). */
    var total: Long = 0
        private set

    /** Adds one sample (ms). NaN / infinite samples are ignored. */
    @Synchronized
    fun add(ms: Double) {
        if (!ms.isFinite()) return
        buf[head] = ms
        head = (head + 1) % capacity
        if (size < capacity) size++
        total++
    }

    /** Samples currently in the window. */
    val count: Int
        @Synchronized get() = size

    @Synchronized
    fun mean(): Double = if (size == 0) Double.NaN else Stats.mean(window())

    @Synchronized
    fun p50(): Double = percentile(50.0)

    @Synchronized
    fun p95(): Double = percentile(95.0)

    @Synchronized
    fun max(): Double = percentile(100.0)

    /** §0 percentile [q] of the window (NaN when empty). */
    @Synchronized
    fun percentile(q: Double): Double {
        if (size == 0) return Double.NaN
        val s = window()
        s.sort()
        return Stats.percentileOfSorted(s, q)
    }

    /** count, mean, p50, p95, max in one pass over one sorted copy. */
    @Synchronized
    fun summary(): LatencySummary {
        if (size == 0) return LatencySummary(0, Double.NaN, Double.NaN, Double.NaN, Double.NaN)
        val s = window()
        val mean = Stats.mean(s)
        s.sort()
        return LatencySummary(
            size, mean, Stats.percentileOfSorted(s, 50.0), Stats.percentileOfSorted(s, 95.0), Stats.percentileOfSorted(s, 100.0),
        )
    }

    @Synchronized
    fun clear() {
        head = 0
        size = 0
        total = 0
    }

    /** The samples in insertion order (oldest first), as a new array. */
    private fun window(): DoubleArray {
        val out = DoubleArray(size)
        val start = (head - size + capacity) % capacity
        for (i in 0 until size) out[i] = buf[(start + i) % capacity]
        return out
    }

    companion object {
        /** Spec §14 ring size. */
        const val DEFAULT_CAPACITY = 512
    }
}
