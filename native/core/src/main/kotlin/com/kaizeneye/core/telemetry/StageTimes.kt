package com.kaizeneye.core.telemetry

/**
 * Per-stage timers (downscale, mask, track, crop, backbone, knn, verdict, …): one [LatencyStats] ring per stage name,
 * created on first use. Thread-safe; the analyzer and judge threads record into the same instance.
 */
class StageTimes(val capacity: Int = LatencyStats.DEFAULT_CAPACITY) {

    private val stages = LinkedHashMap<String, LatencyStats>()

    /** The ring of [stage] (created empty on first use). */
    fun stats(stage: String): LatencyStats = synchronized(stages) { stages.getOrPut(stage) { LatencyStats(capacity) } }

    /** Records one sample of [stage] in ms. */
    fun record(stage: String, ms: Double) = stats(stage).add(ms)

    /** Records one sample of [stage] from a `System.nanoTime()` start. */
    fun recordSince(stage: String, startNanos: Long) = record(stage, (System.nanoTime() - startNanos) / 1e6)

    /** Runs [block] and records its duration under [stage]. */
    inline fun <T> time(stage: String, block: () -> T): T {
        val t0 = System.nanoTime()
        try {
            return block()
        } finally {
            recordSince(stage, t0)
        }
    }

    /** Stage names in first-use order. */
    fun names(): List<String> = synchronized(stages) { stages.keys.toList() }

    /** Summary of every stage, in first-use order. */
    fun snapshot(): Map<String, LatencySummary> {
        val copy = synchronized(stages) { stages.entries.map { it.key to it.value } }
        val out = LinkedHashMap<String, LatencySummary>(copy.size)
        for ((name, s) in copy) out[name] = s.summary()
        return out
    }

    fun clear() = synchronized(stages) { stages.clear() }
}
