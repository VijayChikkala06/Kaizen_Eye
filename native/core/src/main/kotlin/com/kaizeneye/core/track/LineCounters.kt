package com.kaizeneye.core.track

import com.kaizeneye.core.model.Verdict

/**
 * Live-line counters (spec §11.6): judged, pass, defect, not-enrolled, reframe, exited-unjudged, plus parts per minute
 * over a sliding window (default 60 s) of judged parts. Thread-safe (the judge thread writes, the UI reads).
 *
 * The window keeps the last [capacity] judged timestamps; beyond that many parts per window the rate saturates at
 * `capacity` per window (far above any hand-fed line).
 */
class LineCounters(val windowMs: Long = 60_000L, val capacity: Int = 4096) {
    init {
        require(windowMs > 0) { "windowMs must be > 0" }
        require(capacity > 0) { "capacity must be > 0" }
    }

    private val times = LongArray(capacity)
    private var head = 0 // next write slot
    private var size = 0

    var judged = 0
        private set
    var pass = 0
        private set
    var defect = 0
        private set
    var notEnrolled = 0
        private set
    var reframe = 0
        private set
    var exitedUnjudged = 0
        private set

    /** One judged presentation with its final [verdict] at time [tMs] (frame time of the trigger). */
    @Synchronized
    fun onVerdict(tMs: Long, verdict: Verdict) {
        judged++
        when (verdict) {
            Verdict.PASS -> pass++
            Verdict.DEFECT -> defect++
            Verdict.NOT_ENROLLED -> notEnrolled++
            Verdict.REFRAME -> reframe++
        }
        times[head] = tMs
        head = (head + 1) % capacity
        if (size < capacity) size++
    }

    /** A track left the view (§11.2 `EXITED`); counts it if it was never judged. */
    @Synchronized
    fun onExit(judged: Boolean) {
        if (!judged) exitedUnjudged++
    }

    /** Judged parts per minute over the window ending at [nowMs] (`(nowMs − windowMs, nowMs]`). */
    @Synchronized
    fun partsPerMinute(nowMs: Long): Double {
        var n = 0
        val from = nowMs - windowMs
        for (i in 0 until size) {
            val t = times[(head - 1 - i + capacity) % capacity]
            if (t > from && t <= nowMs) n++
        }
        return n * 60_000.0 / windowMs
    }

    @Synchronized
    fun reset() {
        judged = 0
        pass = 0
        defect = 0
        notEnrolled = 0
        reframe = 0
        exitedUnjudged = 0
        head = 0
        size = 0
    }

    @Synchronized
    override fun toString(): String =
        "LineCounters(judged=$judged, pass=$pass, defect=$defect, notEnrolled=$notEnrolled, reframe=$reframe, " +
            "exitedUnjudged=$exitedUnjudged)"
}

/** Judge-queue health (spec §11.6). */
object JudgeQueue {
    /** §11.6 default: more than this many queued jobs = "line too fast". */
    const val DEFAULT_MAX_JOBS = 3

    /** "Line too fast" when the judge queue holds more than [maxJobs] jobs. */
    fun lineTooFast(queuedJobs: Int, maxJobs: Int = DEFAULT_MAX_JOBS): Boolean = queuedJobs > maxJobs
}

/** Overlay helpers (spec §11.6). */
object Overlay {
    /** Displayed box centre x = `cx + vx · latencyMs` (latency = measured camera-to-screen delay). */
    fun displayX(cx: Double, vx: Double, latencyMs: Double): Double = cx + vx * latencyMs

    /** Displayed box centre y = `cy + vy · latencyMs`. */
    fun displayY(cy: Double, vy: Double, latencyMs: Double): Double = cy + vy * latencyMs
}
