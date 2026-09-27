package com.kaizeneye.core.replay

import com.kaizeneye.core.model.Verdict
import kotlin.math.abs

/** One verdict of a replay run: frame time of the trigger, the track (if any), the verdict and its reason (if any). */
data class VerdictRecord(val tMs: Long, val trackId: Int?, val verdict: Verdict, val reason: String? = null)

/** An expected record paired in time with an actual one (indices into the two input lists). */
data class ReplayPair(val expectedIndex: Int, val actualIndex: Int, val dtMs: Long)

/**
 * Result of [ReplayComparator.compare]. [matched] = time-paired records that agree (allowed borderline flips included,
 * also listed in [borderlineFlips]); [mismatched] = time-paired records that disagree; [missing] / [extra] = indices of
 * expected / actual records left unpaired.
 */
data class ReplayReport(
    val matched: List<ReplayPair>,
    val missing: List<Int>,
    val extra: List<Int>,
    val mismatched: List<ReplayPair>,
    val borderlineFlips: List<ReplayPair>,
) {
    val pass: Boolean get() = missing.isEmpty() && extra.isEmpty() && mismatched.isEmpty()

    override fun toString(): String =
        "ReplayReport(pass=$pass, matched=${matched.size}, missing=$missing, extra=$extra, mismatched=$mismatched, " +
            "borderlineFlips=${borderlineFlips.size})"
}

/** Replay regression: compares the verdicts of a replayed clip with the recorded expectation. */
object ReplayComparator {

    /**
     * Pairs [expected] and [actual] records whose times differ by at most [timeToleranceMs] (closest pairs first; ties →
     * lower expected index, then lower actual index), then compares each pair: they agree when the verdicts are equal
     * and the reasons are equal whenever both are given. A disagreeing pair whose expected index is in
     * [borderlineAllowed] (a known borderline presentation) counts as matched and is reported in `borderlineFlips`.
     */
    fun compare(
        expected: List<VerdictRecord>,
        actual: List<VerdictRecord>,
        timeToleranceMs: Long = 150L,
        borderlineAllowed: Set<Int> = emptySet(),
    ): ReplayReport {
        require(timeToleranceMs >= 0) { "timeToleranceMs must be >= 0" }
        data class Cand(val e: Int, val a: Int, val dt: Long)
        val cands = ArrayList<Cand>()
        for (e in expected.indices) for (a in actual.indices) {
            val dt = abs(expected[e].tMs - actual[a].tMs)
            if (dt <= timeToleranceMs) cands.add(Cand(e, a, dt))
        }
        cands.sortWith(compareBy<Cand>({ it.dt }, { it.e }, { it.a }))
        val eUsed = BooleanArray(expected.size)
        val aUsed = BooleanArray(actual.size)
        val pairs = ArrayList<ReplayPair>()
        for (c in cands) {
            if (eUsed[c.e] || aUsed[c.a]) continue
            eUsed[c.e] = true
            aUsed[c.a] = true
            pairs.add(ReplayPair(c.e, c.a, actual[c.a].tMs - expected[c.e].tMs))
        }
        pairs.sortBy { it.expectedIndex }
        val matched = ArrayList<ReplayPair>()
        val mismatched = ArrayList<ReplayPair>()
        val flips = ArrayList<ReplayPair>()
        for (p in pairs) {
            val x = expected[p.expectedIndex]
            val y = actual[p.actualIndex]
            val agree = x.verdict == y.verdict && (x.reason == null || y.reason == null || x.reason == y.reason)
            when {
                agree -> matched.add(p)
                p.expectedIndex in borderlineAllowed -> {
                    matched.add(p)
                    flips.add(p)
                }
                else -> mismatched.add(p)
            }
        }
        return ReplayReport(
            matched = matched,
            missing = expected.indices.filter { !eUsed[it] },
            extra = actual.indices.filter { !aUsed[it] },
            mismatched = mismatched,
            borderlineFlips = flips,
        )
    }
}
