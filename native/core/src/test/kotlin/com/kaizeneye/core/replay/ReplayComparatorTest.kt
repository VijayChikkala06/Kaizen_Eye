package com.kaizeneye.core.replay

import com.kaizeneye.core.model.Verdict
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReplayComparatorTest {

    private fun r(t: Long, v: Verdict, reason: String? = null, track: Int? = null) = VerdictRecord(t, track, v, reason)

    @Test
    fun identicalRunsPass() {
        val e = listOf(r(1_000, Verdict.PASS), r(2_000, Verdict.DEFECT), r(3_000, Verdict.REFRAME, "TOUCHES_BORDER"))
        val rep = ReplayComparator.compare(e, e)
        assertTrue(rep.pass)
        assertEquals(3, rep.matched.size)
    }

    @Test
    fun timeToleranceIsInclusiveAndClosestPairsWin() {
        val e = listOf(r(1_000, Verdict.PASS), r(1_200, Verdict.DEFECT))
        val a = listOf(r(1_150, Verdict.PASS), r(1_190, Verdict.DEFECT))
        val rep = ReplayComparator.compare(e, a, timeToleranceMs = 150)
        // (1, 1) dt 10 is paired first, then (0, 0) dt 150 (inclusive)
        assertTrue(rep.toString(), rep.pass)
        assertEquals(listOf(ReplayPair(0, 0, 150), ReplayPair(1, 1, -10)), rep.matched)
        val late = ReplayComparator.compare(listOf(r(1_000, Verdict.PASS)), listOf(r(1_151, Verdict.PASS)))
        assertFalse(late.pass)
        assertEquals(listOf(0), late.missing)
        assertEquals(listOf(0), late.extra)
    }

    @Test
    fun mismatchesAndBorderlineFlips() {
        val e = listOf(r(1_000, Verdict.DEFECT), r(2_000, Verdict.PASS), r(3_000, Verdict.NOT_ENROLLED, "IDENTITY"))
        val a = listOf(r(1_010, Verdict.PASS), r(2_020, Verdict.PASS), r(3_000, Verdict.NOT_ENROLLED, "SHAPE"))
        val strict = ReplayComparator.compare(e, a)
        assertFalse(strict.pass)
        assertEquals(listOf(0, 2), strict.mismatched.map { it.expectedIndex })
        val lenient = ReplayComparator.compare(e, a, borderlineAllowed = setOf(0))
        assertEquals(listOf(2), lenient.mismatched.map { it.expectedIndex })
        assertEquals(listOf(0), lenient.borderlineFlips.map { it.expectedIndex })
        val ok = ReplayComparator.compare(e, a, borderlineAllowed = setOf(0, 2))
        assertTrue(ok.pass)
        // a missing reason on either side does not count as a mismatch
        assertTrue(ReplayComparator.compare(listOf(r(0, Verdict.REFRAME, "MULTIPLE")), listOf(r(0, Verdict.REFRAME))).pass)
    }
}
