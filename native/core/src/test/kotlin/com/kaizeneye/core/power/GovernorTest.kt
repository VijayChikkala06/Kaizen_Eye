package com.kaizeneye.core.power

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GovernorTest {

    private val nan = Float.NaN

    @Test
    fun targetLevelTable() {
        val g = Governor()
        val table = listOf(
            Triple(0, nan, PowerLevel.L0),
            Triple(1, nan, PowerLevel.L0),
            Triple(2, nan, PowerLevel.L1),
            Triple(3, nan, PowerLevel.L2),
            Triple(6, nan, PowerLevel.L2),
            Triple(0, 0.69f, PowerLevel.L0),
            Triple(0, 0.70f, PowerLevel.L1), // >= 0.70
            Triple(0, 0.85f, PowerLevel.L1), // not > 0.85
            Triple(0, 0.851f, PowerLevel.L2),
            Triple(1, 0.9f, PowerLevel.L2),
            Triple(2, 0.1f, PowerLevel.L1),
            Triple(3, 0.1f, PowerLevel.L2),
        )
        for ((status, headroom, level) in table) assertEquals("status=$status headroom=$headroom", level, g.targetLevel(status, headroom))
    }

    @Test
    fun outputsPerLevelAndIdleOverridesOnlyFps() {
        val g = Governor()
        val l0 = g.update(0, 0, nan, idle = false)
        assertEquals(PowerLevel.L0, l0.level)
        assertEquals(30, l0.fps)
        assertTrue(l0.votingEnabled)
        assertEquals(VlmMode.AUTO, l0.vlmMode)
        assertNull(l0.banner)
        val l1 = g.update(1_000, 2, nan, idle = false)
        assertEquals(PowerLevel.L1, l1.level)
        assertEquals(24, l1.fps)
        assertFalse(l1.votingEnabled)
        assertEquals(VlmMode.ON_TAP, l1.vlmMode)
        assertNull(l1.banner)
        val l2 = g.update(2_000, 3, nan, idle = false)
        assertEquals(PowerLevel.L2, l2.level)
        assertEquals(15, l2.fps)
        assertFalse(l2.votingEnabled)
        assertEquals(VlmMode.PAUSED, l2.vlmMode)
        assertNotNull(l2.banner)
        val idle = g.update(3_000, 3, nan, idle = true)
        assertEquals(PowerLevel.L2, idle.level)
        assertEquals(2, idle.fps)
        assertEquals(VlmMode.PAUSED, idle.vlmMode)
        assertTrue(idle.changed) // fps changed
        assertFalse(idle.levelChanged)
    }

    @Test
    fun risesImmediatelyAndFallsOnlyAfterThirtySecondsOfLowerTarget() {
        val g = Governor()
        assertEquals(PowerLevel.L0, g.update(0, 0, 0.5f, false).level)
        val up = g.update(1_000, 0, 0.9f, false)
        assertEquals(PowerLevel.L2, up.level)
        assertTrue(up.changed)
        assertTrue(up.levelChanged)
        assertTrue(up.reason, up.reason.startsWith("L0→L2"))
        // target drops to L0 at t = 2 s; a blip back to L2 at t = 20 s restarts the clock
        for (t in 2_000L..19_000L step 1_000) assertEquals(PowerLevel.L2, g.update(t, 0, 0.5f, false).level)
        assertEquals(PowerLevel.L2, g.update(20_000, 0, 0.9f, false).level)
        for (t in 21_000L..50_000L step 1_000) assertEquals("t=$t", PowerLevel.L2, g.update(t, 0, 0.5f, false).level)
        // 30 s after the first lower sample (t = 21 s) it drops straight to L0
        val down = g.update(51_000, 0, 0.5f, false)
        assertEquals(PowerLevel.L0, down.level)
        assertTrue(down.levelChanged)
        assertEquals(30, down.fps)
        val again = g.update(52_000, 0, 0.5f, false)
        assertFalse(again.changed)
    }

    @Test
    fun anyLowerTargetKeepsTheCooldownRunningAndTheDropGoesToTheCurrentTarget() {
        val g = Governor()
        g.update(0, 3, nan, false)
        g.update(10_000, 2, nan, false) // L1 target, clock starts
        g.update(25_000, 0, nan, false) // L0 target, clock keeps running
        assertEquals(PowerLevel.L2, g.update(39_999, 0, nan, false).level)
        assertEquals(PowerLevel.L0, g.update(40_000, 0, nan, false).level)
        // unknown headroom is ignored, status alone decides
        val g2 = Governor()
        assertEquals(PowerLevel.L1, g2.update(0, 2, nan, false).level)
    }

    @Test
    fun idleDetectorNeedsThreeSecondsWithoutAComponent() {
        val d = IdleDetector()
        assertFalse(d.update(0, false))
        assertFalse(d.update(2_999, false))
        assertTrue(d.update(3_000, false))
        assertFalse(d.update(3_100, true))
        assertFalse(d.update(6_000, false))
        assertTrue(d.update(6_100, false))
    }
}
