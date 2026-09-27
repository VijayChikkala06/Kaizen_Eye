package com.kaizeneye.core.track

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.Verdict
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.roundToInt
import kotlin.math.roundToLong

class TrackerTest {

    private val w = 320
    private val h = 180

    /** A square detection of side 2·[half] centred at ([cx], [cy]); bbox clipped to the frame, border flag with margin 2. */
    private fun det(cx: Double, cy: Double, sharp: Double = 100.0, ref: Long = 0L, half: Int = 10): Detection {
        val minX = maxOf(0, (cx - half).roundToInt())
        val maxX = minOf(w - 1, (cx + half).roundToInt() - 1)
        val minY = maxOf(0, (cy - half).roundToInt())
        val maxY = minOf(h - 1, (cy + half).roundToInt() - 1)
        val area = maxOf(1, (maxX - minX + 1) * (maxY - minY + 1))
        val border = minX < 2 || minY < 2 || maxX >= w - 2 || maxY >= h - 2
        return Detection(Component(1, area, minX, minY, maxX, maxY, cx, cy, border), sharp, ref)
    }

    private fun Tracker.run(frames: List<Pair<Long, List<Detection>>>): List<FrameUpdate> = frames.map { (t, d) -> update(t, d) }

    private fun lineEvents(updates: List<FrameUpdate>) = updates.flatMap { u -> u.triggers.filter { it.trigger == Trigger.LINE } }
    private fun steadyEvents(updates: List<FrameUpdate>) = updates.flatMap { u -> u.triggers.filter { it.trigger == Trigger.STEADY } }

    @Test
    fun fiftyPartsCrossingAtSpeedWithJitterAndDropoutsFireExactlyOnceEach() {
        val rnd = Random(2026)
        class Part(val start: Int, val y: Double, val speed: Double) {
            var dropLeft = 0
            var cool = 0
            var fired = 0
        }
        val parts = ArrayList<Part>()
        var start = 0
        repeat(50) {
            parts += Part(start, 50.0 + rnd.nextDouble() * 80.0, 8.0 + rnd.nextDouble() * 4.0)
            start += 18 + rnd.nextInt(5)
        }
        val tracker = Tracker(w, h, TriggerConfig()) // ANY direction, vertical line at x = 160, both triggers on
        val trackToPart = HashMap<Int, Part>()
        var drops = 0
        val lastFrame = start + 60
        var lineCount = 0
        for (i in 0..lastFrame) {
            val t = (i * 1000.0 / 30).roundToLong()
            val dets = ArrayList<Detection>()
            val owners = ArrayList<Part>()
            for (p in parts) {
                val k = i - p.start
                if (k < 0) continue
                val x = 5.0 + p.speed * k + (rnd.nextDouble() * 3 - 1.5)
                val y = p.y + (rnd.nextDouble() * 3 - 1.5)
                if (x > w + 15) continue
                if (p.dropLeft > 0) {
                    p.dropLeft--
                    continue
                }
                if (p.cool > 0) p.cool--
                if (k > 2 && p.cool == 0 && rnd.nextDouble() < 0.06) {
                    p.dropLeft = rnd.nextInt(2) // this frame + 0 or 1 more: 1-2 frame drop-outs
                    p.cool = 3 + p.dropLeft
                    drops++
                    continue
                }
                dets += det(x, y, ref = i.toLong())
                owners += p
            }
            val u = tracker.update(t, dets)
            for (d in dets.indices) trackToPart.putIfAbsent(u.assignments[d], owners[d])
            for (f in u.triggers) {
                assertEquals("frame $i: only LINE may fire for moving parts", Trigger.LINE, f.trigger)
                trackToPart.getValue(f.trackId).fired++
                lineCount++
            }
        }
        assertTrue("the scenario must contain drop-outs, had $drops", drops >= 20)
        assertEquals(50, lineCount)
        assertEquals(List(50) { 1 }, parts.map { it.fired })
    }

    @Test
    fun aPartThatStopsOnTheLineAndReversesFiresOnce() {
        for (direction in listOf(Direction.ANY, Direction.POSITIVE)) {
            val tracker = Tracker(w, h, TriggerConfig(direction = direction, steadyEnabled = false))
            val xs = ArrayList<Double>()
            for (i in 0..11) xs += 40.0 + 10 * i // up to 150
            repeat(8) { xs += listOf(158.0, 163.0, 164.0, 157.0, 156.0, 162.0) } // dither across the line
            for (i in 0..12) xs += 160.0 - 10 * i // back upstream
            for (i in 0..15) xs += 40.0 + 15 * i // and across again
            val updates = xs.mapIndexed { i, x -> tracker.update(i * 33L, listOf(det(x, 90.0, ref = i.toLong()))) }
            val lines = lineEvents(updates)
            assertEquals("direction $direction", 1, lines.size)
            assertEquals(1, lines[0].trackId)
            assertTrue(updates.all { it.tracks.size == 1 && it.tracks[0].id == 1 }) // never lost the part
        }
        // NEGATIVE: the same part is first confirmed on the NEGATIVE line's downstream side -> never fires
        val neg = Tracker(w, h, TriggerConfig(direction = Direction.NEGATIVE, steadyEnabled = false))
        val ups = (0..30).map { i -> neg.update(i * 33L, listOf(det(40.0 + 10 * i, 90.0))) }
        assertEquals(0, lineEvents(ups).size)
    }

    @Test
    fun borderTouchingHandsNeverFireSteady() {
        val tracker = Tracker(w, h, TriggerConfig())
        val updates = (0 until 120).map { i ->
            tracker.update(i * 33L, listOf(det(8.0, 120.0, sharp = 300.0, half = 20))) // minX = 0 -> touches the border
        }
        assertTrue(updates.last().tracks.single().touchesBorder)
        assertEquals(0, steadyEvents(updates).size)
        assertEquals(0, lineEvents(updates).size)
    }

    @Test
    fun aStillSharpPartFiresSteadyOnceAfterHalfASecond() {
        val rnd = Random(4)
        val tracker = Tracker(w, h, TriggerConfig())
        val updates = (0 until 60).map { i ->
            tracker.update(i * 40L, listOf(det(220.0 + rnd.nextDouble() * 0.2 - 0.1, 90.0, ref = i.toLong())))
        }
        val fired = updates.withIndex().filter { it.value.triggers.isNotEmpty() }
        assertEquals(1, fired.size)
        // confirmed at frame 1 (t = 40) -> fire at the first t >= 540: frame 14 (t = 560)
        assertEquals(14, fired[0].index)
        val trig = fired[0].value.triggers.single()
        assertEquals(Trigger.STEADY, trig.trigger)
        assertEquals(14L, trig.judgeFrame!!.frameRef)
        assertEquals(TrackState.JUDGED, updates.last().tracks.single().state)
        assertEquals(TrackState.HOLDING, updates[5].tracks.single().state)
    }

    @Test
    fun aBlurDipRestartsTheHoldClock() {
        val tracker = Tracker(w, h, TriggerConfig())
        val updates = (0 until 60).map { i ->
            val sharp = if (i == 8) 50.0 else 100.0 // 50 < 0.7 * 100
            tracker.update(i * 40L, listOf(det(220.0, 90.0, sharp = sharp, ref = i.toLong())))
        }
        val fired = updates.withIndex().filter { it.value.triggers.isNotEmpty() }
        assertEquals(1, fired.size)
        // run restarts at frame 9 (t = 360) -> first t >= 860 is frame 22 (t = 880)
        assertEquals(22, fired[0].index)
        assertEquals(TrackState.TRACKING, updates[8].tracks.single().state)
    }

    @Test
    fun steadyNeedsTheSpeedBelowVStill() {
        val tracker = Tracker(w, h, TriggerConfig())
        // 1 px per 40 ms = 0.025 px/ms > 0.02: a slowly creeping part never holds
        val updates = (0 until 60).map { i -> tracker.update(i * 40L, listOf(det(200.0 + i, 90.0))) }
        assertEquals(0, steadyEvents(updates).size)
    }

    @Test
    fun twoPartsInTheFrameAtOnceFireSeparately() {
        val tracker = Tracker(w, h, TriggerConfig())
        val updates = (0 until 30).map { i ->
            tracker.update(i * 33L, listOf(det(20.0 + 10 * i, 50.0, ref = i.toLong()), det(30.0 + 10 * i, 130.0, ref = i.toLong())))
        }
        val lines = lineEvents(updates)
        assertEquals(2, lines.size)
        assertEquals(setOf(1, 2), lines.map { it.trackId }.toSet())
        assertTrue(updates.all { it.tracks.size == 2 })
        assertArrayEquals(intArrayOf(1, 2), updates[20].assignments)
    }

    @Test
    fun anIdSwitchAfterAPositiveLineDoesNotCountTwice() {
        // (a) the part jumps beyond the gate after the line -> a new track opens downstream
        val a = Tracker(w, h, TriggerConfig(direction = Direction.POSITIVE, steadyEnabled = false))
        val xs = (0..16).map { 40.0 + 10 * it } + (0..8).map { 290.0 - 5 * it } // jump from 200 to 290, then drift
        val ua = xs.mapIndexed { i, x -> a.update(i * 33L, listOf(det(x, 90.0))) }
        assertEquals(1, lineEvents(ua).size)
        assertTrue(ua.last().tracks.any { it.id == 2 && it.confirmed && !it.judged })
        // (b) the part vanishes for longer than maxMissed, then reappears downstream moving on
        val b = Tracker(w, h, TriggerConfig(direction = Direction.POSITIVE, steadyEnabled = false))
        val ub = ArrayList<FrameUpdate>()
        for (i in 0..16) ub += b.update(i * 33L, listOf(det(40.0 + 10 * i, 90.0)))
        for (i in 17..24) ub += b.update(i * 33L, emptyList())
        for (i in 25..30) ub += b.update(i * 33L, listOf(det(250.0 + 5 * (i - 25), 90.0)))
        assertEquals(1, lineEvents(ub).size)
        val exits = ub.flatMap { it.exits }
        assertEquals(listOf(TrackExit(1, judged = true, confirmed = true)), exits)
    }

    @Test
    fun lineFiresWithTheMostCentralSharpFrameFirst() {
        val tracker = Tracker(w, h, TriggerConfig(steadyEnabled = false))
        val updates = (0..20).map { i -> tracker.update(i * 33L, listOf(det(20.0 + 10 * i, 90.0, ref = i.toLong()))) }
        val fired = updates.withIndex().single { it.value.triggers.isNotEmpty() }
        // x = 160 (frame 14) is exactly on the line = upstream; 170 (15) and 180 (16) are downstream -> fires at 16
        assertEquals(16, fired.index)
        val trig = fired.value.triggers.single()
        assertEquals(listOf(14L, 13L, 15L), trig.frames.map { it.frameRef }) // centrality ties -> earlier first
        assertEquals(trig.frames, trig.buffer)
        assertFalse(trig.needsReframe)
    }

    @Test
    fun aTrackThatOnlyEverTouchedTheBorderFiresAsReframe() {
        val tracker = Tracker(w, h, TriggerConfig(steadyEnabled = false))
        val updates = (0..25).map { i -> tracker.update(i * 33L, listOf(det(20.0 + 10 * i, 90.0, half = 90))) } // full height
        val trig = lineEvents(updates).single()
        assertTrue(trig.needsReframe)
        assertEquals(null, trig.judgeFrame)
        assertTrue(updates.all { it.bestCrops.isEmpty() })
    }

    @Test
    fun equidistantCandidatesGoToTheLowerTrackId() {
        val tracker = Tracker(w, h, TriggerConfig())
        tracker.update(0L, listOf(det(100.0, 50.0), det(100.0, 70.0)))
        val u = tracker.update(33L, listOf(det(100.0, 60.0)))
        assertArrayEquals(intArrayOf(1), u.assignments)
        assertEquals(1, u.tracks.first { it.id == 2 }.missed)
    }

    @Test
    fun velocityIsTheFirstDifferenceThenAnEma() {
        val tracker = Tracker(w, h, TriggerConfig())
        tracker.update(0L, listOf(det(100.0, 50.0)))
        val u1 = tracker.update(50L, listOf(det(110.0, 50.0)))
        assertEquals(0.2, u1.tracks[0].vx, 1e-15)
        val u2 = tracker.update(100L, listOf(det(115.0, 51.0)))
        assertEquals(0.5 * 5.0 / 50 + 0.5 * 0.2, u2.tracks[0].vx, 1e-15)
        assertEquals(0.5 * 1.0 / 50, u2.tracks[0].vy, 1e-15)
        val same = tracker.update(100L, listOf(det(120.0, 51.0))) // same timestamp: velocity unchanged
        assertEquals(u2.tracks[0].vx, same.tracks[0].vx, 0.0)
        assertEquals(4, same.tracks[0].hits)
        assertEquals(120.0 + 0.15 * 20, same.tracks[0].displayCx(20.0), 1e-12)
    }

    @Test
    fun tracksExitAfterMoreThanMaxMissedFrames() {
        val tracker = Tracker(w, h, TriggerConfig())
        tracker.update(0L, listOf(det(100.0, 50.0)))
        for (i in 1..5) assertTrue(tracker.update(i * 33L, emptyList()).exits.isEmpty())
        val u = tracker.update(6 * 33L, emptyList())
        assertEquals(listOf(TrackExit(1, judged = false, confirmed = false)), u.exits)
        assertEquals(0, tracker.liveTracks)
        // ids keep increasing; reset() restarts them
        assertEquals(2, tracker.update(7 * 33L, listOf(det(10.0, 10.0))).assignments[0])
        tracker.reset()
        assertEquals(1, tracker.update(8 * 33L, listOf(det(10.0, 10.0))).assignments[0])
    }

    @Test
    fun bestCropNoticesReportInsertionsAndEvictions() {
        val tracker = Tracker(w, h, TriggerConfig(lineEnabled = false, steadyEnabled = false))
        val notices = ArrayList<BestCropNotice>()
        // moving towards the centre: every confirmed frame is a new best
        for (i in 0..5) notices += tracker.update(i * 33L, listOf(det(100.0 + 10 * i, 90.0, ref = i.toLong()))).bestCrops
        assertEquals(listOf(1L, 2L, 3L, 4L, 5L), notices.map { it.frameRef })
        assertEquals(listOf(null, null, null, 1L, 2L), notices.map { it.evictedFrameRef })
    }

    @Test
    fun countersQueueAndOverlayHelpers() {
        val c = LineCounters()
        c.onVerdict(1_000, Verdict.PASS)
        c.onVerdict(2_000, Verdict.DEFECT)
        c.onVerdict(30_000, Verdict.NOT_ENROLLED)
        c.onVerdict(61_500, Verdict.REFRAME)
        c.onExit(judged = false)
        c.onExit(judged = true)
        assertEquals(4, c.judged)
        assertEquals(1, c.pass)
        assertEquals(1, c.defect)
        assertEquals(1, c.notEnrolled)
        assertEquals(1, c.reframe)
        assertEquals(1, c.exitedUnjudged)
        assertEquals(2.0, c.partsPerMinute(62_000), 0.0) // 30 000 and 61 500 are inside (2 000, 62 000]
        assertEquals(3.0, c.partsPerMinute(61_500), 0.0) // 2 000, 30 000, 61 500
        assertFalse(JudgeQueue.lineTooFast(3))
        assertTrue(JudgeQueue.lineTooFast(4))
        assertEquals(103.0, Overlay.displayX(100.0, 0.1, 30.0), 1e-12)
        assertEquals(49.0, Overlay.displayY(50.0, -0.05, 20.0), 1e-12)
    }
}
