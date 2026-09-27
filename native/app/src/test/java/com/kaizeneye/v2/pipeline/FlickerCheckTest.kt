package com.kaizeneye.v2.pipeline

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class FlickerCheckTest {
    private val w = 320
    private val h = 180

    private fun frame(f: (x: Int, y: Int) -> Double) = DoubleArray(w * h) { i -> f(i % w, i / w) }

    @Test
    fun smoothVignettingIsNotFlicker() {
        val frames = List(20) { frame { x, y -> 140.0 - 0.002 * ((x - 160) * (x - 160) + (y - 90) * (y - 90)) } }
        val r = FlickerCheck.analyse(frames, w, h)
        assertFalse(r.describe(), r.flicker)
    }

    @Test
    fun driftingRollingShutterBandsAreFlicker() {
        // 50 Hz mains at 30 fps: the stripe phase moves ~0.33 cycle per frame.
        val frames = List(20) { k -> frame { _, y -> 140.0 + 6.0 * sin(2 * PI * (y / 30.0 + k * 0.33)) } }
        val r = FlickerCheck.analyse(frames, w, h, uniform = false)
        assertTrue(r.describe(), r.flicker)
    }

    @Test
    fun staticBandsOnAUniformSheetAreFlicker() {
        // 60 Hz mains at 30 fps can phase-lock: static stripes; on a plain sheet they can only be banding.
        val frames = List(20) { frame { _, y -> 140.0 + 6.0 * sin(2 * PI * y / 30.0) } }
        assertTrue(FlickerCheck.analyse(frames, w, h, uniform = true).flicker)
    }

    @Test
    fun staticHorizontalTextureOnATexturedBackgroundIsNotFlicker() {
        // The emulator's virtual room: shelves = static horizontal structure, not light flicker.
        val frames = List(20) { frame { _, y -> 120.0 + 25.0 * sin(2 * PI * y / 20.0) } }
        val r = FlickerCheck.analyse(frames, w, h, uniform = false)
        assertFalse(r.describe(), r.flicker)
    }

    @Test
    fun frameToFramePumpingIsFlicker() {
        val frames = List(20) { k -> frame { _, _ -> 140.0 * (1.0 + 0.05 * sin(k * 1.3)) } }
        val r = FlickerCheck.analyse(frames, w, h)
        assertTrue(r.describe(), r.flicker)
    }
}
