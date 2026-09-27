package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.track.Axis
import org.junit.Assert.assertEquals
import org.junit.Test

class MotionAxisTest {
    @Test
    fun portraitCameraFramesSwapTheAxes() {
        // Portrait phone: CameraX delivers landscape sensor frames with rotation 90 → screen left↔right is sensor Y.
        assertEquals(Axis.Y, LineSession.sensorAxisFor("H", 90))
        assertEquals(Axis.X, LineSession.sensorAxisFor("V", 90))
        assertEquals(Axis.Y, LineSession.sensorAxisFor("H", 270))
    }

    @Test
    fun uprightFramesKeepTheAxes() {
        // Landscape clips (rotation 0, e.g. the synthetic self-test clip): screen left↔right is sensor X.
        assertEquals(Axis.X, LineSession.sensorAxisFor("H", 0))
        assertEquals(Axis.Y, LineSession.sensorAxisFor("V", 0))
        assertEquals(Axis.X, LineSession.sensorAxisFor("H", 180))
    }
}
