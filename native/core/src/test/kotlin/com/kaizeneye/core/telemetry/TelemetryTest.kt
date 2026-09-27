package com.kaizeneye.core.telemetry

import com.kaizeneye.core.math.Stats
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class TelemetryTest {

    @Test
    fun ringBufferStatsUseTheSpecPercentileOverTheLastSamples() {
        val s = LatencyStats(capacity = 512)
        assertEquals(0, s.count)
        assertTrue(s.p50().isNaN())
        val rnd = Random(8)
        val all = DoubleArray(700) { 5 + rnd.nextDouble() * 20 }
        all.forEach { s.add(it) }
        val last = all.copyOfRange(700 - 512, 700)
        assertEquals(512, s.count)
        assertEquals(700L, s.total)
        assertEquals(Stats.percentile(last, 50.0), s.p50(), 0.0)
        assertEquals(Stats.percentile(last, 95.0), s.p95(), 0.0)
        assertEquals(last.max(), s.max(), 0.0)
        assertEquals(Stats.mean(last), s.mean(), 1e-12)
        val sum = s.summary()
        assertEquals(512, sum.count)
        assertEquals(s.p95(), sum.p95, 0.0)
        s.add(Double.NaN)
        assertEquals(700L, s.total)
    }

    @Test
    fun smallWindowsAndTheExampleFromSpecZero() {
        val s = LatencyStats(capacity = 4)
        listOf(10.0, 20.0, 30.0, 40.0, 50.0).forEach { s.add(it) } // keeps 20..50
        assertEquals(35.0, s.p50(), 0.0)
        assertEquals(20.0 + 0.95 * 3 * 10, s.p95(), 1e-12)
        assertEquals(50.0, s.max(), 0.0)
    }

    @Test
    fun stageTimesKeepOneRingPerStage() {
        val st = StageTimes()
        st.record("mask", 2.0)
        st.record("mask", 4.0)
        st.record("backbone", 7.0)
        val v = st.time("knn") { 42 }
        assertEquals(42, v)
        val snap = st.snapshot()
        assertEquals(listOf("mask", "backbone", "knn"), snap.keys.toList())
        assertEquals(3.0, snap.getValue("mask").p50, 0.0)
        assertEquals(1, snap.getValue("knn").count)
    }

    @Test
    fun powerFromBatteryReadings() {
        val dis = PowerMath.wattsFrom(-250_000, 3_900, dischargeSign = -1)
        assertEquals(0.975, dis.watts, 1e-12)
        assertFalse(dis.charging)
        val chg = PowerMath.wattsFrom(500_000, 4_200, dischargeSign = -1)
        assertEquals(2.1, chg.watts, 1e-12)
        assertTrue(chg.charging)
        val other = PowerMath.wattsFrom(300_000, 4_000, dischargeSign = 1) // a phone that reports + while discharging
        assertFalse(other.charging)
        assertEquals(1.2, other.watts, 1e-12)
        assertFalse(PowerMath.wattsFrom(0, 4_000, -1).charging)
    }
}
