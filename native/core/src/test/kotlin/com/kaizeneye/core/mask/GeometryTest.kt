package com.kaizeneye.core.mask

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.GeometryFeatures
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class GeometryTest {

    /** Labels one shape given by a pixel-centre predicate; returns (labels, the single component). */
    private fun shape(w: Int, h: Int, inside: (Double, Double) -> Boolean): Pair<IntArray, Component> {
        val m = ByteArray(w * h)
        for (y in 0 until h) for (x in 0 until w) if (inside(x + 0.5, y + 0.5)) m[y * w + x] = 1
        val seg = Segmenter(w, h, MaskParams(minBlobPx = 1))
        val comps = seg.label(m)
        assertEquals(1, comps.size)
        return seg.labels.copyOf() to comps[0]
    }

    private fun feats(w: Int, h: Int, inside: (Double, Double) -> Boolean): GeometryFeatures {
        val (labels, c) = shape(w, h, inside)
        return Geometry.features(labels, w, h, c)
    }

    @Test
    fun discHu1IsOneOverTwoPi() {
        val f = feats(120, 120) { x, y -> (x - 60) * (x - 60) + (y - 60) * (y - 60) <= 40.0 * 40.0 }
        val expected = 1 / (2 * PI)
        assertTrue("hu1 ${f.hu1} vs $expected", abs(f.hu1 - expected) <= 0.02 * expected)
        assertEquals(1.0, f.aspect, 0.01)
        assertTrue("solidity ${f.solidity}", f.solidity > 0.95 && f.solidity <= 1.0)
    }

    @Test
    fun axisAlignedRectangleFillsItsBox() {
        val f = feats(100, 50) { x, y -> x in 20.0..80.0 && y in 15.0..35.0 } // 60 x 20 pixels
        assertEquals(1200.0, f.area, 0.0)
        assertEquals(1.0, f.fill, 1e-12)
        assertEquals(1.0 / 3.0, f.aspect, 0.05)
        assertEquals(1.0, f.solidity, 1e-12)
    }

    private fun rotatedRect(w: Int, h: Int, len: Double, wid: Double, t: Double): Pair<IntArray, Component> {
        val ct = cos(t)
        val st = sin(t)
        return shape(w, h) { x, y ->
            val dx = x - (w / 2 + 0.3)
            val dy = y - (h / 2 + 0.1)
            val u = dx * ct + dy * st
            val v = -dx * st + dy * ct
            abs(u) <= len / 2 && abs(v) <= wid / 2
        }
    }

    @Test
    fun rotatedRectangleStillFillsItsOrientedBox() {
        val t = PI / 6 // 30 degrees
        val (labels, c) = rotatedRect(180, 150, 100.0, 40.0, t)
        val f = Geometry.features(labels, 180, 150, c)
        assertTrue("fill ${f.fill}", abs(f.fill - 1.0) <= 0.06)
        assertEquals(0.4, f.aspect, 0.05)
        assertTrue("area ${f.area}", abs(f.area - 4000) < 80)
        assertEquals(t, Geometry.principalAngle(labels, 180, 150, c), PI / 180)
    }

    @Test
    fun specFillOfSmallRotatedPartsCarriesThePlusOneBias() {
        // §2.7 wu = max − min + 1 over pixel centres: exact for axis-aligned boxes, but on oblique edges the centres already
        // span (almost) the full length, so the +1 over-counts: fill ≈ (L/(L+1))·(S/(S+1)) = 0.937 for 60 × 20 at 30°.
        val (labels, c) = rotatedRect(140, 110, 60.0, 20.0, PI / 6)
        val f = Geometry.features(labels, 140, 110, c)
        assertTrue("fill ${f.fill}", f.fill in 0.92..0.96)
        assertEquals(1.0 / 3.0, f.aspect, 0.05)
    }

    @Test
    fun lShapeOfThreeUnitSquaresHasSoliditySixSevenths() {
        // pixels (0,0), (0,1), (1,1): corner points (0,0) (1,0) (0,2) (2,1) (2,2) ... hull = pentagon of area 3.5
        val w = 6
        val h = 6
        val labels = IntArray(w * h)
        labels[2 * w + 2] = 1
        labels[3 * w + 2] = 1
        labels[3 * w + 3] = 1
        val c = Component(1, 3, 2, 2, 3, 3, 2.0 + 1.0 / 3 + 0.5, 2.0 + 2.0 / 3 + 0.5, false)
        assertEquals(3.5, Geometry.hullArea(labels, w, h, c), 0.0)
        assertEquals(3.0 / 3.5, Geometry.features(labels, w, h, c).solidity, 0.0)
    }

    @Test
    fun singlePixelAndSquareAreDegenerateButDefined() {
        val labels = IntArray(9).also { it[4] = 7 }
        val one = Geometry.features(labels, 3, 3, Component(7, 1, 1, 1, 1, 1, 1.5, 1.5, false))
        assertEquals(1.0, one.area, 0.0)
        assertEquals(0.0, one.hu1, 0.0)
        assertEquals(1.0, one.aspect, 0.0) // λ1 == 0
        assertEquals(1.0, one.fill, 0.0)
        assertEquals(1.0, one.solidity, 0.0)
        val sq = feats(20, 20) { x, y -> x in 5.0..15.0 && y in 5.0..15.0 }
        assertEquals(1.0, sq.aspect, 1e-12)
        assertEquals(1.0, sq.fill, 1e-12)
        assertEquals(1.0, sq.solidity, 1e-12)
    }

    @Test
    fun ringHasLowSolidityAndHullAreaIgnoresTheHole() {
        val (labels, c) = shape(80, 80) { x, y ->
            val r2 = (x - 40) * (x - 40) + (y - 40) * (y - 40)
            r2 <= 30.0 * 30.0 && r2 >= 20.0 * 20.0
        }
        val f = Geometry.features(labels, 80, 80, c)
        val hull = Geometry.hullArea(labels, 80, 80, c)
        assertTrue("hull $hull", hull > PI * 29.5 * 29.5 && hull < PI * 31 * 31)
        assertEquals(f.area / hull, f.solidity, 1e-12)
        assertTrue(f.solidity < 0.6)
    }
}
