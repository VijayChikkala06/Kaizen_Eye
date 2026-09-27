package com.kaizeneye.core.vision

import com.kaizeneye.core.mask.Geometry
import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.CropSquare
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class CanonicalCropTest {

    private val w = 120
    private val h = 90
    private val f = 4

    /** A filled rotated rectangle (length [len] × width [wid] analysis pixels, angle [deg] in image axes) as label 1. */
    private fun rect(cx: Double, cy: Double, len: Double, wid: Double, deg: Double): Pair<IntArray, Component> {
        val lab = IntArray(w * h)
        val c = cos(Math.toRadians(deg))
        val s = sin(Math.toRadians(deg))
        var minX = w; var minY = h; var maxX = -1; var maxY = -1; var n = 0
        var sx = 0.0; var sy = 0.0
        for (y in 0 until h) for (x in 0 until w) {
            val px = x + 0.5 - cx
            val py = y + 0.5 - cy
            val a = px * c + py * s
            val b = -px * s + py * c
            if (abs(a) <= len / 2 && abs(b) <= wid / 2) {
                lab[y * w + x] = 1
                minX = minOf(minX, x); maxX = maxOf(maxX, x); minY = minOf(minY, y); maxY = maxOf(maxY, y)
                n++; sx += x + 0.5; sy += y + 0.5
            }
        }
        return lab to Component(1, n, minX, minY, maxX, maxY, sx / n, sy / n, false)
    }

    @Test
    fun theSquareFollowsTheRotatedExtentSoALongPartIsNeverClipped() {
        for (deg in listOf(0.0, 25.0, 45.0, 70.0, 135.0)) {
            val (lab, comp) = rect(60.0, 45.0, 60.0, 12.0, deg)
            val theta = Geometry.principalAngle(lab, w, h, comp)
            val sq = CanonicalCrop.square(lab, w, h, comp, theta, f, 0.10)
            // length 60 analysis px = 240 full-res px; side = 1.2 × length (+ a pixel of outline) and the part is centred.
            assertTrue("deg $deg side ${sq.side}", sq.side in 1.2 * 236 .. 1.2 * 250)
            assertEquals(f * 60.0, sq.cx, 3.0)
            assertEquals(f * 45.0, sq.cy, 3.0)
            // The whole part lies inside the canonical square: every pixel maps to |a|,|b| <= side/2 in the rotated frame.
            val c = cos(theta)
            val s = sin(theta)
            for (i in lab.indices) if (lab[i] == 1) {
                val px = f * ((i % w) + 0.5) - sq.cx
                val py = f * ((i / w) + 0.5) - sq.cy
                assertTrue(abs(px * c + py * s) <= sq.side / 2 + 1e-6)
                assertTrue(abs(-px * s + py * c) <= sq.side / 2 + 1e-6)
            }
        }
    }

    @Test
    fun coverageIsCanonicalAtAnyPartAngle() {
        val gh = 20
        val gw = 20
        val ref = CanonicalCrop.coverage(rect(60.0, 45.0, 60.0, 12.0, 0.0).let { it.first }, w, h, 1,
            CanonicalCrop.square(rect(60.0, 45.0, 60.0, 12.0, 0.0).first, w, h, rect(60.0, 45.0, 60.0, 12.0, 0.0).second, 0.0, f, 0.10), 0.0, f, gh, gw)
        for (deg in listOf(30.0, 60.0, 120.0, 200.0)) {
            val (lab, comp) = rect(60.0, 45.0, 60.0, 12.0, deg)
            val theta = Geometry.principalAngle(lab, w, h, comp)
            val sq = CanonicalCrop.square(lab, w, h, comp, theta, f, 0.10)
            val a = CanonicalCrop.coverage(lab, w, h, 1, sq, theta, f, gh, gw)
            // The part is a horizontal band in the canonical frame for every angle (same shape as the axis-aligned reference).
            var diff = 0f
            for (i in a.indices) diff += abs(a[i] - ref[i])
            assertTrue("deg $deg total diff $diff", diff < 0.03f * a.size)
            // θ + π is the 180°-rotated view of the same crop.
            val b = CanonicalCrop.coverage(lab, w, h, 1, sq, theta + Math.PI, f, gh, gw)
            var flip = 0f
            for (r in 0 until gh) for (c in 0 until gw) flip += abs(b[r * gw + c] - a[(gh - 1 - r) * gw + (gw - 1 - c)])
            assertTrue("deg $deg flip diff $flip", flip < 0.02f * a.size)
        }
    }

    @Test
    fun atAngleZeroItEqualsThePlainCoverageOfTheSameSquare() {
        val (lab, comp) = rect(50.0, 40.0, 30.0, 20.0, 0.0)
        val sq = CanonicalSquare(4 * 50.0, 4 * 40.0, 4 * 44.0, 0.0)
        val canon = CanonicalCrop.coverage(lab, w, h, 1, sq, 0.0, f, 16, 16)
        val plain = PatchCoverage.coverage(lab, w, h, 1, CropSquare(sq.cx - sq.side / 2, sq.cy - sq.side / 2, sq.side), f, 16, 16)
        assertArrayEquals(plain, canon, 0f)
        assertTrue(comp.area > 0)
    }

    @Test
    fun cellCornersAreTheRotatedCell() {
        val sq = CanonicalSquare(100.0, 80.0, 40.0, Math.PI / 2)
        // θ = 90°: crop "right" (+a) points to the frame's +y, "down" (+b) to the frame's −x.
        val c = CanonicalCrop.cellCorners(sq, Math.PI / 2, 4, 4, 0, 0)
        assertEquals(100.0 - 20.0 * 0 + 20.0, c[0], 1e-9)    // top-left corner of the crop → (cx + 20, cy − 20)
        assertEquals(80.0 - 20.0, c[1], 1e-9)
        assertEquals(CanonicalCrop.foldPi(Math.PI + 0.1), 0.1 - 0.0, 1e-9)
    }
}
