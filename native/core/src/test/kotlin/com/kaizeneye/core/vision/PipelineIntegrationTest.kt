package com.kaizeneye.core.vision

import com.kaizeneye.core.image.ImageOps
import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.image.RgbaFrame
import com.kaizeneye.core.mask.Geometry
import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.mask.Sanity
import com.kaizeneye.core.mask.Segmenter
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.track.Detections
import com.kaizeneye.core.track.Tracker
import com.kaizeneye.core.track.Trigger
import com.kaizeneye.core.track.TriggerConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs

/**
 * The live pipeline end to end on synthetic 1280×720 RGBA frames (a dark disc sliding across a light sheet), exactly as
 * the app wires it: analyzer thread (downscale → segment → grey → detections → tracker, keeping what a best-crop notice
 * asks for) and judge (sanity → crop square → crop bytes → patch coverage → geometry).
 */
class PipelineIntegrationTest {

    private val fullW = 1280
    private val fullH = 720
    private val f = 4
    private val w = fullW / f
    private val h = fullH / f

    private var seed = 0x2545F4914F6CDD1DL
    private fun noise(): Int { // xorshift, ±2
        seed = seed xor (seed shl 13)
        seed = seed xor (seed ushr 7)
        seed = seed xor (seed shl 17)
        return ((seed ushr 33) % 5).toInt() - 2
    }

    private fun render(frame: RgbaFrame, partX: Double?, partY: Double, radius: Double) {
        val d = frame.data
        for (y in 0 until fullH) {
            var p = y * frame.rowStride
            val dy = y + 0.5 - partY
            for (x in 0 until fullW) {
                val dx = x + 0.5 - (partX ?: -1e9)
                val inPart = partX != null && dx * dx + dy * dy <= radius * radius
                if (inPart) {
                    d[p] = (60 + noise()).toByte(); d[p + 1] = (70 + noise()).toByte(); d[p + 2] = (85 + noise()).toByte()
                } else {
                    d[p] = (200 + noise()).toByte(); d[p + 1] = (190 + noise()).toByte(); d[p + 2] = (180 + noise()).toByte()
                }
                d[p + 3] = 255.toByte()
                p += 4
            }
        }
    }

    /** What the analyzer keeps for a buffered frame so the judge can run later (the camera frame is recycled). */
    private class Kept(val frame: RgbaFrame, val labels: IntArray, val components: List<Component>, val component: Component)

    @Test
    fun aPartCrossingTheLineIsJudgedFromItsBestBufferedFrame() {
        val params = MaskParams()
        val frame = RgbaFrame.allocate(fullW, fullH, fullW * 4 + 64)
        val small = RgbImage(w, h)
        val grey = DoubleArray(w * h)

        // Empty-sheet tap: fit the sheet model from two analysis frames.
        val empties = List(2) {
            render(frame, null, 0.0, 0.0)
            RgbImage(w, h).also { img -> ImageOps.downscaleRgba(frame, f, img) }
        }
        val sheet = SheetModel.fit(empties, sigmaMin = params.sigmaMin)

        val seg = Segmenter(w, h, params)
        val tracker = Tracker(w, h, TriggerConfig(steadyEnabled = false))
        val kept = HashMap<Long, Kept>() // frameRef → data kept at best-crop notice time
        var fired: Pair<Int, Long>? = null // (trackId, judge frameRef)
        var fastLoopNs = 0L
        val radius = 64.0
        for (i in 0 until 26) {
            render(frame, 120.0 + 40.0 * i, 360.0, radius)
            val t0 = System.nanoTime()
            ImageOps.downscaleRgba(frame, f, small)
            val comps = seg.segment(small, sheet)
            ImageOps.grey(small, grey)
            val update = tracker.update(i * 33L, Detections.of(comps, grey, w, h, i.toLong()))
            if (i >= 6) fastLoopNs += System.nanoTime() - t0
            for (n in update.bestCrops) {
                val d = update.assignments.indexOfFirst { it == n.trackId }
                val copy = RgbaFrame(frame.width, frame.height, frame.rowStride, frame.data.copyOf())
                kept[n.frameRef] = Kept(copy, seg.labels.copyOf(), comps, comps[d])
                n.evictedFrameRef?.let { kept.remove(it) }
            }
            for (trig in update.triggers) {
                assertEquals(Trigger.LINE, trig.trigger)
                fired = trig.trackId to trig.judgeFrame!!.frameRef
            }
            if (comps.isEmpty()) sheet.adapt(small)
        }
        println("[timing] fast loop (downscale+segment+grey+detections+tracker), 1280x720 in: %.3f ms/frame (JVM mean)"
            .format(fastLoopNs / 20 / 1e6))
        val (trackId, ref) = requireNotNull(fired) { "the part never fired" }
        assertEquals(1, trackId)
        assertTrue("kept frames ${kept.keys}", kept.size <= 3 && ref in kept)

        // Judge.
        val k = kept.getValue(ref)
        val c = k.component
        assertEquals(SanityReason.OK, Sanity.check(c, k.components, w, h, params))
        val crop = CropMath.square(c, f, fullW, fullH, params.cropMargin)
        val partX = 120.0 + 40.0 * ref
        assertEquals(partX, crop.cx, 4.0)
        assertEquals(360.0, crop.cy, 4.0)
        assertEquals(2 * radius * 1.2, crop.side, 10.0)
        val n = 320
        val bytes = ByteArray(n * n * 3)
        ImageOps.cropResize(k.frame, crop.x0, crop.y0, crop.side, n, bytes)
        val centre = ((n / 2) * n + n / 2) * 3
        assertEquals(60.0, (bytes[centre].toInt() and 0xFF).toDouble(), 3.0) // the part in the middle
        assertEquals(200.0, (bytes[0].toInt() and 0xFF).toDouble(), 3.0) // the sheet in the corner
        val cov = PatchCoverage.coverage(k.labels, w, h, c.label, crop, f, 40, 40)
        val sets = PatchCoverage.sets(cov, 40, 40, params.coreThreshold)
        // each analysis pixel spans (4·40/side)² grid cells, so the core (cells ≥ half covered) ≈ area · (160/side)²
        val expectedCore = c.area * (160.0 / crop.side) * (160.0 / crop.side)
        assertEquals(expectedCore, sets.coreCount.toDouble(), 0.05 * expectedCore)
        assertTrue(sets.sCount > sets.coreCount && sets.bCount > sets.sCount)
        val geo = Geometry.features(k.labels, w, h, c)
        assertTrue("hu1 ${geo.hu1}", abs(geo.hu1 - 1 / (2 * PI)) < 0.01)
    }
}
