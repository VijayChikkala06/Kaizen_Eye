package com.kaizeneye.core.image

import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.mask.Segmenter
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.track.Detections
import com.kaizeneye.core.track.Tracker
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test
import java.lang.management.ManagementFactory
import java.util.Random

/** Steady-state allocation of the 30 fps fast loop, measured with the HotSpot per-thread allocation counter. */
class AllocationTest {

    private val bean = ManagementFactory.getThreadMXBean() as? com.sun.management.ThreadMXBean

    @Suppress("DEPRECATION")
    private fun allocatedPerCall(runs: Int, block: () -> Unit): Long {
        val b = bean!!
        repeat(runs) { block() } // warm-up (JIT, lazily grown scratch)
        val id = Thread.currentThread().id
        val before = b.getThreadAllocatedBytes(id)
        repeat(runs) { block() }
        return (b.getThreadAllocatedBytes(id) - before) / runs
    }

    @Test
    fun fastLoopAllocatesOnlyItsSmallResults() {
        Assume.assumeTrue(bean != null && bean.isThreadAllocatedMemorySupported)
        bean!!.isThreadAllocatedMemoryEnabled = true
        val rnd = Random(1)
        val frame = RgbaFrame.allocate(1280, 720)
        rnd.nextBytes(frame.data)
        val small = RgbImage(320, 180)
        val grey = DoubleArray(320 * 180)
        val sheetImg = RgbImage(320, 180).also { it.fill(200, 190, 180) }
        val sheet = SheetModel.fit(listOf(sheetImg))
        val scene = sheetImg.copy()
        for (y in 60 until 120) for (x in 100 until 160) scene.setRgb(x, y, 40, 50, 60)
        val seg = Segmenter(320, 180, MaskParams())
        val tracker = Tracker(320, 180)

        val down = allocatedPerCall(200) { ImageOps.downscaleRgba(frame, 4, small) }
        val gs = allocatedPerCall(200) {
            ImageOps.grey(small, grey)
            ImageOps.laplacianVariance(grey, 320, 180, 99, 59, 161, 121)
        }
        val adaptOnly = allocatedPerCall(200) { sheet.adapt(sheetImg, 0.0) }
        val segment = allocatedPerCall(200) { seg.segment(scene, sheet) }
        var t = 0L
        val comps = seg.segment(scene, sheet)
        val trackStep = allocatedPerCall(200) {
            t += 33
            tracker.update(t, Detections.of(comps, grey, 320, 180, t))
        }
        println("[alloc] bytes/call: downscale=$down grey+sharpness=$gs adapt=$adaptOnly segment=$segment detections+tracker=$trackStep")
        assertTrue("downscale allocates $down B", down < 16)
        assertTrue("grey+sharpness allocates $gs B", gs < 16)
        assertTrue("adapt allocates $adaptOnly B", adaptOnly < 16)
        assertTrue("segment allocates $segment B", segment < 512) // result list + one Component
        assertTrue("tracker allocates $trackStep B", trackStep < 1024) // FrameUpdate + small lists + one TrackView
    }
}
