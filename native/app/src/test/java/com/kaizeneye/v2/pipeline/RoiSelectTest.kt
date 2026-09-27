package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.image.RgbaFrame
import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.mask.ObjectPicker
import com.kaizeneye.core.mask.Sanity
import com.kaizeneye.core.mask.Segmenter
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.model.SanityReason
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * "Circle to select" on a synthetic frame: a part split in two by a glare stripe (sheet-coloured), plus a BIGGER distractor
 * (a shadow / stand foot) elsewhere on the sheet. Without a circle the §2.5 picker takes the distractor; with a circle the
 * part is selected and its two pieces are merged into one object.
 */
class RoiSelectTest {
    private val w = 160
    private val h = 90
    private val params = MaskParams()

    private fun frame(): Analysed {
        val sheetImg = RgbImage(w, h)
        paint(sheetImg, 0, 0, w, h, 200, 200, 200)
        val sheet = SheetModel.fit(listOf(sheetImg, sheetImg, sheetImg))
        val img = RgbImage(w, h)
        paint(img, 0, 0, w, h, 200, 200, 200)
        // The part: 30×20 at (30,30), split by a 6-px glare band (sheet colour) at x = 42..47.
        paint(img, 30, 30, 60, 50, 40, 60, 90)
        paint(img, 42, 30, 48, 50, 200, 200, 200)
        // Distractor: 40×30 dark blob at (105,40) — bigger than the part, fully inside the view.
        paint(img, 105, 40, 145, 70, 30, 30, 30)
        val seg = Segmenter(w, h, params)
        val comps = seg.segment(img, sheet)
        val rgba = RgbaFrame(w * params.analysisFactor, h * params.analysisFactor, w * params.analysisFactor * 4, ByteArray(w * params.analysisFactor * h * params.analysisFactor * 4))
        return Analysed(0L, 0L, rgba, img, DoubleArray(w * h), seg.labels, comps, 0)
    }

    private fun paint(img: RgbImage, x0: Int, y0: Int, x1: Int, y1: Int, r: Int, g: Int, b: Int) {
        for (y in y0 until y1) for (x in x0 until x1) {
            val i = (y * img.width + x) * 3
            img.data[i] = r.toByte(); img.data[i + 1] = g.toByte(); img.data[i + 2] = b.toByte()
        }
    }

    private fun circle(cx: Double, cy: Double, r: Double): Roi {
        val k = 24
        return Roi(DoubleArray(k) { cx + r * kotlin.math.cos(2 * Math.PI * it / k) }, DoubleArray(k) { cy + r * kotlin.math.sin(2 * Math.PI * it / k) })
    }

    @Test
    fun withoutACircleTheBiggerDistractorWins() {
        val a = frame()
        assertEquals(3, a.components.size)
        val main = ObjectPicker.main(a.components)!!
        assertTrue("distractor centre", main.cx > 100)
    }

    @Test
    fun circledPartIsSelectedAndItsPiecesMerged() {
        val a = frame()
        val pick = RoiSelect.apply(a, circle(45.0, 40.0, 22.0), params)
        val obj = assertNotNullAndGet(pick.obj)
        assertEquals(30, obj.minX); assertEquals(59, obj.maxX)
        assertEquals(30, obj.minY); assertEquals(49, obj.maxY)
        assertEquals(24 * 20, obj.area)
        // Both pieces carry the merged label in the label map; the glare column does not.
        assertEquals(obj.label, a.labels[40 * w + 35])
        assertEquals(obj.label, a.labels[40 * w + 55])
        assertEquals(0, a.labels[40 * w + 44])
        // Sane: the distractor is outside the crop square, so no MULTIPLE.
        assertEquals(SanityReason.OK, Sanity.check(obj, pick.analysed.components, w, h, params))
    }

    @Test
    fun aTapOnThePartSelectsItToo() {
        val a = frame()
        val tap = Roi.fromStroke(doubleArrayOf(52.0, 52.5), doubleArrayOf(41.0, 41.0), w, h)!!
        assertTrue(tap.isTap)
        val obj = assertNotNullAndGet(RoiSelect.apply(a, tap, params).obj)
        assertTrue(obj.cx in 30.0..60.0)
    }

    @Test
    fun anEmptyCircleSelectsNothing() {
        val a = frame()
        assertNull(RoiSelect.apply(a, circle(80.0, 15.0, 8.0), params).obj)
    }

    @Test
    fun followerMovesWithThePart() {
        val a = frame()
        val f = RoiFollower(circle(45.0, 40.0, 22.0))
        assertNotNull(f.pick(a, params).obj)
        assertTrue(f.roi.contains(45.0, 40.0))
    }

    private fun assertNotNullAndGet(c: com.kaizeneye.core.model.Component?): com.kaizeneye.core.model.Component {
        assertNotNull(c)
        return c!!
    }
}
