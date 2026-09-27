package com.kaizeneye.v2.export

import com.kaizeneye.core.model.Component
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class ExportEncodingTest {

    @Test
    fun clipNamesFollowTheExportFormat() {
        assertEquals("good-20260927-101500", EvalExport.clipName("good_20260927-101500"))
        assertEquals("a-b-c", EvalExport.clipName("a b/c"))
        assertEquals("clip", EvalExport.clipName("___"))
        assertEquals(32, EvalExport.clipName("x".repeat(80)).length)
    }

    @Test
    fun maskRleAlternatesStartingWithZeroRun() {
        // 4x3 analysis image; component label 2 is an "L": (1,0),(1,1),(1,2),(2,2). bbox x 1..2, y 0..2 → 2x3 window.
        val w = 4
        val labels = IntArray(12)
        labels[0 * w + 1] = 2
        labels[1 * w + 1] = 2
        labels[2 * w + 1] = 2
        labels[2 * w + 2] = 2
        labels[0 * w + 2] = 7 // another component inside the bbox counts as 0
        val c = Component(label = 2, area = 4, minX = 1, minY = 0, maxX = 2, maxY = 2, cx = 1.75, cy = 1.75, touchesBorder = false)
        val m = MaskRle.of(labels, w, c)
        // window row-major: [1,0],[1,0],[1,1] → runs: 0-run 0, 1,1,1,1,2 ... = 0,1,1,1,1,2
        assertEquals(1, m.x); assertEquals(0, m.y); assertEquals(2, m.w); assertEquals(3, m.h)
        assertArrayEquals(intArrayOf(0, 1, 1, 1, 1, 2), m.rle)
        assertEquals(m.w * m.h, m.rle.sum())
    }
}
