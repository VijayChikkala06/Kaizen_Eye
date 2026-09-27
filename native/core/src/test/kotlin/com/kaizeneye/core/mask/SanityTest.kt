package com.kaizeneye.core.mask

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.SanityReason
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SanityTest {

    private fun comp(label: Int, area: Int, minX: Int, minY: Int, maxX: Int, maxY: Int, border: Boolean = false) =
        Component(label, area, minX, minY, maxX, maxY, (minX + maxX + 1) / 2.0, (minY + maxY + 1) / 2.0, border)

    private val w = 320
    private val h = 180
    private val p = MaskParams()

    @Test
    fun mainObjectIsTheLargestNonBorderComponentWithTiesToTheLowestLabel() {
        val a = comp(1, 900, 10, 10, 39, 39)
        val b = comp(2, 1500, 0, 50, 49, 79, border = true)
        val c = comp(3, 900, 100, 10, 129, 39)
        assertEquals(1, ObjectPicker.main(listOf(a, b, c))?.label)
        assertEquals(3, ObjectPicker.main(listOf(b, c))?.label)
        assertNull(ObjectPicker.main(listOf(b)))
        assertNull(ObjectPicker.main(emptyList()))
    }

    @Test
    fun reasonsInOrder() {
        val ok = comp(1, 2000, 140, 70, 179, 119)
        assertEquals(SanityReason.NO_OBJECT, Sanity.check(null, emptyList(), w, h, p))
        // border beats size: a tiny border-touching blob is TOUCHES_BORDER
        val tinyBorder = comp(1, 40, 0, 0, 6, 6, border = true)
        assertEquals(SanityReason.TOUCHES_BORDER, Sanity.check(tinyBorder, listOf(tinyBorder), w, h, p))
        // 1 % of 57600 = 576 px: 575 is too small, 576 is fine
        val small = comp(1, 575, 100, 100, 123, 123)
        assertEquals(SanityReason.TOO_SMALL, Sanity.check(small, listOf(small), w, h, p))
        val edge = comp(1, 576, 100, 100, 123, 123)
        assertEquals(SanityReason.OK, Sanity.check(edge, listOf(edge), w, h, p))
        val large = comp(1, 46081, 2, 2, 317, 177)
        assertEquals(SanityReason.TOO_LARGE, Sanity.check(large, listOf(large), w, h, p))
        assertEquals(SanityReason.OK, Sanity.check(ok, listOf(ok), w, h, p))
    }

    @Test
    fun multipleNeedsAComparableBlobWithItsCentroidInsideTheCropSquare() {
        // chosen: bbox x 140..179, y 70..119 -> full-res bbox 560..720 x 280..480, side = 200*1.2 = 240,
        // centre (640, 380) -> square x0 = 520, y0 = 260 -> analysis [130, 190) x [65, 125)
        val chosen = comp(1, 2000, 140, 70, 179, 119)
        val inside = comp(2, 500, 184, 100, 187, 104) // centroid (186, 102.5) inside, area 500 >= 0.25*2000
        assertEquals(SanityReason.MULTIPLE, Sanity.check(chosen, listOf(chosen, inside), w, h, p))
        val tooSmall = comp(2, 499, 184, 100, 187, 104)
        assertEquals(SanityReason.OK, Sanity.check(chosen, listOf(chosen, tooSmall), w, h, p))
        val outside = comp(2, 900, 190, 100, 199, 109) // centroid x = 195 >= 190
        assertEquals(SanityReason.OK, Sanity.check(chosen, listOf(chosen, outside), w, h, p))
        val onEdge = Component(2, 900, 185, 100, 194, 109, 190.0, 105.0, false) // cx == x0 + side -> outside
        assertEquals(SanityReason.OK, Sanity.check(chosen, listOf(chosen, onEdge), w, h, p))
        val onLowEdge = Component(2, 900, 125, 60, 134, 69, 130.0, 65.0, false) // cx == x0, cy == y0 -> inside
        assertEquals(SanityReason.MULTIPLE, Sanity.check(chosen, listOf(chosen, onLowEdge), w, h, p))
        // a border-touching hand reaching into the square also counts
        val hand = Component(2, 3000, 150, 100, 200, 179, 170.0, 120.0, true)
        assertEquals(SanityReason.MULTIPLE, Sanity.check(chosen, listOf(chosen, hand), w, h, p))
        val (main, reason) = Sanity.checkMain(listOf(hand, chosen), w, h, p)
        assertEquals(chosen, main)
        assertEquals(SanityReason.MULTIPLE, reason)
    }
}
