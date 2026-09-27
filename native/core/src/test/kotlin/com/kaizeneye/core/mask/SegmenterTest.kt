package com.kaizeneye.core.mask

import com.kaizeneye.core.image.RgbImage
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class SegmenterTest {

    private fun labelsOf(vararg rows: String, minBlob: Int = 1, margin: Int = 2): Pair<Segmenter, List<com.kaizeneye.core.model.Component>> {
        val (m, w, h) = MaskTestUtil.grid(*rows)
        val seg = Segmenter(w, h, MaskParams(minBlobPx = minBlob, borderMargin = margin))
        return seg to seg.label(m)
    }

    @Test
    fun labelsFollowTheFirstPixelInRowMajorOrder() {
        // A "U" whose arms start at (0,0) and (4,0) and a middle blob starting at (2,0): the U is label 1 (first pixel
        // (0,0)) even though its right arm opened a later provisional label; the middle blob is label 2.
        val (seg, comps) = labelsOf(
            "#.#.#",
            "#.#.#",
            "#...#",
            "#####",
        )
        assertEquals(2, comps.size)
        val l = seg.labels
        assertEquals(1, l[0])
        assertEquals(1, l[4])
        assertEquals(2, l[2])
        assertEquals(2, l[5 + 2])
        assertEquals(1, l[3 * 5 + 2])
        assertEquals(listOf(11, 2), comps.map { it.area })
    }

    @Test
    fun eightConnectivityJoinsDiagonals() {
        val (seg, comps) = labelsOf(
            "#....#",
            ".#..#.",
            "..##..",
            "......",
            "#.#.#.",
        )
        // the X shape is one component (diagonal and anti-diagonal links); the bottom row has three singletons
        assertEquals(4, comps.size)
        assertEquals(1, seg.labels[0])
        assertEquals(1, seg.labels[5])
        assertEquals(listOf(6, 1, 1, 1), comps.map { it.area })
        assertEquals(listOf(2, 3, 4), (1..3).map { seg.labels[4 * 6 + 2 * (it - 1)] })
    }

    @Test
    fun anAntiDiagonalStaircaseMergesLateIntoTheEarlierLabel() {
        // Right-to-left staircase: each row opens a label that only meets the previous one through the NE neighbour.
        val (seg, comps) = labelsOf(
            "....#",
            "...#.",
            "..#..",
            ".#...",
            "#....",
            "....#",
        )
        assertEquals(2, comps.size)
        for (i in 0 until 5) assertEquals(1, seg.labels[i * 5 + (4 - i)])
        assertEquals(2, seg.labels[5 * 5 + 4])
    }

    @Test
    fun specksBelowMinBlobAreRemovedAndTheRestRenumbered() {
        val (seg, comps) = labelsOf(
            "##......###",
            "##......###",
            "........###",
            ".#.........",
            "......####.",
            "......####.",
            minBlob = 5,
        )
        // raw components in first-pixel order: 2x2 (4 px), 3x3 (9 px), single (1 px), 2x4 (8 px)
        assertArrayEquals(intArrayOf(4, 9, 1, 8), seg.rawAreas())
        assertEquals(listOf(1, 2), comps.map { it.label })
        assertEquals(listOf(9, 8), comps.map { it.area })
        assertEquals(0, seg.labels[0])
        assertEquals(1, seg.labels[8])
        assertEquals(0, seg.labels[3 * 11 + 1])
        assertEquals(2, seg.labels[4 * 11 + 6])
        assertEquals(2, seg.componentCount)
    }

    @Test
    fun componentStatsCentroidBboxAndBorderFlag() {
        val (_, comps) = labelsOf(
            "..........",
            "..........",
            "..###.....",
            "..###....#",
            "..........",
            "..........",
            margin = 2,
        )
        val a = comps[0]
        assertEquals(6, a.area)
        assertEquals(2, a.minX)
        assertEquals(2, a.minY)
        assertEquals(4, a.maxX)
        assertEquals(3, a.maxY)
        assertEquals(3.5, a.cx, 0.0) // mean x = 3, + 0.5
        assertEquals(3.0, a.cy, 0.0) // mean y = 2.5, + 0.5
        assertFalse(a.touchesBorder) // x >= 2, y >= 2, x < 8, y < 4
        val b = comps[1]
        assertTrue(b.touchesBorder) // x = 9 >= W - 2
    }

    @Test
    fun labellingEqualsANaiveFloodFillOnRandomMasks() {
        val rnd = Random(3)
        repeat(150) {
            val w = 1 + rnd.nextInt(30)
            val h = 1 + rnd.nextInt(20)
            val p = rnd.nextDouble()
            val m = ByteArray(w * h) { if (rnd.nextDouble() < p) 1 else 0 }
            val minBlob = 1 + rnd.nextInt(6)
            val seg = Segmenter(w, h, MaskParams(minBlobPx = minBlob))
            val comps = seg.label(m)
            val (expected, areas) = MaskTestUtil.naiveLabels(m, w, h, minBlob)
            assertArrayEquals("labels ${w}x$h p=$p", expected, seg.labels)
            assertArrayEquals(areas.toIntArray(), seg.rawAreas())
            assertEquals(areas.count { it >= minBlob }, comps.size)
        }
    }

    @Test
    fun segmentFindsAPartOnTheSheetAndIgnoresNoise() {
        val w = 64
        val h = 48
        val img = MaskTestUtil.sheet(w, h)
        MaskTestUtil.fillRect(img, 20, 10, 40, 30, 40, 50, 60) // the part: 20x20
        img.setRgb(5, 5, 0, 0, 0) // a one-pixel speck
        val sheet = SheetModel(doubleArrayOf(200.0, 190.0, 180.0), doubleArrayOf(3.0, 3.0, 3.0))
        val seg = Segmenter(w, h, MaskParams())
        val comps = seg.segment(img, sheet)
        assertEquals(1, comps.size)
        val c = comps[0]
        assertEquals(400, c.area)
        assertEquals(20, c.minX)
        assertEquals(39, c.maxX)
        assertEquals(30.0, c.cx, 0.0)
        assertEquals(20.0, c.cy, 0.0)
        assertEquals(1, seg.fgRaw[5 * w + 5].toInt())
        assertEquals(0, seg.afterOpen[5 * w + 5].toInt())
        // d2 exactly as the formula
        val d2 = DoubleArray(w * h)
        seg.distanceSquared(img, sheet, d2)
        val z = doubleArrayOf((40 - 200.0) / 3.0, (50 - 190.0) / 3.0, (60 - 180.0) / 3.0)
        assertEquals(z[0] * z[0] + z[1] * z[1] + z[2] * z[2], d2[15 * w + 25], 0.0)
        assertEquals(0.0, d2[0], 0.0)
    }

    @Test
    fun thresholdIsStrictlyGreaterThanKSigmaSquared() {
        val img = RgbImage(3, 1)
        img.setRgb(0, 0, 112, 100, 100) // d = 12/3 = 4.0 exactly -> d² = 16 -> NOT foreground
        img.setRgb(1, 0, 113, 100, 100) // d² > 16 -> foreground
        img.setRgb(2, 0, 100, 100, 100)
        val seg = Segmenter(3, 1, MaskParams())
        seg.foreground(img, SheetModel(doubleArrayOf(100.0, 100.0, 100.0), doubleArrayOf(3.0, 3.0, 3.0)))
        assertArrayEquals(byteArrayOf(0, 1, 0), seg.fgRaw)
    }

    @Test
    fun timingSegment320x180() {
        val w = 320
        val h = 180
        val img = MaskTestUtil.sheet(w, h)
        val rnd = Random(5)
        for (i in img.data.indices) img.data[i] = ((img.data[i].toInt() and 0xFF) + rnd.nextInt(7) - 3).toByte() // sensor noise
        MaskTestUtil.fillDisc(img, 120.0, 90.0, 35.0, 60, 70, 80)
        MaskTestUtil.fillRect(img, 220, 40, 280, 70, 30, 30, 30)
        MaskTestUtil.fillRect(img, 0, 120, 30, 180, 150, 90, 70) // a "hand" at the border
        val sheet = SheetModel.fit(listOf(MaskTestUtil.sheet(w, h)))
        val seg = Segmenter(w, h, MaskParams())
        repeat(300) { seg.segment(img, sheet) }
        val t = DoubleArray(60)
        for (i in t.indices) {
            val t0 = System.nanoTime()
            seg.segment(img, sheet)
            t[i] = (System.nanoTime() - t0) / 1e6
        }
        t.sort()
        val comps = seg.segment(img, sheet)
        println("[timing] segment 320x180 (threshold+open+close+CC, ${comps.size} blobs): %.3f ms (JVM median)".format(t[t.size / 2]))
        assertEquals(3, comps.size)
        assertTrue(t[t.size / 2] < 40.0)
    }
}
