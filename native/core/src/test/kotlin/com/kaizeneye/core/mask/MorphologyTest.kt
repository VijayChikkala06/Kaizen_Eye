package com.kaizeneye.core.mask

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.util.Random

class MorphologyTest {

    private fun erode(m: ByteArray, w: Int, h: Int) = ByteArray(w * h).also { Morphology.erode(m, it, w, h, ByteArray(w * h)) }
    private fun dilate(m: ByteArray, w: Int, h: Int) = ByteArray(w * h).also { Morphology.dilate(m, it, w, h, ByteArray(w * h)) }

    @Test
    fun outOfImageNeighboursAreIgnored() {
        // An all-ones image stays all ones under erosion (the outside does not count as 0).
        val ones = ByteArray(6 * 4) { 1 }
        assertArrayEquals(ones, erode(ones, 6, 4))
        // A 2-px stripe on the left edge survives erosion as its outer column and is restored by the opening,
        // while the same stripe in the interior is erased.
        val (edge, w, h) = MaskTestUtil.grid(
            "##......",
            "##......",
            "##......",
            "##......",
        )
        val (edgeEroded) = MaskTestUtil.grid(
            "#.......",
            "#.......",
            "#.......",
            "#.......",
        )
        assertArrayEquals(edgeEroded, erode(edge, w, h))
        val opened = ByteArray(w * h)
        Morphology.open(edge, opened, w, h, ByteArray(w * h), ByteArray(w * h))
        assertArrayEquals(edge, opened)
        val (inner) = MaskTestUtil.grid(
            "...##...",
            "...##...",
            "...##...",
            "...##...",
        )
        assertArrayEquals(ByteArray(w * h), erode(inner, w, h))
        // Dilation of a corner pixel only reaches in-image neighbours.
        val (corner) = MaskTestUtil.grid(
            "#.......",
            "........",
            "........",
            "........",
        )
        val (cornerDilated) = MaskTestUtil.grid(
            "##......",
            "##......",
            "........",
            "........",
        )
        assertArrayEquals(cornerDilated, dilate(corner, w, h))
    }

    @Test
    fun openRemovesSpecksAndKeepsA3x3Block() {
        val (m, w, h) = MaskTestUtil.grid(
            "..........",
            ".##.......",
            ".##...###.",
            "......###.",
            "..#...###.",
            "..........",
        )
        val (expected) = MaskTestUtil.grid(
            "..........",
            "..........",
            "......###.",
            "......###.",
            "......###.",
            "..........",
        )
        val open = ByteArray(w * h)
        Morphology.open(m, open, w, h, ByteArray(w * h), ByteArray(w * h))
        assertArrayEquals(expected, open)
    }

    @Test
    fun closeFillsAPinholeAwayFromTheBorder() {
        val (m, w, h) = MaskTestUtil.grid(
            "...........",
            "...........",
            "...#####...",
            "...#####...",
            "...##.##...",
            "...#####...",
            "...#####...",
            "...........",
            "...........",
        )
        val (expected) = MaskTestUtil.grid(
            "...........",
            "...........",
            "...#####...",
            "...#####...",
            "...#####...",
            "...#####...",
            "...#####...",
            "...........",
            "...........",
        )
        val closed = ByteArray(w * h)
        Morphology.close(m, closed, w, h, ByteArray(w * h), ByteArray(w * h))
        assertArrayEquals(expected, closed)
    }

    @Test
    fun closeNextToTheBorderGrowsToTheEdge() {
        // Border rule consequence: the dilated block reaches row 0, and the erosion ignores the missing row -1.
        val (m, w, h) = MaskTestUtil.grid(
            ".......",
            "..###..",
            "..###..",
            "..###..",
            ".......",
        )
        val (expected) = MaskTestUtil.grid(
            "..###..",
            "..###..",
            "..###..",
            "..###..",
            "..###..",
        )
        val closed = ByteArray(w * h)
        Morphology.close(m, closed, w, h, ByteArray(w * h), ByteArray(w * h))
        assertArrayEquals(expected, closed)
    }

    @Test
    fun separablePassesEqualTheDirectDefinitionOnRandomMasks() {
        val rnd = Random(7)
        repeat(200) {
            val w = 1 + rnd.nextInt(12)
            val h = 1 + rnd.nextInt(12)
            val density = rnd.nextDouble()
            val m = ByteArray(w * h) { if (rnd.nextDouble() < density) 1 else 0 }
            assertArrayEquals("erode ${w}x$h", MaskTestUtil.naiveMorph(m, w, h, erode = true), erode(m, w, h))
            assertArrayEquals("dilate ${w}x$h", MaskTestUtil.naiveMorph(m, w, h, erode = false), dilate(m, w, h))
        }
    }
}
