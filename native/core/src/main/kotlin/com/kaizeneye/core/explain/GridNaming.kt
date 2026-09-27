package com.kaizeneye.core.explain

/**
 * Names a patch-grid cell by thirds of the crop: rows → top / middle / bottom band, columns → left / middle / right band,
 * using the cell centre (`band = floor(3·(row + 0.5)/gh)`), so a 40-row grid splits 13 / 14 / 13.
 */
object GridNaming {

    /** The nine region names, `[verticalBand][horizontalBand]`. */
    val REGIONS: List<List<String>> = listOf(
        listOf("upper-left", "top", "upper-right"),
        listOf("left", "centre", "right"),
        listOf("lower-left", "bottom", "lower-right"),
    )

    /** Band 0..2 of index [i] on an axis of [n] cells (by the cell centre). */
    fun band(i: Int, n: Int): Int {
        require(n > 0 && i in 0 until n) { "index $i outside 0 until $n" }
        return minOf(2, (3 * (2 * i + 1)) / (2 * n))
    }

    /** Region of patch ([row], [col]) on a [gh] × [gw] grid (0-based). */
    fun region(row: Int, col: Int, gh: Int, gw: Int): String = REGIONS[band(row, gh)][band(col, gw)]

    /** Vertical band of a region name: 0 = top, 1 = middle, 2 = bottom. */
    fun verticalBand(region: String): Int = REGIONS.indexOfFirst { region in it }.also {
        require(it >= 0) { "unknown region '$region'" }
    }

    /** Horizontal band of a region name: 0 = left, 1 = middle, 2 = right. */
    fun horizontalBand(region: String): Int = REGIONS[verticalBand(region)].indexOf(region)
}
