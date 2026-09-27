package com.kaizeneye.core.mask

// SPEC-QUESTION: (§2.1 ROI) the ROI format is not specified: here it is `[x0, y0, x1, y1)` in analysis pixels
//   (half-open, clamped to the frame); slow adaptation uses the same ROI as the fit.

import com.kaizeneye.core.image.RgbImage

/**
 * Empty-sheet colour model (spec §2.1): per channel `c ∈ {R, G, B}` a centre [mean] `μ_c` (the median of the sheet) and
 * a spread [sigma] `σ_c = max(1.4826 · MAD_c, sigmaMin)`, fitted over the region of interest [roi]
 * (`[x0, y0, x1, y1)` in analysis pixels, half-open; `null` = the whole frame).
 *
 * [mad] is the raw per-channel MAD of the fit (informational; null for a model built from stored values).
 *
 * [adapt] mutates [mean] in place; use the model from one thread (the analyzer) or hand the judge a [copy].
 */
class SheetModel(val mean: DoubleArray, val sigma: DoubleArray, val roi: IntArray? = null, val mad: DoubleArray? = null) {
    init {
        require(mean.size == 3 && sigma.size == 3) { "mean and sigma need 3 channels" }
        require(sigma.all { it > 0.0 }) { "sigma must be > 0, got ${sigma.toList()}" }
        roi?.let { require(it.size == 4 && it[2] > it[0] && it[3] > it[1]) { "roi must be [x0, y0, x1, y1) non-empty" } }
    }

    private val hist = IntArray(256)
    private val adaptMedian = DoubleArray(3)

    /**
     * Slow adaptation (spec §2.1, live only): on a frame with **no** component,
     * `μ_c ← (1−β)·μ_c + β·median_c(frame)` over the same ROI; `σ` unchanged. Zero allocation.
     */
    fun adapt(frame: RgbImage, beta: Double = DEFAULT_BETA) {
        require(beta in 0.0..1.0) { "beta must be in [0, 1], got $beta" }
        for (c in 0 until 3) {
            hist.fill(0)
            accumulate(frame, roi, c, hist)
            adaptMedian[c] = ByteHistogram.median(hist)
        }
        for (c in 0 until 3) mean[c] = (1 - beta) * mean[c] + beta * adaptMedian[c]
    }

    fun copy(): SheetModel = SheetModel(mean.copyOf(), sigma.copyOf(), roi?.copyOf(), mad?.copyOf())

    override fun toString(): String =
        "SheetModel(mean=${mean.toList()}, sigma=${sigma.toList()}, roi=${roi?.toList()})"

    companion object {
        const val MAD_TO_SIGMA = 1.4826
        const val DEFAULT_SIGMA_MIN = 3.0
        const val DEFAULT_BETA = 0.02

        /**
         * Spec §2.1 fit from `K ≥ 1` analysis frames: `μ_c` = median of all ROI values of channel `c` in all frames
         * (exact, from 256-bin histograms), `σ_c = max(1.4826 · MAD_c, sigmaMin)`.
         */
        fun fit(frames: List<RgbImage>, roi: IntArray? = null, sigmaMin: Double = DEFAULT_SIGMA_MIN): SheetModel {
            require(frames.isNotEmpty()) { "need at least one empty-sheet frame" }
            val w = frames[0].width
            val h = frames[0].height
            require(frames.all { it.width == w && it.height == h }) { "all frames must have the same size" }
            val r = clampRoi(roi, w, h)
            val mean = DoubleArray(3)
            val sigma = DoubleArray(3)
            val mads = DoubleArray(3)
            val hist = IntArray(256)
            val scratch = IntArray(ByteHistogram.DEVIATION_BINS)
            for (c in 0 until 3) {
                hist.fill(0)
                for (f in frames) accumulate(f, r, c, hist)
                val med = ByteHistogram.median(hist)
                val mad = ByteHistogram.mad(hist, med, scratch)
                mean[c] = med
                mads[c] = mad
                sigma[c] = maxOf(MAD_TO_SIGMA * mad, sigmaMin)
            }
            return SheetModel(mean, sigma, roi?.copyOf(), mads)
        }

        private fun clampRoi(roi: IntArray?, w: Int, h: Int): IntArray? {
            if (roi == null) return null
            require(roi.size == 4) { "roi must be [x0, y0, x1, y1)" }
            val c = intArrayOf(roi[0].coerceIn(0, w), roi[1].coerceIn(0, h), roi[2].coerceIn(0, w), roi[3].coerceIn(0, h))
            require(c[2] > c[0] && c[3] > c[1]) { "roi ${roi.toList()} is empty inside a ${w}x$h frame" }
            return c
        }

        /** Adds channel [c] of the ROI pixels of [img] to [hist]. */
        internal fun accumulate(img: RgbImage, roi: IntArray?, c: Int, hist: IntArray) {
            val w = img.width
            val h = img.height
            var x0 = 0
            var y0 = 0
            var x1 = w
            var y1 = h
            if (roi != null) {
                x0 = roi[0].coerceIn(0, w)
                y0 = roi[1].coerceIn(0, h)
                x1 = roi[2].coerceIn(0, w)
                y1 = roi[3].coerceIn(0, h)
            }
            val d = img.data
            for (y in y0 until y1) {
                var p = (y * w + x0) * 3 + c
                for (x in x0 until x1) {
                    hist[d[p].toInt() and 0xFF]++
                    p += 3
                }
            }
        }
    }
}
