package com.kaizeneye.v2.pipeline

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Mains-flicker check on the empty sheet (plan: "frame-to-frame brightness over ~1–2 s at the short exposure; if banded,
 * fall back to 10 ms exposure and cap demo speed"). Symptoms of 50/60 Hz lighting at sub-10 ms exposures:
 *  - rolling-shutter BANDING: horizontal stripes → a ripple in the detrended row-mean profile. Bands from 50 Hz mains at
 *    30 fps DRIFT between frames (3.33 flicker cycles per frame), so the temporal change of the profile ([temporalBanding])
 *    is the robust signal; the static ripple ([bandingLevels]) is only trusted on a UNIFORM sheet, because any static
 *    horizontal structure in a textured background looks the same (found on the emulator's virtual room scene);
 *  - frame-to-frame PUMPING: the mean brightness varies between frames.
 * Input: grey analysis frames (row-major, values 0..255). Pure Kotlin, unit-tested on the JVM.
 */
object FlickerCheck {
    data class Result(
        val bandingLevels: Double,
        val temporalBanding: Double,
        val pumpingPct: Double,
        val uniform: Boolean,
        val flicker: Boolean,
    ) {
        fun describe(): String {
            val nums = "banding %.1f, drift %.1f, pumping %.1f %%".format(bandingLevels, temporalBanding, pumpingPct)
            return if (flicker) "FLICKER ($nums) → use 10 ms exposure" else "no flicker ($nums)"
        }
    }

    /**
     * [uniform] = the background is plain (small sheet σ), so a static row ripple can only be banding.
     * Thresholds: static ripple > 1.5 grey levels (uniform sheets only), profile drift > 1.0 level, pumping CV > 2 %.
     */
    fun analyse(
        frames: List<DoubleArray>,
        w: Int,
        h: Int,
        uniform: Boolean = true,
        bandingThreshold: Double = 1.5,
        driftThreshold: Double = 1.0,
        pumpingThresholdPct: Double = 2.0,
    ): Result {
        if (frames.isEmpty() || h < 8) return Result(0.0, 0.0, 0.0, uniform, false)
        val window = max(5, h / 12)
        val means = DoubleArray(frames.size)
        val profiles = ArrayList<DoubleArray>(frames.size)
        var banding = 0.0
        for ((fi, g) in frames.withIndex()) {
            val rows = DoubleArray(h)
            var total = 0.0
            for (y in 0 until h) {
                var s = 0.0
                val o = y * w
                for (x in 0 until w) s += g[o + x]
                rows[y] = s / w
                total += s
            }
            means[fi] = total / (w.toDouble() * h)
            val d = detrend(rows, window)
            profiles += d
            banding += rms(d)
        }
        banding /= frames.size
        // Temporal drift: per row, the std over frames of the detrended profile, averaged over rows.
        var drift = 0.0
        if (profiles.size >= 2) {
            for (y in 0 until h) {
                var m = 0.0
                for (p in profiles) m += p[y]
                m /= profiles.size
                var v = 0.0
                for (p in profiles) v += (p[y] - m) * (p[y] - m)
                drift += sqrt(v / profiles.size)
            }
            drift /= h
        }
        val mean = means.average()
        val cv = if (mean > 1e-9) sqrt(means.sumOf { (it - mean) * (it - mean) } / means.size) / mean * 100.0 else 0.0
        val flicker = drift > driftThreshold || cv > pumpingThresholdPct || (uniform && banding > bandingThreshold)
        return Result(banding, drift, cv, uniform, flicker)
    }

    /** [v] minus its centred moving average (removes vignetting / light gradients, keeps stripes). */
    fun detrend(v: DoubleArray, window: Int): DoubleArray {
        val n = v.size
        val half = window / 2
        return DoubleArray(n) { i ->
            val lo = max(0, i - half)
            val hi = minOf(n - 1, i + half)
            var s = 0.0
            for (k in lo..hi) s += v[k]
            v[i] - s / (hi - lo + 1)
        }
    }

    private fun rms(v: DoubleArray): Double = sqrt(v.sumOf { it * it } / max(1, v.size))

    /** Kept for callers of the previous API. */
    fun detrendedStd(v: DoubleArray, window: Int): Double = rms(detrend(v, window))
}
