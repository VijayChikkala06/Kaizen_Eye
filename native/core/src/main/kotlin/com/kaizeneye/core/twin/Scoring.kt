package com.kaizeneye.core.twin

import kotlin.math.ceil
import kotlin.math.max

/** Frame score rules (spec §6.3). */
enum class ScoreRule {
    /** `raw = max_{p ∈ S} sm(p)` — default, legacy-proven. */
    SMOOTHED_MAX,

    /** `raw = mean of the n largest d(p), p ∈ S`, `n = max(1, ceil(0.01·|S|))` — challenger. */
    TOP1_MEAN,
}

/**
 * Score of one crop against a Twin (spec §6.2–6.4).
 * [smoothed] is the masked-smoothed map `sm` over the whole grid (`[gh·gw]`, NaN outside S) for the heat map;
 * [dmap] is a copy of the distance map `d(p)` that was scored (NaN where it was not computed; only S matters).
 */
class ScoreResult(
    val rule: ScoreRule,
    val raw: Double,
    /** Normalised score `s = raw / (τ · sensitivity)`; `s > 1` = further from normal than calibrated. */
    val s: Double,
    val tau: Double,
    val sensitivity: Double,
    /** `argmax_{p ∈ S} sm(p)` (ties → lowest p). */
    val peak: Int,
    val peakRow: Int,
    val peakCol: Int,
    /** `|{p ∈ core : sm(p) > τ·sensitivity}| / |core|`. */
    val anomalousFraction: Double,
    /** `|{p ∈ core : sm(p) > τ·sensitivity}|`. */
    val anomalousCount: Int,
    /** `|core|`. */
    val coreCount: Int,
    /** `n` of TOP1_MEAN (0 for SMOOTHED_MAX). */
    val topN: Int,
    val gh: Int,
    val gw: Int,
    val smoothed: DoubleArray,
    val dmap: DoubleArray,
    /** FIT statistic (Fit.kt): mean of `d(p)` over the core (NaN when not computed). Not part of the spec. */
    val fit: Double = Double.NaN,
) {
    /** `100 · anomalousFraction` (spec §6.4). */
    val areaPct: Double get() = 100.0 * anomalousFraction
}

/** Masked smoothing and frame scores (spec §6.2–6.4). */
object Scoring {

    /** Fraction of S averaged by [ScoreRule.TOP1_MEAN] (spec §6.3). */
    const val DEFAULT_TOP_FRACTION: Double = 0.01

    /**
     * Masked 3×3 smoothing (spec §6.2): for p ∈ S, `sm(p) = mean{ d(q) : q ∈ N(p) ∩ S }` (float64, index order);
     * NaN outside S. Background patches never leak into the score. Returns [out].
     */
    fun smoothMasked(d: DoubleArray, gh: Int, gw: Int, s: BooleanArray, out: DoubleArray = DoubleArray(gh * gw)): DoubleArray {
        require(d.size >= gh * gw && s.size == gh * gw && out.size >= gh * gw) { "grid size mismatch" }
        for (r in 0 until gh) {
            for (c in 0 until gw) {
                val p = r * gw + c
                if (!s[p]) {
                    out[p] = Double.NaN
                    continue
                }
                var sum = 0.0
                var n = 0
                for (rr in maxOf(0, r - 1)..minOf(gh - 1, r + 1)) {
                    for (cc in maxOf(0, c - 1)..minOf(gw - 1, c + 1)) {
                        val q = rr * gw + cc
                        if (s[q]) {
                            sum += d[q]
                            n++
                        }
                    }
                }
                out[p] = sum / n
            }
        }
        return out
    }

    /**
     * `raw` frame score of a distance map (spec §6.3) with the crop's own sets. [smoothed] may be passed when already
     * computed (SMOOTHED_MAX). Only `d(p), p ∈ S` is read.
     */
    fun raw(
        d: DoubleArray,
        sets: PatchSets,
        rule: ScoreRule,
        smoothed: DoubleArray? = null,
        topFraction: Double = DEFAULT_TOP_FRACTION,
    ): Double {
        require(!sets.isEmpty) { "empty patch core (NO_CORE)" }
        return when (rule) {
            ScoreRule.SMOOTHED_MAX -> {
                val sm = smoothed ?: smoothMasked(d, sets.gh, sets.gw, sets.s)
                var best = Double.NEGATIVE_INFINITY
                // A non-finite distance anywhere in S poisons the score (NaN -> the verdict engine reframes, never PASS).
                for (p in sets.sIdx) if (sm[p].isNaN() || sm[p].isInfinite()) return Double.NaN
                for (p in sets.sIdx) if (sm[p] > best) best = sm[p]
                best
            }
            ScoreRule.TOP1_MEAN -> if (sets.sIdx.any { d[it].isNaN() || d[it].isInfinite() }) Double.NaN else topMean(d, sets.sIdx, topFraction)
        }
    }

    /**
     * Mean of the `n = max(1, ceil(fraction·|idx|))` largest `d(p)`, p ∈ [idx] (ties → lowest index), summed in index
     * order (spec §6.3 TOP1_MEAN).
     */
    fun topMean(d: DoubleArray, idx: IntArray, fraction: Double = DEFAULT_TOP_FRACTION): Double {
        require(idx.isNotEmpty()) { "empty set" }
        val n = topN(idx.size, fraction)
        val vals = DoubleArray(idx.size) { d[idx[it]] }
        require(vals.none { it.isNaN() }) { "NaN distance in the score set" }
        vals.sort()
        val cut = vals[vals.size - n]                   // n-th largest value
        var greater = 0
        for (v in vals) if (v > cut) greater++
        var equalNeeded = n - greater
        var sum = 0.0
        for (p in idx) {
            val v = d[p]
            if (v > cut) {
                sum += v
            } else if (v == cut && equalNeeded > 0) {
                sum += v
                equalNeeded--
            }
        }
        return sum / n
    }

    /** `n = max(1, ceil(fraction · size))` of TOP1_MEAN (capped at [size]). */
    fun topN(size: Int, fraction: Double = DEFAULT_TOP_FRACTION): Int =
        max(1, ceil(fraction * size).toInt()).coerceAtMost(size)

    /**
     * Full score of a crop (spec §6.2–6.4) from its distance map [d] (`[gh·gw]`, only S read), sets, `τ` and sensitivity.
     */
    fun score(
        d: DoubleArray,
        sets: PatchSets,
        tau: Double,
        sensitivity: Double = 1.0,
        rule: ScoreRule = ScoreRule.SMOOTHED_MAX,
        topFraction: Double = DEFAULT_TOP_FRACTION,
    ): ScoreResult {
        require(!sets.isEmpty) { "empty patch core (NO_CORE)" }
        val gh = sets.gh
        val gw = sets.gw
        val sm = smoothMasked(d, gh, gw, sets.s)
        val raw = raw(d, sets, rule, sm, topFraction)
        val limit = tau * sensitivity
        var peak = -1
        for (p in sets.sIdx) if (peak < 0 || sm[p] > sm[peak]) peak = p
        var hot = 0
        for (p in sets.coreIdx) if (sm[p] > limit) hot++
        return ScoreResult(
            rule = rule,
            raw = raw,
            s = raw / limit,
            tau = tau,
            sensitivity = sensitivity,
            peak = peak,
            peakRow = peak / gw,
            peakCol = peak % gw,
            anomalousFraction = hot.toDouble() / sets.coreIdx.size,
            anomalousCount = hot,
            coreCount = sets.coreIdx.size,
            topN = if (rule == ScoreRule.TOP1_MEAN) topN(sets.sIdx.size, topFraction) else 0,
            gh = gh,
            gw = gw,
            smoothed = sm,
            dmap = d.copyOf(gh * gw),
            fit = fit(d, sets),
        )
    }

    /** Mean of `d(p)` over the patch core (the FIT statistic); NaN if any core distance is NaN. */
    fun fit(d: DoubleArray, sets: PatchSets): Double {
        if (sets.coreIdx.isEmpty()) return Double.NaN
        var sum = 0.0
        for (p in sets.coreIdx) {
            if (d[p].isNaN() || d[p].isInfinite()) return Double.NaN
            sum += d[p]
        }
        return sum / sets.coreIdx.size
    }

    /**
     * Rotation challenger CANONICAL (spec §6.5): the component is scored at θ and θ + π and the lower `raw` wins
     * (ties → the first, θ).
     */
    fun lowerRaw(atTheta: ScoreResult, atThetaPlusPi: ScoreResult): ScoreResult =
        if (atThetaPlusPi.raw < atTheta.raw) atThetaPlusPi else atTheta
}
