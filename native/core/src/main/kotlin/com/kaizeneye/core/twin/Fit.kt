package com.kaizeneye.core.twin

/*
 * The FIT gate (Kaizen Eye 2 accuracy package, docs/verification/accuracy-v2.md). NOT part of the frozen twin-spec: it is an
 * extra NOT_ENROLLED gate that only exists for Twins that carry a `fit.json` (older Twins and the golden Twins have none, so
 * every spec number stays exactly as verified).
 *
 * Why: the score gate `raw = max smoothed nearest-patch distance` (spec §6.3) answers "is any part of this crop unseen?",
 * which is right for a local defect but blind to a DIFFERENT object made of the same materials (a second black earbud
 * case matches every local patch of the first one). The fit `mean over the core of the nearest-patch distance` answers "does
 * the WHOLE crop look like the taught part?": it rises a little everywhere for a look-alike, and hardly moves for a small defect.
 * Measured on real earbud-case captures (DINOv2 + canonical rotation): AUROC 1.000 for every pair tried, against 0.97-0.99
 * for the max-based score.
 */

import com.kaizeneye.core.math.Stats
import kotlin.math.min

/** Numbers of the fit gate. */
data class FitParams(
    /** Without negatives: `τ_fit = factor · max(same-part fits)` (leave-segment-out fits of the keyframes ∪ calibration fits). */
    val factor: Double = 1.10,
    /** Negatives closer than `hi · overlapFactor` are treated as indistinguishable (kept out of the midpoint rule). */
    val overlapFactor: Double = 1.0,
)

/**
 * The fit threshold of a Twin. [pos] = same-part fits (teach leave-segment-out fits, then calibration fits), [neg] = fits of
 * known DIFFERENT objects (other Twins' keyframes, the negatives library). [margin] = `min(neg) − max(pos)` (null without
 * negatives); a margin ≤ 0 means at least one negative looks like the part (the gate cannot tell them apart).
 */
class FitThreshold(
    val tauFit: Double,
    val pos: DoubleArray,
    val neg: DoubleArray,
    val rule: String,
    val margin: Double?,
    /** Fits of the calibration parts (real good parts presented on the line); part of the same-part spread. */
    val cal: DoubleArray = DoubleArray(0),
) {
    /** The largest same-part fit seen (teach ∪ calibration). */
    val hi: Double get() = (pos + cal).filter { !it.isNaN() }.maxOrNull() ?: Double.NaN

    fun passes(fit: Double, sensitivity: Double = 1.0): Boolean = fit.isNaN() || fit <= tauFit * scale(sensitivity)

    companion object {
        const val RULE_LSO = "lso"
        const val RULE_MIDPOINT = "midpoint"
        const val RULE_OVERLAP = "overlap"

        /** The slider (sensitivity, 0.5-2.5× on τ) moves the fit gate by its square root, so one control stays usable. */
        fun scale(sensitivity: Double): Double = kotlin.math.sqrt(sensitivity)

        /**
         * `hi = max(pos)`. No negatives: `τ_fit = factor · hi`. With negatives, `lo = min(neg)`: `lo > hi` →
         * `τ_fit = min(factor · hi, (hi + lo) / 2)` (never looser than the midpoint to the nearest wrong object, always above
         * the same-part maximum); `lo ≤ hi` (a negative inside the same-part spread) → `factor · hi`, rule "overlap".
         */
        fun derive(pos: DoubleArray, neg: DoubleArray, params: FitParams = FitParams(), cal: DoubleArray = DoubleArray(0)): FitThreshold {
            val p = pos.filter { !it.isNaN() }.toDoubleArray()
            val c = cal.filter { !it.isNaN() }.toDoubleArray()
            require(p.isNotEmpty() || c.isNotEmpty()) { "fit threshold needs at least one same-part fit" }
            val hi = (p + c).max()
            val n = neg.filter { !it.isNaN() }.toDoubleArray()
            if (n.isEmpty()) return FitThreshold(params.factor * hi, p, n, RULE_LSO, null, c)
            val lo = n.min()
            val margin = lo - hi
            if (lo <= hi * params.overlapFactor) return FitThreshold(params.factor * hi, p, n, RULE_OVERLAP, margin, c)
            val tau = min(params.factor * hi, (hi + lo) / 2.0)          // always in (hi, lo)
            return FitThreshold(tau, p, n, RULE_MIDPOINT, margin, c)
        }

        /** Percentile helper for reports. */
        fun p5(v: DoubleArray): Double = Stats.percentile(v, 5.0)
    }
}
