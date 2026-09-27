package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Stats
import kotlin.math.max

/*
 * SPEC-QUESTION: §7.3 names the rule string only for the no-negatives case ("no-negatives"). With negatives the rule is
 * written as "midpoint" (same word as tools/lab/make_golden_twin.py; tauIdRule in twin.json).
 */

/** Numbers of the identity threshold (spec §7.3). */
data class IdentityParams(
    val posPercentile: Double = 5.0,
    val negPercentile: Double = 99.0,
    /** Without negatives: `τ_id = P5(pos) − max(minGap, madK · 1.4826 · MAD(pos))`. */
    val minGap: Double = 0.02,
    val madK: Double = 3.0,
    val madScale: Double = 1.4826,
)

/**
 * Identity threshold `τ_id` (spec §7.3). Identity passes iff `sim ≥ τ_id`, `sim = max_k g_q · g_k`.
 * [margin] = `P5(pos) − P99(neg)` (null without negatives); [overlap] = `margin ≤ 0` (null without negatives).
 */
data class IdentityThreshold(
    val tauId: Double,
    val rule: String,
    val margin: Double?,
    val overlap: Boolean?,
) {
    fun passes(sim: Double): Boolean = sim >= tauId

    companion object {
        const val RULE_NO_NEGATIVES = "no-negatives"
        const val RULE_NEGATIVES = "midpoint"

        /** Derives `τ_id` from positive similarities [pos] and per-negative similarities [neg] (spec §7.3). */
        fun derive(pos: DoubleArray, neg: DoubleArray, params: IdentityParams = IdentityParams()): IdentityThreshold {
            require(pos.isNotEmpty()) { "identity threshold needs at least one positive similarity" }
            val lo = Stats.percentile(pos, params.posPercentile)
            if (neg.isEmpty()) {
                val gap = max(params.minGap, params.madK * params.madScale * Stats.mad(pos))
                return IdentityThreshold(lo - gap, RULE_NO_NEGATIVES, null, null)
            }
            val hi = Stats.percentile(neg, params.negPercentile)
            val margin = lo - hi
            return IdentityThreshold((lo + hi) / 2.0, RULE_NEGATIVES, margin, margin <= 0.0)
        }
    }
}
