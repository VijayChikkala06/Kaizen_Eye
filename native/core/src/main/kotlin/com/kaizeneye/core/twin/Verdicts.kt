package com.kaizeneye.core.twin

/*
 * SPEC-QUESTION: §9 borderline voting takes "the crop whose s is the median"; when several crops share the median
 * value the earliest crop wins (§0 ties rule). A buffered crop that cannot be scored (empty patch core) is not an
 * "available" s value. The identity / geometry gates of the primary crop are kept (only steps 3–5 are re-applied).
 */

import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.model.Verdict

/**
 * Everything the UI and the log need about one judged presentation (spec §9).
 * Fields that were not computed (the score of a REFRAME) are NaN / -1 / null.
 */
data class Judgement(
    val verdict: Verdict,
    /**
     * REFRAME: the §2.6 sanity name (e.g. "TOUCHES_BORDER"); NOT_ENROLLED: [VerdictEngine.REASON_IDENTITY],
     * [VerdictEngine.REASON_SHAPE] or [VerdictEngine.REASON_COVERAGE]; PASS / DEFECT: null.
     */
    val reason: String?,
    val sanity: SanityReason,
    /** `sim = max_k g_q · g_k`. */
    val sim: Double,
    val tauId: Double,
    val identityOk: Boolean,
    val geometryOk: Boolean,
    /** Failing geometry features in check order (reported even when identity already failed). */
    val geometryFailures: List<String>,
    val raw: Double,
    /** Normalised score `raw / (τ · sensitivity)`. */
    val s: Double,
    val tau: Double,
    val sensitivity: Double,
    val anomalousFraction: Double,
    val areaPct: Double,
    val coreCount: Int,
    val peakRow: Int,
    val peakCol: Int,
    val gh: Int,
    val gw: Int,
    /** Masked-smoothed distance map (`[gh·gw]`, NaN outside S) for the heat map; null for REFRAME. */
    val smoothed: DoubleArray?,
    /** Distance map `d(p)` (`[gh·gw]`); null for REFRAME. */
    val dmap: DoubleArray?,
    /** TOPK_VIEWS: the retrieved keyframes, best first; null for the coreset bank. */
    val views: IntArray? = null,
    /** Number of crops whose `s` entered the borderline vote (1 = no vote). */
    val votes: Int = 1,
    /** FIT statistic of the judged crop and its gate (Fit.kt); NaN when the Twin has no fit gate. */
    val fit: Double = Double.NaN,
    val tauFit: Double = Double.NaN,
    val fitOk: Boolean = true,
    /** CANONICAL rotation: 0 = the crop at θ was judged, 1 = the crop at θ + π. */
    val orientation: Int = 0,
) {
    /** True when the crop was scored (not REFRAME). */
    val scored: Boolean get() = !s.isNaN()

    /** Peak patch index `row·gw + col` (-1 when not scored). */
    val peakIndex: Int get() = if (peakRow < 0) -1 else peakRow * gw + peakCol
}

/** Numbers of the borderline vote (spec §9). */
data class VoteParams(
    /** Vote when the verdict is DEFECT with `lower < s ≤ upper`. */
    val lower: Double = 1.0,
    val upper: Double = 1.15,
    /** Extra buffered crops re-scored at most. */
    val maxExtra: Int = 2,
)

/**
 * Outcome of a borderline vote: [s] per crop (primary first, NaN = not available), the [chosen] crop index and the
 * final [judgement] (steps 3–5 re-applied with that crop's score).
 */
class VoteOutcome(val s: DoubleArray, val chosen: Int, val judgement: Judgement)

/** Verdict rules for one presentation (spec §9; order matters). */
object VerdictEngine {
    const val REASON_IDENTITY = "identity"
    const val REASON_SHAPE = "shape"
    const val REASON_COVERAGE = "coverage"
    /** The whole crop is further from the taught part than any same-part crop (a look-alike / another object). */
    const val REASON_FIT = "fit"

    /** Step 1: a presentation whose sanity failed (reason = the §2.6 name). */
    fun reframe(sanity: SanityReason, gh: Int = 0, gw: Int = 0): Judgement {
        require(sanity != SanityReason.OK) { "REFRAME needs a failing sanity reason" }
        return Judgement(
            verdict = Verdict.REFRAME, reason = sanity.name, sanity = sanity, sim = Double.NaN, tauId = Double.NaN,
            identityOk = false, geometryOk = false, geometryFailures = emptyList(), raw = Double.NaN, s = Double.NaN,
            tau = Double.NaN, sensitivity = Double.NaN, anomalousFraction = Double.NaN, areaPct = Double.NaN,
            coreCount = 0, peakRow = -1, peakCol = -1, gh = gh, gw = gw, smoothed = null, dmap = null,
        )
    }

    /**
     * Steps 2–5 for a sane, scored presentation: `sim < τ_id` → NOT_ENROLLED "identity" (shape failures still
     * reported); geometry fails → NOT_ENROLLED "shape"; `s ≤ 1` → PASS; `a > coverageCut` → NOT_ENROLLED "coverage";
     * else DEFECT with the peak and area %.
     */
    fun decide(
        sim: Double,
        tauId: Double,
        geometry: GeometryCheck,
        score: ScoreResult,
        coverageCut: Double,
        views: IntArray? = null,
        tauFit: Double = Double.NaN,
        orientation: Int = 0,
    ): Judgement {
        val idOk = sim >= tauId
        val fitOk = tauFit.isNaN() || score.fit.isNaN() || score.fit <= tauFit
        val (verdict, reason) = when {
            !idOk -> Verdict.NOT_ENROLLED to REASON_IDENTITY
            !geometry.ok -> Verdict.NOT_ENROLLED to REASON_SHAPE
            !fitOk -> Verdict.NOT_ENROLLED to REASON_FIT
            else -> scoreVerdict(score.s, score.anomalousFraction, coverageCut)
        }
        return Judgement(
            verdict = verdict, reason = reason, sanity = SanityReason.OK, sim = sim, tauId = tauId, identityOk = idOk,
            geometryOk = geometry.ok, geometryFailures = geometry.failures, raw = score.raw, s = score.s, tau = score.tau,
            sensitivity = score.sensitivity, anomalousFraction = score.anomalousFraction, areaPct = score.areaPct,
            coreCount = score.coreCount, peakRow = score.peakRow, peakCol = score.peakCol, gh = score.gh, gw = score.gw,
            smoothed = score.smoothed, dmap = score.dmap, views = views,
            fit = score.fit, tauFit = tauFit, fitOk = fitOk, orientation = orientation,
        )
    }

    /** Steps 3–5 on a score: PASS, NOT_ENROLLED ("coverage") or DEFECT; reason null for PASS / DEFECT. */
    fun scoreVerdict(s: Double, anomalousFraction: Double, coverageCut: Double): Pair<Verdict, String?> = when {
        s <= 1.0 -> Verdict.PASS to null
        anomalousFraction > coverageCut -> Verdict.NOT_ENROLLED to REASON_COVERAGE
        else -> Verdict.DEFECT to null
    }

    /** Borderline voting applies (spec §9): the verdict is DEFECT with `1 < s ≤ 1.15`. */
    fun needsVote(j: Judgement, params: VoteParams = VoteParams()): Boolean =
        j.verdict == Verdict.DEFECT && j.s > params.lower && j.s <= params.upper

    /**
     * Borderline vote (spec §9): [first] is the judged primary crop, [extras] the scores of up to
     * [VoteParams.maxExtra] more buffered crops of the same track (next best quality first; null = not available, e.g.
     * no core). The median of the available `s` values (1 to 3) becomes the final `s`, and steps 3–5 are re-applied
     * with that crop's `a` and peak: 3 values → the middle one, 2 values → the higher one; ties → the earlier crop.
     * The identity / geometry results stay those of [first].
     */
    fun vote(first: Judgement, extras: List<ScoreResult?>, coverageCut: Double, params: VoteParams = VoteParams()): VoteOutcome {
        require(first.scored) { "the first crop must be scored" }
        val used = extras.take(params.maxExtra)
        val s = DoubleArray(1 + used.size) { if (it == 0) first.s else used[it - 1]?.s ?: Double.NaN }
        val avail = s.indices.filter { !s[it].isNaN() }
        val chosen = when (avail.size) {
            1 -> 0
            2 -> if (s[avail[1]] > s[avail[0]]) avail[1] else avail[0]
            else -> {
                val median = avail.map { s[it] }.sorted()[1]
                avail.first { s[it] == median }                  // ties → the earliest crop with the median value
            }
        }
        val judgement = if (chosen == 0) {
            first.copy(votes = avail.size)
        } else {
            val sc = used[chosen - 1]!!
            val (verdict, reason) = scoreVerdict(sc.s, sc.anomalousFraction, coverageCut)
            first.copy(
                verdict = verdict, reason = reason, raw = sc.raw, s = sc.s, tau = sc.tau, sensitivity = sc.sensitivity,
                anomalousFraction = sc.anomalousFraction, areaPct = sc.areaPct, coreCount = sc.coreCount,
                peakRow = sc.peakRow, peakCol = sc.peakCol, smoothed = sc.smoothed, dmap = sc.dmap, votes = avail.size,
            )
        }
        val final = if (chosen == 0) {
            val (verdict, reason) = scoreVerdict(first.s, first.anomalousFraction, coverageCut)
            judgement.copy(verdict = verdict, reason = reason)
        } else judgement
        return VoteOutcome(s, chosen, final)
    }
}
