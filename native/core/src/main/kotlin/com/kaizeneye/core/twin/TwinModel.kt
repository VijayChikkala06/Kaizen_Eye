package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Half
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import kotlinx.serialization.Serializable
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.math.max
import kotlin.math.sqrt

/** Teach statistics stored in twin.json `teach` (spec §8). */
@Serializable
data class TeachStats(
    val framesSeen: Int,
    val framesAccepted: Int,
    val framesKept: Int,
    val keyframes: Int,
    val segmentsUsed: Int,
    val durationMs: Long,
    val bankRows: Int,
    val pooledRows: Int,
)

/** One keyframe of a Twin (spec §8 `keyframes`). [file] is relative to the Twin folder. */
@Serializable
data class KeyframeInfo(val index: Int, val tMs: Long, val segment: Int, val sharpness: Double, val file: String) {
    companion object {
        fun fileName(index: Int): String = "keyframes/kf_%02d.jpg".format(Locale.ROOT, index)
    }
}

/** Thresholds of a Twin (spec §8 `thresholds`). `τ = τ_teach` until calibration. */
data class Thresholds(
    val tau: Double,
    val tauTeach: Double,
    val tauFactor: Double,
    val calibrated: Boolean,
    val tauId: Double,
    val tauIdRule: String,
    val identityMargin: Double?,
    val coverageCut: Double,
    /** Sensitivity slider (0.8–2.0× on τ), persisted per Twin. */
    val sensitivity: Double,
    val geometry: GeometryModel,
    /** FIT gate (Fit.kt); null = the Twin has none (spec-only Twin). Stored in `fit.json`, not in twin.json. */
    val fit: FitThreshold? = null,
)

/** Identity of a Twin: folder id, display name, creation time. */
data class TwinMeta(val id: String, val name: String, val createdAtMs: Long) {
    companion object {
        /** New id like `20260927-031500-a1b2` (UTC time + 4 random hex digits). */
        fun create(name: String, createdAtMs: Long = System.currentTimeMillis(), random: java.util.Random = java.util.Random()): TwinMeta =
            TwinMeta(newId(createdAtMs, random), name, createdAtMs)

        fun newId(createdAtMs: Long, random: java.util.Random = java.util.Random()): String {
            val fmt = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }
            return fmt.format(Date(createdAtMs)) + "-" + "%04x".format(Locale.ROOT, random.nextInt(0x10000))
        }
    }
}

/** Scoring numbers that are not part of the pipeline fingerprint (spec §6.3, §6.5, §9). */
data class JudgeParams(
    val topFraction: Double = Scoring.DEFAULT_TOP_FRACTION,
    val topKViews: Int = ViewRetrieval.DEFAULT_K,
    val vote: VoteParams = VoteParams(),
    /**
     * Compute `d(p)` for every patch (spec §6.1). When false only the score set S is searched (same score, less CPU
     * work; the heat map is defined on S only).
     */
    val fullDistanceMap: Boolean = true,
)

/** A crop to score: its feature map and patch coverage grid (spec §4). */
class CropInput(val features: FeatureMap, val cov: FloatArray)

/** Score of one crop against a Twin, before the verdict (spec §6). */
class CropScore(val sets: PatchSets, val global: DoubleArray, val sim: Double, val score: ScoreResult, val views: IntArray?)

/**
 * An in-memory Visual Twin (spec §7–§9). Feature rows are float32 values that are exactly binary16-representable
 * (rounded at teach time, spec §7 step 6), so a Twin saved and reloaded reproduces every number exactly.
 *
 * Layouts: [globals] `[K, D]`, [bank] `[M, D]`, [bankKf]/[bankSeg] `[M]`, [kfFeats] `[K, P, D]`, [kfCov] `[K, P]`,
 * [negatives] `[Nneg, D]`, [negativeSims] `[Nneg]`.
 */
class TwinModel(
    val id: String,
    val name: String,
    val createdAtMs: Long,
    val pipeline: PipelineInfo,
    val fingerprint: String,
    val teach: TeachStats,
    val keyframes: List<KeyframeInfo>,
    val globals: FloatArray,
    val bank: FloatArray,
    val bankKf: IntArray,
    val bankSeg: IntArray,
    val kfFeats: FloatArray,
    val kfCov: FloatArray,
    val lso: DoubleArray,
    val positives: DoubleArray,
    val negativeSims: DoubleArray,
    val negatives: FloatArray,
    val thresholds: Thresholds,
    val certificate: Certificate?,
) {
    val gh: Int get() = pipeline.gh
    val gw: Int get() = pipeline.gw
    val dim: Int get() = pipeline.dim
    val patches: Int get() = pipeline.gh * pipeline.gw
    val keyframeCount: Int get() = keyframes.size
    val bankRows: Int get() = bankKf.size
    val negativeCount: Int get() = negativeSims.size

    init {
        val k = keyframes.size
        val d = pipeline.dim
        val p = pipeline.gh * pipeline.gw
        require(globals.size == k * d) { "globals: ${globals.size} != $k x $d" }
        require(bank.size == bankKf.size * d && bankSeg.size == bankKf.size) { "bank/bankKf/bankSeg size mismatch" }
        require(kfFeats.size == k * p * d) { "kfFeats: ${kfFeats.size} != $k x $p x $d" }
        require(kfCov.size == k * p) { "kfCov: ${kfCov.size} != $k x $p" }
        require(negatives.size == negativeSims.size * d) { "negatives/negativeSims size mismatch" }
        require(bankKf.all { it in 0 until k }) { "bankKf out of range" }
    }

    /** Patch sets of every keyframe from [kfCov] (used by TOPK_VIEWS). */
    val keyframeSets: Array<PatchSets> by lazy {
        Array(keyframes.size) { k ->
            PatchSets.fromCov(kfCov.copyOfRange(k * patches, (k + 1) * patches), gh, gw, pipeline.mask.coreThreshold)
        }
    }

    /** `sim = max_k g_q · g_k` over the keyframe globals (spec §7.3). */
    fun similarity(gq: DoubleArray): Double {
        var best = Double.NEGATIVE_INFINITY
        for (k in keyframes.indices) best = max(best, Descriptor.dot(gq, globals, k))
        return best
    }

    /**
     * Scores one crop (spec §6): patch sets from [cov], global descriptor and identity similarity, distance map against
     * the bank ([bankRule]) and the frame score. Returns null when the patch core is empty (NO_CORE).
     * [features] must already be in pipeline space (DINOv2: L2-normalised per patch, [Descriptor.l2NormalizePatches]).
     */
    fun scoreCrop(
        features: FeatureMap,
        cov: FloatArray,
        knn: KnnBackend = Knn.CPU,
        sensitivity: Double = thresholds.sensitivity,
        scoreRule: ScoreRule = pipeline.scoreRule,
        bankRule: BankRule = pipeline.bankRule,
        params: JudgeParams = JudgeParams(),
    ): CropScore? {
        require(features.gh == gh && features.gw == gw && features.dim == dim) {
            "feature map ${features.gh}x${features.gw}x${features.dim} does not match the Twin's ${gh}x${gw}x$dim"
        }
        val sets = PatchSets.fromCov(cov, gh, gw, pipeline.mask.coreThreshold)
        if (sets.isEmpty) return null
        val p = patches
        val gq = Descriptor.global(features.data, p, dim, sets.coreIdx)
        val sim = similarity(gq)
        var views: IntArray? = null
        val (searchBank, rows) = when (bankRule) {
            BankRule.CORESET -> bank to bankRows
            BankRule.TOPK_VIEWS -> {
                val v = ViewRetrieval.topK(gq, globals, keyframeCount, params.topKViews)
                views = v
                val b = ViewRetrieval.gatherBank(kfFeats, keyframeSets, v, p, dim)
                b to b.size / dim
            }
        }
        val d = DoubleArray(p) { Double.NaN }
        if (params.fullDistanceMap) {
            val out = DoubleArray(p)
            knn.minSq(features.data, p, searchBank, rows, dim, null, out)
            for (i in 0 until p) d[i] = sqrt(max(0.0, out[i]))
        } else {
            val idx = sets.sIdx
            val q = FloatArray(idx.size * dim)
            for (i in idx.indices) System.arraycopy(features.data, idx[i] * dim, q, i * dim, dim)
            val out = DoubleArray(idx.size)
            knn.minSq(q, idx.size, searchBank, rows, dim, null, out)
            for (i in idx.indices) d[idx[i]] = sqrt(max(0.0, out[i]))
        }
        val score = Scoring.score(d, sets, thresholds.tau, sensitivity, scoreRule, params.topFraction)
        return CropScore(sets, gq, sim, score, views)
    }

    /**
     * Judges one presentation (spec §9): REFRAME on a failing [sanity] or an empty patch core (NO_CORE), else identity,
     * geometry and score gates in that order. [geometry] is required for a sane presentation.
     */
    fun judge(
        features: FeatureMap?,
        cov: FloatArray?,
        geometry: GeometryFeatures?,
        sanity: SanityReason,
        knn: KnnBackend = Knn.CPU,
        sensitivity: Double = thresholds.sensitivity,
        scoreRule: ScoreRule = pipeline.scoreRule,
        bankRule: BankRule = pipeline.bankRule,
        params: JudgeParams = JudgeParams(),
    ): Judgement {
        if (sanity != SanityReason.OK) return VerdictEngine.reframe(sanity, gh, gw)
        requireNotNull(features) { "a sane presentation needs its feature map" }
        requireNotNull(cov) { "a sane presentation needs its coverage grid" }
        requireNotNull(geometry) { "a sane presentation needs its geometry features" }
        val crop = scoreCrop(features, cov, knn, sensitivity, scoreRule, bankRule, params)
            ?: return VerdictEngine.reframe(SanityReason.NO_CORE, gh, gw)
        val geo = thresholds.geometry.check(geometry)
        return VerdictEngine.decide(crop.sim, thresholds.tauId, geo, crop.score, thresholds.coverageCut, crop.views, tauFitFor(sensitivity))
    }

    /** Effective FIT gate for the slider position (NaN when the Twin has no fit gate). */
    fun tauFitFor(sensitivity: Double = thresholds.sensitivity): Double =
        thresholds.fit?.let { it.tauFit * FitThreshold.scale(sensitivity) } ?: Double.NaN

    /**
     * Rotation challenger CANONICAL (spec §6.5): the component was cropped at its principal angle θ and at θ + π
     * ([atTheta], [atThetaPlusPi], both from `cropResizeRotated`); the crop with the lower `raw` (ties → θ) is judged.
     * REFRAME rules as in [judge].
     */
    fun judgeCanonical(
        atTheta: CropInput?,
        atThetaPlusPi: CropInput?,
        geometry: GeometryFeatures?,
        sanity: SanityReason,
        knn: KnnBackend = Knn.CPU,
        sensitivity: Double = thresholds.sensitivity,
        scoreRule: ScoreRule = pipeline.scoreRule,
        bankRule: BankRule = pipeline.bankRule,
        params: JudgeParams = JudgeParams(),
    ): Judgement {
        if (sanity != SanityReason.OK) return VerdictEngine.reframe(sanity, gh, gw)
        requireNotNull(atTheta) { "a sane presentation needs its θ crop" }
        requireNotNull(atThetaPlusPi) { "a sane presentation needs its θ + π crop" }
        requireNotNull(geometry) { "a sane presentation needs its geometry features" }
        val a = scoreCrop(atTheta.features, atTheta.cov, knn, sensitivity, scoreRule, bankRule, params)
        val b = scoreCrop(atThetaPlusPi.features, atThetaPlusPi.cov, knn, sensitivity, scoreRule, bankRule, params)
        // With a FIT gate the orientation that fits the taught part best is judged (a whole-crop statistic is more stable than the
        // max-based raw); without one, the spec rule (lower raw, ties → θ).
        val useFit = thresholds.fit != null && a != null && b != null && !a.score.fit.isNaN() && !b.score.fit.isNaN()
        val second = when {
            a == null && b == null -> return VerdictEngine.reframe(SanityReason.NO_CORE, gh, gw)
            a == null -> true
            b == null -> false
            useFit -> b.score.fit < a.score.fit
            else -> Scoring.lowerRaw(a.score, b.score) === b.score
        }
        val crop = if (second) b!! else a!!
        val geo = thresholds.geometry.check(geometry)
        return VerdictEngine.decide(
            crop.sim, thresholds.tauId, geo, crop.score, thresholds.coverageCut, crop.views, tauFitFor(sensitivity),
            orientation = if (second) 1 else 0,
        )
    }

    /**
     * Borderline vote (spec §9) for a DEFECT with `1 < s ≤ 1.15`: re-scores up to 2 more buffered crops of the same
     * track ([extras], next best quality first) and applies [VerdictEngine.vote]. Call only when
     * [VerdictEngine.needsVote] is true and the governor allows it.
     */
    fun vote(
        first: Judgement,
        extras: List<CropInput>,
        knn: KnnBackend = Knn.CPU,
        sensitivity: Double = first.sensitivity,
        scoreRule: ScoreRule = pipeline.scoreRule,
        bankRule: BankRule = pipeline.bankRule,
        params: JudgeParams = JudgeParams(),
    ): VoteOutcome {
        val scores = extras.take(params.vote.maxExtra).map {
            scoreCrop(it.features, it.cov, knn, sensitivity, scoreRule, bankRule, params)?.score
        }
        return VerdictEngine.vote(first, scores, thresholds.coverageCut, params.vote)
    }

    /** FIT statistic of a crop against this Twin's bank (negatives / other Twins' keyframes); NaN when the crop has no core. */
    fun fitOf(features: FeatureMap, cov: FloatArray, knn: KnnBackend = Knn.CPU): Double =
        scoreCrop(features, cov, knn, 1.0, params = JudgeParams(fullDistanceMap = false))?.score?.fit ?: Double.NaN

    /**
     * CANONICAL variant of [vote]: every extra crop is scored at θ and θ + π and the orientation with the lower fit (raw when
     * the Twin has no fit gate) is used.
     */
    fun voteCanonical(
        first: Judgement,
        extras: List<Pair<CropInput, CropInput>>,
        knn: KnnBackend = Knn.CPU,
        sensitivity: Double = first.sensitivity,
        scoreRule: ScoreRule = pipeline.scoreRule,
        bankRule: BankRule = pipeline.bankRule,
        params: JudgeParams = JudgeParams(),
    ): VoteOutcome {
        val scores = extras.take(params.vote.maxExtra).map { (t, tp) ->
            val a = scoreCrop(t.features, t.cov, knn, sensitivity, scoreRule, bankRule, params)?.score
            val b = scoreCrop(tp.features, tp.cov, knn, sensitivity, scoreRule, bankRule, params)?.score
            when {
                a == null -> b
                b == null -> a
                thresholds.fit != null && !a.fit.isNaN() && !b.fit.isNaN() -> if (b.fit < a.fit) b else a
                else -> if (Scoring.lowerRaw(a, b) === b) b else a
            }
        }
        return VerdictEngine.vote(first, scores, thresholds.coverageCut, params.vote)
    }

    /** Number of distinct segments that contain keyframes (`B_used`, spec §10.2). */
    val segmentsUsed: Int get() = keyframes.map { it.segment }.distinct().size

    /**
     * Calibrates this Twin (spec §10.1) and returns the calibrated copy (τ never lowered, τ_id and geometry re-derived,
     * certificate attached) together with the full result.
     */
    fun calibrate(
        samples: List<CalibrationSample>,
        params: CalibrationParams = CalibrationParams(),
        latenciesMs: DoubleArray = DoubleArray(0),
        accelerator: String? = null,
    ): Pair<TwinModel, CalibrationResult> {
        val r = Calibration.calibrate(
            thresholds.tauTeach, positives, negativeSims, thresholds.geometry, segmentsUsed, samples, params,
            latenciesMs, accelerator,
        )
        // The FIT gate learns from the real good parts too (their fits join the same-part spread; negatives are kept).
        val fit = thresholds.fit?.let { f ->
            val cal = (f.cal.toList() + samples.filter { it.sanityOk }.mapNotNull { it.fit }).toDoubleArray()
            FitThreshold.derive(f.pos, f.neg, FitParams(), cal)
        }
        val th = thresholds.copy(
            tau = r.tauCal, calibrated = true, tauId = r.identity.tauId, tauIdRule = r.identity.rule,
            identityMargin = r.identity.margin, geometry = r.geometry, fit = fit,
        )
        return copy(thresholds = th, certificate = r.certificate) to r
    }

    /**
     * Copy with a new negatives library (spec §7.3): each global is rounded through binary16 (as stored),
     * `negativeSims[i] = max_k g_neg_i · g_k` over the keyframe globals, and τ_id is re-derived from the teach
     * [positives] and these similarities. τ and `calibrated` are kept; a calibrated Twin loses its certificate
     * (its identity gate counts are outdated — re-calibrate).
     */
    fun withNegatives(negativeGlobals: List<FloatArray>, params: IdentityParams = IdentityParams()): TwinModel {
        val d = dim
        val neg = FloatArray(negativeGlobals.size * d)
        negativeGlobals.forEachIndexed { i, g ->
            require(g.size == d) { "negative $i has ${g.size} values, expected $d" }
            g.copyInto(neg, i * d)
        }
        Half.roundInPlace(neg)
        val sims = DoubleArray(negativeGlobals.size) { i ->
            var best = Double.NEGATIVE_INFINITY
            for (k in 0 until keyframeCount) best = max(best, Descriptor.dot(neg, i, globals, k, d))
            best
        }
        val identity = IdentityThreshold.derive(positives, sims, params)
        val th = thresholds.copy(tauId = identity.tauId, tauIdRule = identity.rule, identityMargin = identity.margin)
        return TwinModel(
            id, name, createdAtMs, pipeline, fingerprint, teach, keyframes, globals, bank, bankKf, bankSeg, kfFeats,
            kfCov, lso, positives, sims, neg, th, if (thresholds.calibrated) null else certificate,
        )
    }

    /**
     * Copy with the fits of known DIFFERENT objects ([negativeFits]: other Twins' keyframes, the negatives library scored
     * against this Twin's bank) and the fit threshold re-derived (midpoint rule). No-op for a Twin without a fit gate.
     */
    fun withFitNegatives(negativeFits: DoubleArray, params: FitParams = FitParams()): TwinModel {
        val f = thresholds.fit ?: return this
        return copy(thresholds = thresholds.copy(fit = FitThreshold.derive(f.pos, negativeFits, params, f.cal)))
    }

    /** Copy with a new sensitivity (slider, 0.5–2.5× on τ). */
    fun withSensitivity(sensitivity: Double): TwinModel = copy(thresholds = thresholds.copy(sensitivity = sensitivity))

    /** Copy with a new display name. */
    fun withName(name: String): TwinModel = copy(name = name)

    /** Within-part bound text for an uncalibrated Twin (spec §10.2). */
    fun withinPartLine(conf: Double = 0.95): String = CertificateText.withinPart(segmentsUsed, conf)

    fun copy(
        id: String = this.id,
        name: String = this.name,
        thresholds: Thresholds = this.thresholds,
        certificate: Certificate? = this.certificate,
    ): TwinModel = TwinModel(
        id, name, createdAtMs, pipeline, fingerprint, teach, keyframes, globals, bank, bankKf, bankSeg, kfFeats, kfCov,
        lso, positives, negativeSims, negatives, thresholds, certificate,
    )
}
