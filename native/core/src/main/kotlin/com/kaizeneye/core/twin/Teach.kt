package com.kaizeneye.core.twin

/*
 * SPEC-QUESTION: §7 step 1 accepts frames whose sanity is OK; a frame passed as OK whose cov gives an empty patch core
 * is NO_CORE by §2.6 item 6, so it is not accepted either.
 * SPEC-QUESTION: §7 step 4 lists three stop rules; when |sel| = Kmax and all kept frames are selected at the same time,
 * the stop is reported as KMAX (text order). kcNext = the next largest min-distance at the stop (null for ALL).
 * SPEC-QUESTION: §7 step 7 does not say what happens when every keyframe is skipped (no bank row qualifies for any of
 * them): τ would be undefined, so teach fails with NO_LSO. Likewise fewer than 2 keyframes leave no identity positive
 * (§7 step 8): fail with TOO_FEW_KEYFRAMES (impossible with the default minKept/kMin).
 * SPEC-QUESTION: §7 step 6 lists bank rows, keyframe globals and keyframe feature maps as the rows rounded through
 * binary16; the negative globals are stored too (negatives.f16), so they are rounded before their similarities are
 * computed (the list is read as non-exhaustive: a reloaded Twin must reproduce negSims exactly; golden_twin.json agrees).
 * kfCov is also stored through binary16 (lossless: cov values are multiples of 1/16).
 * SPEC-QUESTION: twin.json teach.durationMs is not defined; it is the recording length (last − first frame time of
 * all frames seen).
 */

import com.kaizeneye.core.math.Half
import com.kaizeneye.core.math.Stats
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * One recorded teach frame (spec §7 input). For sane frames [cov] (`[gh·gw]`, spec §4), [geometry] (§2.7),
 * [features] (the crop's feature map, already in pipeline space) and [sharpness] (§1.3 on the grey crop) are required.
 */
class TeachFrame(
    val tMs: Long,
    val sanity: SanityReason,
    val sharpness: Double,
    val cov: FloatArray?,
    val geometry: GeometryFeatures?,
    val features: FeatureMap?,
)

/** All numbers of spec §7 (defaults = the spec). */
data class TeachParams(
    val keepPercentile: Double = 40.0,
    val minKept: Int = 8,
    val kMin: Int = 16,
    val kMax: Int = 32,
    val kcEps: Double = 0.002,
    val segments: Int = 5,
    val ratio: Double = 0.10,
    val minK: Int = 256,
    val maxRows: Int = 2400,
    val tauFactor: Double = 1.4,
    val coverageCut: Double = 0.5,
    val sensitivity: Double = 1.0,
    /** Patch core threshold (spec §4); must equal the pipeline's `mask.coreThreshold`. */
    val coreThreshold: Double = 0.5,
    /** Frame score rule of the LSO scores (spec §6.3); must equal the pipeline's `scoreRule`. */
    val scoreRule: ScoreRule = ScoreRule.SMOOTHED_MAX,
    val topFraction: Double = Scoring.DEFAULT_TOP_FRACTION,
    val geometry: GeometryParams = GeometryParams(),
    val identity: IdentityParams = IdentityParams(),
    /**
     * CPU threads for the coreset and (with the exact CPU backend) the LSO distance maps. Not a spec number: every
     * result is identical for any value (1 = the calling thread only).
     */
    val threads: Int = 1,
    /** Enables the FIT gate (Fit.kt): the Twin gets `thresholds.fit`, derived from the leave-segment-out fits. Off = spec-only Twin. */
    val fit: FitParams? = null,
) {
    companion object {
        /** Defaults with the pipeline's core threshold and score rule. */
        fun forPipeline(pipeline: PipelineInfo): TeachParams =
            TeachParams(coreThreshold = pipeline.mask.coreThreshold, scoreRule = pipeline.scoreRule)
    }
}

/** Why a teach failed. */
enum class TeachFailure { NOT_ENOUGH_FRAMES, TOO_FEW_KEYFRAMES, NO_LSO }

/** Why keyframe selection stopped (spec §7 step 4). */
enum class KeyframeStop { KMAX, EPS, ALL }

/** Leave-segment-out mode (spec §7 step 7): by segment, or the fallback by keyframe (< 2 segments used). */
enum class LsoMode { SEGMENT, KEYFRAME }

/** Everything teach decided, for the log and the golden tests. Arrays are empty when teach stopped before a step. */
class TeachDiagnostics(
    val framesSeen: Int,
    /** Indices (into the frame list) of accepted frames. */
    val accepted: IntArray,
    /** `P_keepPercentile` of the accepted frames' sharpness (NaN when none). */
    val sharpnessCut: Double,
    val kept: IntArray,
    /** Frame index of each keyframe, in selection order (keyframe index = position). */
    val keyframeFrames: IntArray,
    /** Min-distance `1 − g·g` at which keyframes 1..K−1 were selected. */
    val kcMinDist: DoubleArray,
    val kcStop: KeyframeStop?,
    val kcNext: Double?,
    val t0: Long,
    val t1: Long,
    /** Segment of each keyframe. */
    val segments: IntArray,
    val segmentsUsed: Int,
    val pooledRows: Int,
    /** Pooled-row indices of the bank, in selection order. */
    val coresetIndices: IntArray,
    val lsoMode: LsoMode?,
    /** LSO scores of the keyframes that were not skipped, keyframe order. */
    val lso: DoubleArray,
    val lsoKeyframes: IntArray,
    val positives: DoubleArray,
    val negativeSims: DoubleArray,
    /** Wall time per stage (ms), for the log. */
    val timingsMs: Map<String, Double>,
)

/** Result of [TeachBuilder.build]. */
sealed class TeachResult {
    abstract val diagnostics: TeachDiagnostics

    class Success(val twin: TwinModel, override val diagnostics: TeachDiagnostics) : TeachResult()

    class Failure(val reason: TeachFailure, override val diagnostics: TeachDiagnostics) : TeachResult()
}

/** Builds a Twin from a recording (spec §7). Deterministic. */
object TeachBuilder {

    /**
     * Teach (spec §7): accept → keep sharp frames → greedy k-center keyframes → segments → pooled B-set bank + coreset
     * (rounded through binary16) → leave-segment-out τ → identity positives + τ_id → geometry model.
     *
     * @param negatives negative-object globals (`[dim]` each) for τ_id (§7.3), rounded through binary16 like every
     *   stored row; may be empty.
     * @param knn backend for the LSO distance maps (LiteRT graph on the phone, exact CPU by default).
     * @param augmenter AUGMENT4 only: rotated views of each keyframe (§6.5).
     */
    fun build(
        frames: List<TeachFrame>,
        pipeline: PipelineInfo,
        meta: TwinMeta,
        params: TeachParams = TeachParams.forPipeline(pipeline),
        negatives: List<FloatArray> = emptyList(),
        knn: KnnBackend = Knn.CPU,
        augmenter: KeyframeAugmenter? = null,
    ): TeachResult {
        require(params.coreThreshold == pipeline.mask.coreThreshold) { "params.coreThreshold differs from the pipeline's" }
        require(params.scoreRule == pipeline.scoreRule) { "params.scoreRule differs from the pipeline's" }
        require(pipeline.rotation != RotationRule.AUGMENT4 || augmenter != null) { "AUGMENT4 needs a KeyframeAugmenter" }
        val gh = pipeline.gh
        val gw = pipeline.gw
        val dim = pipeline.dim
        val np = gh * gw
        val timings = LinkedHashMap<String, Double>()
        var clock = System.nanoTime()
        fun lap(stage: String) {
            val now = System.nanoTime()
            timings[stage] = (now - clock) / 1e6
            clock = now
        }

        // 1. Accept sane frames (with a non-empty patch core).
        val sets = arrayOfNulls<PatchSets>(frames.size)
        val acceptedList = ArrayList<Int>()
        for ((i, f) in frames.withIndex()) {
            if (f.sanity != SanityReason.OK) continue
            val fm = requireNotNull(f.features) { "sane frame $i has no feature map" }
            val cov = requireNotNull(f.cov) { "sane frame $i has no cov" }
            requireNotNull(f.geometry) { "sane frame $i has no geometry" }
            require(fm.gh == gh && fm.gw == gw && fm.dim == dim) { "frame $i feature map does not match the pipeline" }
            require(!f.sharpness.isNaN()) { "sane frame $i has NaN sharpness" }
            val s = PatchSets.fromCov(cov, gh, gw, params.coreThreshold)
            if (s.isEmpty) continue
            sets[i] = s
            acceptedList.add(i)
        }
        val accepted = acceptedList.toIntArray()

        // 3. Keep the sharp ones.
        val cut = if (accepted.isEmpty()) Double.NaN
        else Stats.percentile(DoubleArray(accepted.size) { frames[accepted[it]].sharpness }, params.keepPercentile)
        val kept = accepted.filter { frames[it].sharpness >= cut }.toIntArray()
        if (kept.size < params.minKept || kept.isEmpty()) {
            return TeachResult.Failure(
                TeachFailure.NOT_ENOUGH_FRAMES, failureDiagnostics(frames.size, accepted, cut, kept, timings),
            )
        }
        lap("accept")

        // 4. Keyframes: greedy k-center on the kept frames' globals (float64, not yet rounded).
        val nk = kept.size
        val g = Array(nk) { i ->
            val f = frames[kept[i]]
            Descriptor.global(f.features!!.data, np, dim, sets[kept[i]]!!.coreIdx)
        }
        val sel = ArrayList<Int>()               // positions in kept
        val kcMinDist = ArrayList<Double>()
        val minD = DoubleArray(nk) { Double.POSITIVE_INFINITY }
        val chosen = BooleanArray(nk)
        var first = 0
        for (i in 1 until nk) if (frames[kept[i]].sharpness > frames[kept[first]].sharpness) first = i
        fun addKeyframe(j: Int) {
            sel.add(j)
            chosen[j] = true
            for (i in 0 until nk) {
                if (chosen[i]) continue
                val d = 1.0 - dot(g[i], g[j])
                if (d < minD[i]) minD[i] = d
            }
        }
        addKeyframe(first)
        var stop: KeyframeStop
        var next: Double?
        while (true) {
            var j = -1
            for (i in 0 until nk) if (!chosen[i] && (j < 0 || minD[i] > minD[j])) j = i
            if (sel.size >= params.kMax) {
                stop = KeyframeStop.KMAX
                next = if (j >= 0) minD[j] else null
                break
            }
            if (j < 0) {
                stop = KeyframeStop.ALL
                next = null
                break
            }
            if (sel.size >= params.kMin && minD[j] < params.kcEps) {
                stop = KeyframeStop.EPS
                next = minD[j]
                break
            }
            kcMinDist.add(minD[j])
            addKeyframe(j)
        }
        val kCount = sel.size
        val kfFrame = IntArray(kCount) { kept[sel[it]] }
        lap("keyframes")

        // 5. Segments.
        val t0 = frames[kept.first()].tMs
        val t1 = frames[kept.last()].tMs
        val nSeg = params.segments
        val seg = IntArray(kCount) {
            val t = frames[kfFrame[it]].tMs
            min(nSeg - 1, floor(nSeg.toDouble() * (t - t0).toDouble() / ((t1 - t0).toDouble() + 1e-9)).toInt())
        }
        val segmentsUsed = seg.distinct().size

        // 6. Bank: pool the B-set patches of every keyframe (plus AUGMENT4 views), coreset, round through binary16.
        val views: List<List<AugmentedView>> = if (pipeline.rotation == RotationRule.AUGMENT4) {
            List(kCount) { augmenter!!.rotatedViews(kfFrame[it]) }
        } else List(kCount) { emptyList() }
        val viewSets = views.map { vs ->
            vs.map { v ->
                require(v.features.gh == gh && v.features.gw == gw && v.features.dim == dim) { "augmented view shape" }
                PatchSets.fromCov(v.cov, gh, gw, params.coreThreshold)
            }
        }
        var pooledRows = 0
        for (k in 0 until kCount) {
            pooledRows += sets[kfFrame[k]]!!.bIdx.size
            for (vs in viewSets[k]) pooledRows += vs.bIdx.size
        }
        val pooled = FloatArray(pooledRows * dim)
        val pooledKf = IntArray(pooledRows)
        var row = 0
        fun pool(data: FloatArray, bIdx: IntArray, k: Int) {
            for (p in bIdx) {
                System.arraycopy(data, p * dim, pooled, row * dim, dim)
                pooledKf[row] = k
                row++
            }
        }
        for (k in 0 until kCount) {
            pool(frames[kfFrame[k]].features!!.data, sets[kfFrame[k]]!!.bIdx, k)
            views[k].forEachIndexed { vi, v -> pool(v.features.data, viewSets[k][vi].bIdx, k) }
        }
        var bankK = min(params.maxRows, max(params.minK, floor(params.ratio * pooledRows).toInt()))
        bankK = min(bankK, pooledRows)
        lap("pool")
        val coreset = Coreset.greedy(pooled, pooledRows, dim, bankK, 0, params.threads)
        lap("coreset")
        val m = coreset.size
        val bank = FloatArray(m * dim)
        val bankKf = IntArray(m)
        val bankSeg = IntArray(m)
        for (i in 0 until m) {
            System.arraycopy(pooled, coreset[i] * dim, bank, i * dim, dim)
            bankKf[i] = pooledKf[coreset[i]]
            bankSeg[i] = seg[bankKf[i]]
        }
        Half.roundInPlace(bank)
        val globals = FloatArray(kCount * dim)
        for (k in 0 until kCount) {
            val gk = g[sel[k]]
            for (t in 0 until dim) globals[k * dim + t] = Half.round(gk[t])
        }
        val kfFeats = FloatArray(kCount * np * dim)
        val kfCov = FloatArray(kCount * np)
        for (k in 0 until kCount) {
            val f = frames[kfFrame[k]]
            System.arraycopy(f.features!!.data, 0, kfFeats, k * np * dim, np * dim)
            System.arraycopy(f.cov!!, 0, kfCov, k * np, np)
        }
        Half.roundInPlace(kfFeats)
        Half.roundInPlace(kfCov)
        lap("round")

        // 7. Leave-segment-out scores on the rounded keyframe maps (NaN = skipped: no bank row qualifies).
        val lsoMode = if (segmentsUsed < 2) LsoMode.KEYFRAME else LsoMode.SEGMENT
        val lsoAll = DoubleArray(kCount) { Double.NaN }
        val lsoFitAll = DoubleArray(kCount) { Double.NaN }
        fun lsoOf(k: Int) {
            val extra = FloatArray(m)
            var any = false
            for (b in 0 until m) {
                val excluded = if (lsoMode == LsoMode.KEYFRAME) bankKf[b] == k else bankSeg[b] == seg[k]
                extra[b] = if (excluded) KNN_MASK else 0f
                if (!excluded) any = true
            }
            if (!any) return
            val ks = PatchSets.fromCov(kfCov.copyOfRange(k * np, (k + 1) * np), gh, gw, params.coreThreshold)
            if (ks.isEmpty) return
            val idx = ks.sIdx
            val query = FloatArray(idx.size * dim)
            val out = DoubleArray(idx.size)
            for (i in idx.indices) System.arraycopy(kfFeats, (k * np + idx[i]) * dim, query, i * dim, dim)
            knn.minSq(query, idx.size, bank, m, dim, extra, out)
            val d = DoubleArray(np) { Double.NaN }
            for (i in idx.indices) d[idx[i]] = sqrt(max(0.0, out[i]))
            lsoAll[k] = Scoring.raw(d, ks, params.scoreRule, topFraction = params.topFraction)
            lsoFitAll[k] = Scoring.fit(d, ks)
        }
        // Keyframes are independent; only the stateless exact CPU backend is called from several threads.
        val lsoParts = if (knn === Knn.CPU) minOf(params.threads, kCount) else 1
        Workers(lsoParts).use { w -> w.run(lsoParts) { part -> for (k in part until kCount step lsoParts) lsoOf(k) } }
        val lsoKeyframes = lsoAll.indices.filter { !lsoAll[it].isNaN() }.toIntArray()
        val lso = DoubleArray(lsoKeyframes.size) { lsoAll[lsoKeyframes[it]] }
        lap("lso")

        // 8. Identity positives (on the rounded globals) and the negatives' similarities.
        val positives = DoubleArray(kCount) { k ->
            var best = Double.NEGATIVE_INFINITY
            for (j in 0 until kCount) {
                if (j == k) continue
                if (lsoMode == LsoMode.SEGMENT && seg[j] == seg[k]) continue
                best = max(best, Descriptor.dot(globals, k, globals, j, dim))
            }
            best
        }
        val negFlat = FloatArray(negatives.size * dim)
        negatives.forEachIndexed { i, v ->
            require(v.size == dim) { "negative $i has ${v.size} values, expected $dim" }
            System.arraycopy(v, 0, negFlat, i * dim, dim)
        }
        Half.roundInPlace(negFlat)                  // stored as negatives.f16
        val negSims = DoubleArray(negatives.size) { i ->
            var best = Double.NEGATIVE_INFINITY
            for (k in 0 until kCount) best = max(best, Descriptor.dot(negFlat, i, globals, k, dim))
            best
        }
        lap("identity")

        val diagnostics = TeachDiagnostics(
            framesSeen = frames.size, accepted = accepted, sharpnessCut = cut, kept = kept, keyframeFrames = kfFrame,
            kcMinDist = kcMinDist.toDoubleArray(), kcStop = stop, kcNext = next, t0 = t0, t1 = t1, segments = seg,
            segmentsUsed = segmentsUsed, pooledRows = pooledRows, coresetIndices = coreset, lsoMode = lsoMode, lso = lso,
            lsoKeyframes = lsoKeyframes, positives = positives, negativeSims = negSims, timingsMs = timings,
        )
        if (kCount < 2) return TeachResult.Failure(TeachFailure.TOO_FEW_KEYFRAMES, diagnostics)
        if (lso.isEmpty()) return TeachResult.Failure(TeachFailure.NO_LSO, diagnostics)

        val tauTeach = params.tauFactor * lso.max()
        val identity = IdentityThreshold.derive(positives, negSims, params.identity)
        // 9. Geometry model from all kept frames.
        val geometry = GeometryModel.fit(kept.map { frames[it].geometry!! }, params.geometry)

        val keyframes = List(kCount) { k ->
            KeyframeInfo(k, frames[kfFrame[k]].tMs, seg[k], frames[kfFrame[k]].sharpness, KeyframeInfo.fileName(k))
        }
        val twin = TwinModel(
            id = meta.id,
            name = meta.name,
            createdAtMs = meta.createdAtMs,
            pipeline = pipeline,
            fingerprint = pipeline.fingerprint(),
            teach = TeachStats(
                framesSeen = frames.size,
                framesAccepted = accepted.size,
                framesKept = kept.size,
                keyframes = kCount,
                segmentsUsed = segmentsUsed,
                durationMs = if (frames.isEmpty()) 0 else frames.last().tMs - frames.first().tMs,
                bankRows = m,
                pooledRows = pooledRows,
            ),
            keyframes = keyframes,
            globals = globals,
            bank = bank,
            bankKf = bankKf,
            bankSeg = bankSeg,
            kfFeats = kfFeats,
            kfCov = kfCov,
            lso = lso,
            positives = positives,
            negativeSims = negSims,
            negatives = negFlat,
            thresholds = Thresholds(
                tau = tauTeach,
                tauTeach = tauTeach,
                tauFactor = params.tauFactor,
                calibrated = false,
                tauId = identity.tauId,
                tauIdRule = identity.rule,
                identityMargin = identity.margin,
                coverageCut = params.coverageCut,
                sensitivity = params.sensitivity,
                geometry = geometry,
                fit = params.fit?.let { fp ->
                    val fits = lsoKeyframes.map { lsoFitAll[it] }.filter { !it.isNaN() }.toDoubleArray()
                    if (fits.isEmpty()) null else FitThreshold.derive(fits, DoubleArray(0), fp)
                },
            ),
            certificate = null,
        )
        lap("model")
        return TeachResult.Success(twin, diagnostics)
    }

    private fun dot(a: DoubleArray, b: DoubleArray): Double {
        var s = 0.0
        for (t in a.indices) s += a[t] * b[t]
        return s
    }

    private fun failureDiagnostics(
        framesSeen: Int,
        accepted: IntArray,
        cut: Double,
        kept: IntArray,
        timings: Map<String, Double>,
    ) = TeachDiagnostics(
        framesSeen = framesSeen, accepted = accepted, sharpnessCut = cut, kept = kept, keyframeFrames = IntArray(0),
        kcMinDist = DoubleArray(0), kcStop = null, kcNext = null, t0 = 0, t1 = 0, segments = IntArray(0),
        segmentsUsed = 0, pooledRows = 0, coresetIndices = IntArray(0), lsoMode = null, lso = DoubleArray(0),
        lsoKeyframes = IntArray(0), positives = DoubleArray(0), negativeSims = DoubleArray(0), timingsMs = timings,
    )
}
