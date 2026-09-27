package com.kaizeneye.core.twin

import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import java.util.Random
import kotlin.math.hypot

/** Synthetic Twin inputs for the unit tests (not a spec; just plausible, deterministic data). */
object Synth {

    fun pipeline(
        gh: Int,
        gw: Int,
        dim: Int,
        scoreRule: ScoreRule = ScoreRule.SMOOTHED_MAX,
        bankRule: BankRule = BankRule.CORESET,
        rotation: RotationRule = RotationRule.NONE,
    ) = PipelineInfo(
        backboneId = "synthetic", backboneSha256 = "0".repeat(64), inputSize = 8 * gh, gh = gh, gw = gw, dim = dim,
        knnGraphId = "knn_p${gh * gw}_d$dim", scoreRule = scoreRule, bankRule = bankRule, rotation = rotation,
    )

    /** Disc coverage: 1 inside radius r, 0.5 on the rim ring (r, r+1], 0 outside (multiples of 1/16 like §4). */
    fun discCov(gh: Int, gw: Int, cr: Double, cc: Double, r: Double): FloatArray = FloatArray(gh * gw) {
        val d = hypot(it / gw + 0.5 - cr, it % gw + 0.5 - cc)
        when {
            d <= r -> 1f
            d <= r + 1 -> 0.5f
            else -> 0f
        }
    }

    /**
     * An object pattern: one base vector per patch, `mean(t) + N(0, 1)` (post-ReLU features share a strong common
     * mean, which dominates the mean-pooled global descriptor; a different [mean] profile = a different object class).
     */
    fun pattern(gh: Int, gw: Int, dim: Int, seed: Long, mean: (Int) -> Float = { 3f }): FloatArray {
        val rnd = Random(seed)
        return FloatArray(gh * gw * dim) { mean(it % dim) + rnd.nextGaussian().toFloat() }
    }

    /** A view of [base] with Gaussian noise [sigma] (and optionally a defect on [defect] patches). */
    fun view(base: FloatArray, sigma: Double, rnd: Random, defect: IntArray = IntArray(0), dim: Int = 0, shift: Float = 3f): FloatArray {
        val f = FloatArray(base.size) { base[it] + (rnd.nextGaussian() * sigma).toFloat() }
        for (p in defect) for (t in 0 until dim) f[p * dim + t] += shift
        return f
    }

    fun geometry(rnd: Random, area: Double = 1500.0) = GeometryFeatures(
        area = area * (1 + 0.03 * rnd.nextGaussian()),
        fill = 0.62 + 0.01 * rnd.nextGaussian(),
        aspect = 0.9 + 0.01 * rnd.nextGaussian(),
        hu1 = 0.16 * (1 + 0.01 * rnd.nextGaussian()),
        solidity = 0.95 + 0.005 * rnd.nextGaussian(),
    )

    /**
     * [n] teach frames at [stepMs] spacing: object pattern [base] + noise, a disc coverage, every [insaneEvery]-th frame
     * insane (0 = none).
     */
    fun teachFrames(
        n: Int,
        gh: Int,
        gw: Int,
        dim: Int,
        seed: Long,
        base: FloatArray = pattern(gh, gw, dim, seed + 1000),
        sigma: Double = 0.25,
        stepMs: Long = 400,
        insaneEvery: Int = 7,
        radius: Double = minOf(gh, gw) / 4.0,
    ): List<TeachFrame> {
        val rnd = Random(seed)
        return List(n) { i ->
            if (insaneEvery > 0 && i % insaneEvery == insaneEvery - 1) {
                TeachFrame(i * stepMs, SanityReason.TOUCHES_BORDER, Double.NaN, null, null, null)
            } else {
                val cov = discCov(gh, gw, gh / 2.0 + rnd.nextDouble() - 0.5, gw / 2.0 + rnd.nextDouble() - 0.5, radius)
                TeachFrame(
                    tMs = i * stepMs,
                    sanity = SanityReason.OK,
                    sharpness = 200.0 + 800.0 * rnd.nextDouble(),
                    cov = cov,
                    geometry = geometry(rnd),
                    features = FeatureMap(gh, gw, dim, view(base, sigma, rnd)),
                )
            }
        }
    }

    fun meta(name: String = "synthetic") = TwinMeta("20260927-000000-abcd", name, 1_790_000_000_000L)
}
