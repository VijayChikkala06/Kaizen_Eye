package com.kaizeneye.core.twin

/*
 * SPEC-QUESTION: §6.5 AUGMENT4 says keyframes are added to the bank at 0°, 90°, 180°, 270° before the coreset but not
 * in which order. Chosen: per keyframe (index order), its 0° B-set rows first, then the rotated views in the order the
 * KeyframeAugmenter returns them (90°, 180°, 270°); every row keeps its keyframe index (bankKf) and segment.
 * SPEC-QUESTION: §7.7 defines the leave-segment-out scores on the coreset bank only; under TOPK_VIEWS they are still
 * computed that way (τ is shared by both bank rules).
 */

import com.kaizeneye.core.model.FeatureMap

/** One rotated view of a keyframe crop (AUGMENT4, spec §6.5): its feature map and patch coverage grid. */
class AugmentedView(val features: FeatureMap, val cov: FloatArray)

/**
 * AUGMENT4 hook (spec §6.5): supplies the rotated views (90°, 180°, 270°) of a teach frame that became a keyframe.
 * The caller crops/rotates and runs the backbone; [frameIndex] is the index into the teach frame list.
 */
fun interface KeyframeAugmenter {
    fun rotatedViews(frameIndex: Int): List<AugmentedView>
}

/** Bank challenger TOPK_VIEWS (spec §6.5): retrieve the keyframes most similar to the query, then exact k-NN. */
object ViewRetrieval {

    /** Default number of retrieved keyframes (spec §6.5: k = 3). */
    const val DEFAULT_K = 3

    /**
     * The [k] keyframes with the highest `g_q · g_k` (float64), best first, ties → lowest index. [globals] is
     * `[K, dim]` with `dim = gq.size`.
     */
    fun topK(gq: DoubleArray, globals: FloatArray, keyframes: Int, k: Int = DEFAULT_K): IntArray {
        val sims = DoubleArray(keyframes) { Descriptor.dot(gq, globals, it) }
        val n = minOf(k, keyframes)
        val out = IntArray(n)
        val taken = BooleanArray(keyframes)
        for (i in 0 until n) {
            var best = -1
            for (j in 0 until keyframes) {
                if (taken[j]) continue
                if (best < 0 || sims[j] > sims[best]) best = j
            }
            out[i] = best
            taken[best] = true
        }
        return out
    }

    /**
     * Pools the B-set patches (float32 rows of [kfFeats], `[K, P, dim]`) of the keyframes [views], in the given order,
     * patches row-major. Returns `[rows, dim]`.
     */
    fun gatherBank(kfFeats: FloatArray, kfSets: Array<PatchSets>, views: IntArray, patches: Int, dim: Int): FloatArray {
        var rows = 0
        for (k in views) rows += kfSets[k].bIdx.size
        val out = FloatArray(rows * dim)
        var o = 0
        for (k in views) {
            val base = k * patches * dim
            for (p in kfSets[k].bIdx) {
                System.arraycopy(kfFeats, base + p * dim, out, o, dim)
                o += dim
            }
        }
        return out
    }
}
