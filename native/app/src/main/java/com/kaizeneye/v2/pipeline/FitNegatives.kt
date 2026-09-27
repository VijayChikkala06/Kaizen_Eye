package com.kaizeneye.v2.pipeline

import android.util.Log
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.twin.KnnBackend
import com.kaizeneye.core.twin.TwinLoadResult
import com.kaizeneye.core.twin.TwinModel
import com.kaizeneye.v2.twin.TwinRepo
import kotlin.math.max

/**
 * Known DIFFERENT objects for the FIT gate (Fit.kt): how far are they from this Twin's bank?
 *  - the negatives library (wrong objects the operator showed; both orientations stored, the better fit counts);
 *  - every other Twin taught with the same pipeline: its keyframes are real pictures of another part. Per other Twin the
 *    closest keyframe counts (its best view), so a Twin taught twice by mistake shows up as an "overlap", not as a wrong object.
 * With them the gate sits at the midpoint between the same-part spread and the nearest wrong object instead of a blind
 * `1.1 × spread`, which is what makes two look-alike parts separable.
 */
object FitNegatives {

    /** Keyframes of another Twin that are scored (spread over its keyframes); each costs one k-NN search. */
    const val KEYFRAMES_PER_TWIN = 8

    class Result(val fits: DoubleArray, val manual: Int, val twins: Int)

    fun compute(twin: TwinModel, repo: TwinRepo, knn: KnnBackend): Result {
        val fits = ArrayList<Double>()
        var manual = 0
        var twins = 0
        val np = twin.patches
        val d = twin.dim
        try {
            for (n in repo.negatives.list(twin.fingerprint)) {
                val maps = repo.negatives.maps(n.id)
                val best = maps.map { m -> twin.fitOf(FeatureMap(twin.gh, twin.gw, d, m.features), m.cov, knn) }.filter { !it.isNaN() }.minOrNull()
                if (best != null) {
                    fits += best
                    manual++
                }
            }
        } catch (t: Throwable) {
            Log.w("KaizenFit", "manual negatives skipped", t)
        }
        try {
            for (s in repo.store.list()) {
                if (s.id == twin.id || s.fingerprint != twin.fingerprint) continue
                val other = (repo.store.load(s.id, null) as? TwinLoadResult.Loaded)?.twin ?: continue
                val k = other.keyframeCount
                val step = max(1, k / KEYFRAMES_PER_TWIN)
                var best = Double.NaN
                var i = 0
                while (i < k) {
                    val f = twin.fitOf(
                        FeatureMap(twin.gh, twin.gw, d, other.kfFeats.copyOfRange(i * np * d, (i + 1) * np * d)),
                        other.kfCov.copyOfRange(i * np, (i + 1) * np), knn,
                    )
                    if (!f.isNaN() && (best.isNaN() || f < best)) best = f
                    i += step
                }
                if (!best.isNaN()) {
                    fits += best
                    twins++
                }
            }
        } catch (t: Throwable) {
            Log.w("KaizenFit", "other twins skipped", t)
        }
        return Result(fits.toDoubleArray(), manual, twins)
    }
}
