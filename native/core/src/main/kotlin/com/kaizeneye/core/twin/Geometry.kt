package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Stats
import com.kaizeneye.core.model.GeometryFeatures
import kotlin.math.max

/** Numbers of the geometry model (spec §7.4). */
data class GeometryParams(
    val kGeo: Double = 3.0,
    val floorFill: Double = 0.04,
    val floorAspect: Double = 0.04,
    val floorSolidity: Double = 0.03,
    /** The hu1 floor is relative: `floorHu1Rel · mean(hu1)`. */
    val floorHu1Rel: Double = 0.04,
    /** Area passes iff `meanArea/areaFactor ≤ area ≤ areaFactor·meanArea`. */
    val areaFactor: Double = 2.5,
)

/** `[mean, sigma-after-floor]` of one geometry feature (spec §7.4, stored as a 2-element list in twin.json). */
data class FeatureBand(val mean: Double, val sigma: Double) {
    fun lower(k: Double): Double = mean - k * sigma
    fun upper(k: Double): Double = mean + k * sigma
    fun contains(x: Double, k: Double): Boolean = lower(k) <= x && x <= upper(k)
}

/** Result of the geometry gate: passes iff all five checks pass; [failures] lists the failing features in check order. */
data class GeometryCheck(val ok: Boolean, val failures: List<String>)

/**
 * Geometry model and gate (spec §7.4): per feature φ ∈ {fill, aspect, hu1, solidity} bounds `[m − kGeo·s, m + kGeo·s]`
 * with `s = max(std, floor_φ)`; area passes iff `meanArea/2.5 ≤ area ≤ 2.5·meanArea`.
 */
data class GeometryModel(
    val kGeo: Double,
    val meanArea: Double,
    val fill: FeatureBand,
    val aspect: FeatureBand,
    val hu1: FeatureBand,
    val solidity: FeatureBand,
    val areaFactor: Double = GeometryParams().areaFactor,
) {
    /** The gate (spec §7.4); failing feature names in the order fill, aspect, hu1, solidity, area. */
    fun check(g: GeometryFeatures): GeometryCheck {
        val failures = ArrayList<String>(5)
        if (!fill.contains(g.fill, kGeo)) failures.add(FILL)
        if (!aspect.contains(g.aspect, kGeo)) failures.add(ASPECT)
        if (!hu1.contains(g.hu1, kGeo)) failures.add(HU1)
        if (!solidity.contains(g.solidity, kGeo)) failures.add(SOLIDITY)
        if (!(meanArea / areaFactor <= g.area && g.area <= areaFactor * meanArea)) failures.add(AREA)
        return GeometryCheck(failures.isEmpty(), failures)
    }

    companion object {
        const val FILL = "fill"
        const val ASPECT = "aspect"
        const val HU1 = "hu1"
        const val SOLIDITY = "solidity"
        const val AREA = "area"

        /** Fits the model over [samples] (spec §7.4; mean and population std in float64, index order). */
        fun fit(samples: List<GeometryFeatures>, params: GeometryParams = GeometryParams()): GeometryModel {
            require(samples.isNotEmpty()) { "geometry model needs at least one sample" }
            fun band(values: DoubleArray, floor: (Double) -> Double): FeatureBand {
                val m = Stats.mean(values)
                return FeatureBand(m, max(Stats.std(values), floor(m)))
            }
            val fill = band(DoubleArray(samples.size) { samples[it].fill }) { params.floorFill }
            val aspect = band(DoubleArray(samples.size) { samples[it].aspect }) { params.floorAspect }
            val hu1 = band(DoubleArray(samples.size) { samples[it].hu1 }) { m -> params.floorHu1Rel * m }
            val solidity = band(DoubleArray(samples.size) { samples[it].solidity }) { params.floorSolidity }
            val meanArea = Stats.mean(DoubleArray(samples.size) { samples[it].area })
            return GeometryModel(params.kGeo, meanArea, fill, aspect, hu1, solidity, params.areaFactor)
        }
    }
}
