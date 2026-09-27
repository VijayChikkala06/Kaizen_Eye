package com.kaizeneye.core.model

/*
 * Shared contract types (docs/verification/twin-spec.md). Owned by the main integrator: helpers read them but do not change
 * them — ask for a change instead, so the segmentation side (mask/track) and the Twin side (teach/score/verdict) stay in sync.
 */

/** A connected component of the foreground mask, analysis resolution (spec §2.4). Centroid in continuous coordinates. */
data class Component(
    val label: Int,
    val area: Int,
    val minX: Int,
    val minY: Int,
    val maxX: Int,
    val maxY: Int,
    val cx: Double,
    val cy: Double,
    val touchesBorder: Boolean,
) {
    val width: Int get() = maxX - minX + 1
    val height: Int get() = maxY - minY + 1
}

/** Scale/rotation/translation-invariant shape features of a component (spec §2.7) plus its raw pixel area. */
data class GeometryFeatures(
    val area: Double,
    val fill: Double,
    val aspect: Double,
    val hu1: Double,
    val solidity: Double,
)

/** Why a presentation cannot be judged (spec §2.6). OK = sane. Order = check order. */
enum class SanityReason { OK, NO_OBJECT, TOUCHES_BORDER, TOO_SMALL, TOO_LARGE, MULTIPLE, NO_CORE }

/** The four verdicts (spec §9). */
enum class Verdict { PASS, DEFECT, NOT_ENROLLED, REFRAME }

/** Square crop in full-resolution continuous pixel coordinates (spec §3). */
data class CropSquare(val x0: Double, val y0: Double, val side: Double) {
    val cx: Double get() = x0 + side / 2
    val cy: Double get() = y0 + side / 2
}

/** Backbone output for one crop: [gh, gw, dim] row-major (spec §0). */
class FeatureMap(val gh: Int, val gw: Int, val dim: Int, val data: FloatArray) {
    init {
        require(data.size == gh * gw * dim) { "feature map has ${data.size} values, expected ${gh * gw * dim}" }
    }
    val patches: Int get() = gh * gw
}
