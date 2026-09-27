package com.kaizeneye.core.mask

// SPEC-QUESTION: (§2.6 MULTIPLE "another component D ≠ C") every other component of the frame counts, including
//   border-touching ones (a hand reaching over the part is a reason to reframe); D ≠ C is decided by label.

import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.vision.CropMath

/**
 * REFRAME reasons 1–5 of spec §2.6, checked in order. Reason 6 (`NO_CORE`) needs the patch core of the crop (§4) and is
 * decided by the caller afterwards (`PatchSets.isCoreEmpty`).
 */
object Sanity {

    /**
     * Checks the [chosen] component (`null` = no component / no candidate) of a `w×h` analysis frame whose components
     * are [components] (all of them, including border-touching ones):
     * 1. `NO_OBJECT` — [chosen] is null;
     * 2. `TOUCHES_BORDER`;
     * 3. `TOO_SMALL` — `area / (W·H) < minAreaFrac`;
     * 4. `TOO_LARGE` — `area / (W·H) > maxAreaFrac`;
     * 5. `MULTIPLE` — another component `D` with `D.area ≥ multipleRatio · C.area` whose centroid lies inside `C`'s crop
     *    square (§3, taken in analysis coordinates: the full-res square divided by `f`).
     *
     * Returns [SanityReason.OK] if none applies. [fullW]/[fullH] default to `f·w`, `f·h` (§1.1).
     */
    fun check(
        chosen: Component?,
        components: List<Component>,
        w: Int,
        h: Int,
        params: MaskParams = MaskParams(),
        fullW: Int = w * params.analysisFactor,
        fullH: Int = h * params.analysisFactor,
    ): SanityReason {
        if (chosen == null) return SanityReason.NO_OBJECT
        if (chosen.touchesBorder) return SanityReason.TOUCHES_BORDER
        val frac = chosen.area.toDouble() / (w.toDouble() * h)
        if (frac < params.minAreaFrac) return SanityReason.TOO_SMALL
        if (frac > params.maxAreaFrac) return SanityReason.TOO_LARGE
        val f = params.analysisFactor
        val square = CropMath.toAnalysis(CropMath.square(chosen, f, fullW, fullH, params.cropMargin), f)
        for (i in components.indices) {
            val d = components[i]
            if (d.label == chosen.label) continue
            if (d.area >= params.multipleRatio * chosen.area && CropMath.contains(square, d.cx, d.cy)) {
                return SanityReason.MULTIPLE
            }
        }
        return SanityReason.OK
    }

    /** Single-object modes (teach, steady-hold, calibration): §2.5 main object, then [check]. */
    fun checkMain(
        components: List<Component>,
        w: Int,
        h: Int,
        params: MaskParams = MaskParams(),
    ): Pair<Component?, SanityReason> {
        val main = ObjectPicker.main(components)
        return main to check(main, components, w, h, params)
    }
}
