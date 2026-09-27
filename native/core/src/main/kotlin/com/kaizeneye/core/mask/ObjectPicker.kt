package com.kaizeneye.core.mask

import com.kaizeneye.core.model.Component

/** The main object of a frame for teach, steady-hold and calibration (spec §2.5). */
object ObjectPicker {

    /**
     * Spec §2.5: among the components with `touchesBorder == false`, the one with the largest `area` (ties → lowest
     * label); `null` if there is no candidate. Zero allocation.
     */
    fun main(components: List<Component>): Component? {
        var best: Component? = null
        for (i in components.indices) {
            val c = components[i]
            if (c.touchesBorder) continue
            val b = best
            if (b == null || c.area > b.area || (c.area == b.area && c.label < b.label)) best = c
        }
        return best
    }
}
