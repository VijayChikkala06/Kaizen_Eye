package com.kaizeneye.core.track

import com.kaizeneye.core.image.ImageOps
import com.kaizeneye.core.model.Component

/**
 * One tracker input (spec §11.1): a component of the frame (§2.4, border-touching ones included), its [sharpness]
 * (§1.3 on the analysis grey image over the bbox expanded by 1 px) and an opaque [frameRef] chosen by the caller (e.g. a
 * frame sequence number or pool slot) that the tracker hands back in best-crop notices and triggers.
 */
data class Detection(val component: Component, val sharpness: Double, val frameRef: Long = 0L)

/** Builds §11.1 detections. */
object Detections {

    /** §11.1 sharpness of [c]: Laplacian variance over its bbox expanded by 1 px, clamped to the `w×h` image. */
    fun sharpness(grey: DoubleArray, w: Int, h: Int, c: Component): Double =
        ImageOps.laplacianVariance(grey, w, h, c.minX - 1, c.minY - 1, c.maxX + 2, c.maxY + 2)

    /** All components of one frame as detections (allocates the small result list). */
    fun of(components: List<Component>, grey: DoubleArray, w: Int, h: Int, frameRef: Long): List<Detection> {
        val out = ArrayList<Detection>(components.size)
        for (i in components.indices) {
            val c = components[i]
            out.add(Detection(c, sharpness(grey, w, h, c), frameRef))
        }
        return out
    }
}
