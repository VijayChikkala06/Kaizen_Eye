package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.image.ImageOps
import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.v2.camera.CameraFrame

/**
 * Empty-sheet tap (plan "Mask"): collects [frames] analysis frames of the EMPTY sheet, fits the spec §2.1 sheet model
 * (per-channel median / 1.4826·MAD, σ floor) and runs the flicker check on the same frames. Frame-source thread only.
 */
class SheetCapture(private val analysis: FrameAnalysis, private val frames: Int = 30) {
    companion object {
        /** Robust σ (grey levels) above which the "sheet" is textured rather than plain. */
        const val UNIFORM_SIGMA = 12.0
    }

    private val collected = ArrayList<RgbImage>(frames)
    private val greys = ArrayList<DoubleArray>(frames)

    val progress: Float get() = collected.size.toFloat() / frames
    val done: Boolean get() = collected.size >= frames

    /** Returns true when enough frames have been collected. */
    fun offer(frame: CameraFrame): Boolean {
        if (done) return true
        val img = analysis.downscale(frame).copy()
        collected += img
        greys += ImageOps.grey(img)
        return done
    }

    class Result(val sheet: SheetModel, val flicker: FlickerCheck.Result, val w: Int, val h: Int) {
        /** A plain matte sheet has a small robust σ; a textured background makes the mask unreliable. */
        val uniform: Boolean get() = sheet.sigma.max() <= UNIFORM_SIGMA

        fun describe(): String =
            "μ=(%.0f, %.0f, %.0f) σ=(%.1f, %.1f, %.1f)".format(sheet.mean[0], sheet.mean[1], sheet.mean[2], sheet.sigma[0], sheet.sigma[1], sheet.sigma[2]) +
                if (uniform) "" else " — background NOT plain: use a matte, single-colour sheet (the part mask will be unreliable)"
    }

    fun finish(sigmaMin: Double = analysis.params.sigmaMin): Result {
        check(collected.isNotEmpty()) { "no sheet frames" }
        val w = collected[0].width
        val h = collected[0].height
        val sheet = SheetModel.fit(collected, null, sigmaMin)
        val flicker = FlickerCheck.analyse(greys, w, h, uniform = sheet.sigma.max() <= UNIFORM_SIGMA)
        return Result(sheet, flicker, w, h)
    }
}
