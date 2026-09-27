package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.mask.Sanity
import com.kaizeneye.core.model.SanityReason

/**
 * "What the camera sees" for single-object screens (Teach, before and while recording): one box around the part the
 * app would use, green when it is usable, amber with the reason when it is not, plus a plain-language next step.
 */
object LiveView {

    /** [pick] = the circled part (null = no circle: the §2.5 main object is used). */
    fun of(a: Analysed, params: MaskParams, pick: RoiPick? = null): InspectUi {
        val (main, sanity) = if (pick != null) {
            pick.obj to Sanity.check(pick.obj, a.components, a.w, a.h, params, a.fullW, a.fullH)
        } else {
            Sanity.checkMain(a.components, a.w, a.h, params)
        }
        // No usable candidate: still show the biggest blob (usually a hand or the part touching the edge) with the reason.
        val shown = if (pick != null) pick.obj else main ?: a.components.maxByOrNull { it.area }
        val reason = if (main == null && shown != null) SanityReason.TOUCHES_BORDER else sanity
        val f = params.analysisFactor
        val box = shown?.let { c ->
            OverlayBox(
                -1,
                (c.minX * f).toFloat(), (c.minY * f).toFloat(), ((c.maxX + 1) * f).toFloat(), ((c.maxY + 1) * f).toFloat(),
                if (reason == SanityReason.OK) OverlayState.PASS else OverlayState.JUDGING,
                label(reason),
                null,
            )
        }
        return InspectUi(
            analysisW = a.fullW, analysisH = a.fullH, rotation = a.rotation,
            boxes = listOfNotNull(box),
            hint = if (pick != null && pick.obj == null) {
                "Nothing inside your circle — keep the part where you circled it, or circle it again."
            } else {
                hint(reason, if (pick != null) 1 else a.components.size)
            },
            circled = pick != null,
        )
    }

    fun label(r: SanityReason): String = when (r) {
        SanityReason.OK -> "part ✓"
        SanityReason.NO_OBJECT -> "no part"
        SanityReason.TOUCHES_BORDER -> "touches the edge"
        SanityReason.TOO_SMALL -> "too small"
        SanityReason.TOO_LARGE -> "too big"
        SanityReason.MULTIPLE -> "more than one object"
        SanityReason.NO_CORE -> "too small for the grid"
    }

    fun hint(r: SanityReason, components: Int): String = when {
        components >= LineSession.MANY_BLOBS -> "Many blobs: the background is not plain or the light changed — clear the sheet and RE-LEARN SHEET."
        r == SanityReason.OK -> "Part detected ✓ — keep it fully inside the view."
        r == SanityReason.NO_OBJECT -> "No part seen. Put the part on the sheet (it must contrast with the sheet colour)."
        r == SanityReason.TOUCHES_BORDER -> "The part or your hand touches the edge of the view — move the part inward and take your hand away."
        r == SanityReason.TOO_SMALL -> "Part too small in the picture — move the phone closer."
        r == SanityReason.TOO_LARGE -> "Part too big in the picture — move the phone further away."
        r == SanityReason.MULTIPLE -> "More than one object near the part — show one part at a time."
        else -> "Part too small for the model grid — move the phone closer."
    }
}
