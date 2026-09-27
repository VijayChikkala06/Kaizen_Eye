package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.image.ImageOps
import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.image.RgbaFrame
import com.kaizeneye.core.mask.Geometry
import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.mask.Sanity
import com.kaizeneye.core.mask.Segmenter
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.CropSquare
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.telemetry.StageTimes
import com.kaizeneye.core.track.Detection
import com.kaizeneye.core.track.Detections
import com.kaizeneye.core.vision.CanonicalCrop
import com.kaizeneye.core.vision.CanonicalSquare
import com.kaizeneye.core.vision.CropMath
import com.kaizeneye.core.vision.PatchCoverage
import com.kaizeneye.v2.camera.CameraFrame

/** Result of the fast analysis of one frame (buffers are reused by the next frame: copy what you keep). */
class Analysed(
    val tMs: Long,
    val index: Long,
    val rgba: RgbaFrame,
    val small: RgbImage,
    val grey: DoubleArray,
    val labels: IntArray,
    val components: List<Component>,
    val rotation: Int,
) {
    val w: Int get() = small.width
    val h: Int get() = small.height
    val fullW: Int get() = rgba.width
    val fullH: Int get() = rgba.height

    fun detections(): List<Detection> = Detections.of(components, grey, w, h, index)

    /** The same frame with a different component list (a circled ROI applied, see [RoiSelect]). */
    fun withComponents(list: List<Component>): Analysed = Analysed(tMs, index, rgba, small, grey, labels, list, rotation)

    /**
     * Components big enough to be a part (area ≥ minAreaFrac of the frame, twin-spec §2.6 TOO_SMALL): specks, dust and
     * background texture below that can never be judged, so they are not tracked at all (no boxes, no REFRAME noise).
     */
    fun partCandidates(minAreaFrac: Double): List<Component> {
        val min = minAreaFrac * w * h
        return components.filter { it.area >= min }
    }
}

/**
 * Everything the Twin needs about one presentation, captured at the moment the frame is available (the camera buffer and
 * the label map are recycled on the next frame): the §1.4 crop bytes, §4 coverage, §2.7 geometry + principal angle,
 * §2.6 sanity (incl. NO_CORE) and the crop square for the overlay.
 */
class CropSnapshot(
    val tMs: Long,
    val frameRef: Long,
    val n: Int,
    val crop: ByteArray,
    val square: CropSquare,
    val cov: FloatArray,
    val geometry: GeometryFeatures,
    val theta: Double,
    val sanity: SanityReason,
    val component: Component,
    val sharpness: Double,
    /** Eval-export capture (mask RLE + crop JPEG at the export crop size), only when requested. */
    val export: com.kaizeneye.v2.export.ExportFrame? = null,
    /**
     * CANONICAL rotation (accuracy mode): [crop] / [cov] above are the crop at θ (principal axis horizontal) and this holds the
     * crop at θ + π; [square] is then the canonical square (centre and side of the rotated crop, unrotated anchor). Null =
     * plain axis-aligned crop (spec pipeline, eval export).
     */
    val canon: Canon? = null,
) {
    /** The same snapshot without the θ + π crop (teach frames only need θ; saves ~0.6 MB per frame with DINOv2). */
    fun withoutAlt(): CropSnapshot = if (canon == null) this else CropSnapshot(
        tMs, frameRef, n, crop, square, cov, geometry, theta, sanity, component, sharpness, export,
        Canon(canon.square, ByteArray(0), FloatArray(0)),
    )
}

/** The second orientation of a canonical crop and the geometry needed to draw its heat map. */
class Canon(val square: CanonicalSquare, val altCrop: ByteArray, val altCov: FloatArray) {
    /** Angle of the crop that was judged: 0 = θ, 1 = θ + π. */
    fun angle(orientation: Int): Double = square.theta + if (orientation == 1) Math.PI else 0.0
}

/**
 * The fast loop's shared first half (plan "FAST LOOP"): RGBA frame → f×f box downscale → empty-sheet mask → morphology →
 * connected components, all into preallocated buffers (sized on the first frame; re-sized only if the frame size changes).
 * Single-threaded: call only from the frame source thread.
 */
class FrameAnalysis(val params: MaskParams = MaskParams(), val stages: StageTimes? = null) {
    private var small: RgbImage? = null
    private var grey = DoubleArray(0)
    private var segmenter: Segmenter? = null
    private var lastW = -1
    private var lastH = -1

    val factor: Int get() = params.analysisFactor

    /** The last analysis image (replay preview); overwritten by the next frame. */
    val lastSmall: RgbImage? get() = small

    /** Wraps the camera buffer as a core [RgbaFrame] whose size is a multiple of the analysis factor. */
    fun wrap(frame: CameraFrame): RgbaFrame {
        val w = frame.width - frame.width % factor
        val h = frame.height - frame.height % factor
        return RgbaFrame(w, h, frame.rowStride, frame.data)
    }

    /** Downscale only (sheet capture / flicker check). */
    fun downscale(frame: CameraFrame): RgbImage {
        val rgba = wrap(frame)
        ensure(rgba.width / factor, rgba.height / factor)
        val img = small!!
        ImageOps.downscaleRgba(rgba, factor, img)
        return img
    }

    fun analyse(frame: CameraFrame, sheet: SheetModel): Analysed {
        val t0 = System.nanoTime()
        val rgba = wrap(frame)
        val w = rgba.width / factor
        val h = rgba.height / factor
        ensure(w, h)
        val img = small!!
        ImageOps.downscaleRgba(rgba, factor, img)
        val t1 = System.nanoTime()
        val comps = segmenter!!.segment(img, sheet)
        val t2 = System.nanoTime()
        ImageOps.grey(img, grey)
        stages?.record("downscale", (t1 - t0) / 1e6)
        stages?.record("segment", (t2 - t1) / 1e6)
        return Analysed(frame.tMs, frame.index, rgba, img, grey, segmenter!!.labels, comps, frame.rotationDegrees)
    }

    private fun ensure(w: Int, h: Int) {
        if (w == lastW && h == lastH && small != null) return
        small = RgbImage(w, h)
        grey = DoubleArray(w * h)
        segmenter = Segmenter(w, h, params)
        lastW = w
        lastH = h
    }

    /**
     * Captures a [CropSnapshot] of component [c] from the current frame: crop-resize to n×n (rounded bytes, §1.4), patch
     * coverage on the gh×gw grid (§4), geometry (§2.7), sanity (§2.6 incl. NO_CORE).
     */
    fun snapshot(
        a: Analysed, c: Component, n: Int, gh: Int, gw: Int, dst: ByteArray? = null, exportCropSize: Int = 0,
        canonical: Boolean = false,
    ): CropSnapshot {
        val t0 = System.nanoTime()
        val geo = Geometry.features(a.labels, a.w, a.h, c)
        val theta = Geometry.principalAngle(a.labels, a.w, a.h, c)
        if (canonical && exportCropSize == 0) return canonicalSnapshot(a, c, n, gh, gw, dst, geo, theta, t0)
        val sq = CropMath.square(c, factor, a.fullW, a.fullH, params.cropMargin)
        val crop = dst ?: ByteArray(n * n * 3)
        ImageOps.cropResize(a.rgba, sq.x0, sq.y0, sq.side, n, crop)
        val cov = PatchCoverage.coverage(a.labels, a.w, a.h, c.label, sq, factor, gh, gw)
        var sanity = Sanity.check(c, a.components, a.w, a.h, params, a.fullW, a.fullH)
        if (sanity == SanityReason.OK && PatchCoverage.sets(cov, gh, gw, params.coreThreshold).isCoreEmpty) sanity = SanityReason.NO_CORE
        val sharp = Detections.sharpness(a.grey, a.w, a.h, c)
        stages?.record("snapshot", (System.nanoTime() - t0) / 1e6)
        val export = if (exportCropSize > 0) exportOf(a, c, sanity, sq, cov, geo, theta, crop, n, exportCropSize) else null
        return CropSnapshot(a.tMs, a.index, n, crop, sq, cov, geo, theta, sanity, c, sharp, export)
    }

    /**
     * CANONICAL snapshot (accuracy mode): the crop is rotated so the part's principal axis is horizontal, at θ and at θ + π,
     * with the square fitted to the part's rotated extent (CanonicalCrop). Patch coverage follows the rotation. Sanity is the
     * same §2.6 check as the plain crop (it looks at components, not at the crop).
     */
    private fun canonicalSnapshot(
        a: Analysed, c: Component, n: Int, gh: Int, gw: Int, dst: ByteArray?, geo: GeometryFeatures, theta0: Double, t0: Long,
    ): CropSnapshot {
        // Round / square parts have no stable principal axis (spec §2.7 θ is then noise): crop them at θ = 0 like the spec,
        // so teach and judge always agree. Elongated parts get their long axis horizontal.
        val theta = if (geo.aspect >= CanonicalCrop.MIN_ASPECT_FOR_AXIS) 0.0 else theta0
        val sq = CanonicalCrop.square(a.labels, a.w, a.h, c, theta, factor, params.cropMargin)
        val crop = dst ?: ByteArray(n * n * 3)
        ImageOps.cropResizeRotated(a.rgba, sq.cx, sq.cy, sq.side, theta, n, crop)
        val cov = CanonicalCrop.coverage(a.labels, a.w, a.h, c.label, sq, theta, factor, gh, gw)
        val alt = ByteArray(n * n * 3)
        ImageOps.cropResizeRotated(a.rgba, sq.cx, sq.cy, sq.side, theta + Math.PI, n, alt)
        val altCov = CanonicalCrop.coverage(a.labels, a.w, a.h, c.label, sq, theta + Math.PI, factor, gh, gw)
        var sanity = Sanity.check(c, a.components, a.w, a.h, params, a.fullW, a.fullH)
        // §2.6 MULTIPLE on the canonical square too (it can be larger than the axis-aligned one for a diagonal part).
        if (sanity == SanityReason.OK) {
            for (d in a.components) {
                if (d.label == c.label || d.area < params.multipleRatio * c.area) continue
                if (CanonicalCrop.contains(sq, d.cx * factor, d.cy * factor)) { sanity = SanityReason.MULTIPLE; break }
            }
        }
        val sets = PatchCoverage.sets(cov, gh, gw, params.coreThreshold)
        val altSets = PatchCoverage.sets(altCov, gh, gw, params.coreThreshold)
        if (sanity == SanityReason.OK && (sets.isCoreEmpty || altSets.isCoreEmpty)) sanity = SanityReason.NO_CORE
        val sharp = Detections.sharpness(a.grey, a.w, a.h, c)
        stages?.record("snapshot", (System.nanoTime() - t0) / 1e6)
        return CropSnapshot(a.tMs, a.index, n, crop, sq.asCropSquare, cov, geo, theta, sanity, c, sharp, null, Canon(sq, alt, altCov))
    }

    private fun exportOf(
        a: Analysed, c: Component, sanity: SanityReason, sq: CropSquare, cov: FloatArray, geo: GeometryFeatures, theta: Double,
        crop: ByteArray, n: Int, cropSize: Int,
    ): com.kaizeneye.v2.export.ExportFrame {
        val big = if (cropSize == n) crop else ByteArray(cropSize * cropSize * 3).also { ImageOps.cropResize(a.rgba, sq.x0, sq.y0, sq.side, cropSize, it) }
        return com.kaizeneye.v2.export.ExportFrame(
            a.tMs, sanity, com.kaizeneye.v2.export.ExportFrame.cropSharpness(crop, n), sq,
            com.kaizeneye.v2.export.MaskRle.of(a.labels, a.w, c), cov, geo, theta,
            com.kaizeneye.v2.util.Images.jpeg(big, cropSize, cropSize, 95),
        )
    }

    /**
     * Export record for a teach frame (export-format §5: every analysed frame, rejected ones too). A frame without a main
     * component writes only tMs + sanity.
     */
    fun teachExport(a: Analysed, n: Int, gh: Int, gw: Int, cropSize: Int): com.kaizeneye.v2.export.ExportFrame {
        val (main, sanity0) = Sanity.checkMain(a.components, a.w, a.h, params)
        if (main == null) return com.kaizeneye.v2.export.ExportFrame(a.tMs, sanity0, null, null, null, null, null, null, null)
        return snapshot(a, main, n, gh, gw, null, cropSize).export!!
    }

    /**
     * Single-object snapshot of an explicitly [chosen] component (the circled part) instead of the §2.5 main object:
     * §2.6 sanity against the frame's other components, then the snapshot. Null crop when not sane.
     */
    fun chosenSnapshot(a: Analysed, chosen: Component?, n: Int, gh: Int, gw: Int, canonical: Boolean = false): Pair<SanityReason, CropSnapshot?> {
        val sanity = Sanity.check(chosen, a.components, a.w, a.h, params, a.fullW, a.fullH)
        if (chosen == null || sanity != SanityReason.OK) return sanity to null
        val s = snapshot(a, chosen, n, gh, gw, canonical = canonical)
        return s.sanity to (if (s.sanity == SanityReason.OK) s else null)
    }

    /** Snapshot for single-object modes (teach, steady hold): §2.5 main object + §2.6 sanity. Null crop when not sane. */
    fun mainSnapshot(a: Analysed, n: Int, gh: Int, gw: Int, canonical: Boolean = false): Pair<SanityReason, CropSnapshot?> {
        val (main, sanity) = Sanity.checkMain(a.components, a.w, a.h, params)
        if (main == null || sanity != SanityReason.OK) return sanity to null
        val s = snapshot(a, main, n, gh, gw, canonical = canonical)
        return s.sanity to (if (s.sanity == SanityReason.OK) s else null)
    }
}
