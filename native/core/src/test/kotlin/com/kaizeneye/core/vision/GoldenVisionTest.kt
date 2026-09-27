package com.kaizeneye.core.vision

import com.kaizeneye.core.image.ImageOps
import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.mask.Geometry
import com.kaizeneye.core.mask.MaskParams
import com.kaizeneye.core.mask.ObjectPicker
import com.kaizeneye.core.mask.Sanity
import com.kaizeneye.core.mask.Segmenter
import com.kaizeneye.core.mask.SheetModel
import com.kaizeneye.core.model.Component
import com.kaizeneye.core.model.CropSquare
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.power.Governor
import com.kaizeneye.core.power.GovernorParams
import com.kaizeneye.core.track.Axis
import com.kaizeneye.core.track.Detection
import com.kaizeneye.core.track.Direction
import com.kaizeneye.core.track.Tracker
import com.kaizeneye.core.track.TriggerConfig
import com.kaizeneye.core.vision.GoldenVision.Mismatches
import com.kaizeneye.core.vision.GoldenVision.arr
import com.kaizeneye.core.vision.GoldenVision.bool
import com.kaizeneye.core.vision.GoldenVision.dbl
import com.kaizeneye.core.vision.GoldenVision.doubles
import com.kaizeneye.core.vision.GoldenVision.has
import com.kaizeneye.core.vision.GoldenVision.int
import com.kaizeneye.core.vision.GoldenVision.ints
import com.kaizeneye.core.vision.GoldenVision.long
import com.kaizeneye.core.vision.GoldenVision.name
import com.kaizeneye.core.vision.GoldenVision.nullableDbl
import com.kaizeneye.core.vision.GoldenVision.nullableInt
import com.kaizeneye.core.vision.GoldenVision.nullableStr
import com.kaizeneye.core.vision.GoldenVision.obj
import com.kaizeneye.core.vision.GoldenVision.str
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.Test

/** Golden tests (spec §15) for the image, mask, crop_patch, tracker and governor sections of golden_twin.json. */
class GoldenVisionTest {

    private fun ByteArray.u8(): IntArray = IntArray(size) { this[it].toInt() and 0xFF }

    // ------------------------------------------------------------------------------------------------------------------
    @Test
    fun image() {
        val m = Mismatches("image")
        for (case in GoldenVision.cases("image")) {
            val name = case.name()
            val inp = case.obj("inputs")
            val exp = case.obj("expected")
            when (case.str("kind")) {
                "downscale" -> {
                    val src = GoldenVision.rgbaFrame(inp.obj("rgba"))
                    val f = inp.int("factor")
                    val e = exp.obj("rgb")
                    val dst = RgbImage(src.width / f, src.height / f)
                    ImageOps.downscaleRgba(src, f, dst)
                    m.eq("$name size", e.int("w") to e.int("h"), dst.width to dst.height)
                    m.ints("$name rgb", e.getValue("data").ints(), dst.data.u8())
                }
                "sharpness" -> {
                    val img = GoldenVision.rgbImage(inp.obj("rgb"))
                    val grey = ImageOps.grey(img)
                    m.f64s("$name grey", exp.getValue("grey").doubles(), grey)
                    val rects = inp.arr("rects")
                    val counts = exp.getValue("count").ints()
                    val sharp = exp.getValue("sharpness").doubles()
                    for ((i, r) in rects.withIndex()) {
                        val o = r.jsonObject
                        val x0 = o.int("x0")
                        val y0 = o.int("y0")
                        val x1 = o.int("x1")
                        val y1 = o.int("y1")
                        m.eq("$name rect[$i] count", counts[i], ImageOps.sharpnessPixelCount(img.width, img.height, x0, y0, x1, y1))
                        m.f64("$name rect[$i] sharpness", sharp[i], ImageOps.laplacianVariance(grey, img.width, img.height, x0, y0, x1, y1))
                    }
                }
                "crop" -> {
                    val img = GoldenVision.rgbImage(inp.obj("rgb"))
                    val out = DoubleArray(3)
                    val es = exp.arr("samples")
                    for ((i, s) in inp.arr("samples").withIndex()) {
                        val o = s.jsonObject
                        ImageOps.sample(img, o.dbl("u"), o.dbl("v"), out)
                        m.f64s("$name sample[$i] (u=${o.dbl("u")}, v=${o.dbl("v")})", es[i].doubles(), out)
                    }
                    val ec = exp.arr("crops")
                    for ((i, c) in inp.arr("crops").withIndex()) {
                        val o = c.jsonObject
                        val n = o.int("n")
                        val vals = DoubleArray(n * n * 3)
                        val bytes = ByteArray(n * n * 3)
                        ImageOps.cropResizeFloat(img, o.dbl("x0"), o.dbl("y0"), o.dbl("side"), n, vals)
                        ImageOps.cropResize(img, o.dbl("x0"), o.dbl("y0"), o.dbl("side"), n, bytes)
                        val e = ec[i].jsonObject
                        m.f64s("$name crop[$i] values", e.getValue("values").doubles(), vals)
                        m.ints("$name crop[$i] bytes", e.getValue("bytes").ints(), bytes.u8())
                    }
                    val er = exp.arr("rotated")
                    for ((i, c) in inp.arr("rotated").withIndex()) {
                        val o = c.jsonObject
                        val n = o.int("n")
                        val vals = DoubleArray(n * n * 3)
                        val bytes = ByteArray(n * n * 3)
                        ImageOps.cropResizeRotatedFloat(img, o.dbl("cx"), o.dbl("cy"), o.dbl("side"), o.dbl("theta"), n, vals)
                        ImageOps.cropResizeRotated(img, o.dbl("cx"), o.dbl("cy"), o.dbl("side"), o.dbl("theta"), n, bytes)
                        val e = er[i].jsonObject
                        m.f64s("$name rotated[$i] values", e.getValue("values").doubles(), vals)
                        m.ints("$name rotated[$i] bytes", e.getValue("bytes").ints(), bytes.u8())
                    }
                }
                else -> m.check(false) { "$name: unknown image kind ${case.str("kind")}" }
            }
        }
        m.assertNone()
    }

    // ------------------------------------------------------------------------------------------------------------------
    private fun maskParams(p: JsonObject) = MaskParams(
        kSigma = p.dbl("kSigma"),
        sigmaMin = p.dbl("sigmaMin"),
        minBlobPx = p.int("minBlobPx"),
        borderMargin = p.int("borderMargin"),
        minAreaFrac = p.dbl("minAreaFrac"),
        maxAreaFrac = p.dbl("maxAreaFrac"),
        multipleRatio = p.dbl("multipleRatio"),
        cropMargin = p.dbl("cropMargin"),
        analysisFactor = p.int("analysisFactor"),
        coreThreshold = p.dbl("coreThreshold"),
    )

    /** Spec §2.6 steps 2–6 for [c] as the chosen component (NO_CORE from the gh × gw coverage of its crop). */
    private fun fullSanity(
        c: Component, comps: List<Component>, labels: IntArray, w: Int, h: Int, params: MaskParams,
        fullW: Int, fullH: Int, gh: Int, gw: Int,
    ): Triple<SanityReason, CropSquare, Int> {
        val crop = CropMath.square(c, params.analysisFactor, fullW, fullH, params.cropMargin)
        val cov = PatchCoverage.coverage(labels, w, h, c.label, crop, params.analysisFactor, gh, gw)
        val core = PatchCoverage.sets(cov, gh, gw, params.coreThreshold).coreCount
        var reason = Sanity.check(c, comps, w, h, params, fullW, fullH)
        if (reason == SanityReason.OK && core == 0) reason = SanityReason.NO_CORE
        return Triple(reason, crop, core)
    }

    @Test
    fun mask() {
        val m = Mismatches("mask")
        for (case in GoldenVision.cases("mask")) {
            val name = case.name()
            val inp = case.obj("inputs")
            val exp = case.obj("expected")
            val frame = GoldenVision.rgbImage(inp.obj("frame"))
            val w = frame.width
            val h = frame.height
            val p = inp.obj("params")
            val params = maskParams(p)
            val fullW = p.int("fullW")
            val fullH = p.int("fullH")
            val gh = p.int("gh")
            val gw = p.int("gw")
            val sheet = if (inp.has("sheetFrames")) {
                SheetModel.fit(inp.arr("sheetFrames").map { GoldenVision.rgbImage(it.jsonObject) }, null, params.sigmaMin)
            } else {
                val s = inp.obj("sheet")
                SheetModel(s.getValue("mu").doubles(), s.getValue("sigma").doubles())
            }
            val es = exp.obj("sheet")
            m.f64s("$name sheet.mu", es.getValue("mu").doubles(), sheet.mean)
            m.f64s("$name sheet.sigma", es.getValue("sigma").doubles(), sheet.sigma)
            if (es.has("mad")) m.f64s("$name sheet.mad", es.getValue("mad").doubles(), sheet.mad ?: DoubleArray(0))

            val seg = Segmenter(w, h, params)
            val d2 = DoubleArray(w * h)
            seg.distanceSquared(frame, sheet, d2)
            m.f64s("$name d2", exp.getValue("d2").doubles(), d2)
            val comps = seg.segment(frame, sheet)
            m.ints("$name foreground", exp.getValue("foreground").ints(), seg.fgRaw.u8())
            m.ints("$name afterOpen", exp.getValue("afterOpen").ints(), seg.afterOpen.u8())
            m.ints("$name afterClose", exp.getValue("afterClose").ints(), seg.afterClose.u8())
            m.ints("$name rawAreas", exp.getValue("rawAreas").ints(), seg.rawAreas())
            m.ints("$name labels", exp.getValue("labels").ints(), seg.labels)

            val ec = exp.arr("components")
            m.eq("$name component count", ec.size, comps.size)
            val eg = exp.arr("geometry")
            for (i in 0 until minOf(ec.size, comps.size)) {
                val e = ec[i].jsonObject
                val c = comps[i]
                val at = "$name component[$i]"
                m.eq("$at ints", listOf(e.int("label"), e.int("area"), e.int("minX"), e.int("minY"), e.int("maxX"), e.int("maxY")),
                    listOf(c.label, c.area, c.minX, c.minY, c.maxX, c.maxY))
                m.f64("$at cx", e.dbl("cx"), c.cx)
                m.f64("$at cy", e.dbl("cy"), c.cy)
                m.eq("$at touchesBorder", e.bool("touchesBorder"), c.touchesBorder)
                if (i < eg.size) {
                    val g = eg[i].jsonObject
                    val f = Geometry.features(seg.labels, w, h, c)
                    m.f64("$at geometry.area", g.dbl("area"), f.area)
                    m.f64("$at geometry.fill", g.dbl("fill"), f.fill)
                    m.f64("$at geometry.aspect", g.dbl("aspect"), f.aspect)
                    m.f64("$at geometry.hu1", g.dbl("hu1"), f.hu1)
                    m.f64("$at geometry.solidity", g.dbl("solidity"), f.solidity)
                    if (g.has("hullArea")) m.f64("$at geometry.hullArea", g.dbl("hullArea"), Geometry.hullArea(seg.labels, w, h, c))
                    if (g.has("theta")) m.f64("$at geometry.theta", g.dbl("theta"), Geometry.principalAngle(seg.labels, w, h, c))
                }
            }
            val main = ObjectPicker.main(comps)
            m.eq("$name mainObject", exp.int("mainObject"), main?.label ?: 0)
            val per = exp.arr("perComponent")
            for (pc in per) {
                val o = pc.jsonObject
                val label = o.int("label")
                val c = comps.firstOrNull { it.label == label }
                m.check(c != null) { "$name perComponent: no component with label $label" }
                if (c == null) continue
                val (reason, crop, core) = fullSanity(c, comps, seg.labels, w, h, params, fullW, fullH, gh, gw)
                val ecrop = o.obj("crop")
                m.f64("$name perComponent[$label] crop.x0", ecrop.dbl("x0"), crop.x0)
                m.f64("$name perComponent[$label] crop.y0", ecrop.dbl("y0"), crop.y0)
                m.f64("$name perComponent[$label] crop.side", ecrop.dbl("side"), crop.side)
                m.eq("$name perComponent[$label] coreCount", o.int("coreCount"), core)
                m.eq("$name perComponent[$label] sanity", o.str("sanity"), reason.name)
            }
            val mainReason = if (main == null) SanityReason.NO_OBJECT
            else fullSanity(main, comps, seg.labels, w, h, params, fullW, fullH, gh, gw).first
            m.eq("$name sanity", exp.str("sanity"), mainReason.name)
        }
        m.assertNone()
    }

    // ------------------------------------------------------------------------------------------------------------------
    @Test
    fun cropPatch() {
        val m = Mismatches("crop_patch")
        for (case in GoldenVision.cases("crop_patch")) {
            val name = case.name()
            val inp = case.obj("inputs")
            val exp = case.obj("expected")
            val lab = inp.obj("labels")
            val w = lab.int("w")
            val h = lab.int("h")
            val labels = lab.getValue("data").ints()
            val label = inp.int("label")
            val p = inp.obj("params")
            val f = p.int("analysisFactor")
            val gh = p.int("gh")
            val gw = p.int("gw")
            val crop = if (inp.has("component")) {
                val c = inp.obj("component")
                val comp = Component(label, 1, c.int("minX"), c.int("minY"), c.int("maxX"), c.int("maxY"), 0.0, 0.0, false)
                CropMath.square(comp, f, p.int("fullW"), p.int("fullH"), p.dbl("cropMargin"))
            } else {
                val c = inp.obj("crop")
                CropSquare(c.dbl("x0"), c.dbl("y0"), c.dbl("side"))
            }
            val ec = exp.obj("crop")
            m.f64("$name crop.x0", ec.dbl("x0"), crop.x0)
            m.f64("$name crop.y0", ec.dbl("y0"), crop.y0)
            m.f64("$name crop.side", ec.dbl("side"), crop.side)
            val cov = PatchCoverage.coverage(labels, w, h, label, crop, f, gh, gw)
            val ecov = exp.getValue("cov").doubles()
            m.check(ecov.size == cov.size && ecov.indices.all { ecov[it] == cov[it].toDouble() }) {
                "$name cov: expected ${ecov.toList()}, got ${cov.toList()}"
            }
            val sets = PatchCoverage.sets(cov, gh, gw, p.dbl("coreThreshold"))
            m.ints("$name core", exp.getValue("core").ints(), IntArray(gh * gw) { if (sets.core[it]) 1 else 0 })
            m.ints("$name S", exp.getValue("S").ints(), IntArray(gh * gw) { if (sets.s[it]) 1 else 0 })
            m.ints("$name B", exp.getValue("B").ints(), IntArray(gh * gw) { if (sets.b[it]) 1 else 0 })
        }
        m.assertNone()
    }

    // ------------------------------------------------------------------------------------------------------------------
    @Test
    fun tracker() {
        val m = Mismatches("tracker")
        for (case in GoldenVision.cases("tracker")) {
            val name = case.name()
            val inp = case.obj("inputs")
            val exp = case.obj("expected")
            val p = inp.obj("params")
            val config = TriggerConfig(
                axis = Axis.valueOf(p.str("lineAxis")),
                position = p.dbl("linePos"),
                direction = Direction.valueOf(p.str("lineDirection")),
                lineEnabled = p.bool("lineEnabled"),
                steadyEnabled = p.bool("steadyEnabled"),
                gateBase = p.dbl("gateBase"),
                gateArea = p.dbl("gateArea"),
                gateSpeed = p.dbl("gateSpeed"),
                velocityAlpha = p.dbl("velAlpha"),
                minHits = p.int("minHits"),
                maxMissed = p.int("maxMissed"),
                crossFrames = p.int("crossFrames"),
                vStill = p.dbl("vStill"),
                holdMs = p.dbl("holdMs"),
                sharpRatio = p.dbl("holdSharpRatio"),
                bestFrames = p.int("bestK"),
            )
            val tracker = Tracker(p.int("w"), p.int("h"), config)
            val frames = inp.arr("frames")
            val ef = exp.arr("frames")
            m.eq("$name frame count", frames.size, ef.size)
            val fired = ArrayList<List<Any?>>()
            val exited = ArrayList<List<Any?>>()
            for ((fi, fr) in frames.withIndex()) {
                val fo = fr.jsonObject
                val dets = fo.arr("detections").mapIndexed { di, d ->
                    val o = d.jsonObject
                    Detection(
                        Component(di + 1, o.int("area"), o.int("minX"), o.int("minY"), o.int("maxX"), o.int("maxY"),
                            o.dbl("cx"), o.dbl("cy"), o.bool("touchesBorder")),
                        o.dbl("sharpness"),
                        fi.toLong(),
                    )
                }
                val u = tracker.update(fo.long("tMs"), dets)
                if (fi >= ef.size) continue
                val e = ef[fi].jsonObject
                val at = "$name frame[$fi] t=${fo.long("tMs")}"
                m.ints("$at assign", e.getValue("assign").ints(), u.assignments)
                // events: FIRED by ascending track id, then EXITED by ascending track id
                val got = ArrayList<List<Any?>>()
                for (t in u.triggers.sortedBy { it.trackId }) {
                    val judge = t.judgeFrame?.frameRef?.toInt()
                    got += listOf("FIRED", t.trackId, t.trigger.name, judge, t.buffer.map { it.frameRef.toInt() },
                        if (judge == null) "TOUCHES_BORDER" else null)
                    fired += listOf(fi, t.trackId, t.trigger.name, judge)
                }
                for (x in u.exits.sortedBy { it.trackId }) {
                    got += listOf("EXITED", x.trackId, x.judged, x.confirmed)
                    exited += listOf(fi, x.trackId, x.judged)
                }
                val want = e.arr("events").map { ev ->
                    val o = ev.jsonObject
                    if (o.str("type") == "FIRED") {
                        listOf("FIRED", o.int("trackId"), o.str("trigger"), o.nullableInt("judgeFrame"),
                            o.getValue("buffer").ints().toList(), o.nullableStr("reframe"))
                    } else {
                        listOf("EXITED", o.int("trackId"), o.bool("judged"), o.bool("confirmed"))
                    }
                }
                m.eq("$at events", want, got)
                val et = e.arr("tracks")
                m.eq("$at live track ids", et.map { it.jsonObject.int("id") }, u.tracks.map { it.id })
                if (et.size == u.tracks.size) {
                    for ((k, tv) in u.tracks.withIndex()) {
                        val o = et[k].jsonObject
                        val tat = "$at track ${tv.id}"
                        m.f64("$tat cx", o.dbl("cx"), tv.cx)
                        m.f64("$tat cy", o.dbl("cy"), tv.cy)
                        m.f64("$tat vx", o.dbl("vx"), tv.vx)
                        m.f64("$tat vy", o.dbl("vy"), tv.vy)
                        m.eq("$tat hits/missed/confirmed/judged", listOf(o.int("hits"), o.int("missed"), o.bool("confirmed"), o.bool("judged")),
                            listOf(tv.hits, tv.missed, tv.confirmed, tv.judged))
                    }
                }
            }
            if (exp.has("fired")) {
                val want = exp.arr("fired").map { val o = it.jsonObject; listOf(o.int("frame"), o.int("trackId"), o.str("trigger"), o.nullableInt("judgeFrame")) }
                m.eq("$name fired summary", want, fired)
            }
            if (exp.has("exited")) {
                val want = exp.arr("exited").map { val o = it.jsonObject; listOf(o.int("frame"), o.int("trackId"), o.bool("judged")) }
                m.eq("$name exited summary", want, exited)
            }
        }
        m.assertNone()
    }

    // ------------------------------------------------------------------------------------------------------------------
    @Test
    fun governor() {
        val m = Mismatches("governor")
        for (case in GoldenVision.cases("governor")) {
            val name = case.name()
            val inp = case.obj("inputs")
            val p = inp.obj("params")
            val g = Governor(
                GovernorParams(
                    l2Status = p.int("statusL2"),
                    l1Status = p.int("statusL1"),
                    l2Headroom = p.dbl("headroomL2").toFloat(),
                    l1Headroom = p.dbl("headroomL1").toFloat(),
                    cooldownMs = p.long("cooldownMs"),
                    fpsL0 = p.int("fpsL0"),
                    fpsL1 = p.int("fpsL1"),
                    fpsL2 = p.int("fpsL2"),
                    fpsIdle = p.int("fpsIdle"),
                ),
            )
            val steps = case.obj("expected").arr("steps")
            for ((i, s) in inp.arr("samples").withIndex()) {
                val o = s.jsonObject
                val headroom = o.nullableDbl("headroom")?.toFloat() ?: Float.NaN
                val target = g.targetLevel(o.int("status"), headroom)
                val st = g.update(o.long("tMs"), o.int("status"), headroom, o.bool("idle"))
                if (i >= steps.size) continue
                val e = steps[i].jsonObject
                val at = "$name step[$i] t=${o.long("tMs")} status=${o.int("status")} headroom=$headroom idle=${o.bool("idle")}"
                m.eq("$at target", e.str("target"), target.name)
                m.eq("$at level", e.str("level"), st.level.name)
                m.eq("$at changed", e.bool("changed"), st.levelChanged)
                m.eq("$at fps", e.int("fps"), st.fps)
                m.eq("$at voting", e.bool("voting"), st.votingEnabled)
                m.eq("$at vlm", e.str("vlm"), st.vlmMode.name)
                m.eq("$at banner", e.bool("banner"), st.banner != null)
            }
        }
        m.assertNone()
    }
}
