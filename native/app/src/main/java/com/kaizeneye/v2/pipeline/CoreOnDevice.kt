package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.legacy.PatchCoreLegacy
import com.kaizeneye.core.math.Binomial
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.twin.GeometryParams
import com.kaizeneye.core.twin.IdentityParams
import com.kaizeneye.core.twin.MaskParams
import com.kaizeneye.core.twin.PipelineInfo
import com.kaizeneye.core.twin.ScoreRule
import com.kaizeneye.core.twin.TeachBuilder
import com.kaizeneye.core.twin.TeachFrame
import com.kaizeneye.core.twin.TeachParams
import com.kaizeneye.core.twin.TeachResult
import com.kaizeneye.core.twin.TwinMeta
import com.kaizeneye.v2.AppGraph
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.json.JSONObject
import kotlin.math.abs

/**
 * On-device run of the core golden vectors (plan "self-test … core"): the phone's ART must reproduce the numbers the JVM
 * unit tests check. Bundled copies of testdata/golden_core.json, golden_app.json (legacy PatchCore port) and the
 * golden_twin.json sections "binomial" and "teach" (Visual Twin §7 end to end) under assets/selftest/golden/.
 */
class CoreOnDevice(private val g: AppGraph) {
    private var checks = 0
    private val failures = ArrayList<String>()

    private fun check(name: String, ok: Boolean, detail: () -> String = { "" }) {
        checks++
        if (!ok && failures.size < 25) failures += "$name ${detail()}"
    }

    private fun f32(name: String, got: Double, want: Double) =
        check(name, abs(got - want) <= 1e-4 * abs(want) + 1e-6) { "got $got want $want" }

    private fun f64(name: String, got: Double, want: Double) =
        check(name, abs(got - want) <= 1e-6 * abs(want) + 1e-9) { "got $got want $want" }

    suspend fun run(out: JSONObject): String = withContext(Dispatchers.Default) {
        val sections = ArrayList<String>()
        asset("golden_core.json")?.let { legacy("golden_core", it); sections += "legacy-core" }
        asset("golden_app.json")?.let { legacy("golden_app", it); sections += "legacy-app" }
        asset("golden_twin.json")?.let { root ->
            root["binomial"]?.let { binomial(it.jsonObject); sections += "binomial" }
            root["teach"]?.let { teach(it.jsonObject); sections += "teach" }
        }
        out.put("sections", sections.joinToString())
        out.put("checks", checks)
        out.put("failures", org.json.JSONArray(failures))
        when {
            sections.isEmpty() -> "SKIP golden vectors not bundled (assets/selftest/golden)"
            failures.isEmpty() -> "PASS $checks checks (${sections.joinToString()}) reproduce the laptop numbers on ART"
            else -> "FAIL ${failures.size} of $checks checks: ${failures.first()}"
        }
    }

    private fun asset(name: String): JsonObject? = try {
        g.context.assets.open("selftest/golden/$name").use { Json.parseToJsonElement(it.readBytes().decodeToString().removePrefix("﻿")).jsonObject }
    } catch (t: Throwable) {
        null
    }

    // ------------------------------------------------------------------------------------------ legacy PatchCore port
    private fun legacy(name: String, j: JsonObject) {
        val p = j.obj("params")
        val gh = p.i("gh")
        val gw = p.i("gw")
        val d = p.i("D")
        val frames = j.arr("frames").map { nested(it, gh * gw * d) }
        val opts = PatchCoreLegacy.EnrolOptions(
            ratio = p.d("ratio"), minK = p.i("minK"), margin = p.d("margin"),
            border = p["border"]?.jsonPrimitive?.int ?: 0, smooth = p["smooth"]?.jsonPrimitive?.content == "true",
        )
        val prof = PatchCoreLegacy.enrol(frames, gh, gw, d, opts, createdAt = 0L)
        val x = FloatArray(frames.size * gh * gw * d).also { buf -> frames.forEachIndexed { i, f -> f.copyInto(buf, i * f.size) } }
        val k = PatchCoreLegacy.bankSize(frames.size * gh * gw, opts.ratio, opts.minK)
        val sel = PatchCoreLegacy.greedyCoreset(x, d, k, 0)
        val want = j.arr("coreset_indices").map { it.jsonPrimitive.int }
        check("$name coreset", sel.toList() == want) { "first mismatch at ${sel.toList().zip(want).indexOfFirst { it.first != it.second }}" }
        val loo = j.arr("loo_scores").map { it.jsonPrimitive.double }
        loo.forEachIndexed { i, v -> f32("$name loo[$i]", prof.looScores[i], v) }
        f32("$name tau", prof.tau, j.d("tau"))
        val r = PatchCoreLegacy.score(nested(j.getValue("test_frame"), gh * gw * d), prof)
        f32("$name raw", r.raw, j.d("raw_score"))
        f32("$name normalised", r.raw / prof.tau, j.d("normalised_score"))
    }

    // ------------------------------------------------------------------------------------------ binomial §10.2
    private fun binomial(sec: JsonObject) {
        for (c in sec.arr("cases")) {
            val inp = c.jsonObject.obj("inputs")
            val e = c.jsonObject.obj("expected")
            val conf = inp.d("conf")
            inp.arr("alphaM").zip(e.arr("alpha")).forEach { (m, a) -> f64("alpha m=$m", Binomial.orderStatisticAlpha(m.jsonPrimitive.int, conf), a.jsonPrimitive.double) }
            inp.arr("cpUpper").zip(e.arr("cpUpper")).forEach { (rn, u) ->
                val o = rn.jsonObject
                val got = Binomial.clopperPearsonUpper(o.i("r"), o.i("n"), conf)
                check("cpUpper $o", abs(got - u.jsonPrimitive.double) <= 1e-9) { "got $got want $u" }
            }
            inp.arr("twoSided").zip(e.arr("twoSided")).forEach { (kn, lu) ->
                val o = kn.jsonObject
                val got = Binomial.clopperPearsonInterval(o.i("k"), o.i("n"), conf)
                val w = lu.jsonObject
                check("twoSided $o", abs(got.lower - w.d("lower")) <= 1e-9 && abs(got.upper - w.d("upper")) <= 1e-9) { "got $got want $w" }
            }
        }
    }

    // ------------------------------------------------------------------------------------------ teach §7 end to end
    private fun teach(sec: JsonObject) {
        for (c in sec.arr("cases")) {
            val co = c.jsonObject
            val e = co.obj("expected")
            if (e.containsKey("error")) continue // failure cases are covered by the JVM tests
            val name = co["name"]?.jsonPrimitive?.content ?: "teach"
            val inp = co.obj("inputs")
            val p = inp.obj("params")
            val params = TeachParams(
                keepPercentile = p.d("keepPercentile"), minKept = p.i("minKept"), kMin = p.i("kMin"), kMax = p.i("kMax"),
                kcEps = p.d("kcEps"), segments = p.i("segments"), ratio = p.d("ratio"), minK = p.i("minK"),
                maxRows = p.i("maxRows"), tauFactor = p.d("tauFactor"), coverageCut = p.d("coverageCut"),
                coreThreshold = p.d("coreThreshold"), scoreRule = ScoreRule.valueOf(p.s("scoreRule")), topFraction = p.d("topFraction"),
                geometry = GeometryParams(
                    kGeo = p.d("kGeo"), floorFill = p.d("fillFloor"), floorAspect = p.d("aspectFloor"),
                    floorSolidity = p.d("solidityFloor"), floorHu1Rel = p.d("hu1FloorFrac"), areaFactor = p.d("areaFactor"),
                ),
                identity = IdentityParams(posPercentile = p.d("posPercentile"), negPercentile = p.d("negPercentile"), minGap = p.d("idMinGap"), madK = p.d("idMadK")),
            )
            val gh = p.i("gh")
            val gw = p.i("gw")
            val dim = p.i("dim")
            val pipeline = PipelineInfo(
                backboneId = "golden", backboneSha256 = "0".repeat(64), inputSize = 8 * gh, gh = gh, gw = gw, dim = dim,
                mask = MaskParams(coreThreshold = params.coreThreshold), knnGraphId = "cpu", scoreRule = params.scoreRule,
            )
            val neg = inp.obj("negatives")
            val negRows = neg.i("rows")
            val negCols = neg.i("cols")
            val negData = floats(neg.getValue("data"))
            val negatives = List(negRows) { negData.copyOfRange(it * negCols, (it + 1) * negCols) }
            val frames = inp.arr("frames").map { fe ->
                val f = fe.jsonObject
                val sane = f["sane"]?.jsonPrimitive?.content == "true"
                TeachFrame(
                    tMs = f.getValue("tMs").jsonPrimitive.long,
                    sanity = if (sane) SanityReason.OK else SanityReason.NO_OBJECT,
                    sharpness = f.dOrNull("sharpness") ?: Double.NaN,
                    cov = f.present("cov")?.let { floats(it) },
                    geometry = f.present("geometry")?.jsonObject?.let { gj ->
                        GeometryFeatures(gj.d("area"), gj.d("fill"), gj.d("aspect"), gj.d("hu1"), gj.d("solidity"))
                    },
                    features = f.present("features")?.jsonObject?.let { fm -> FeatureMap(fm.i("gh"), fm.i("gw"), fm.i("dim"), floats(fm.getValue("data"))) },
                )
            }
            val r = TeachBuilder.build(frames, pipeline, TwinMeta("golden-$name", name, 0L), params, negatives)
            val dg = r.diagnostics
            check("$name accepted", dg.accepted.toList() == e.arr("accepted").map { it.jsonPrimitive.int })
            check("$name kept", dg.kept.toList() == e.arr("kept").map { it.jsonPrimitive.int })
            if (r !is TeachResult.Success) {
                check("$name success", false) { "teach failed: ${(r as TeachResult.Failure).reason}" }
                continue
            }
            val t = r.twin
            f64("$name sharpnessCut", dg.sharpnessCut, e.d("sharpnessCut"))
            val kfs = e.arr("keyframes")
            check("$name keyframes", kfs.size == t.keyframeCount) { "got ${t.keyframeCount} want ${kfs.size}" }
            kfs.forEachIndexed { k, kf -> if (k < dg.keyframeFrames.size) check("$name kf[$k]", kf.jsonObject.i("frame") == dg.keyframeFrames[k]) }
            check("$name pooledRows", dg.pooledRows == e.i("pooledRows")) { "got ${dg.pooledRows} want ${e.i("pooledRows")}" }
            check("$name coreset", dg.coresetIndices.toList() == e.arr("coresetIndices").map { it.jsonPrimitive.int })
            val lso = e.arr("lso").map { it.jsonPrimitive.double }
            lso.forEachIndexed { i, v -> if (i < dg.lso.size) f32("$name lso[$i]", dg.lso[i], v) }
            f32("$name tau", t.thresholds.tau, e.d("tau"))
            e.arr("positives").forEachIndexed { i, v -> if (i < t.positives.size) f32("$name positives[$i]", t.positives[i], v.jsonPrimitive.double) }
            val ident = e.obj("identity")
            val want = if (negRows == 0) ident.obj("noNegatives") else ident.obj("withNegatives")
            f32("$name tauId", t.thresholds.tauId, want.d("tauId"))
        }
    }

    // ------------------------------------------------------------------------------------------ JSON helpers
    private fun JsonObject.obj(k: String) = getValue(k).jsonObject
    private fun JsonObject.arr(k: String) = getValue(k).jsonArray
    private fun JsonObject.i(k: String) = getValue(k).jsonPrimitive.int
    private fun JsonObject.d(k: String) = getValue(k).jsonPrimitive.double
    private fun JsonObject.s(k: String) = getValue(k).jsonPrimitive.content
    private fun JsonObject.dOrNull(k: String): Double? = this[k]?.takeIf { it !is JsonNull }?.jsonPrimitive?.double
    private fun JsonObject.present(k: String): JsonElement? = this[k]?.takeIf { it !is JsonNull }

    private fun floats(e: JsonElement): FloatArray {
        val a = e.jsonArray
        return FloatArray(a.size) { a[it].jsonPrimitive.double.toFloat() }
    }

    /** Flattens nested lists of numbers (legacy golden frames [gh][gw][D]) into a FloatArray of [size]. */
    private fun nested(e: JsonElement, size: Int): FloatArray {
        val out = FloatArray(size)
        var i = 0
        fun walk(x: JsonElement) {
            if (x is JsonArray) x.forEach { walk(it) } else out[i++] = x.jsonPrimitive.double.toFloat()
        }
        walk(e)
        check("nested size", i == size) { "got $i want $size" }
        return out
    }
}
