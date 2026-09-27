package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Auroc
import com.kaizeneye.core.math.Binomial
import com.kaizeneye.core.math.Half
import com.kaizeneye.core.math.Stats
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.model.Verdict
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import com.kaizeneye.core.twin.GoldenTwin.beta
import com.kaizeneye.core.twin.GoldenTwin.f32
import com.kaizeneye.core.twin.GoldenTwin.f64

/** My sections of testdata/golden_twin.json (spec §15); each test is skipped when the file/section is missing. */
class GoldenTwinTest {

    @get:org.junit.Rule
    val testName = org.junit.rules.TestName()

    @org.junit.After
    fun printWorstError() {
        val s = GoldenTwin.summary()
        if (s.isNotEmpty()) println("[golden] ${testName.methodName}: $s")
    }

    private fun name(c: JsonObject) = c.sOrNull("name") ?: "?"

    // ---------------------------------------------------------------------------------------------------- stats §0
    @Test
    fun stats() {
        for (c in GoldenTwin.cases("stats")) {
            val n = name(c)
            val x = c.obj("inputs").getValue("x").doubles()
            val qs = c.obj("inputs").getValue("qs").doubles()
            val e = c.obj("expected")
            val ps = e.getValue("percentiles").doubles()
            qs.forEachIndexed { i, q -> f64("$n P$q", Stats.percentile(x, q), ps[i]) }
            f64("$n median", Stats.median(x), e.d("median"))
            f64("$n mad", Stats.mad(x), e.d("mad"))
            f64("$n robustSigma", 1.4826 * Stats.mad(x), e.d("robustSigma"))
            f64("$n mean", Stats.mean(x), e.d("mean"))
            f64("$n std", Stats.std(x), e.d("std"))
        }
    }

    // ----------------------------------------------------------------------------------------------------- half §0
    @Test
    fun half() {
        for (c in GoldenTwin.cases("half")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            when (c.sOrNull("kind")) {
                "encode" -> {
                    val bits = inp.getValue("f32Bits").jsonArray.map { it.jsonPrimitive.long.toInt() }
                    val labels = inp["labels"]?.strings() ?: List(bits.size) { "" }
                    val want = e.getValue("f16Bits").ints()
                    val wantDec = e.getValue("decodedF32Bits").jsonArray.map { it.jsonPrimitive.long.toInt() }
                    bits.forEachIndexed { i, b ->
                        val f = Float.fromBits(b)
                        val h = Half.fromFloat(f)
                        if (labels[i].startsWith("nan")) {
                            assertTrue("$n[$i] ${labels[i]}", Half.toFloat(h).isNaN())
                        } else {
                            assertEquals("$n[$i] ${labels[i]} f16", "%04x".format(want[i]), "%04x".format(h))
                            assertEquals("$n[$i] ${labels[i]} widened", "%08x".format(wantDec[i]), "%08x".format(Half.toFloat(h).toRawBits()))
                            assertEquals("$n[$i] double path", "%04x".format(want[i]), "%04x".format(Half.fromDouble(f.toDouble())))
                        }
                    }
                }
                "decode" -> {
                    val hs = inp.getValue("f16Bits").ints()
                    val want = e.getValue("f32Bits").jsonArray.map { it.jsonPrimitive.long.toInt() }
                    hs.forEachIndexed { i, h ->
                        val got = Half.toFloat(h)
                        if (Float.fromBits(want[i]).isNaN()) assertTrue("$n[$i]", got.isNaN())
                        else assertEquals("$n[$i] ${"%04x".format(h)}", "%08x".format(want[i]), "%08x".format(got.toRawBits()))
                    }
                }
                else -> error("$n: unknown half kind")
            }
        }
    }

    // ----------------------------------------------------------------------------------------------- binomial §10.2
    @Test
    fun binomial() {
        for (c in GoldenTwin.cases("binomial")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            val conf = inp.d("conf")
            val ms = inp.getValue("alphaM").ints()
            val alpha = e.getValue("alpha").doubles()
            ms.forEachIndexed { i, m -> f64("$n alpha m=$m", Binomial.orderStatisticAlpha(m, conf), alpha[i]) }
            val cp = e.getValue("cpUpper").doubles()
            inp.arr("cpUpper").forEachIndexed { i, rn ->
                val o = rn.jsonObject
                beta("$n cpUpper(${o.i("r")},${o.i("n")})", Binomial.clopperPearsonUpper(o.i("r"), o.i("n"), conf), cp[i])
            }
            val two = e.arr("twoSided")
            inp.arr("twoSided").forEachIndexed { i, kn ->
                val o = kn.jsonObject
                val iv = Binomial.clopperPearsonInterval(o.i("k"), o.i("n"), conf)
                beta("$n twoSided(${o.i("k")},${o.i("n")}).lower", iv.lower, two[i].jsonObject.d("lower"))
                beta("$n twoSided(${o.i("k")},${o.i("n")}).upper", iv.upper, two[i].jsonObject.d("upper"))
            }
            val bi = e.getValue("betaInc").doubles()
            inp.arr("betaInc").forEachIndexed { i, xab ->
                val o = xab.jsonObject
                beta("$n I_${o.d("x")}(${o.d("a")},${o.d("b")})", Binomial.regularizedIncompleteBeta(o.d("x"), o.d("a"), o.d("b")), bi[i])
            }
            val binv = e.getValue("betaInv").doubles()
            inp.arr("betaInv").forEachIndexed { i, pab ->
                val o = pab.jsonObject
                beta("$n betaInv(${o.d("p")};${o.d("a")},${o.d("b")})", Binomial.betaInv(o.d("p"), o.d("a"), o.d("b")), binv[i])
            }
            val wp = e.getValue("withinPart").doubles()
            inp.getValue("withinPartBUsed").ints().forEachIndexed { i, b ->
                f64("$n withinPart B=$b", Calibration.withinPartAlpha(b, conf), wp[i])
            }
        }
    }

    // --------------------------------------------------------------------------------------------------- auroc §14
    @Test
    fun auroc() {
        for (c in GoldenTwin.cases("auroc")) {
            val inp = c.obj("inputs")
            f64(name(c), Auroc.auroc(inp.getValue("pos").doubles(), inp.getValue("neg").doubles()), c.obj("expected").d("auroc"))
        }
    }

    // ------------------------------------------------------------------------------------------------------- knn §5
    @Test
    fun knn() {
        for (c in GoldenTwin.cases("knn")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            val feats = inp.getValue("feats").fmap()
            val bank = inp.getValue("bank").mat()
            val mask = inp.d("knnMask").toFloat()
            assertEquals(KNN_MASK, mask, 0f)
            val excluded = inp.getValue("excluded").ints()
            val extra = FloatArray(bank.rows) { if (excluded[it] != 0) KNN_MASK else 0f }
            val p = feats.gh * feats.gw
            val out = DoubleArray(p)
            Knn.CPU.minSq(feats.data, p, bank.data, bank.rows, bank.cols, extra, out)
            val want = e.getValue("minSq").doubles()
            val wantD = e.getValue("d").doubles()
            if (e.b("allExcluded")) {
                for (i in 0 until p) assertTrue("$n minSq[$i] = ${out[i]}", out[i] >= 1e30)
            } else {
                for (i in 0 until p) {
                    f32("$n minSq[$i]", out[i], want[i])
                    f32("$n d[$i]", Knn.distance(out[i]), wantD[i])
                }
            }
        }
    }

    // --------------------------------------------------------------------------------------------- scoring §6.2-6.4
    @Test
    fun scoring() {
        for (c in GoldenTwin.cases("scoring")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            val gh = inp.i("gh")
            val gw = inp.i("gw")
            val sets = PatchSets.fromCore(inp.getValue("core").mask(), gh, gw)
            assertArrayEquals("$n S", inp.getValue("S").mask(), sets.s)
            val rule = ScoreRule.valueOf(inp.s("scoreRule"))
            val top = inp["params"]?.jsonObject?.dOrNull("topFraction") ?: Scoring.DEFAULT_TOP_FRACTION
            val r = Scoring.score(inp.getValue("dmap").doubles(), sets, inp.d("tau"), inp.d("sensitivity"), rule, top)
            val sm = e.getValue("sm").doubles()
            for (p in sets.sIdx) f64("$n sm[$p]", r.smoothed[p], sm[p])
            f64("$n raw", r.raw, e.d("raw"))
            f64("$n s", r.s, e.d("s"))
            assertEquals("$n topN", e.i("topN"), r.topN)
            val pk = e.obj("peak")
            assertEquals("$n peak", pk.i("index"), r.peak)
            assertEquals("$n peak row", pk.i("row"), r.peakRow)
            assertEquals("$n peak col", pk.i("col"), r.peakCol)
            assertEquals("$n coreCount", e.i("coreCount"), r.coreCount)
            assertEquals("$n anomalousCount", e.i("anomalousCount"), r.anomalousCount)
            f64("$n anomalousFraction", r.anomalousFraction, e.d("anomalousFraction"))
            f64("$n areaPct", r.areaPct, e.d("areaPct"))
        }
    }

    // ------------------------------------------------------------------------------------------------------- teach §7
    private fun teachParams(p: JsonObject) = TeachParams(
        keepPercentile = p.d("keepPercentile"), minKept = p.i("minKept"), kMin = p.i("kMin"), kMax = p.i("kMax"),
        kcEps = p.d("kcEps"), segments = p.i("segments"), ratio = p.d("ratio"), minK = p.i("minK"),
        maxRows = p.i("maxRows"), tauFactor = p.d("tauFactor"), coverageCut = p.d("coverageCut"),
        coreThreshold = p.d("coreThreshold"), scoreRule = ScoreRule.valueOf(p.s("scoreRule")),
        topFraction = p.d("topFraction"),
        geometry = GeometryParams(
            kGeo = p.d("kGeo"), floorFill = p.d("fillFloor"), floorAspect = p.d("aspectFloor"),
            floorSolidity = p.d("solidityFloor"), floorHu1Rel = p.d("hu1FloorFrac"), areaFactor = p.d("areaFactor"),
        ),
        identity = IdentityParams(
            posPercentile = p.d("posPercentile"), negPercentile = p.d("negPercentile"), minGap = p.d("idMinGap"),
            madK = p.d("idMadK"),
        ),
    )

    private fun goldenPipeline(gh: Int, gw: Int, dim: Int, coreThreshold: Double, rule: ScoreRule) = PipelineInfo(
        backboneId = "golden", backboneSha256 = "0".repeat(64), inputSize = 8 * gh, gh = gh, gw = gw, dim = dim,
        mask = MaskParams(coreThreshold = coreThreshold), knnGraphId = "cpu", scoreRule = rule,
    )

    private fun geomodel(what: String, got: GeometryModel, e: JsonObject) {
        f64("$what kGeo", got.kGeo, e.d("kGeo"))
        f64("$what meanArea", got.meanArea, e.d("meanArea"))
        e.dOrNull("areaFactor")?.let { f64("$what areaFactor", got.areaFactor, it) }
        for ((k, band) in listOf("fill" to got.fill, "aspect" to got.aspect, "hu1" to got.hu1, "solidity" to got.solidity)) {
            val v = e.getValue(k).doubles()
            f64("$what $k mean", band.mean, v[0])
            f64("$what $k sigma", band.sigma, v[1])
        }
    }

    private fun geomodelFrom(e: JsonObject) = GeometryModel(
        kGeo = e.d("kGeo"), meanArea = e.d("meanArea"),
        fill = e.getValue("fill").doubles().let { FeatureBand(it[0], it[1]) },
        aspect = e.getValue("aspect").doubles().let { FeatureBand(it[0], it[1]) },
        hu1 = e.getValue("hu1").doubles().let { FeatureBand(it[0], it[1]) },
        solidity = e.getValue("solidity").doubles().let { FeatureBand(it[0], it[1]) },
        areaFactor = e.dOrNull("areaFactor") ?: GeometryParams().areaFactor,
    )

    private fun identity(what: String, got: IdentityThreshold, e: JsonObject, compareRule: Boolean) {
        f32("$what tauId", got.tauId, e.d("tauId"))
        if (compareRule) assertEquals("$what rule", e.s("rule"), got.rule)
        val m = e.dOrNull("margin")
        if (m == null) assertNull("$what margin", got.margin) else f32("$what margin", got.margin!!, m)
        val o = e["overlap"]
        if (o == null || o is JsonNull) assertNull("$what overlap", got.overlap)
        else assertEquals("$what overlap", o.jsonPrimitive.content.toBoolean(), got.overlap)
    }

    @Test
    fun teach() {
        for (c in GoldenTwin.cases("teach")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            val p = inp.obj("params")
            val params = teachParams(p)
            val gh = p.i("gh")
            val gw = p.i("gw")
            val dim = p.i("dim")
            assertEquals(KNN_MASK, p.d("knnMask").toFloat(), 0f)
            val pipeline = goldenPipeline(gh, gw, dim, params.coreThreshold, params.scoreRule)
            val negMat = inp.getValue("negatives").mat()
            val negatives = List(negMat.rows) { negMat.data.copyOfRange(it * negMat.cols, (it + 1) * negMat.cols) }
            val frames = inp.arr("frames").map { fe ->
                val f = fe.jsonObject
                val sane = f.b("sane")
                TeachFrame(
                    tMs = f.l("tMs"),
                    sanity = if (sane) SanityReason.OK else SanityReason.NO_OBJECT,
                    sharpness = f.dOrNull("sharpness") ?: Double.NaN,
                    cov = if (f.has("cov")) f.getValue("cov").floats() else null,
                    geometry = if (f.has("geometry")) f.getValue("geometry").geometry() else null,
                    features = if (f.has("features")) f.getValue("features").fmap() else null,
                )
            }
            val r = TeachBuilder.build(frames, pipeline, TwinMeta("golden-$n", n, 0L), params, negatives)
            assertArrayEquals("$n accepted", e.getValue("accepted").ints(), r.diagnostics.accepted)
            assertArrayEquals("$n kept", e.getValue("kept").ints(), r.diagnostics.kept)
            if (c.sOrNull("kind") == "fail" || e.has("error")) {
                assertTrue("$n should fail", r is TeachResult.Failure)
                assertEquals("$n error", e.s("error"), (r as TeachResult.Failure).reason.name)
                continue
            }
            assertTrue("$n failed: ${(r as? TeachResult.Failure)?.reason}", r is TeachResult.Success)
            val t = (r as TeachResult.Success).twin
            val dg = r.diagnostics
            f64("$n sharpnessCut", dg.sharpnessCut, e.d("sharpnessCut"))
            val kfs = e.arr("keyframes")
            assertEquals("$n keyframe count", kfs.size, t.keyframeCount)
            kfs.forEachIndexed { k, kf ->
                val o = kf.jsonObject
                assertEquals("$n kf[$k] index", o.i("index"), k)
                assertEquals("$n kf[$k] frame", o.i("frame"), dg.keyframeFrames[k])
                assertEquals("$n kf[$k] tMs", o.l("tMs"), t.keyframes[k].tMs)
                assertEquals("$n kf[$k] segment", o.i("segment"), t.keyframes[k].segment)
                f64("$n kf[$k] sharpness", t.keyframes[k].sharpness, o.d("sharpness"))
            }
            val kc = e.getValue("kcMinDist").doubles()
            assertEquals("$n kcMinDist size", kc.size, dg.kcMinDist.size)
            kc.forEachIndexed { i, v -> f32("$n kcMinDist[$i]", dg.kcMinDist[i], v) }
            assertEquals("$n kcStop", e.s("kcStop"), dg.kcStop!!.name)
            val next = e.dOrNull("kcNext")
            if (next == null) assertNull("$n kcNext", dg.kcNext) else f32("$n kcNext", dg.kcNext!!, next)
            assertEquals("$n t0", e.l("t0"), dg.t0)
            assertEquals("$n t1", e.l("t1"), dg.t1)
            assertEquals("$n segmentsUsed", e.i("segmentsUsed"), dg.segmentsUsed)
            assertEquals("$n pooledRows", e.i("pooledRows"), dg.pooledRows)
            assertEquals("$n bankRows", e.i("bankRows"), t.bankRows)
            assertArrayEquals("$n coresetIndices", e.getValue("coresetIndices").ints(), dg.coresetIndices)
            val bank = e.getValue("bank").mat()
            assertEquals("$n bank rows", bank.rows, t.bankRows)
            for (i in bank.data.indices) f32("$n bank[$i]", t.bank[i].toDouble(), bank.data[i].toDouble())
            assertArrayEquals("$n bankKf", e.getValue("bankKf").ints(), t.bankKf)
            assertArrayEquals("$n bankSeg", e.getValue("bankSeg").ints(), t.bankSeg)
            val globals = e.getValue("globals").mat()
            for (i in globals.data.indices) f32("$n globals[$i]", t.globals[i].toDouble(), globals.data[i].toDouble())
            assertEquals("$n lsoMode", e.s("lsoMode"), dg.lsoMode!!.name)
            val lso = e.getValue("lso").doubles()
            assertArrayEquals("$n lsoKeyframes", e.getValue("lsoKeyframes").ints(), dg.lsoKeyframes)
            lso.forEachIndexed { i, v -> f32("$n lso[$i]", dg.lso[i], v) }
            f32("$n tauTeach", t.thresholds.tauTeach, e.d("tauTeach"))
            f32("$n tau", t.thresholds.tau, e.d("tau"))
            val pos = e.getValue("positives").doubles()
            assertEquals("$n positives size", pos.size, t.positives.size)
            pos.forEachIndexed { i, v -> f32("$n positives[$i]", t.positives[i], v) }
            val idE = e.obj("identity")
            identity("$n noNegatives", IdentityThreshold.derive(t.positives, DoubleArray(0), params.identity), idE.obj("noNegatives"), true)
            val wn = idE["withNegatives"]
            if (wn != null && wn !is JsonNull) {
                val w = wn.jsonObject
                val sims = w.getValue("negSims").doubles()
                sims.forEachIndexed { i, v -> f32("$n negSims[$i]", dg.negativeSims[i], v) }
                identity("$n withNegatives", IdentityThreshold.derive(t.positives, dg.negativeSims, params.identity), w, false)
                f32("$n twin tauId", t.thresholds.tauId, w.d("tauId"))
            } else {
                f32("$n twin tauId", t.thresholds.tauId, idE.obj("noNegatives").d("tauId"))
            }
            geomodel("$n geometry", t.thresholds.geometry, e.obj("geometry"))
            f64("$n coverageCut", t.thresholds.coverageCut, e.d("coverageCut"))
        }
    }

    // ---------------------------------------------------------------------------------------------------- verdicts §9
    @Test
    fun verdicts() {
        for (c in GoldenTwin.cases("verdicts")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            val tw = inp.obj("twin")
            val pr = inp.obj("params")
            assertEquals(KNN_MASK, pr.d("knnMask").toFloat(), 0f)
            val gh = tw.i("gh")
            val gw = tw.i("gw")
            val dim = tw.i("dim")
            val np = gh * gw
            val bank = tw.getValue("bank").mat()
            val globals = tw.getValue("globals").mat()
            val kfFeats = tw.getValue("kfFeats").mat()
            val kfCov = tw.getValue("kfCov").mat()
            val bankKf = tw.getValue("bankKf").ints()
            val bankSeg = tw.getValue("bankSeg").ints()
            val k = globals.rows
            val segOf = IntArray(k).also { s -> bankKf.forEachIndexed { i, kf -> s[kf] = bankSeg[i] } }
            val geo = geomodelFrom(tw.obj("geometry"))
            val pipeline = goldenPipeline(gh, gw, dim, pr.d("coreThreshold"), ScoreRule.SMOOTHED_MAX)
            val twin = TwinModel(
                id = "golden", name = "golden", createdAtMs = 0L, pipeline = pipeline, fingerprint = pipeline.fingerprint(),
                teach = TeachStats(0, 0, 0, k, segOf.distinct().size, 0, bank.rows, 0),
                keyframes = List(k) { KeyframeInfo(it, 0L, segOf[it], 0.0, KeyframeInfo.fileName(it)) },
                globals = globals.data, bank = bank.data, bankKf = bankKf, bankSeg = bankSeg, kfFeats = kfFeats.data,
                kfCov = kfCov.data, lso = DoubleArray(0), positives = DoubleArray(0), negativeSims = DoubleArray(0),
                negatives = FloatArray(0),
                thresholds = Thresholds(
                    tau = tw.d("tau"), tauTeach = tw.d("tau"), tauFactor = 1.4, calibrated = false, tauId = tw.d("tauId"),
                    tauIdRule = "golden", identityMargin = null, coverageCut = tw.d("coverageCut"), sensitivity = 1.0,
                    geometry = geo,
                ),
                certificate = null,
            )
            assertEquals(np * dim, kfFeats.cols)
            val vp = VoteParams(pr.d("voteLo"), pr.d("voteHi"), pr.i("voteMaxExtra"))
            val jp = JudgeParams(topFraction = pr.d("topFraction"), topKViews = pr.i("topKViews"), vote = vp)
            val results = e.arr("results")
            inp.arr("queries").forEachIndexed { qi, qe ->
                val q = qe.jsonObject
                val qn = "$n/${q.sOrNull("name") ?: qi}"
                val want = results[qi].jsonObject
                val sanity = SanityReason.valueOf(q.s("sanity"))
                val rule = ScoreRule.valueOf(q.s("scoreRule"))
                val bankRule = BankRule.valueOf(q.s("bankRule"))
                val features = if (q.has("features")) q.getValue("features").fmap() else null
                val cov = if (q.has("cov")) q.getValue("cov").floats() else null
                val geometry = if (q.has("geometry")) q.getValue("geometry").geometry() else null
                val j = twin.judge(features, cov, geometry, sanity, Knn.CPU, q.d("sensitivity"), rule, bankRule, jp)
                assertEquals("$qn verdict", want.s("verdict"), j.verdict.name)
                assertEquals("$qn reason", want.sOrNull("reason"), j.reason?.uppercase())
                if (j.verdict == Verdict.REFRAME) {
                    assertTrue("$qn REFRAME numeric fields are null", want["s"] == null || want["s"] is JsonNull)
                    assertEquals("$qn finalVerdict", want.s("finalVerdict"), j.verdict.name)
                    return@forEachIndexed
                }
                f32("$qn sim", j.sim, want.d("sim"))
                f32("$qn raw", j.raw, want.d("raw"))
                f32("$qn s", j.s, want.d("s"))
                assertEquals("$qn idOk", want.b("idOk"), j.identityOk)
                assertEquals("$qn geoOk", want.b("geoOk"), j.geometryOk)
                assertEquals("$qn geoFailures", want.getValue("geoFailures").strings(), j.geometryFailures)
                val pk = want.obj("peak")
                assertEquals("$qn peak", pk.i("index"), j.peakIndex)
                assertEquals("$qn peak row", pk.i("row"), j.peakRow)
                assertEquals("$qn peak col", pk.i("col"), j.peakCol)
                assertEquals("$qn coreCount", want.i("coreCount"), j.coreCount)
                f32("$qn anomalousFraction", j.anomalousFraction, want.d("anomalousFraction"))
                f32("$qn areaPct", j.areaPct, want.d("areaPct"))
                val dm = want.getValue("dmap").doubles()
                for (p in 0 until np) f32("$qn dmap[$p]", j.dmap!![p], dm[p])
                val views = want["views"]
                if (views == null || views is JsonNull) assertNull("$qn views", j.views)
                else assertArrayEquals("$qn views", views.ints(), j.views)
                var final = j.verdict
                val voteE = want["vote"]
                val voted = q.b("voting") && VerdictEngine.needsVote(j, vp)
                assertEquals("$qn vote ran", voteE != null && voteE !is JsonNull, voted)
                if (voted) {
                    val extras = q.arr("extraCrops").map { x ->
                        val o = x.jsonObject
                        CropInput(o.getValue("features").fmap(), o.getValue("cov").floats())
                    }
                    val v = twin.vote(j, extras, Knn.CPU, q.d("sensitivity"), rule, bankRule, jp)
                    val ve = voteE!!.jsonObject
                    val ss = ve.getValue("s").doubles()
                    assertEquals("$qn vote s count", ss.size, v.s.count { !it.isNaN() })
                    ss.forEachIndexed { i, sv -> f32("$qn vote s[$i]", v.s[i], sv) }
                    assertEquals("$qn vote chosen", ve.i("chosen"), v.chosen)
                    f32("$qn vote finalS", v.judgement.s, ve.d("finalS"))
                    assertEquals("$qn vote verdict", ve.s("verdict"), v.judgement.verdict.name)
                    assertEquals("$qn vote reason", ve.sOrNull("reason"), v.judgement.reason?.uppercase())
                    val vpk = ve.obj("peak")
                    assertEquals("$qn vote peak", vpk.i("index"), v.judgement.peakIndex)
                    f32("$qn vote anomalousFraction", v.judgement.anomalousFraction, ve.d("anomalousFraction"))
                    f32("$qn vote areaPct", v.judgement.areaPct, ve.d("areaPct"))
                    final = v.judgement.verdict
                }
                assertEquals("$qn finalVerdict", want.s("finalVerdict"), final.name)
            }
        }
    }

    // ------------------------------------------------------------------------------------------------- calibration §10
    @Test
    fun calibration() {
        for (c in GoldenTwin.cases("calibration")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            val p = inp.obj("params")
            val params = CalibrationParams(
                conf = p.d("conf"),
                geometry = GeometryParams(
                    kGeo = p.d("kGeo"), floorFill = p.d("fillFloor"), floorAspect = p.d("aspectFloor"),
                    floorSolidity = p.d("solidityFloor"), floorHu1Rel = p.d("hu1FloorFrac"), areaFactor = p.d("areaFactor"),
                ),
                identity = IdentityParams(
                    posPercentile = p.d("posPercentile"), negPercentile = p.d("negPercentile"), minGap = p.d("idMinGap"),
                    madK = p.d("idMadK"),
                ),
                latencyWindow = p.i("latencyWindow"),
            )
            val samples = inp.arr("samples").map { se ->
                val s = se.jsonObject
                CalibrationSample(
                    sanityOk = s.b("sanityOk"), idOk = s.b("idOk"), geoOk = s.b("geoOk"), raw = s.dOrNull("raw"),
                    sim = s.dOrNull("sim"), geometry = if (s.has("geometry")) s.getValue("geometry").geometry() else null,
                )
            }
            val teachGeo = GeometryModel(3.0, 1.0, FeatureBand(0.0, 1.0), FeatureBand(0.0, 1.0), FeatureBand(0.0, 1.0), FeatureBand(0.0, 1.0))
            val r = Calibration.calibrate(
                inp.d("tauTeach"), inp.getValue("positives").doubles(), inp.getValue("negSims").doubles(), teachGeo,
                inp.i("segmentsUsed"), samples, params, inp.getValue("latencyMs").doubles(),
            )
            val cert = r.certificate
            assertEquals("$n n", e.i("n"), cert.n)
            assertEquals("$n nSanityOk", e.i("nSanityOk"), cert.nSanityOk)
            assertEquals("$n m", e.i("m"), cert.m)
            f64("$n tauCal", r.tauCal, e.d("tauCal"))
            e.dOrNull("calMax").let { if (it == null) assertNull(r.calMax) else f64("$n calMax", r.calMax!!, it) }
            e.dOrNull("alpha").let { if (it == null) assertNull(cert.alpha) else f64("$n alpha", cert.alpha!!, it) }
            val idE = e.obj("identity")
            f64("$n tauId", r.identity.tauId, idE.d("tauId"))
            idE.dOrNull("margin").let { if (it == null) assertNull(r.identity.margin) else f64("$n margin", r.identity.margin!!, it) }
            if (r.identity.margin == null) assertEquals("$n rule", idE.s("rule"), r.identity.rule)
            val ov = idE["overlap"]
            if (ov == null || ov is JsonNull) assertNull(r.identity.overlap) else assertEquals("$n overlap", ov.jsonPrimitive.content.toBoolean(), r.identity.overlap)
            if (e.has("geometry")) geomodel("$n geometry", r.geometry, e.obj("geometry"))
            val ce = e.obj("certificate")
            for ((key, got) in listOf("sanity" to cert.sanity, "identity" to cert.identity, "geometry" to cert.geometry)) {
                val g = ce.obj(key)
                assertEquals("$n $key k", g.i("k"), got.k)
                assertEquals("$n $key n", g.i("n"), got.n)
                assertEquals("$n $key rejections", g.i("rejections"), got.rejections)
                beta("$n $key upper", got.upper, g.d("upper"))
            }
            ce.dOrNull("identityMargin").let { if (it == null) assertNull(cert.identityMargin) else f64("$n identityMargin", cert.identityMargin!!, it) }
            f64("$n withinPart", cert.withinPart, ce.d("withinPart"))
            ce.dOrNull("latencyP50")?.let { f64("$n latencyP50", cert.latencyP50Ms!!, it) }
            ce.dOrNull("latencyP95")?.let { f64("$n latencyP95", cert.latencyP95Ms!!, it) }
            val sens = inp.getValue("sensitivities").doubles()
            val valid = e.getValue("valid").jsonArray.map { it.jsonPrimitive.content.toBoolean() }
            sens.forEachIndexed { i, s -> assertEquals("$n valid[$i] (sensitivity $s)", valid[i], cert.isValid(r.tauCal, s)) }
        }
    }

    // ------------------------------------------------------------------------------------------------- twin_json §8
    @Test
    fun twinJson() {
        for (c in GoldenTwin.cases("twin_json")) {
            val n = name(c)
            val inp = c.obj("inputs")
            val e = c.obj("expected")
            when (c.sOrNull("kind")) {
                "pipeline" -> {
                    val pj = inp.getValue("pipeline")
                    val pipeline = Json.decodeFromJsonElement(PipelineInfo.serializer(), pj)
                    assertEquals("$n canonical (typed)", e.s("canonical"), pipeline.canonicalJson())
                    assertEquals("$n canonical (tree)", e.s("canonical"), CanonicalJson.write(pj))
                    assertEquals("$n fingerprint", e.s("fingerprint"), pipeline.fingerprint())
                }
                "f16File" -> {
                    val rows = inp.i("rows")
                    val cols = inp.i("cols")
                    val values = inp.getValue("values").floats()
                    val bytes = F16File.encode(rows, cols, values)
                    assertEquals("$n bytes", e.s("bytesHex"), bytes.joinToString("") { "%02x".format(it) })
                    val bits = e.getValue("f16Bits").ints()
                    values.forEachIndexed { i, v -> assertEquals("$n f16[$i]", bits[i], Half.fromFloat(v)) }
                    val back = F16File.decode(bytes)
                    assertEquals(rows, back.rows)
                    assertEquals(cols, back.cols)
                    for (i in values.indices) assertEquals(Half.toFloat(bits[i]), back.data[i], 0f)
                }
                else -> error("$n: unknown twin_json kind")
            }
        }
    }

}
