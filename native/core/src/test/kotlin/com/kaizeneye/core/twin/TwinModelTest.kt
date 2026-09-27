package com.kaizeneye.core.twin

import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.model.Verdict
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.util.Random

class TwinModelTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private val gh = 8
    private val gw = 8
    private val dim = 16
    private val base = Synth.pattern(gh, gw, dim, 1000 + 11)

    private fun twin(pipeline: PipelineInfo = Synth.pipeline(gh, gw, dim)): TwinModel {
        val frames = Synth.teachFrames(40, gh, gw, dim, seed = 11, base = base)
        return (TeachBuilder.build(frames, pipeline, Synth.meta()) as TeachResult.Success).twin
    }

    private val cov = Synth.discCov(gh, gw, 4.0, 4.0, 2.0)
    private val geo = GeometryFeatures(1500.0, 0.62, 0.9, 0.16, 0.95)

    @Test
    fun judgePaths() {
        val t = twin()
        val rnd = Random(12)
        // A clean view of the taught part passes.
        val good = FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd))
        val pass = t.judge(good, cov, geo, SanityReason.OK)
        assertEquals(Verdict.PASS, pass.verdict)
        assertNull(pass.reason)
        assertTrue(pass.identityOk && pass.geometryOk)
        assertTrue(pass.s <= 1.0)
        // The same part with a local defect (one core patch shifted) is a DEFECT at the defect.
        val bad = FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd, defect = intArrayOf(27), dim = dim, shift = 6f))
        val defect = t.judge(bad, cov, geo, SanityReason.OK)
        assertEquals(Verdict.DEFECT, defect.verdict)
        assertTrue(defect.peakIndex in listOf(18, 19, 20, 26, 27, 28, 29, 34, 35, 36, 37))
        assertTrue(defect.areaPct > 0 && defect.areaPct <= 50.0)
        // A different object fails identity.
        val otherBase = Synth.pattern(gh, gw, dim, 999) { if (it % 2 == 0) 0f else 6f }
        val other = FeatureMap(gh, gw, dim, Synth.view(otherBase, 0.05, rnd))
        val wrong = t.judge(other, cov, geo, SanityReason.OK)
        assertEquals(Verdict.NOT_ENROLLED, wrong.verdict)
        assertEquals(VerdictEngine.REASON_IDENTITY, wrong.reason)
        // Right texture, wrong shape.
        val shape = t.judge(good, cov, geo.copy(area = 6000.0, fill = 0.2), SanityReason.OK)
        assertEquals(Verdict.NOT_ENROLLED, shape.verdict)
        assertEquals(VerdictEngine.REASON_SHAPE, shape.reason)
        assertEquals(listOf("fill", "area"), shape.geometryFailures)
        // Sanity and empty core → REFRAME.
        assertEquals(Verdict.REFRAME, t.judge(null, null, null, SanityReason.MULTIPLE).verdict)
        val noCore = t.judge(good, FloatArray(gh * gw), geo, SanityReason.OK)
        assertEquals(Verdict.REFRAME, noCore.verdict)
        assertEquals("NO_CORE", noCore.reason)
        // Scoring only S gives the same score as the full map (§6.2: background never leaks in).
        val fast = t.judge(bad, cov, geo, SanityReason.OK, params = JudgeParams(fullDistanceMap = false))
        assertEquals(defect.raw, fast.raw, 0.0)
        assertEquals(defect.peakIndex, fast.peakIndex)
        // Sensitivity scales s.
        val lax = t.judge(bad, cov, geo, SanityReason.OK, sensitivity = 2.0)
        assertEquals(defect.s / 2.0, lax.s, 1e-12)
    }

    @Test
    fun canonicalRotationJudgesTheLowerRawCrop() {
        val t = twin()
        val rnd = Random(16)
        val good = CropInput(FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd)), cov)
        val flipped = CropInput(FeatureMap(gh, gw, dim, Synth.view(Synth.pattern(gh, gw, dim, 555), 0.05, rnd)), cov)
        val a = t.judgeCanonical(good, flipped, geo, SanityReason.OK)
        val b = t.judgeCanonical(flipped, good, geo, SanityReason.OK)
        val direct = t.judge(good.features, good.cov, geo, SanityReason.OK)
        assertEquals(direct.raw, a.raw, 0.0)
        assertEquals(direct.raw, b.raw, 0.0)
        assertEquals(Verdict.PASS, a.verdict)
        assertEquals(Verdict.REFRAME, t.judgeCanonical(null, null, null, SanityReason.TOUCHES_BORDER).verdict)
    }

    @Test
    fun prepareFeaturesNormalisesDinoPatchesOnly() {
        val raw = FloatArray(2 * 2 * 3) { (it % 5).toFloat() + 1f }
        val r18 = Synth.pipeline(2, 2, 3)
        assertArrayEquals(raw, r18.prepareFeatures(FeatureMap(2, 2, 3, raw.copyOf())).data, 0f)
        val dino = r18.copy(l2NormalizePatches = true)
        val out = dino.prepareFeatures(FeatureMap(2, 2, 3, raw.copyOf())).data
        for (p in 0 until 4) {
            var n = 0.0
            for (t in 0 until 3) n += out[p * 3 + t].toDouble() * out[p * 3 + t]
            assertEquals(1.0, n, 1e-6)
        }
        assertTrue(dino.fingerprint() != r18.fingerprint())
    }

    @Test
    fun topKViewsRetrievesKeyframes() {
        val t = twin()
        val rnd = Random(13)
        val good = FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd))
        val j = t.judge(good, cov, geo, SanityReason.OK, bankRule = BankRule.TOPK_VIEWS)
        assertEquals(3, j.views!!.size)
        assertEquals(3, j.views!!.toSet().size)
        assertTrue(j.dmap!!.all { !it.isNaN() })
        // The best view has the highest similarity.
        val gq = Descriptor.global(good.data, gh * gw, dim, PatchSets.fromCov(cov, gh, gw).coreIdx)
        val sims = DoubleArray(t.keyframeCount) { Descriptor.dot(gq, t.globals, it) }
        assertEquals(sims.indices.maxByOrNull { sims[it] }, j.views!![0])
    }

    @Test
    fun voteUsesTheMedianCrop() {
        val t = twin()
        val rnd = Random(14)
        val bad = FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd, defect = intArrayOf(27), dim = dim, shift = 6f))
        val first = t.judge(bad, cov, geo, SanityReason.OK)
        val extras = listOf(
            CropInput(FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd)), cov),
            CropInput(bad, FloatArray(gh * gw)),       // no core → not available
        )
        val v = t.vote(first, extras)
        assertEquals(3, v.s.size)
        assertTrue(v.s[2].isNaN())
        // two available values → the higher one
        val want = if (v.s[1] > v.s[0]) 1 else 0
        assertEquals(want, v.chosen)
        assertEquals(2, v.judgement.votes)
    }

    @Test
    fun withNegativesRederivesTheIdentityThreshold() {
        val t = twin()
        // No negatives → the teach τ_id.
        val none = t.withNegatives(emptyList())
        assertEquals(t.thresholds.tauId, none.thresholds.tauId, 0.0)
        assertEquals(IdentityThreshold.RULE_NO_NEGATIVES, none.thresholds.tauIdRule)
        assertNull(none.thresholds.identityMargin)
        // With negatives → midpoint rule on rounded globals.
        val rnd = Random(15)
        val negs = List(4) { Descriptor.l2NormalizeInPlace(DoubleArray(dim) { rnd.nextGaussian() }).map { it.toFloat() }.toFloatArray() }
        val w = t.withNegatives(negs)
        assertEquals(IdentityThreshold.RULE_NEGATIVES, w.thresholds.tauIdRule)
        assertEquals(4, w.negativeCount)
        val rounded = negs.map { com.kaizeneye.core.math.Half.roundedCopy(it) }
        for (i in 0 until 4) {
            var best = Double.NEGATIVE_INFINITY
            for (k in 0 until t.keyframeCount) best = maxOf(best, Descriptor.dot(rounded[i], 0, t.globals, k, dim))
            assertEquals(best, w.negativeSims[i], 0.0)
        }
        val id = IdentityThreshold.derive(t.positives, w.negativeSims)
        assertEquals(id.tauId, w.thresholds.tauId, 0.0)
        assertEquals(id.margin!!, w.thresholds.identityMargin!!, 0.0)
        assertEquals(t.thresholds.tau, w.thresholds.tau, 0.0)
        // Store round trip keeps everything.
        val store = TwinStore(tmp.newFolder("twins"))
        store.save(w, List(w.keyframeCount) { byteArrayOf(1, 2, 3) })
        val back = store.load(w.id, w.pipeline).getOrThrow()
        assertArrayEquals(w.negatives, back.negatives, 0f)
        assertArrayEquals(w.negativeSims, back.negativeSims, 0.0)
        assertEquals(w.thresholds, back.thresholds)
        // A calibrated Twin keeps τ but loses its certificate.
        val sample = CalibrationSample(true, true, true, 0.5, 0.99, geo)
        val (cal, _) = t.calibrate(List(3) { sample })
        val cn = cal.withNegatives(negs)
        assertTrue(cn.thresholds.calibrated)
        assertEquals(cal.thresholds.tau, cn.thresholds.tau, 0.0)
        assertNull(cn.certificate)
    }
}
