package com.kaizeneye.core.twin

import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.model.Verdict
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.Random

/** The FIT gate (Fit.kt): threshold rules, verdict order, judging, calibration, negatives, storage. */
class FitGateTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private val gh = 16
    private val gw = 16
    private val dim = 16
    private val base = Synth.pattern(gh, gw, dim, 1000 + 21)
    private val cov = Synth.discCov(gh, gw, 8.0, 8.0, 6.0)
    private val geo = GeometryFeatures(1500.0, 0.62, 0.9, 0.16, 0.95)

    private fun twin(fit: Boolean = true): TwinModel {
        val frames = Synth.teachFrames(48, gh, gw, dim, seed = 21, base = base, radius = 6.0)
        val params = TeachParams(fit = if (fit) FitParams() else null)
        return (TeachBuilder.build(frames, Synth.pipeline(gh, gw, dim), Synth.meta(), params) as TeachResult.Success).twin
    }

    @Test
    fun thresholdRules() {
        val pos = doubleArrayOf(0.30, 0.35, 0.50, 0.40)
        // No negatives: factor × the largest same-part fit.
        val a = FitThreshold.derive(pos, DoubleArray(0))
        assertEquals(0.55, a.tauFit, 1e-12)
        assertEquals(FitThreshold.RULE_LSO, a.rule)
        assertNull(a.margin)
        // Nearest negative far away: never looser than factor × hi.
        val far = FitThreshold.derive(pos, doubleArrayOf(0.9, 1.2))
        assertEquals(0.55, far.tauFit, 1e-12)
        assertEquals(FitThreshold.RULE_MIDPOINT, far.rule)
        assertEquals(0.4, far.margin!!, 1e-12)
        // Nearest negative just above the same-part maximum: the midpoint (between hi and lo, so it still rejects the negative).
        val near = FitThreshold.derive(pos, doubleArrayOf(0.54, 0.7))
        assertEquals(0.52, near.tauFit, 1e-12)
        assertTrue(near.tauFit > 0.50 && near.tauFit < 0.54)
        // A negative inside the same-part spread cannot be separated: keep factor × hi and say so.
        val overlap = FitThreshold.derive(pos, doubleArrayOf(0.45))
        assertEquals(FitThreshold.RULE_OVERLAP, overlap.rule)
        assertTrue(overlap.margin!! < 0)
        assertEquals(0.55, overlap.tauFit, 1e-12)
        // Calibration fits join the same-part spread.
        val cal = FitThreshold.derive(pos, DoubleArray(0), FitParams(), doubleArrayOf(0.6))
        assertEquals(0.66, cal.tauFit, 1e-12)
        // The slider moves the gate by its square root.
        assertTrue(a.passes(0.55, 1.0))
        assertFalse(a.passes(0.56, 1.0))
        assertTrue(a.passes(0.56, 1.44))
        assertTrue(a.passes(Double.NaN))
    }

    @Test
    fun fitComesBeforeTheScoreVerdictAndAfterIdentityAndShape() {
        val t = twin()
        val rnd = Random(22)
        val good = FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd))
        val j = t.judge(good, cov, geo, SanityReason.OK)
        assertEquals(Verdict.PASS, j.verdict)
        assertTrue(j.fitOk)
        assertFalse(j.fit.isNaN())
        assertEquals(t.tauFitFor(), j.tauFit, 0.0)
        // Forcing a tiny fit gate turns a PASS into NOT_ENROLLED "fit" (shape / identity still come first).
        val strict = t.copy(thresholds = t.thresholds.copy(fit = FitThreshold.derive(doubleArrayOf(0.01), DoubleArray(0))))
        val r = strict.judge(good, cov, geo, SanityReason.OK)
        assertEquals(Verdict.NOT_ENROLLED, r.verdict)
        assertEquals(VerdictEngine.REASON_FIT, r.reason)
        assertFalse(r.fitOk)
        val shape = strict.judge(good, cov, geo.copy(area = 6000.0, fill = 0.2), SanityReason.OK)
        assertEquals(VerdictEngine.REASON_SHAPE, shape.reason)
        // The slider: 2.0× tolerance widens the fit gate by √2 only.
        assertEquals(strict.tauFitFor(1.0) * Math.sqrt(2.0), strict.tauFitFor(2.0), 1e-12)
    }

    @Test
    fun aLookAlikeIsRejectedAndALocalDefectIsStillADefect() {
        val t = twin()
        val rnd = Random(23)
        val same = t.judge(FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd)), cov, geo, SanityReason.OK)
        assertEquals(Verdict.PASS, same.verdict)
        // A look-alike: same mean texture (identity passes), but every patch is a bit different.
        val other = Synth.view(base, 1.2, rnd)
        val look = t.judge(FeatureMap(gh, gw, dim, other), cov, geo, SanityReason.OK)
        assertEquals(Verdict.NOT_ENROLLED, look.verdict)
        assertEquals(VerdictEngine.REASON_FIT, look.reason)
        assertTrue(look.identityOk && look.geometryOk)
        // The same look-alike sails through a spec-only Twin (score gate only): this is the failure the gate fixes.
        val plain = twin(fit = false).judge(FeatureMap(gh, gw, dim, other), cov, geo, SanityReason.OK)
        assertTrue(plain.verdict != Verdict.NOT_ENROLLED || plain.reason != VerdictEngine.REASON_FIT)
        // One damaged patch is a local defect, not a different object: the fit hardly moves.
        val bad = t.judge(FeatureMap(gh, gw, dim, Synth.view(base, 0.05, rnd, defect = intArrayOf(8 * 16 + 8), dim = dim, shift = 8.0f)), cov, geo, SanityReason.OK)
        assertEquals(Verdict.DEFECT, bad.verdict)
        assertTrue("defect fit ${bad.fit} vs look-alike ${look.fit} tauFit ${bad.tauFit}", bad.fitOk && bad.fit < look.fit / 2)
    }

    @Test
    fun negativesMoveTheGateToTheMidpointAndCalibrationRaisesIt() {
        val t = twin()
        val f = t.thresholds.fit!!
        val hi = f.hi
        val withNeg = t.withFitNegatives(doubleArrayOf(hi * 1.05, hi * 2.0))
        val f2 = withNeg.thresholds.fit!!
        assertEquals(FitThreshold.RULE_MIDPOINT, f2.rule)
        assertEquals(hi * 1.025, f2.tauFit, 1e-9)
        assertTrue(f2.margin!! > 0)
        // Calibration: real good parts with a larger fit widen the same-part spread; the negatives are kept.
        val samples = List(12) { CalibrationSample(true, true, true, 0.5, 0.99, geo, fit = hi * 0.9 + 0.01 * it) }
        val (cal, _) = withNeg.calibrate(samples)
        val f3 = cal.thresholds.fit!!
        assertEquals(12, f3.cal.size)
        assertEquals(f2.neg.size, f3.neg.size)
        assertTrue(f3.tauFit >= f2.tauFit - 1e-12)
        // No fit gate → nothing to derive.
        val plain = twin(fit = false)
        assertNull(plain.thresholds.fit)
        assertTrue(plain.withFitNegatives(doubleArrayOf(1.0)) === plain)
    }

    @Test
    fun fitIsStoredInASidecarFileAndSpecTwinsHaveNone() {
        val store = TwinStore(tmp.newFolder("twins"))
        val t = twin()
        val dir = store.save(t, null)
        assertTrue(File(dir, TwinStore.FIT_JSON).isFile)
        val back = (store.load(t.id) as TwinLoadResult.Loaded).twin
        val a = t.thresholds.fit!!
        val b = back.thresholds.fit
        assertNotNull(b)
        assertEquals(a.tauFit, b!!.tauFit, 1e-12)
        assertEquals(a.pos.size, b.pos.size)
        // Negatives are never persisted (they are recomputed from the other Twins).
        val neg = t.withFitNegatives(doubleArrayOf(a.hi * 3))
        val dir2 = store.save(neg.copy(name = "again"), null)
        val back2 = (store.load(t.id) as TwinLoadResult.Loaded).twin
        assertEquals(0, back2.thresholds.fit!!.neg.size)
        assertTrue(File(dir2, TwinStore.FIT_JSON).isFile)
        // A spec-only Twin writes no fit.json and loads without a gate.
        val plain = twin(fit = false).copy(name = "plain")
        val plainMeta = TwinMeta("20260927-000001-abcd", "plain", 1L)
        val plain2 = TeachBuilder.build(Synth.teachFrames(48, gh, gw, dim, seed = 21, base = base, radius = 6.0), Synth.pipeline(gh, gw, dim), plainMeta) as TeachResult.Success
        val d3 = store.save(plain2.twin, null)
        assertFalse(File(d3, TwinStore.FIT_JSON).exists())
        assertNull((store.load(plainMeta.id) as TwinLoadResult.Loaded).twin.thresholds.fit)
        assertNotNull(plain)
    }
}
