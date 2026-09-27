package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Binomial
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.model.Verdict
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Identity threshold, geometry model, verdict rules, voting and calibration (spec §7.3, §7.4, §9, §10). */
class GatesTest {

    private val pos = doubleArrayOf(0.95, 0.9, 0.99, 0.97, 0.98)

    @Test
    fun identityThresholdByHand() {
        // P5 of [0.90, 0.95, 0.97, 0.98, 0.99]: h = 0.2 → 0.90 + 0.2·0.05 = 0.91. MAD = 0.02 → 3·1.4826·0.02 = 0.088956.
        val none = IdentityThreshold.derive(pos, DoubleArray(0))
        assertEquals(0.91 - 3 * 1.4826 * 0.02, none.tauId, 1e-12)
        assertEquals("no-negatives", none.rule)
        assertNull(none.margin)
        assertNull(none.overlap)
        // Tight positives → the 0.02 minimum gap.
        val tight = IdentityThreshold.derive(doubleArrayOf(0.99, 0.99, 0.99), DoubleArray(0))
        assertEquals(0.97, tight.tauId, 1e-12)
        // With negatives: P99 of [0.5, 0.6] = 0.599 → midpoint.
        val w = IdentityThreshold.derive(pos, doubleArrayOf(0.6, 0.5))
        assertEquals((0.91 + 0.599) / 2, w.tauId, 1e-12)
        assertEquals(0.91 - 0.599, w.margin!!, 1e-12)
        assertEquals(false, w.overlap)
        val o = IdentityThreshold.derive(pos, doubleArrayOf(0.95))
        assertEquals(true, o.overlap)
        assertTrue(w.passes(w.tauId))
        assertFalse(w.passes(Math.nextDown(w.tauId)))
    }

    @Test
    fun geometryModelFloorsAndGate() {
        val g = List(4) { GeometryFeatures(1000.0 + it, 0.6, 0.9, 0.16, 0.95) }   // zero spread → floors apply
        val m = GeometryModel.fit(g)
        assertEquals(0.6, m.fill.mean, 1e-12)
        assertEquals(0.04, m.fill.sigma, 0.0)
        assertEquals(0.04, m.aspect.sigma, 0.0)
        assertEquals(0.03, m.solidity.sigma, 0.0)
        assertEquals(0.04 * 0.16, m.hu1.sigma, 1e-15)
        assertEquals(1001.5, m.meanArea, 0.0)
        val top = m.fill.upper(m.kGeo)
        assertTrue(m.check(GeometryFeatures(1001.5, top, 0.9, 0.16, 0.95)).ok)            // the bound itself passes
        assertFalse(m.check(GeometryFeatures(1001.5, Math.nextUp(top), 0.9, 0.16, 0.95)).ok)
        val bad = m.check(GeometryFeatures(1001.5 * 2.6, 0.6 + 0.121, 0.7, 0.16 * 1.2, 0.95 - 0.1))
        assertFalse(bad.ok)
        assertEquals(listOf("fill", "aspect", "hu1", "solidity", "area"), bad.failures)
        assertTrue(m.check(GeometryFeatures(1001.5 / 2.5, 0.6, 0.9, 0.16, 0.95)).ok)
        assertFalse(m.check(GeometryFeatures(Math.nextDown(1001.5 / 2.5), 0.6, 0.9, 0.16, 0.95)).ok)
        // Real spread above the floor.
        val m2 = GeometryModel.fit(listOf(GeometryFeatures(1.0, 0.2, 0.9, 0.16, 0.95), GeometryFeatures(1.0, 0.6, 0.9, 0.16, 0.95)))
        assertEquals(0.2, m2.fill.sigma, 1e-15)
    }

    private fun score(dmapValue: Double, tau: Double = 1.0, core: Int = 9, hot: Int = 0): ScoreResult {
        val c = BooleanArray(9) { it < core }
        val sets = PatchSets.fromCore(c, 3, 3)
        val d = DoubleArray(9) { dmapValue }
        for (i in 0 until hot) d[i] = 100.0
        return Scoring.score(d, sets, tau)
    }

    private val ok = GeometryCheck(true, emptyList())

    @Test
    fun verdictOrder() {
        val pass = VerdictEngine.decide(0.95, 0.9, ok, score(1.0), 0.5)
        assertEquals(Verdict.PASS, pass.verdict)              // s == 1 passes
        assertNull(pass.reason)
        val id = VerdictEngine.decide(0.85, 0.9, GeometryCheck(false, listOf("area")), score(5.0), 0.5)
        assertEquals(Verdict.NOT_ENROLLED, id.verdict)
        assertEquals("identity", id.reason)
        assertEquals(listOf("area"), id.geometryFailures)   // shape failures still reported
        val shape = VerdictEngine.decide(0.95, 0.9, GeometryCheck(false, listOf("fill")), score(0.1), 0.5)
        assertEquals("shape", shape.reason)
        val coverage = VerdictEngine.decide(0.95, 0.9, ok, score(2.0), 0.5)   // every patch hot → a = 1
        assertEquals(Verdict.NOT_ENROLLED, coverage.verdict)
        assertEquals("coverage", coverage.reason)
        val defect = VerdictEngine.decide(0.95, 0.9, ok, score(0.1, core = 9, hot = 1), 0.5)
        assertEquals(Verdict.DEFECT, defect.verdict)
        assertNull(defect.reason)
        assertEquals(0, defect.peakIndex)
        val reframe = VerdictEngine.reframe(SanityReason.TOO_SMALL)
        assertEquals(Verdict.REFRAME, reframe.verdict)
        assertEquals("TOO_SMALL", reframe.reason)
        assertTrue(reframe.s.isNaN())
    }

    private fun judged(s: Double, a: Double = 0.1): Judgement {
        val sc = score(0.1, core = 9, hot = 1)
        return VerdictEngine.decide(1.0, 0.0, ok, sc, 0.5).copy(verdict = Verdict.DEFECT, s = s, anomalousFraction = a)
    }

    private fun scoreWith(s: Double, a: Double = 0.1): ScoreResult {
        val sc = score(0.1, core = 9, hot = 1)
        return ScoreResult(sc.rule, s, s, 1.0, 1.0, sc.peak, sc.peakRow, sc.peakCol, a, 1, 9, 0, 3, 3, sc.smoothed, sc.dmap)
    }

    @Test
    fun borderlineVote() {
        assertTrue(VerdictEngine.needsVote(judged(1.1)))
        assertTrue(VerdictEngine.needsVote(judged(1.15)))
        assertFalse(VerdictEngine.needsVote(judged(1.1500001)))
        assertFalse(VerdictEngine.needsVote(judged(1.0)))
        // three values → the median crop
        val v3 = VerdictEngine.vote(judged(1.1), listOf(scoreWith(0.9), scoreWith(1.3)), 0.5)
        assertEquals(0, v3.chosen)
        assertEquals(Verdict.DEFECT, v3.judgement.verdict)
        val v3b = VerdictEngine.vote(judged(1.1), listOf(scoreWith(0.95), scoreWith(0.9)), 0.5)
        assertEquals(1, v3b.chosen)
        assertEquals(Verdict.PASS, v3b.judgement.verdict)
        assertEquals(0.95, v3b.judgement.s, 0.0)
        // two values → the higher one; ties → the earlier crop
        val v2 = VerdictEngine.vote(judged(1.1), listOf(scoreWith(0.9)), 0.5)
        assertEquals(0, v2.chosen)
        val v2b = VerdictEngine.vote(judged(1.1), listOf(null, scoreWith(1.12, a = 0.8)), 0.5)
        assertEquals(2, v2b.chosen)
        assertEquals(Verdict.NOT_ENROLLED, v2b.judgement.verdict)   // steps 3–5 with that crop's a
        assertEquals("coverage", v2b.judgement.reason)
        val tie = VerdictEngine.vote(judged(1.1), listOf(scoreWith(1.1), scoreWith(1.1)), 0.5)
        assertEquals(0, tie.chosen)
        // one value
        val v1 = VerdictEngine.vote(judged(1.1), emptyList(), 0.5)
        assertEquals(0, v1.chosen)
        assertEquals(1, v1.judgement.votes)
    }

    @Test
    fun calibrationNeverLowersTauAndCountsGates() {
        val geo = GeometryFeatures(1000.0, 0.6, 0.9, 0.16, 0.95)
        val teachGeo = GeometryModel.fit(listOf(geo))
        val samples = listOf(
            CalibrationSample(true, true, true, 1.2, 0.97, geo),
            CalibrationSample(true, true, true, 0.8, 0.96, geo),
            CalibrationSample(true, false, true, 3.0, 0.80, geo),     // identity fail: raw ignored
            CalibrationSample(false, false, false, null, null, null),
            CalibrationSample(true, true, false, 2.0, 0.95, geo.copy(area = 1100.0)),
        )
        val r = Calibration.calibrate(1.0, pos, DoubleArray(0), teachGeo, 3, samples, latenciesMs = doubleArrayOf(10.0, 20.0, 30.0))
        assertEquals(1.2, r.tauCal, 0.0)
        assertEquals(1.2, r.calMax!!, 0.0)
        val c = r.certificate
        assertEquals(5, c.n)
        assertEquals(4, c.nSanityOk)
        assertEquals(2, c.m)
        assertEquals(Binomial.orderStatisticAlpha(2), c.alpha!!, 0.0)
        assertEquals(GateCount(4, 5, 1, Binomial.clopperPearsonUpper(1, 5)), c.sanity)
        assertEquals(GateCount(3, 4, 1, Binomial.clopperPearsonUpper(1, 4)), c.identity)
        assertEquals(GateCount(3, 4, 1, Binomial.clopperPearsonUpper(1, 4)), c.geometry)
        assertNull(c.identityMargin)
        assertEquals(Binomial.orderStatisticAlpha(3), c.withinPart, 0.0)
        assertEquals(20.0, c.latencyP50Ms!!, 0.0)
        assertEquals(29.0, c.latencyP95Ms!!, 1e-12)
        // τ_id from teach positives ∪ sanity-ok sims; geometry from the sanity-ok samples.
        val allPos = pos + doubleArrayOf(0.97, 0.96, 0.80, 0.95)
        assertEquals(IdentityThreshold.derive(allPos, DoubleArray(0)).tauId, r.identity.tauId, 0.0)
        assertEquals(GeometryModel.fit(samples.filter { it.sanityOk }.map { it.geometry!! }), r.geometry)
        assertTrue(c.isValid(1.2, 1.0))
        assertFalse(c.isValid(1.2, 0.99))
        // Never lowered: all valid raws below τ_teach.
        val low = Calibration.calibrate(5.0, pos, DoubleArray(0), teachGeo, 3, samples.take(2))
        assertEquals(5.0, low.tauCal, 0.0)
        assertEquals(1.2, low.calMax!!, 0.0)
        // No valid sample → no α.
        val none = Calibration.calibrate(5.0, pos, DoubleArray(0), teachGeo, 3, samples.subList(2, 4))
        assertNull(none.certificate.alpha)
        assertNull(none.calMax)
        assertEquals(5.0, none.tauCal, 0.0)
    }

    @Test
    fun certificateLines() {
        val geo = GeometryFeatures(1000.0, 0.6, 0.9, 0.16, 0.95)
        val samples = List(29) { CalibrationSample(true, true, true, 1.0, 0.97, geo) }
        val r = Calibration.calibrate(1.0, pos, doubleArrayOf(0.5), GeometryModel.fit(listOf(geo)), 5, samples,
            latenciesMs = doubleArrayOf(12.0), accelerator = "NPU")
        val lines = CertificateText.lines(r.certificate)
        assertEquals(4, lines.size)
        assertTrue(lines[0], lines[0].contains("29 valid parts") && lines[0].contains("9.8 %") && lines[0].contains("score gate only"))
        assertTrue(lines[1], lines[1].contains("sanity 29/29") && lines[1].contains("identity 29/29"))
        assertTrue(lines[2], lines[2].startsWith("Identity margin"))
        assertTrue(lines[3], lines[3].contains("NPU"))
        val within = CertificateText.withinPart(5)
        assertTrue(within, within.contains("5 time blocks") && within.contains("not a false-alarm guarantee"))
    }
}
