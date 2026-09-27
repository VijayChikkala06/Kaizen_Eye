package com.kaizeneye.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AccelRulesTest {
    private val f32 = ModelAsset("r18_320_f32", "backbone_r18_320.tflite", "")
    private val i8 = ModelAsset("r18_320_int8", "backbone_r18_320_int8.tflite", "", int8 = true)
    private val r18 = CandidateSet(listOf(f32, i8))

    private val good = AccelRules.Comparison(cosine = 0.99995, patchCosine = 0.9999, relErr = 0.002, finite = true)

    private fun t(ms: Double) = DoubleArray(10) { ms }

    private fun m(set: CandidateSet, accel: Accel, v: ModelAsset, ms: Double?, cmp: AccelRules.Comparison? = good, error: String? = null) =
        Measurement(set.candidate(accel, v), ms?.let { t(it) }, 50.0, if (ms == null) null else cmp, error)

    private fun result(d: AccelRules.Decision, label: String) = d.results.first { it.name == label }

    @Test
    fun candidateLabelsAndOrders() {
        assertEquals(listOf("NPU", "NPU (int8)", "GPU", "CPU", "CPU (int8)"), r18.all.map { it.label })
        assertEquals(listOf("CPU", "CPU (int8)", "GPU", "NPU", "NPU (int8)"), r18.runOrder.map { it.label })
        assertEquals(f32, r18.reference)
        assertTrue(r18.referenceCpu.isReference)
        assertEquals("CPU (int8)", r18.candidate(Accel.NPU, i8).baselineLabel)
        assertEquals("GPU (int8)", r18.byLabel("GPU (int8)")?.label)   // exists for forced runs, not a benchmark candidate
    }

    @Test
    fun twoFloatVariantsGetSuffixTags() {
        val a = ModelAsset("dinov2_s14_448_fp16w", "a.tflite", "")
        val b = ModelAsset("dinov2_s14_448_fp32", "b.tflite", "")
        val s = CandidateSet(listOf(a, b))
        assertEquals(a, s.reference)
        assertEquals(listOf("NPU", "NPU (fp32)", "GPU", "GPU (fp32)", "CPU", "CPU (fp32)"), s.all.map { it.label })
        // colliding suffixes fall back to the whole id
        val c = ModelAsset("x_fp32", "c.tflite", "")
        val d = ModelAsset("y_fp32", "d.tflite", "")
        val s2 = CandidateSet(listOf(a, c, d))
        assertEquals(setOf("CPU", "CPU (x_fp32)", "CPU (y_fp32)"), s2.all.filter { it.accel == Accel.CPU }.map { it.label }.toSet())
    }

    @Test(expected = IllegalArgumentException::class)
    fun int8OnlyHasNoReference() {
        CandidateSet(listOf(i8))
    }

    @Test
    fun fastAccurateNpuIsChosen() {
        val s = CandidateSet(listOf(f32))
        val d = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 38.0), m(s, Accel.GPU, f32, 9.0), m(s, Accel.NPU, f32, 4.1)))
        assertEquals("NPU", d.chosen.label)
        assertEquals(38.0 / 4.1, d.speedupVsCpu!!, 1e-9)
        assertEquals("NPU 4.1 ms (9.3× CPU)", d.badge)
        assertEquals("slower than NPU", result(d, "GPU").note)
        assertEquals("NPU is faster", result(d, "CPU").note)
        assertEquals(1.0, result(d, "CPU").cosine!!, 0.0)
        assertTrue(result(d, "NPU").ok)
    }

    @Test
    fun acceleratorWithoutSpeedupIsNotUsed() {
        val s = CandidateSet(listOf(f32))
        val d = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 38.0), m(s, Accel.NPU, f32, 36.0)))
        assertEquals("CPU", d.chosen.label)
        assertNull(d.speedupVsCpu)
        assertEquals("CPU 38 ms", d.badge)
        val npu = result(d, "NPU")
        assertTrue(npu.ok)                                   // accurate ...
        assertTrue(npu.note!!.startsWith("NPU requested - no speed-up (ran on CPU?)"))   // ... but not faster
    }

    @Test
    fun speedRuleBoundaryIsInclusive() {
        val s = CandidateSet(listOf(f32))
        val d = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 40.0), m(s, Accel.GPU, f32, 36.0)))
        assertEquals("GPU", d.chosen.label)                  // 36 <= 0.9 * 40
        val d2 = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 40.0), m(s, Accel.GPU, f32, 36.01)))
        assertEquals("CPU", d2.chosen.label)
    }

    @Test
    fun inaccurateAcceleratorIsRejectedEvenIfFast() {
        val s = CandidateSet(listOf(f32))
        val bad = AccelRules.Comparison(0.99, 0.99, 0.05, true)
        val d = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 38.0), m(s, Accel.NPU, f32, 2.0, bad)))
        assertEquals("CPU", d.chosen.label)
        assertFalse(result(d, "NPU").ok)
        assertTrue(result(d, "NPU").error!!.startsWith("output differs from CPU"))
    }

    @Test
    fun nonFiniteOutputIsRejected() {
        val s = CandidateSet(listOf(f32))
        val nan = AccelRules.Comparison(Double.NaN, Double.NaN, Double.NaN, false)
        val d = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 38.0), m(s, Accel.GPU, f32, 2.0, nan)))
        assertEquals("CPU", d.chosen.label)
        assertTrue(result(d, "GPU").error!!.contains("non-finite"))
    }

    @Test
    fun int8AcceleratorIsComparedWithInt8Cpu() {
        // NPU int8 (19 ms) beats float CPU (38 ms) but not CPU int8 (20 ms): not an NPU speed-up.
        val d = AccelRules.decide(
            r18,
            listOf(m(r18, Accel.CPU, f32, 38.0), m(r18, Accel.CPU, i8, 20.0), m(r18, Accel.NPU, i8, 19.0)),
        )
        assertEquals("CPU", d.chosen.label)
        assertTrue(result(d, "NPU (int8)").note!!.contains("on CPU (int8)"))
        // ... while a real int8 speed-up wins, with the badge naming the same-variant baseline.
        val d2 = AccelRules.decide(
            r18,
            listOf(m(r18, Accel.CPU, f32, 38.0), m(r18, Accel.CPU, i8, 20.0), m(r18, Accel.NPU, f32, 4.1), m(r18, Accel.NPU, i8, 3.0)),
        )
        assertEquals("NPU (int8)", d2.chosen.label)
        assertEquals("r18_320_int8", d2.chosen.variant.id)
        assertEquals("NPU int8 3.0 ms (6.7× CPU int8)", d2.badge)
        assertEquals(20.0 / 3.0, d2.speedupVsCpu!!, 1e-9)
        assertEquals("slower than NPU (int8)", result(d2, "NPU").note)
    }

    @Test
    fun int8WithoutBaselineIsNotUsed() {
        val d = AccelRules.decide(
            r18,
            listOf(m(r18, Accel.CPU, f32, 38.0), m(r18, Accel.CPU, i8, null, error = "failed"), m(r18, Accel.NPU, i8, 3.0)),
        )
        assertEquals("CPU", d.chosen.label)
        assertTrue(result(d, "NPU (int8)").note!!.contains("baseline unavailable"))
    }

    @Test
    fun int8OnCpuIsNeverChosen() {
        val d = AccelRules.decide(r18, listOf(m(r18, Accel.CPU, f32, 38.0), m(r18, Accel.CPU, i8, 10.0)))
        assertEquals("CPU", d.chosen.label)
        assertTrue(result(d, "CPU (int8)").note!!.contains("not selectable"))
    }

    @Test
    fun int8GateUsesPatchCosine() {
        val passesInt8 = AccelRules.Comparison(cosine = 0.99, patchCosine = 0.995, relErr = 0.1, finite = true)
        val failsInt8 = AccelRules.Comparison(cosine = 0.999, patchCosine = 0.985, relErr = 0.01, finite = true)
        assertTrue(AccelRules.accurate(passesInt8, int8 = true))
        assertFalse(AccelRules.accurate(passesInt8, int8 = false))
        assertFalse(AccelRules.accurate(failsInt8, int8 = true))
        assertTrue(AccelRules.gateText(failsInt8, true).startsWith("too far from the float CPU output"))
        // float gate edges
        assertTrue(AccelRules.accurate(AccelRules.Comparison(0.999, 0.9, 0.01, true), false))
        assertFalse(AccelRules.accurate(AccelRules.Comparison(0.999, 0.9, 0.0101, true), false))
    }

    @Test
    fun fastestEligibleByMedianAndTiesGoToCpu() {
        val s = CandidateSet(listOf(f32))
        val d = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 10.0), m(s, Accel.GPU, f32, 5.0), m(s, Accel.NPU, f32, 4.0)))
        assertEquals("NPU", d.chosen.label)
        val tie = AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, 10.0), m(s, Accel.GPU, f32, 10.0)))
        assertEquals("CPU", tie.chosen.label)                 // 10 > 0.9*10: not even eligible
    }

    @Test
    fun skippedCandidatesKeepTheirReason() {
        val s = CandidateSet(listOf(f32))
        val d = AccelRules.decide(
            s,
            listOf(m(s, Accel.CPU, f32, 38.0), m(s, Accel.NPU, f32, null, error = "crashed during an earlier attempt - skipped")),
        )
        assertEquals("CPU", d.chosen.label)
        assertEquals("crashed during an earlier attempt - skipped", result(d, "NPU").error)
        assertFalse(result(d, "NPU").ok)
        assertNull(result(d, "NPU").medianMs)
        assertEquals(listOf("NPU", "CPU"), d.results.map { it.name })   // GPU was not measured: not listed
    }

    @Test(expected = IllegalArgumentException::class)
    fun referenceMustHaveRun() {
        val s = CandidateSet(listOf(f32))
        AccelRules.decide(s, listOf(m(s, Accel.CPU, f32, null, error = "boom"), m(s, Accel.NPU, f32, 4.0)))
    }

    @Test
    fun dinoTwoFloatVariantsUseTheirOwnCpuBaselines() {
        val a = ModelAsset("dinov2_s14_448_fp16w", "a.tflite", "")
        val b = ModelAsset("dinov2_s14_448_fp32", "b.tflite", "")
        val s = CandidateSet(listOf(a, b))
        val d = AccelRules.decide(
            s,
            listOf(m(s, Accel.CPU, a, 200.0), m(s, Accel.CPU, b, 180.0), m(s, Accel.NPU, a, 20.0), m(s, Accel.NPU, b, 15.0)),
        )
        assertEquals("NPU (fp32)", d.chosen.label)
        assertEquals("dinov2_s14_448_fp32", d.chosen.variant.id)
        assertEquals("NPU fp32 15 ms (12.0× CPU fp32)", d.badge)
        // fp16w on NPU (20 ms vs its own CPU 200 ms) was eligible but slower
        assertEquals("slower than NPU (fp32)", result(d, "NPU").note)
        // a float variant failing the float gate against the reference is not a valid baseline user
        val off = AccelRules.Comparison(0.998, 0.998, 0.02, true)
        val d2 = AccelRules.decide(s, listOf(m(s, Accel.CPU, a, 200.0), m(s, Accel.CPU, b, 180.0, off), m(s, Accel.NPU, b, 15.0, off)))
        assertEquals("CPU", d2.chosen.label)
        assertFalse(result(d2, "NPU (fp32)").ok)
    }

    @Test
    fun compareMatchesDefinitions() {
        val a = floatArrayOf(1f, 2f, 3f, 4f)
        val same = AccelRules.compare(a, a.copyOf(), 2)
        assertEquals(1.0, same.cosine, 1e-12)
        assertEquals(1.0, same.patchCosine, 1e-12)
        assertEquals(0.0, same.relErr, 0.0)
        assertTrue(same.finite)
        val twice = AccelRules.compare(FloatArray(4) { a[it] * 2 }, a, 2)
        assertEquals(1.0, twice.cosine, 1e-12)
        assertEquals(1.0, twice.relErr, 1e-12)               // ||2a - a|| / ||a||
        val flipped = AccelRules.compare(floatArrayOf(1f, 0f, 0f, 1f), floatArrayOf(1f, 0f, 1f, 0f), 2)
        assertEquals(0.5, flipped.patchCosine, 1e-12)        // patch 0 identical, patch 1 orthogonal
        assertEquals(0.5, flipped.cosine, 1e-12)
        assertFalse(AccelRules.compare(floatArrayOf(1f), floatArrayOf(1f, 2f)).finite)
        assertFalse(AccelRules.compare(floatArrayOf(Float.NaN, 1f), floatArrayOf(1f, 1f)).finite)
        assertEquals(1.0, AccelRules.compare(FloatArray(4), FloatArray(4)).cosine, 0.0)
    }

    @Test
    fun timingUsesSpecPercentiles() {
        val t = AccelRules.timing(doubleArrayOf(4.0, 1.0, 3.0, 2.0))!!
        assertEquals(2.5, t.medianMs, 1e-12)
        assertEquals(1.0, t.minMs, 0.0)
        assertEquals(3.85, t.p95Ms, 1e-12)                   // h = 3*0.95 = 2.85 -> 3 + 0.85*(4-3)
        val one = AccelRules.timing(doubleArrayOf(7.0))!!
        assertEquals(7.0, one.medianMs, 0.0)
        assertEquals(7.0, one.p95Ms, 0.0)
        assertNull(AccelRules.timing(DoubleArray(0)))
        assertNull(AccelRules.timing(null))
    }

    @Test
    fun badgeFormats() {
        val s = CandidateSet(listOf(f32, i8))
        assertEquals("CPU 38 ms", AccelRules.badge(s.referenceCpu, 38.2, null))
        assertEquals("GPU 9.0 ms (2.1× CPU)", AccelRules.badge(s.candidate(Accel.GPU, f32), 9.0, 2.1))
        assertEquals("NPU 4.1 ms (7.9× CPU)", AccelRules.badge(s.candidate(Accel.NPU, f32), 4.1, 7.9))
        assertEquals("NPU int8 12 ms (3.0× CPU int8)", AccelRules.badge(s.candidate(Accel.NPU, i8), 12.4, 3.0))
        assertEquals("CPU int8", AccelRules.badge(s.cpu(i8), null, null))
        val cpu = CandidateResult("CPU", "r18_320_f32", true, 38.0, 37.0, 40.0, 1.0, 1.0, 0.0, 10.0)
        val slowNpu = CandidateResult("NPU", "r18_320_f32", true, 36.0, 35.0, 37.0, 0.9999, 0.9999, 0.001, 10.0)
        assertEquals(
            "NPU requested - no speed-up (ran on CPU?) 36 ms (forced)",
            AccelRules.forcedBadge(s.candidate(Accel.NPU, f32), slowNpu, cpu),
        )
        val fastNpu = slowNpu.copy(medianMs = 4.0, minMs = 3.9)
        assertEquals("NPU 4.0 ms (9.5× CPU) (forced)", AccelRules.forcedBadge(s.candidate(Accel.NPU, f32), fastNpu, cpu))
        assertEquals("CPU 38 ms (forced)", AccelRules.forcedBadge(s.referenceCpu, cpu, cpu))
    }

    @Test
    fun describeMentionsEveryCandidate() {
        val d = AccelRules.decide(
            r18,
            listOf(m(r18, Accel.CPU, f32, 38.0), m(r18, Accel.CPU, i8, 20.0), m(r18, Accel.NPU, f32, 4.1), m(r18, Accel.GPU, f32, null, error = "no GPU")),
        )
        val r = AcceleratorReport("backbone_r18_320", d.chosen.label, d.chosen.accel, d.chosen.variant.id, d.badge, d.speedupVsCpu, d.results, false, 0L, "k")
        val line = AccelRules.describe(r)
        assertTrue(line, line.contains("NPU 4.1 ms (chosen)"))
        assertTrue(line, line.contains("GPU: failed (no GPU)"))
        assertTrue(line, line.contains("CPU (int8)"))
        assertNotNull(r.speedupVsCpu)
    }
}
