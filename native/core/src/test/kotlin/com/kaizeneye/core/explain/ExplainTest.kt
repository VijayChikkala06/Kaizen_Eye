package com.kaizeneye.core.explain

import com.kaizeneye.core.model.Verdict
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExplainTest {

    private val defect = Facts(
        partName = "M8 nut", verdict = Verdict.DEFECT, peakRow = 6, peakCol = 8, gh = 40, gw = 40,
        areaPct = 3.1, score = 1.34, tau = 1.43, rejectNumber = 42,
    )

    private fun at(region: String): Facts {
        // a patch in the middle of the requested region on a 40x40 grid
        val v = GridNaming.verticalBand(region)
        val h = GridNaming.horizontalBand(region)
        return defect.copy(peakRow = 6 + 13 * v, peakCol = 6 + 13 * h)
    }

    @Test
    fun gridThirdsUseTheCellCentre() {
        assertEquals(List(13) { 0 } + List(14) { 1 } + List(13) { 2 }, (0 until 40).map { GridNaming.band(it, 40) })
        assertEquals(listOf(0, 1, 2), (0 until 3).map { GridNaming.band(it, 3) })
        assertEquals("upper-left", GridNaming.region(0, 0, 40, 40))
        assertEquals("top", GridNaming.region(0, 20, 40, 40))
        assertEquals("upper-right", GridNaming.region(12, 39, 40, 40))
        assertEquals("left", GridNaming.region(20, 0, 40, 40))
        assertEquals("centre", GridNaming.region(20, 20, 40, 40))
        assertEquals("right", GridNaming.region(26, 27, 40, 40))
        assertEquals("lower-left", GridNaming.region(39, 0, 40, 40))
        assertEquals("bottom", GridNaming.region(27, 13, 40, 40))
        assertEquals("lower-right", GridNaming.region(31, 39, 32, 40))
    }

    @Test
    fun factsLineMatchesTheCardFormat() {
        assertEquals(
            "M8 nut · DEFECT · upper-left (row 7/40, col 9/40) · 3.1 % of part · 1.34× limit",
            FactsLine.render(defect),
        )
        assertEquals("M8 nut · PASS · 0.82× limit", FactsLine.render(Facts("M8 nut", Verdict.PASS, score = 0.82)))
        assertEquals("M8 nut · NOT ENROLLED · identity", FactsLine.render(Facts("M8 nut", Verdict.NOT_ENROLLED, reason = "IDENTITY")))
        assertEquals(
            "M8 nut · REFRAME · touches border · NPU 4.1 ms",
            FactsLine.render(Facts("M8 nut", Verdict.REFRAME, reason = "TOUCHES_BORDER", accelLabel = "NPU 4.1 ms")),
        )
    }

    @Test
    fun templateSentences() {
        assertEquals(
            "Reject #042: anomaly covering 3.1 % of the part at the upper-left, 1.34× the learned limit.",
            TemplateExplainer.sentence(defect),
        )
        assertEquals(
            "Reject: anomaly covering 3.1 % of the part at the centre, 1.34× the learned limit.",
            TemplateExplainer.sentence(at("centre").copy(rejectNumber = null)),
        )
        assertEquals("Pass: within the learned limit (0.82×).", TemplateExplainer.sentence(Facts("M8 nut", Verdict.PASS, score = 0.82)))
        assertEquals(
            "Not the enrolled part: it does not look like the taught M8 nut.",
            TemplateExplainer.sentence(Facts("M8 nut", Verdict.NOT_ENROLLED, reason = "identity")),
        )
        assertEquals(
            "Not the enrolled part: its shape does not match the taught M8 nut.",
            TemplateExplainer.sentence(Facts("M8 nut", Verdict.NOT_ENROLLED, reason = "SHAPE")),
        )
        assertEquals(
            "Not the enrolled part: most of its surface differs from the taught M8 nut (61.0 % of the part).",
            TemplateExplainer.sentence(Facts("M8 nut", Verdict.NOT_ENROLLED, reason = "coverage", areaPct = 61.0)),
        )
        assertEquals(
            "Reframe: more than one object — show one part at a time.",
            TemplateExplainer.sentence(Facts("M8 nut", Verdict.REFRAME, reason = "MULTIPLE")),
        )
        assertEquals("Reframe: no part in view.", TemplateExplainer.sentence(Facts("M8 nut", Verdict.REFRAME, reason = "NO_OBJECT")))
        for (r in listOf("TOUCHES_BORDER", "TOO_SMALL", "TOO_LARGE", "NO_CORE")) {
            assertTrue(TemplateExplainer.sentence(Facts("M8 nut", Verdict.REFRAME, reason = r)).startsWith("Reframe: "))
        }
    }

    @Test
    fun locationGuardAcceptTable() {
        val cases = listOf(
            "upper-left" to "A dent near the upper left edge.",
            "upper-left" to "Scratch at the top-left corner of the thread.",
            "upper-left" to "A chip on the left side.",
            "upper-left" to "Crack along the top edge.",
            "upper-left" to "A small burr is visible.",
            "top" to "Burr at the top.",
            "top" to "A dent in the top centre of the face.",
            "top" to "Missing material above the hole.",
            "centre" to "Deformed thread in the middle.",
            "centre" to "A scratch across the centre of the part.",
            "right" to "Scratch on the right-hand side.",
            "right" to "Crack at the middle of the right edge.",
            "lower-right" to "Chipped corner at the bottom right.",
            "lower-right" to "The lower edge shows a dent.",
            "bottom" to "Burr below the flange.",
        )
        for ((region, sentence) in cases) {
            val r = LocationGuard.check(sentence, at(region))
            assertEquals("$region: $sentence -> ${r.offendingWord}", GuardDecision.ACCEPT, r.decision)
        }
    }

    @Test
    fun locationGuardRejectTable() {
        val cases = listOf(
            Triple("upper-left", "A dent at the bottom of the part.", "bottom"),
            Triple("upper-left", "Scratch at the upper right corner.", "right"),
            Triple("upper-left", "A chip in the centre.", "centre"),
            Triple("top", "A dent at the top left.", "left"),
            Triple("top", "A dent in the center.", "center"),
            Triple("centre", "Crack near the top edge.", "top"),
            Triple("centre", "A scratch on the Left side.", "left"),
            Triple("right", "A chip on the lower right.", "lower"),
            Triple("lower-right", "Burr in the middle of the face.", "middle"),
            Triple("bottom", "A dent above the hole.", "above"),
        )
        for ((region, sentence, word) in cases) {
            val r = LocationGuard.check(sentence, at(region))
            assertEquals("$region: $sentence", GuardDecision.REJECT, r.decision)
            assertEquals("$region: $sentence", word, r.offendingWord)
        }
        // facts without a peak accept no location at all
        val pass = Facts("M8 nut", Verdict.PASS, score = 0.5)
        assertEquals(GuardDecision.ACCEPT, LocationGuard.check("The part looks fine.", pass).decision)
        assertEquals(GuardResult(GuardDecision.REJECT, "top"), LocationGuard.check("A mark at the top.", pass))
    }

    @Test
    fun vlmPromptNamesTheRegionAndTheWordLimit() {
        val p = VlmPrompt.build(defect)
        assertTrue(p, p.contains("upper-left"))
        assertTrue(p, p.contains("one sentence of at most 25 words"))
        assertTrue(p, p.contains("Do not mention any other location"))
        assertTrue(VlmPrompt.build(Facts("M8 nut", Verdict.PASS)).contains("Do not mention any location"))
        assertEquals(5, VlmPrompt.wordCount(" a  dent at the top "))
        assertEquals("A dent at the top.", VlmPrompt.clip("A dent at the top. It is deep."))
        val long = (1..40).joinToString(" ") { "w$it" }
        assertEquals(25, VlmPrompt.wordCount(VlmPrompt.clip(long)))
    }
}
