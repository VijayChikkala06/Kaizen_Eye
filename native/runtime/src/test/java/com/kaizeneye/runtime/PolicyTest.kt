package com.kaizeneye.runtime

import android.app.ApplicationExitInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PolicyTest {
    // --------------------------------------------------------------------------------------------------- crash guard
    @Test
    fun exitReasonsAreClassified() {
        assertEquals(CrashGuard.Death.CRASH, CrashGuard.classify(ApplicationExitInfo.REASON_CRASH_NATIVE))
        assertEquals(CrashGuard.Death.CRASH, CrashGuard.classify(ApplicationExitInfo.REASON_CRASH))
        assertEquals(CrashGuard.Death.CRASH, CrashGuard.classify(ApplicationExitInfo.REASON_ANR))
        assertEquals(CrashGuard.Death.EXTERNAL, CrashGuard.classify(ApplicationExitInfo.REASON_LOW_MEMORY))
        assertEquals(CrashGuard.Death.EXTERNAL, CrashGuard.classify(ApplicationExitInfo.REASON_USER_REQUESTED))
        assertEquals(CrashGuard.Death.UNKNOWN, CrashGuard.classify(ApplicationExitInfo.REASON_SIGNALED))
        assertEquals(CrashGuard.Death.UNKNOWN, CrashGuard.classify(null))
        assertEquals("native crash", CrashGuard.reasonName(ApplicationExitInfo.REASON_CRASH_NATIVE))
    }

    @Test
    fun probeDeathsCountAsCrashesUnlessTheSystemKilledIt() {
        assertEquals(CrashGuard.CRASHED, CrashGuard.strikesAfterProbeDeath(0, CrashGuard.Death.CRASH, false))
        assertEquals(CrashGuard.CRASHED, CrashGuard.strikesAfterProbeDeath(0, CrashGuard.Death.UNKNOWN, false))
        assertEquals(CrashGuard.CRASHED, CrashGuard.strikesAfterProbeDeath(0, CrashGuard.Death.EXTERNAL, timedOut = true))
        assertEquals(1, CrashGuard.strikesAfterProbeDeath(0, CrashGuard.Death.EXTERNAL, false))   // low memory: retry once
        assertEquals(CrashGuard.CRASHED, CrashGuard.strikesAfterProbeDeath(1, CrashGuard.Death.EXTERNAL, false))
        assertFalse(CrashGuard.isCrashed(1))
        assertTrue(CrashGuard.isCrashed(2))
        assertFalse(CrashGuard.isCrashed(null))
    }

    @Test
    fun staleMarkersAreJudgedByPhase() {
        val p = CrashGuard.Phase
        // the app died while creating/warming the accelerated model itself
        assertEquals(CrashGuard.CRASHED, CrashGuard.strikesAfterStaleMarker(0, p.MAIN, CrashGuard.Death.CRASH))
        assertEquals(0, CrashGuard.strikesAfterStaleMarker(0, p.MAIN, CrashGuard.Death.EXTERNAL))   // swiped away / LMK
        assertEquals(1, CrashGuard.strikesAfterStaleMarker(0, p.MAIN, CrashGuard.Death.UNKNOWN))    // e.g. reboot
        assertEquals(CrashGuard.CRASHED, CrashGuard.strikesAfterStaleMarker(1, p.MAIN, CrashGuard.Death.UNKNOWN))
        // the app died while the separate probe ran: the candidate is not implicated unless nothing is known
        assertEquals(0, CrashGuard.strikesAfterStaleMarker(0, p.PROBE, CrashGuard.Death.CRASH))
        assertEquals(0, CrashGuard.strikesAfterStaleMarker(0, p.PROBE, CrashGuard.Death.EXTERNAL))
        assertEquals(1, CrashGuard.strikesAfterStaleMarker(0, p.PROBE, CrashGuard.Death.UNKNOWN))
        // the VLM process died initialising an engine
        assertEquals(CrashGuard.CRASHED, CrashGuard.strikesAfterStaleMarker(0, p.VLM, CrashGuard.Death.CRASH))
        assertEquals(0, CrashGuard.strikesAfterStaleMarker(0, p.VLM, CrashGuard.Death.EXTERNAL))
    }

    // ----------------------------------------------------------------------------------------------------- cache key
    @Test
    fun decisionKeyHasEveryComponent() {
        val base = CacheKeys.decisionKey(
            "backbone_r18_320", listOf("r18_320_f32" to "ABC", "r18_320_int8" to "def"), 7, "vivo/PD2505/fp:16/x", "Dispatch=1,QnnHtp=2",
        )
        assertEquals("backbone_r18_320#r18_320_f32:abc+r18_320_int8:def@v7|vivo/PD2505/fp:16/x|qnn2.49.0|Dispatch=1,QnnHtp=2|d${CacheKeys.DECISION_VERSION}", base)
        val variants = listOf(
            CacheKeys.decisionKey("backbone_r18_320", listOf("r18_320_f32" to "abd", "r18_320_int8" to "def"), 7, "vivo/PD2505/fp:16/x", "Dispatch=1,QnnHtp=2"),
            CacheKeys.decisionKey("backbone_r18_320", listOf("r18_320_f32" to "abc", "r18_320_int8" to "def"), 8, "vivo/PD2505/fp:16/x", "Dispatch=1,QnnHtp=2"),
            CacheKeys.decisionKey("backbone_r18_320", listOf("r18_320_f32" to "abc", "r18_320_int8" to "def"), 7, "vivo/PD2505/fp:17/x", "Dispatch=1,QnnHtp=2"),
            CacheKeys.decisionKey("backbone_r18_320", listOf("r18_320_f32" to "abc", "r18_320_int8" to "def"), 7, "vivo/PD2505/fp:16/x", "Dispatch=1,QnnHtp=3"),
            CacheKeys.decisionKey("backbone_r18_320", listOf("r18_320_f32" to "abc"), 7, "vivo/PD2505/fp:16/x", "Dispatch=1,QnnHtp=2"),
            CacheKeys.decisionKey("backbone_r18_320", listOf("r18_320_f32" to "abc", "r18_320_int8" to "def"), 7, "vivo/PD2505/fp:16/x", "Dispatch=1,QnnHtp=2", qnn = "qnn2.47.0"),
        )
        for (k in variants) assertNotEquals(base, k)
    }

    // ------------------------------------------------------------------------------------------------------ vlm plan
    @Test
    fun vlmPlanOrdersModelsAndBackends() {
        val models = listOf(
            "FastVLM-0.5B.qualcomm.sm8850.litertlm" to "/ext/FastVLM-0.5B.qualcomm.sm8850.litertlm",
            "gemma-4-E2B-it-gpu.litertlm" to "/ext/gemma-4-E2B-it-gpu.litertlm",
        )
        val plan = VlmPlan.plan(models)
        assertEquals(
            listOf(
                "FastVLM-0.5B.qualcomm.sm8850.litertlm|NPU", "FastVLM-0.5B.qualcomm.sm8850.litertlm|GPU",
                "FastVLM-0.5B.qualcomm.sm8850.litertlm|CPU", "gemma-4-E2B-it-gpu.litertlm|GPU", "gemma-4-E2B-it-gpu.litertlm|CPU",
            ),
            plan.map { it.id },
        )
        val skipNpu = VlmPlan.plan(models) { it.backend == VlmPlan.NPU }
        assertEquals("FastVLM-0.5B.qualcomm.sm8850.litertlm|GPU", skipNpu.first().id)
        assertTrue(VlmPlan.plan(emptyList()).isEmpty())
        assertEquals("One scratch near the rim.", VlmPlan.joinText(listOf(" One scratch", " near the rim.", "\n")))
        assertEquals("v3|fp|942374912", VlmPlan.stamp(3, "fp", 942374912))
    }

    // ------------------------------------------------------------------------------------------------------ SoC rule
    @Test
    fun socAllowListAndAdspPath() {
        assertTrue(NpuSoc.supported("SM8850"))
        assertTrue(NpuSoc.supported("SM8850-AC"))
        assertTrue(NpuSoc.supported(" sm8850p "))
        assertFalse(NpuSoc.supported("SM8750"))
        assertFalse(NpuSoc.supported(""))
        assertFalse(NpuSoc.supported(null))
        assertEquals(
            "/data/app/x/lib/arm64;/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp",
            NpuSoc.adspPath("/data/app/x/lib/arm64"),
        )
        assertTrue(NpuSoc.adspPath(null).startsWith("/odm/lib/rfsa/adsp;"))
    }

    // --------------------------------------------------------------------------------------------------- misc rules
    @Test
    fun outputsArePickedByElementCount() {
        assertEquals(0 to 1, OutputPick.pick(listOf(393216, 384), 393216, 384))
        assertEquals(1 to 0, OutputPick.pick(listOf(384, 393216), 393216, 384))   // signature order may put CLS first
        assertEquals(0 to -1, OutputPick.pick(listOf(204800), 204800, 128))
        assertEquals(0 to -1, OutputPick.pick(listOf(204800, 7), 204800, 0))
        try {
            OutputPick.pick(listOf(10, 20), 30, 10)
            throw AssertionError("must fail")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("gh*gw*dim = 30"))
        }
    }

    @Test
    fun probeTimeoutScalesWithModelSize() {
        assertEquals(60_000L, BackboneLoader.probeTimeoutMs(11_335_136))
        assertTrue(BackboneLoader.probeTimeoutMs(87_762_856) > 150_000L)
    }
}
