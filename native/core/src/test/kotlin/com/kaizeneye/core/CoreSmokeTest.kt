package com.kaizeneye.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class CoreSmokeTest {

    @Test
    fun orderStatisticBoundMatchesResearchTable() {
        assertEquals(0.1391, KaizenCore.orderStatisticAlpha(20), 0.0005)   // "today's 20-photo leave-one-out max bounds FPR at ~13.9 %"
        assertEquals(0.0981, KaizenCore.orderStatisticAlpha(29), 0.0005)   // <= 10 % needs 29 independent good units
        assertEquals(0.0495, KaizenCore.orderStatisticAlpha(59), 0.0005)   // <= 5 % needs 59
        assertEquals(0.0100, KaizenCore.orderStatisticAlpha(299), 0.0002)  // <= 1 % needs 299
    }

    @Test
    fun goldenVectorsAreReachableFromTheJvmTestRunner() {
        val dir = File(requireNotNull(System.getProperty("testdata.dir")) { "testdata.dir system property missing" })
        assertTrue("missing ${File(dir, "golden_core.json")}", File(dir, "golden_core.json").isFile)
        assertTrue("missing ${File(dir, "golden_app.json")}", File(dir, "golden_app.json").isFile)
    }
}
