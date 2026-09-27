package com.kaizeneye.core.twin

import com.kaizeneye.core.model.FeatureMap
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Optional check against REAL captures (not committed: ~30 MB of twin folders pulled from the phone with
 * `adb exec-out run-as com.kaizeneye.v2 tar -c -C files twins`). Set KZ_REAL_TWINS to the folder holding them and
 * KZ_FIT_REF to the JSON list written by tools/lab/real_twin_probe.py (`fit_reference`); skipped when unset.
 * Verifies that the Kotlin FIT statistic equals the numpy one on real ResNet18 features (max |Δ| ≤ 1e-4).
 */
class RealTwinFitTest {

    @Test
    fun kotlinFitEqualsTheNumpyFitOnRealTwins() {
        val root = System.getenv("KZ_REAL_TWINS")
        val ref = System.getenv("KZ_FIT_REF")
        assumeTrue("KZ_REAL_TWINS / KZ_FIT_REF not set", root != null && ref != null && File(root).isDirectory && File(ref).isFile)
        val store = TwinStore(File(root!!))
        val a = (store.load("20260926-234828-06b8") as TwinLoadResult.Loaded).twin
        val b = (store.load("20260926-235659-96bf") as TwinLoadResult.Loaded).twin
        val want = File(ref!!).readText().trim().removePrefix("[").removeSuffix("]").split(",").map { it.trim().toDouble() }
        val np = b.patches
        val d = b.dim
        var worst = 0.0
        for (k in 0 until b.keyframeCount) {
            val f = a.fitOf(
                FeatureMap(b.gh, b.gw, d, b.kfFeats.copyOfRange(k * np * d, (k + 1) * np * d)),
                b.kfCov.copyOfRange(k * np, (k + 1) * np),
            )
            worst = maxOf(worst, Math.abs(f - want[k]))
            assertEquals("keyframe $k", want[k], f, 1e-4)
        }
        println("RealTwinFitTest: ${b.keyframeCount} keyframes, max |kotlin - numpy| = $worst")
    }
}
