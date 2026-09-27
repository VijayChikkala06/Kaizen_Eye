package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Half
import com.kaizeneye.core.math.Stats
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.SanityReason
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

class TeachTest {

    private val gh = 8
    private val gw = 8
    private val dim = 16
    private val pipeline = Synth.pipeline(gh, gw, dim)

    private fun success(r: TeachResult): TeachResult.Success {
        assertTrue("teach failed: ${(r as? TeachResult.Failure)?.reason}", r is TeachResult.Success)
        return r as TeachResult.Success
    }

    @Test
    fun teachFollowsTheSpecSteps() {
        val frames = Synth.teachFrames(40, gh, gw, dim, seed = 1)
        val params = TeachParams()
        val r = success(TeachBuilder.build(frames, pipeline, Synth.meta(), params))
        val dg = r.diagnostics
        val t = r.twin

        // 1–3: accepted = sane frames; kept = sharpness >= P40 of the accepted ones.
        val sane = frames.indices.filter { frames[it].sanity == SanityReason.OK }
        assertArrayEquals(sane.toIntArray(), dg.accepted)
        val cut = Stats.percentile(DoubleArray(sane.size) { frames[sane[it]].sharpness }, 40.0)
        assertEquals(cut, dg.sharpnessCut, 0.0)
        assertArrayEquals(sane.filter { frames[it].sharpness >= cut }.toIntArray(), dg.kept)

        // 4: the first keyframe is the sharpest kept frame; keyframes are distinct kept frames.
        val sharpest = dg.kept.maxByOrNull { frames[it].sharpness }!!
        assertEquals(sharpest, dg.keyframeFrames[0])
        assertEquals(dg.keyframeFrames.size, dg.keyframeFrames.toSet().size)
        assertTrue(dg.keyframeFrames.all { it in dg.kept })
        assertEquals(dg.keyframeFrames.size - 1, dg.kcMinDist.size)
        assertTrue(dg.kcStop != null)

        // 5: segments from the kept time range.
        val t0 = frames[dg.kept.first()].tMs
        val t1 = frames[dg.kept.last()].tMs
        for ((k, f) in dg.keyframeFrames.withIndex()) {
            val want = min(4, floor(5.0 * (frames[f].tMs - t0) / ((t1 - t0) + 1e-9)).toInt())
            assertEquals(want, dg.segments[k])
            assertEquals(want, t.keyframes[k].segment)
        }

        // 6: bank size rule, f16 rounding, bankKf / bankSeg consistency.
        val n = dg.pooledRows
        val k = min(min(2400, max(256, floor(0.1 * n).toInt())), n)
        assertEquals(k, t.bankRows)
        assertEquals(k, dg.coresetIndices.size)
        assertEquals(0, dg.coresetIndices[0])
        for (v in t.bank) assertEquals(Half.round(v), v, 0f)
        for (v in t.globals) assertEquals(Half.round(v), v, 0f)
        for (v in t.kfFeats) assertEquals(Half.round(v), v, 0f)
        for (i in 0 until t.bankRows) assertEquals(t.keyframes[t.bankKf[i]].segment, t.bankSeg[i])

        // 7: τ = 1.4 · max LSO.
        assertEquals(1.4 * dg.lso.max(), t.thresholds.tau, 0.0)
        assertEquals(t.thresholds.tau, t.thresholds.tauTeach, 0.0)
        assertFalse(t.thresholds.calibrated)
        assertEquals(if (dg.segmentsUsed >= 2) LsoMode.SEGMENT else LsoMode.KEYFRAME, dg.lsoMode)

        // 8: positives against other segments; τ_id without negatives.
        assertEquals(t.keyframeCount, t.positives.size)
        val id = IdentityThreshold.derive(t.positives, DoubleArray(0))
        assertEquals(id.tauId, t.thresholds.tauId, 0.0)
        assertEquals(IdentityThreshold.RULE_NO_NEGATIVES, t.thresholds.tauIdRule)
        // 9–10
        assertEquals(GeometryModel.fit(dg.kept.map { frames[it].geometry!! }), t.thresholds.geometry)
        assertEquals(0.5, t.thresholds.coverageCut, 0.0)
        assertEquals(t.fingerprint, pipeline.fingerprint())
        assertEquals(frames.size, t.teach.framesSeen)
        assertEquals(dg.kept.size, t.teach.framesKept)
    }

    @Test
    fun teachIsDeterministic() {
        val frames = Synth.teachFrames(40, gh, gw, dim, seed = 2)
        val negs = listOf(Descriptor.l2NormalizeInPlace(DoubleArray(dim) { it.toDouble() }).map { it.toFloat() }.toFloatArray())
        val a = success(TeachBuilder.build(frames, pipeline, Synth.meta(), negatives = negs)).twin
        val b = success(TeachBuilder.build(frames, pipeline, Synth.meta(), negatives = negs)).twin
        assertArrayEquals(a.bank, b.bank, 0f)
        assertArrayEquals(a.bankKf, b.bankKf)
        assertArrayEquals(a.globals, b.globals, 0f)
        assertArrayEquals(a.kfFeats, b.kfFeats, 0f)
        assertArrayEquals(a.lso, b.lso, 0.0)
        assertArrayEquals(a.positives, b.positives, 0.0)
        assertArrayEquals(a.negativeSims, b.negativeSims, 0.0)
        assertEquals(a.thresholds, b.thresholds)
        assertEquals(a.keyframes, b.keyframes)
        assertEquals(IdentityThreshold.RULE_NEGATIVES, a.thresholds.tauIdRule)
    }

    @Test
    fun notEnoughFramesFails() {
        val frames = Synth.teachFrames(9, gh, gw, dim, seed = 3, insaneEvery = 0)
        val r = TeachBuilder.build(frames, pipeline, Synth.meta())
        assertTrue(r is TeachResult.Failure)
        r as TeachResult.Failure
        assertEquals(TeachFailure.NOT_ENOUGH_FRAMES, r.reason)
        assertEquals(9, r.diagnostics.accepted.size)
        assertTrue(r.diagnostics.kept.size < 8)
        // Frames that say OK but whose cov has no core are NO_CORE, not accepted.
        val empty = frames.map { TeachFrame(it.tMs, it.sanity, it.sharpness, FloatArray(gh * gw), it.geometry, it.features) }
        val r2 = TeachBuilder.build(empty, pipeline, Synth.meta()) as TeachResult.Failure
        assertEquals(0, r2.diagnostics.accepted.size)
    }

    @Test
    fun singleSegmentFallsBackToLeaveOneKeyframeOut() {
        val frames = Synth.teachFrames(40, gh, gw, dim, seed = 4)
        val r = success(TeachBuilder.build(frames, pipeline, Synth.meta(), TeachParams(segments = 1)))
        assertEquals(LsoMode.KEYFRAME, r.diagnostics.lsoMode)
        assertEquals(1, r.diagnostics.segmentsUsed)
        assertEquals(r.twin.keyframeCount, r.diagnostics.lso.size)
    }

    @Test
    fun keyframesWithoutAnAllowedBankRowAreSkipped() {
        val frames = Synth.teachFrames(40, gh, gw, dim, seed = 5)
        val r = success(TeachBuilder.build(frames, pipeline, Synth.meta(), TeachParams(maxRows = 1, minK = 1)))
        assertEquals(1, r.twin.bankRows)
        val seg0 = r.twin.bankSeg[0]
        val want = r.diagnostics.segments.indices.filter { r.diagnostics.segments[it] != seg0 }
        assertArrayEquals(want.toIntArray(), r.diagnostics.lsoKeyframes)
    }

    @Test
    fun augment4PoolsRotatedViewsBeforeTheCoreset() {
        val pipe = Synth.pipeline(gh, gw, dim, rotation = RotationRule.AUGMENT4)
        val frames = Synth.teachFrames(40, gh, gw, dim, seed = 6)
        val aug = KeyframeAugmenter { i ->
            val f = frames[i]
            List(3) { AugmentedView(FeatureMap(gh, gw, dim, f.features!!.data.copyOf()), f.cov!!.copyOf()) }
        }
        val plain = success(TeachBuilder.build(frames, pipeline, Synth.meta()))
        val r = success(TeachBuilder.build(frames, pipe, Synth.meta(), negatives = emptyList(), augmenter = aug))
        assertEquals(4 * plain.diagnostics.pooledRows, r.diagnostics.pooledRows)
    }

    /**
     * Performance smoke test of spec §7 at phone scale: 32 keyframes of 40×40×128 with ~900 B-set patches each
     * (≈ 28,800 pooled rows → 2,400 bank rows), exact CPU k-NN for the LSO step.
     */
    @Test
    fun teachTimingSmoke() {
        val g = 40
        val d = 128
        val pipe = Synth.pipeline(g, g, d)
        val base = Synth.pattern(g, g, d, 77)
        val rnd = Random(78)
        val frames = List(40) { i ->
            val cov = Synth.discCov(g, g, 20.0 + rnd.nextDouble() - 0.5, 20.0 + rnd.nextDouble() - 0.5, 12.0)
            TeachFrame(
                tMs = i * 300L, sanity = SanityReason.OK, sharpness = 500.0 + i, cov = cov, geometry = Synth.geometry(rnd),
                features = FeatureMap(g, g, d, Synth.view(base, 0.35, rnd)),
            )
        }
        var first: TwinModel? = null
        for (threads in intArrayOf(1, 4)) {
            val t0 = System.nanoTime()
            val r = success(
                TeachBuilder.build(frames, pipe, Synth.meta(), TeachParams(keepPercentile = 0.0, kcEps = 0.0, threads = threads)),
            )
            val ms = (System.nanoTime() - t0) / 1e6
            val dg = r.diagnostics
            println(
                "[timing] teach 40 frames 40x40x128, %d thread(s): %d keyframes, %d pooled rows (%.0f per keyframe) -> %d bank rows in %.0f ms; stages %s"
                    .format(threads, r.twin.keyframeCount, dg.pooledRows, dg.pooledRows.toDouble() / r.twin.keyframeCount,
                        r.twin.bankRows, ms, dg.timingsMs.entries.joinToString { "${it.key}=%.0f".format(it.value) }),
            )
            assertEquals(32, r.twin.keyframeCount)
            assertEquals(2400, r.twin.bankRows)
            val f = first
            if (f == null) {
                first = r.twin
            } else {
                // Deterministic for any thread count.
                assertArrayEquals(f.bank, r.twin.bank, 0f)
                assertArrayEquals(f.bankKf, r.twin.bankKf)
                assertArrayEquals(f.lso, r.twin.lso, 0.0)
                assertEquals(f.thresholds, r.twin.thresholds)
            }
        }
    }
}
