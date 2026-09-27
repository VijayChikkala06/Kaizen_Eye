package com.kaizeneye.core.legacy

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.Random
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** The legacy port against the Python reference's golden vectors (same checks as mobile/scripts/verify-core.ts). */
class PatchCoreLegacyTest {

    private fun golden(name: String): JsonObject? {
        val dir = System.getProperty("testdata.dir") ?: return null
        val f = File(dir, name)
        if (!f.isFile) return null
        return Json.parseToJsonElement(f.readText(Charsets.UTF_8).removePrefix("﻿")).jsonObject
    }

    private fun flatten(e: JsonElement, out: MutableList<Float>) {
        when (e) {
            is JsonArray -> e.forEach { flatten(it, out) }
            is JsonPrimitive -> out.add(e.double.toFloat())
            else -> error("unexpected $e")
        }
    }

    private fun floats(e: JsonElement): FloatArray = ArrayList<Float>().also { flatten(e, it) }.toFloatArray()

    private var maxRel = 0.0

    private fun close(name: String, got: Double, want: Double, rel: Double = 1e-4) {
        val r = abs(got - want) / max(abs(want), 1e-12)
        maxRel = max(maxRel, r)
        assertTrue("$name: got $got, want $want (rel $r)", r <= rel)
    }

    private fun check(file: String) {
        val g = golden(file)
        assumeTrue("$file missing", g != null)
        g!!
        val p = g.getValue("params").jsonObject
        val gh = p.getValue("gh").jsonPrimitive.int
        val gw = p.getValue("gw").jsonPrimitive.int
        val d = p.getValue("D").jsonPrimitive.int
        val ratio = p.getValue("ratio").jsonPrimitive.double
        val minK = p.getValue("minK").jsonPrimitive.int
        val margin = p.getValue("margin").jsonPrimitive.double
        val start = p.getValue("start").jsonPrimitive.int
        val smooth = p["smooth"]?.jsonPrimitive?.boolean ?: false
        val border = p["border"]?.jsonPrimitive?.int ?: 0

        val frames = g.getValue("frames").jsonArray.map { floats(it) }
        val x = FloatArray(frames.sumOf { it.size })
        var o = 0
        for (f in frames) {
            f.copyInto(x, o)
            o += f.size
        }
        val k = max(minK, kotlin.math.floor(ratio * frames.size * gh * gw).toInt())
        val sel = PatchCoreLegacy.greedyCoreset(x, d, k, start)
        val want = g.getValue("coreset_indices").jsonArray.map { it.jsonPrimitive.int }.toIntArray()
        assertArrayEquals("$file coreset_indices", want, sel)

        val prof = PatchCoreLegacy.enrol(
            frames, gh, gw, d,
            PatchCoreLegacy.EnrolOptions(ratio = ratio, minK = minK, margin = margin, smooth = smooth, border = border),
        )
        val loo = g.getValue("loo_scores").jsonArray.map { it.jsonPrimitive.double }
        assertEquals("$file loo length", loo.size, prof.looScores.size)
        loo.forEachIndexed { i, v -> close("$file loo[$i]", prof.looScores[i], v) }
        close("$file tau", prof.tau, g.getValue("tau").jsonPrimitive.double)

        val res = PatchCoreLegacy.score(floats(g.getValue("test_frame")), prof, 1.0)
        val pd = floats(g.getValue("patch_dist"))
        pd.forEachIndexed { i, v -> close("$file patch_dist[$i]", res.dmap[i].toDouble(), v.toDouble()) }
        close("$file raw_score", res.raw, g.getValue("raw_score").jsonPrimitive.double)
        close("$file normalised_score", res.patchScore, g.getValue("normalised_score").jsonPrimitive.double)
        println("$file (grid ${gh}x$gw, smooth $smooth, border $border): coreset EXACT (${sel.size}), max rel error $maxRel")
    }

    @Test
    fun goldenCore() = check("golden_core.json")

    @Test
    fun goldenApp() = check("golden_app.json")

    /** Unpruned float32-storage greedy k-centre (the TS semantics without any pruning). */
    private fun bruteCoreset(x: FloatArray, dim: Int, k: Int, start: Int): IntArray {
        val n = x.size / dim
        val kk = min(k, n)
        val sel = IntArray(kk)
        sel[0] = start
        val mind = FloatArray(n) { PatchCoreLegacy.sqdistBounded(x, it * dim, x, start * dim, dim, Double.POSITIVE_INFINITY).toFloat() }
        for (s in 1 until kk) {
            var j = 0
            for (i in 1 until n) if (mind[i] > mind[j]) j = i
            sel[s] = j
            for (i in 0 until n) {
                val dd = PatchCoreLegacy.sqdistBounded(x, i * dim, x, j * dim, dim, Double.POSITIVE_INFINITY)
                if (dd < mind[i]) mind[i] = dd.toFloat()
            }
        }
        return sel
    }

    @Test
    fun prunedCoresetEqualsBruteForce() {
        val rnd = Random(11)
        for (trial in 0 until 6) {
            val dim = intArrayOf(8, 13, 32)[trial % 3]
            val n = 500 + rnd.nextInt(700)
            val centres = Array(12) { FloatArray(dim) { (rnd.nextGaussian() * 3).toFloat() } }
            val x = FloatArray(n * dim)
            for (i in 0 until n) {
                val c = centres[rnd.nextInt(centres.size)]
                for (t in 0 until dim) x[i * dim + t] = c[t] + (rnd.nextGaussian() * 0.3).toFloat()
            }
            // a few exact duplicates (ties)
            for (i in 0 until 20) x.copyInto(x, (n - 1 - i) * dim, i * dim, i * dim + dim)
            val k = 60 + rnd.nextInt(120)
            assertArrayEquals(bruteCoreset(x, dim, k, 0), PatchCoreLegacy.greedyCoreset(x, dim, k, 0))
        }
    }

    @Test
    fun outlierFramesAndHelpers() {
        assertEquals(emptyList<Int>(), PatchCoreLegacy.outlierFrames(listOf(1.0, 2.0, 3.0)))
        val loo = listOf(1.0, 1.1, 0.9, 1.0, 1.05, 5.0, 0.95, 1.02, 9.0, 1.0)
        assertEquals(listOf(8, 5), PatchCoreLegacy.outlierFrames(loo))
        assertEquals(256, PatchCoreLegacy.bankSize(1000))
        assertEquals(500, PatchCoreLegacy.bankSize(10000))
        val m = floatArrayOf(1f, 2f, 3f, 4f)
        val s = PatchCoreLegacy.smooth3(m, 2, 2)
        // edge replication: (0,0) averages 1,1,2 / 1,1,2 / 3,3,4 = 18/9; (1,1) averages 1,2,2 / 3,4,4 / 3,4,4 = 27/9
        assertEquals(2.0f, s[0], 1e-6f)
        assertEquals(3.0f, s[3], 1e-6f)
        val g = PatchCoreLegacy.globalDescriptor(floatArrayOf(3f, 0f, 0f, 4f), 2, 2)
        assertEquals(0.6f, g[0], 1e-7f)
        assertEquals(0.8f, g[1], 1e-7f)
    }
}
