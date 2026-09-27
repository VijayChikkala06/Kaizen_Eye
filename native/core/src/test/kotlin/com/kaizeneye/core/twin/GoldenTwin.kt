package com.kaizeneye.core.twin

import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.math.abs

/**
 * Reader for `testdata/golden_twin.json` (layout: docstring of tools/lab/make_golden_twin.py). Tests call [cases],
 * which skips (JUnit Assume) when the file or the section is missing.
 */
object GoldenTwin {
    val file: File? = System.getProperty("testdata.dir")?.let { File(it, "golden_twin.json") }

    val root: JsonObject? by lazy {
        val f = file
        if (f == null || !f.isFile) null
        else Json.parseToJsonElement(f.readText(Charsets.UTF_8).removePrefix("﻿")).jsonObject
    }

    /** The cases of [section]; skips the calling test when the file or section is absent. */
    fun cases(section: String): List<JsonObject> {
        val r = root
        assumeTrue("golden_twin.json not found", r != null)
        val sec = r!![section]
        assumeTrue("golden_twin.json has no section '$section'", sec is JsonObject)
        val cases = (sec as JsonObject)["cases"]
        assumeTrue("section '$section' has no cases", cases is JsonArray && cases.isNotEmpty())
        return (cases as JsonArray).map { it.jsonObject }
    }

    // ------------------------------------------------------------------------------------------------ tolerances
    const val F64_REL = 1e-6
    const val F64_ABS = 1e-9
    const val F32_REL = 1e-4
    const val F32_ABS = 1e-6
    const val BETA_ABS = 1e-9

    /** Largest error seen per tolerance class, as a fraction of the allowed error (for the summary line). */
    private val worst = java.util.concurrent.ConcurrentHashMap<String, Double>()

    private fun track(cls: String, err: Double, allowed: Double) {
        worst.merge(cls, err / allowed) { a, b -> maxOf(a, b) }
    }

    /** "f32 0.3% of tolerance, ..." then reset. */
    fun summary(): String = worst.entries.sortedBy { it.key }
        .joinToString { "${it.key} %.3g%% of tolerance".format(100 * it.value) }.also { worst.clear() }

    fun f64(what: String, got: Double, exp: Double) {
        val allowed = F64_REL * abs(exp) + F64_ABS
        assertTrue("$what: got $got, expected $exp", abs(got - exp) <= allowed)
        track("f64", abs(got - exp), allowed)
    }

    fun f32(what: String, got: Double, exp: Double) {
        val allowed = F32_REL * abs(exp) + F32_ABS
        assertTrue("$what: got $got, expected $exp", abs(got - exp) <= allowed)
        track("f32", abs(got - exp), allowed)
    }

    fun beta(what: String, got: Double, exp: Double) {
        assertTrue("$what: got $got, expected $exp", abs(got - exp) <= BETA_ABS)
        track("beta", abs(got - exp), BETA_ABS)
    }
}

// ---------------------------------------------------------------------------------------------------- JSON helpers

val JsonElement.isNull: Boolean get() = this is JsonNull
fun JsonObject.obj(k: String): JsonObject = getValue(k).jsonObject
fun JsonObject.arr(k: String): JsonArray = getValue(k).jsonArray
fun JsonObject.d(k: String): Double = getValue(k).jsonPrimitive.double
fun JsonObject.dOrNull(k: String): Double? = this[k]?.takeUnless { it is JsonNull }?.jsonPrimitive?.double
fun JsonObject.i(k: String): Int = getValue(k).jsonPrimitive.int
fun JsonObject.l(k: String): Long = getValue(k).jsonPrimitive.long
fun JsonObject.b(k: String): Boolean = getValue(k).jsonPrimitive.boolean
fun JsonObject.s(k: String): String = getValue(k).jsonPrimitive.content
fun JsonObject.sOrNull(k: String): String? = this[k]?.takeUnless { it is JsonNull }?.jsonPrimitive?.content
fun JsonObject.has(k: String): Boolean = this[k] != null && this[k] !is JsonNull

fun JsonElement.doubles(): DoubleArray = jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
fun JsonElement.floats(): FloatArray = jsonArray.map { it.jsonPrimitive.double.toFloat() }.toFloatArray()
fun JsonElement.ints(): IntArray = jsonArray.map { it.jsonPrimitive.int }.toIntArray()
fun JsonElement.longs(): LongArray = jsonArray.map { it.jsonPrimitive.long }.toLongArray()
fun JsonElement.bools(): BooleanArray = jsonArray.map { (it as JsonPrimitive).let { p -> p.content == "true" || p.content == "1" } }.toBooleanArray()
fun JsonElement.strings(): List<String> = jsonArray.map { it.jsonPrimitive.content }

/** FMAP = {"gh","gw","dim","data"}. */
fun JsonElement.fmap(): FeatureMap {
    val o = jsonObject
    return FeatureMap(o.i("gh"), o.i("gw"), o.i("dim"), o.getValue("data").floats())
}

/** MAT = {"rows","cols","data"} → (rows, cols, float32 data). */
class GoldenMat(val rows: Int, val cols: Int, val data: FloatArray)

fun JsonElement.mat(): GoldenMat {
    val o = jsonObject
    return GoldenMat(o.i("rows"), o.i("cols"), o.getValue("data").floats())
}

/** GEOMETRY (inputs carry the five features). */
fun JsonElement.geometry(): GeometryFeatures {
    val o = jsonObject
    return GeometryFeatures(o.d("area"), o.d("fill"), o.d("aspect"), o.d("hu1"), o.d("solidity"))
}

/** MASK = flat 0/1 list. */
fun JsonElement.mask(): BooleanArray = jsonArray.map { it.jsonPrimitive.int != 0 }.toBooleanArray()
