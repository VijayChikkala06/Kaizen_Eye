package com.kaizeneye.core.vision

import com.kaizeneye.core.image.RgbImage
import com.kaizeneye.core.image.RgbaFrame
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
import org.junit.Assume
import java.io.File
import kotlin.math.abs

/**
 * Reader for `testdata/golden_twin.json` (spec §15; layout in the docstring of tools/lab/make_golden_twin.py), for the
 * H1b sections: image, mask, crop_patch, tracker, governor. Tests call [cases], which skips (JUnit Assume) while the file
 * or the section is missing.
 */
object GoldenVision {

    const val F64_REL = 1e-6
    const val F64_ABS = 1e-9

    val file: File? = System.getProperty("testdata.dir")?.let { File(it, "golden_twin.json") }

    private val root: JsonObject? by lazy {
        val f = file
        if (f == null || !f.isFile) null else Json.parseToJsonElement(f.readText(Charsets.UTF_8)).jsonObject
    }

    /** The cases of [section]; skips the calling test when the golden file or the section is absent. */
    fun cases(section: String): List<JsonObject> {
        val r = root
        Assume.assumeTrue("golden_twin.json not found at $file", r != null)
        val sec = r!![section]
        Assume.assumeTrue("golden_twin.json has no section '$section'", sec != null && sec !is JsonNull)
        return sec!!.jsonObject.getValue("cases").jsonArray.map { it.jsonObject }
    }

    // --- JSON helpers ------------------------------------------------------------------------------------------------

    fun JsonObject.obj(key: String): JsonObject = getValue(key).jsonObject
    fun JsonObject.arr(key: String): JsonArray = getValue(key).jsonArray
    fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int
    fun JsonObject.long(key: String): Long = getValue(key).jsonPrimitive.long
    fun JsonObject.dbl(key: String): Double = getValue(key).jsonPrimitive.double
    fun JsonObject.bool(key: String): Boolean = getValue(key).jsonPrimitive.boolean
    fun JsonObject.str(key: String): String = getValue(key).jsonPrimitive.content
    fun JsonObject.has(key: String): Boolean = this[key] != null && this[key] !is JsonNull
    fun JsonObject.nullableInt(key: String): Int? = if (has(key)) int(key) else null
    fun JsonObject.nullableDbl(key: String): Double? = if (has(key)) dbl(key) else null
    fun JsonObject.nullableStr(key: String): String? = if (has(key)) str(key) else null
    fun JsonObject.name(): String = (this["name"] as? JsonPrimitive)?.content ?: "?"

    fun JsonElement.ints(): IntArray = jsonArray.let { a -> IntArray(a.size) { a[it].jsonPrimitive.int } }
    fun JsonElement.doubles(): DoubleArray = jsonArray.let { a -> DoubleArray(a.size) { a[it].jsonPrimitive.double } }
    fun JsonElement.bytes(): ByteArray = jsonArray.let { a -> ByteArray(a.size) { a[it].jsonPrimitive.int.toByte() } }

    /** IMAGE with c = 3 → [RgbImage]. */
    fun rgbImage(o: JsonObject): RgbImage {
        require(o.int("c") == 3) { "expected an RGB image, got c=${o.int("c")}" }
        return RgbImage(o.int("w"), o.int("h"), o.getValue("data").bytes())
    }

    /** IMAGE with c = 4 and rowStride → [RgbaFrame] (padding bytes included). */
    fun rgbaFrame(o: JsonObject): RgbaFrame {
        require(o.int("c") == 4) { "expected an RGBA image, got c=${o.int("c")}" }
        val stride = if (o.has("rowStride")) o.int("rowStride") else o.int("w") * 4
        return RgbaFrame(o.int("w"), o.int("h"), stride, o.getValue("data").bytes())
    }

    // --- comparison --------------------------------------------------------------------------------------------------

    fun f64Close(expected: Double, got: Double): Boolean = abs(got - expected) <= F64_REL * abs(expected) + F64_ABS

    /** Collects every mismatch of a golden test so one run reports them all. */
    class Mismatches(private val section: String) {
        private val items = ArrayList<String>()
        var checks = 0
            private set

        fun check(ok: Boolean, what: () -> String) {
            checks++
            if (!ok) items += what()
        }

        fun f64(where: String, expected: Double, got: Double) =
            check(f64Close(expected, got)) { "$where: expected $expected, got $got (Δ=${got - expected})" }

        fun f64s(where: String, expected: DoubleArray, got: DoubleArray) {
            check(expected.size == got.size) { "$where: length ${got.size} != expected ${expected.size}" }
            if (expected.size != got.size) return
            var bad = 0
            for (i in expected.indices) {
                if (!f64Close(expected[i], got[i])) {
                    if (bad < 5) items += "$where[$i]: expected ${expected[i]}, got ${got[i]} (Δ=${got[i] - expected[i]})"
                    bad++
                }
            }
            checks++
            if (bad > 5) items += "$where: ... $bad mismatching values in total"
        }

        fun <T> eq(where: String, expected: T, got: T) = check(expected == got) { "$where: expected $expected, got $got" }

        fun ints(where: String, expected: IntArray, got: IntArray) {
            check(expected.size == got.size) { "$where: length ${got.size} != expected ${expected.size}" }
            if (expected.size != got.size) return
            var bad = 0
            for (i in expected.indices) {
                if (expected[i] != got[i]) {
                    if (bad < 5) items += "$where[$i]: expected ${expected[i]}, got ${got[i]}"
                    bad++
                }
            }
            checks++
            if (bad > 5) items += "$where: ... $bad mismatching values in total"
        }

        fun assertNone() {
            println("[golden] $section: $checks checks, ${items.size} mismatches")
            if (items.isNotEmpty()) {
                throw AssertionError("golden_twin.json section '$section': ${items.size} mismatches\n" + items.take(60).joinToString("\n"))
            }
        }
    }
}
