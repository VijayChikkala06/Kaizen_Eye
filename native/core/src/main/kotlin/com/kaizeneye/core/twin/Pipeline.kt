package com.kaizeneye.core.twin

/*
 * SPEC-QUESTION: §8 defines the canonical pipeline JSON's numbers as "shortest round-trip form ... like Kotlin/JSON 0.1,
 * 4.0". Kotlin and Python disagree outside [1e-3, 1e7) (Kotlin "1.0E-4", Python "0.0001"). Chosen: the shortest
 * round-trip digits, written like Python's repr (plain notation for 1e-4 <= |v| < 1e16, else "1e-05" / "1.5e+16"),
 * which equals Kotlin's form for every value in the default pipeline. Strings are escaped like Python's json.dumps
 * (ensure_ascii: non-ASCII as \uXXXX).
 */

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.encodeToJsonElement
import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode
import java.security.MessageDigest

/** Bank challengers (spec §6.5). */
enum class BankRule { CORESET, TOPK_VIEWS }

/** Rotation challengers (spec §6.5). */
enum class RotationRule { NONE, CANONICAL, AUGMENT4 }

/** Mask parameters recorded in the pipeline (spec §2–§4, §8). */
@Serializable
data class MaskParams(
    val analysisFactor: Int = 4,
    val kSigma: Double = 4.0,
    val sigmaMin: Double = 3.0,
    val minBlobPx: Int = 30,
    val borderMargin: Int = 2,
    val cropMargin: Double = 0.10,
    val coreThreshold: Double = 0.5,
)

/**
 * Everything that changes the numbers of a Twin (spec §8 `pipeline`). Two pipelines with different [fingerprint]s must
 * never share a Twin ("Rebuild from stored crops").
 */
@Serializable
data class PipelineInfo(
    val backboneId: String,
    val backboneSha256: String,
    val inputSize: Int,
    val gh: Int,
    val gw: Int,
    val dim: Int,
    val precision: String = "fp32",
    val l2NormalizePatches: Boolean = false,
    val preprocessVersion: Int = 1,
    val mask: MaskParams = MaskParams(),
    val knnGraphId: String,
    val scoreRule: ScoreRule = ScoreRule.SMOOTHED_MAX,
    val bankRule: BankRule = BankRule.CORESET,
    val rotation: RotationRule = RotationRule.NONE,
) {
    val patches: Int get() = gh * gw

    /**
     * Puts a backbone output into pipeline space, in place (spec §5): DINOv2 patch vectors are L2-normalised per patch
     * "before anything else" when [l2NormalizePatches]; ResNet18 maps are left unchanged. Call once per feature map,
     * right after the backbone, before teach / judge.
     */
    fun prepareFeatures(fm: com.kaizeneye.core.model.FeatureMap): com.kaizeneye.core.model.FeatureMap {
        require(fm.gh == gh && fm.gw == gw && fm.dim == dim) { "feature map ${fm.gh}x${fm.gw}x${fm.dim} is not ${gh}x${gw}x$dim" }
        if (l2NormalizePatches) Descriptor.l2NormalizePatches(fm.data, fm.patches, fm.dim)
        return fm
    }

    /** Canonical pipeline JSON (spec §8): keys sorted at every level, no whitespace, shortest round-trip numbers. */
    fun canonicalJson(): String = CanonicalJson.write(CanonicalJson.json.encodeToJsonElement(this))

    /** SHA-256 hex (lowercase) of [canonicalJson] in UTF-8 (spec §8). */
    fun fingerprint(): String = CanonicalJson.sha256Hex(canonicalJson())
}

/** Canonical JSON writer + SHA-256 used for the pipeline fingerprint (spec §8). */
object CanonicalJson {
    internal val json = Json { encodeDefaults = true; explicitNulls = true }

    /** Serialises [e] with keys sorted lexicographically at every level and no whitespace. */
    internal fun write(e: JsonElement): String = StringBuilder().also { write(e, it) }.toString()

    private fun write(e: JsonElement, sb: StringBuilder) {
        when (e) {
            is JsonNull -> sb.append("null")
            is JsonObject -> {
                sb.append('{')
                var first = true
                for (k in e.keys.sorted()) {
                    if (!first) sb.append(',')
                    first = false
                    writeString(k, sb)
                    sb.append(':')
                    write(e.getValue(k), sb)
                }
                sb.append('}')
            }
            is JsonArray -> {
                sb.append('[')
                e.forEachIndexed { i, v ->
                    if (i > 0) sb.append(',')
                    write(v, sb)
                }
                sb.append(']')
            }
            is JsonPrimitive -> when {
                e.isString -> writeString(e.content, sb)
                e.content == "true" || e.content == "false" -> sb.append(e.content)
                else -> sb.append(formatNumber(e.content))
            }
        }
    }

    /** Integers stay integers; anything with a fraction or exponent is a double in its shortest round-trip form. */
    internal fun formatNumber(literal: String): String {
        val isDouble = literal.any { it == '.' || it == 'e' || it == 'E' }
        return if (isDouble) formatDouble(literal.toDouble()) else BigDecimal(literal).toBigIntegerExact().toString()
    }

    /** Shortest round-trip decimal of a double, Python-repr layout (see SPEC-QUESTION at the top of this file). */
    fun formatDouble(v: Double): String {
        require(!v.isNaN() && !v.isInfinite()) { "JSON has no NaN/inf" }
        if (v == 0.0) return if (1.0 / v < 0) "-0.0" else "0.0"
        val bd = shortest(v)
        val digits = bd.unscaledValue().abs().toString()
        val exp10 = digits.length - 1 - bd.scale()           // v = d.ddd × 10^exp10
        val sign = if (v < 0) "-" else ""
        return if (exp10 >= -4 && exp10 < 16) {
            // plain notation, at least one digit after the point
            val plain = bd.abs().toPlainString()
            sign + if (plain.contains('.')) plain else "$plain.0"
        } else {
            val mant = if (digits.length == 1) digits else digits[0] + "." + digits.substring(1)
            val e = if (exp10 < 0) "-" + (-exp10).toString().padStart(2, '0') else "+" + exp10.toString().padStart(2, '0')
            "$sign${mant}e$e"
        }
    }

    private fun shortest(v: Double): BigDecimal {
        val exact = BigDecimal(v)
        for (p in 1..17) {
            val r = exact.round(MathContext(p, RoundingMode.HALF_EVEN))
            if (r.toDouble() == v) return r.stripTrailingZeros()
        }
        return exact.round(MathContext(17, RoundingMode.HALF_EVEN)).stripTrailingZeros()
    }

    private fun writeString(s: String, sb: StringBuilder) {
        sb.append('"')
        for (ch in s) {
            when {
                ch == '"' -> sb.append("\\\"")
                ch == '\\' -> sb.append("\\\\")
                ch == '\n' -> sb.append("\\n")
                ch == '\r' -> sb.append("\\r")
                ch == '\t' -> sb.append("\\t")
                ch == '\b' -> sb.append("\\b")
                ch == '\u000C' -> sb.append("\\f")
                ch < ' ' || ch.code > 0x7E -> sb.append("\\u").append(ch.code.toString(16).padStart(4, '0'))
                else -> sb.append(ch)
            }
        }
        sb.append('"')
    }

    /** Lowercase hex SHA-256 of the UTF-8 bytes of [s]. */
    fun sha256Hex(s: String): String {
        val d = MessageDigest.getInstance("SHA-256").digest(s.toByteArray(Charsets.UTF_8))
        val sb = StringBuilder(64)
        for (b in d) {
            val v = b.toInt() and 0xFF
            sb.append(HEX[v ushr 4]).append(HEX[v and 0xF])
        }
        return sb.toString()
    }

    private const val HEX = "0123456789abcdef"
}
