package com.kaizeneye.core.twin

/*
 * SPEC-QUESTION: §10.1 re-derives the geometry model "from the sanity-ok samples' geometry"; with no sanity-ok sample
 * there is nothing to fit, so the teach geometry model is kept (and τ_id is re-derived from the teach positives only).
 * SPEC-QUESTION: §8 fixes "certificate": null before calibration but not the calibrated object's keys; its layout is
 * CertificateJson in TwinStore.kt.
 */

import com.kaizeneye.core.math.Binomial
import com.kaizeneye.core.math.Stats
import com.kaizeneye.core.model.GeometryFeatures
import java.util.Locale
import kotlin.math.max

/**
 * One distinct good presentation used for calibration (spec §10.1). [raw], [sim] and [geometry] are null when sanity
 * failed. [idOk] / [geoOk] are the gate results against the Twin being calibrated.
 */
data class CalibrationSample(
    val sanityOk: Boolean,
    val idOk: Boolean,
    val geoOk: Boolean,
    val raw: Double?,
    val sim: Double?,
    val geometry: GeometryFeatures?,
) {
    /** Valid = all three gates pass (spec §10.1). */
    val valid: Boolean get() = sanityOk && idOk && geoOk
}

/** Numbers of calibration and the certificate (spec §10). */
data class CalibrationParams(
    val conf: Double = 0.95,
    val geometry: GeometryParams = GeometryParams(),
    val identity: IdentityParams = IdentityParams(),
    /** Latency statistics use the last this-many samples (spec §14 ring buffer). */
    val latencyWindow: Int = 512,
)

/** Pass count `k/n` of one gate with the Clopper–Pearson upper bound on its rejection rate (spec §10.3 line 2). */
data class GateCount(val k: Int, val n: Int, val rejections: Int, val upper: Double) {
    companion object {
        fun of(k: Int, n: Int, conf: Double): GateCount = GateCount(k, n, n - k, Binomial.clopperPearsonUpper(n - k, n, conf))
    }
}

/** The certificate data (spec §10.2–10.3); [CertificateText] renders it. */
data class Certificate(
    val conf: Double,
    /** Number of calibration presentations `n`. */
    val n: Int,
    /** Sanity-ok presentations `n'`. */
    val nSanityOk: Int,
    /** Valid presentations `m` (all gates passed). */
    val m: Int,
    /** Order-statistic α of the score gate (null when m = 0). */
    val alpha: Double?,
    val sanity: GateCount,
    val identity: GateCount,
    val geometry: GateCount,
    /** `P5(pos) − P99(neg)`, null = no negatives captured. */
    val identityMargin: Double?,
    /** Segments that contain keyframes (the within-part bound's time blocks). */
    val bUsed: Int,
    /** `α_within = 1 − (1−conf)^(1/B_used)` (spec §10.2). */
    val withinPart: Double,
    val latencyP50Ms: Double?,
    val latencyP95Ms: Double?,
    /** Accelerator that ran the backbone on this phone (e.g. "NPU"), if known. */
    val accelerator: String?,
    /** `max raw` over valid samples (null when m = 0). */
    val calMax: Double?,
    val tauCal: Double,
) {
    /** The certificate is valid while `τ · sensitivity ≥ calMax` (spec §10.1); the UI marks it void below. */
    fun isValid(tau: Double, sensitivity: Double): Boolean = calMax == null || tau * sensitivity >= calMax
}

/** Result of [Calibration.calibrate]. */
class CalibrationResult(
    val tauCal: Double,
    val calMax: Double?,
    val identity: IdentityThreshold,
    val geometry: GeometryModel,
    val certificate: Certificate,
)

/** Calibration (spec §10.1) and the certificate numbers (§10.2–10.3). */
object Calibration {

    /**
     * Calibrates from distinct good presentations (spec §10.1):
     * `τ_cal = max(τ_teach, max raw over valid samples)` (never lowered), `τ_id` re-derived with
     * `pos = teach positives ∪ sims of sanity-ok samples`, geometry re-derived from the sanity-ok samples.
     */
    fun calibrate(
        tauTeach: Double,
        teachPositives: DoubleArray,
        negativeSims: DoubleArray,
        teachGeometry: GeometryModel,
        segmentsUsed: Int,
        samples: List<CalibrationSample>,
        params: CalibrationParams = CalibrationParams(),
        latenciesMs: DoubleArray = DoubleArray(0),
        accelerator: String? = null,
    ): CalibrationResult {
        val conf = params.conf
        val n = samples.size
        val sane = samples.filter { it.sanityOk }
        val valid = samples.filter { it.valid }
        val m = valid.size
        var calMax: Double? = null
        for (v in valid) {
            val r = requireNotNull(v.raw) { "a valid sample needs raw" }
            calMax = if (calMax == null) r else max(calMax, r)
        }
        val tauCal = if (calMax == null) tauTeach else max(tauTeach, calMax)

        val pos = DoubleArray(teachPositives.size + sane.size)
        teachPositives.copyInto(pos)
        sane.forEachIndexed { i, sm -> pos[teachPositives.size + i] = requireNotNull(sm.sim) { "a sanity-ok sample needs sim" } }
        val identity = IdentityThreshold.derive(pos, negativeSims, params.identity)
        val geometry = if (sane.isEmpty()) teachGeometry else GeometryModel.fit(
            sane.map { requireNotNull(it.geometry) { "a sanity-ok sample needs geometry" } }, params.geometry,
        )

        val window = latenciesMs.copyOfRange(max(0, latenciesMs.size - params.latencyWindow), latenciesMs.size)
        val certificate = Certificate(
            conf = conf,
            n = n,
            nSanityOk = sane.size,
            m = m,
            alpha = if (m >= 1) Binomial.orderStatisticAlpha(m, conf) else null,
            sanity = GateCount.of(sane.size, n, conf),
            identity = GateCount.of(sane.count { it.idOk }, sane.size, conf),
            geometry = GateCount.of(sane.count { it.geoOk }, sane.size, conf),
            identityMargin = identity.margin,
            bUsed = segmentsUsed,
            withinPart = withinPartAlpha(segmentsUsed, conf),
            latencyP50Ms = if (window.isEmpty()) null else Stats.percentile(window, 50.0),
            latencyP95Ms = if (window.isEmpty()) null else Stats.percentile(window, 95.0),
            accelerator = accelerator,
            calMax = calMax,
            tauCal = tauCal,
        )
        return CalibrationResult(tauCal, calMax, identity, geometry, certificate)
    }

    /** Within-part bound before calibration (spec §10.2): `α_within = 1 − (1−conf)^(1/B_used)`. */
    fun withinPartAlpha(bUsed: Int, conf: Double = 0.95): Double = Binomial.orderStatisticAlpha(bUsed, conf)
}

/** Plain-text certificate lines (spec §10.3); the UI renders them. */
object CertificateText {

    private fun pct(v: Double): String = String.format(Locale.ROOT, "%.1f %%", 100.0 * v)
    private fun num(v: Double): String = String.format(Locale.ROOT, "%.3f", v)
    private fun ms(v: Double): String = String.format(Locale.ROOT, "%.1f ms", v)

    /** The within-part bound line shown before calibration (spec §10.2). */
    fun withinPart(bUsed: Int, conf: Double = 0.95): String =
        "Within-part bound from $bUsed time blocks — not a false-alarm guarantee: " +
            "${pct(Calibration.withinPartAlpha(bUsed, conf))} at ${pct(conf)} confidence"

    /** The four certificate lines of spec §10.3. */
    fun lines(c: Certificate): List<String> {
        val conf = pct(c.conf)
        val line1 = if (c.alpha != null) {
            "Score gate: ${c.m} valid parts → false alarms ≤ ${pct(c.alpha)} at $conf confidence (score gate only)"
        } else {
            "Score gate: no valid calibration parts — no false-alarm bound"
        }
        val line2 = "Gate pass counts over ${c.n} presentations: " +
            "sanity ${c.sanity.k}/${c.sanity.n} (rejection ≤ ${pct(c.sanity.upper)}), " +
            "identity ${c.identity.k}/${c.identity.n} (≤ ${pct(c.identity.upper)}), " +
            "geometry ${c.geometry.k}/${c.geometry.n} (≤ ${pct(c.geometry.upper)}) at $conf"
        val line3 = if (c.identityMargin != null) {
            "Identity margin (P5 positives − P99 negatives): ${num(c.identityMargin)}" +
                if (c.identityMargin <= 0.0) " — overlap" else ""
        } else {
            "Identity margin: no negatives captured"
        }
        val line4 = if (c.latencyP50Ms != null && c.latencyP95Ms != null) {
            "Decision latency p50 ${ms(c.latencyP50Ms)} / p95 ${ms(c.latencyP95Ms)} on ${c.accelerator ?: "this phone"}"
        } else {
            "Decision latency: not measured"
        }
        return listOf(line1, line2, line3, line4)
    }
}
