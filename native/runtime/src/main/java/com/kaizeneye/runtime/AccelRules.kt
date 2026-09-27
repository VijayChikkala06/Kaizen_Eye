package com.kaizeneye.runtime

import java.util.Locale
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sqrt

/*
 * Accelerator honesty rules (twin-spec §13, ported from mobile/src/ml/accelerator.ts and generalised to N model variants).
 * Pure Kotlin: no Android types, so every rule is unit-tested on the JVM (AccelRulesTest).
 *
 *  - Candidates = accelerator x model variant. The first (usable) non-int8 variant is the REFERENCE variant; "CPU" on it
 *    (4 threads) is the accuracy reference and the fallback. Every other variant also runs on the CPU, as the speed baseline
 *    for accelerators running that variant ("CPU (int8)", "CPU (fp32)", ...); those CPU runs are never chosen.
 *    GPU runs float variants only (like the legacy app); NPU runs every variant.
 *  - Float (non-int8) candidates pass iff all outputs are finite, cosine >= 0.999 and relative L2 error <= 1 % vs the
 *    reference output; int8 candidates pass iff the mean per-patch cosine vs the reference output is >= 0.99.
 *  - A non-CPU candidate is used only if its best (min) timed run is <= 0.9x the best CPU run of the SAME variant.
 *    Otherwise it is reported "<X> requested - no speed-up (ran on CPU?)" and not used: CompiledModel.Options(NPU) silently
 *    runs unsupported parts on the CPU, so a measured speed-up is the only honest evidence that the accelerator did the work.
 *  - Among the eligible candidates (and the reference CPU) the lowest median wins; ties go to the CPU.
 * Candidate labels: "NPU", "GPU", "CPU" for the reference variant, "<ACCEL> (<tag>)" for the others, where tag = "int8" for
 * the (single) int8 variant, else the variant id's suffix after the last '_' (e.g. "fp32"), else the whole id.
 */

/** One benchmark candidate: [accel] running [variant]. [tag] is null for the reference variant. */
internal data class Candidate(val accel: Accel, val variant: ModelAsset, val tag: String?) {
    val label: String get() = if (tag == null) accel.name else "${accel.name} ($tag)"
    /** Badge name, e.g. "NPU int8". */
    val short: String get() = if (tag == null) accel.name else "${accel.name} $tag"
    val int8: Boolean get() = variant.int8
    val isReference: Boolean get() = accel == Accel.CPU && tag == null
    /** Name of the CPU run of the same variant (the speed baseline). */
    val baselineLabel: String get() = if (tag == null) "CPU" else "CPU ($tag)"
    val baselineShort: String get() = if (tag == null) "CPU" else "CPU $tag"
}

/** The candidates for a list of (usable) variants, in spec order. */
internal class CandidateSet(val variants: List<ModelAsset>, npuAllowed: Boolean = true, gpuAllowed: Boolean = true) {
    val reference: ModelAsset = variants.firstOrNull { !it.int8 }
        ?: throw IllegalArgumentException("no float (non-int8) variant among ${variants.map { it.id }}")

    private val tags: Map<String, String?> = tagsFor(variants, reference)

    fun candidate(accel: Accel, variant: ModelAsset): Candidate = Candidate(accel, variant, tags[variant.id])

    fun cpu(variant: ModelAsset): Candidate = candidate(Accel.CPU, variant)

    val referenceCpu: Candidate get() = candidate(Accel.CPU, reference)

    /** Report order: NPU x variants, GPU x float variants, CPU x variants (reference first within each group). */
    val all: List<Candidate> = buildList {
        val ordered = listOf(reference) + variants.filter { it.id != reference.id }
        if (npuAllowed) ordered.forEach { add(candidate(Accel.NPU, it)) }
        if (gpuAllowed) ordered.filter { !it.int8 }.forEach { add(candidate(Accel.GPU, it)) }
        ordered.forEach { add(candidate(Accel.CPU, it)) }
    }

    /** Probe order: CPU runs first (reference first), then GPU, then NPU (riskiest last). */
    val runOrder: List<Candidate> = all.filter { it.accel == Accel.CPU } + all.filter { it.accel == Accel.GPU } +
        all.filter { it.accel == Accel.NPU }

    fun byLabel(label: String): Candidate? = all.firstOrNull { it.label == label } ?: allAccels().firstOrNull { it.label == label }

    /** Every accelerator x variant combination (also the ones that are not benchmark candidates, e.g. for forced runs). */
    fun allAccels(): List<Candidate> = Accel.entries.flatMap { a -> variants.map { candidate(a, it) } }

    companion object {
        fun tagsFor(variants: List<ModelAsset>, reference: ModelAsset): Map<String, String?> {
            val others = variants.filter { it.id != reference.id }
            val int8Count = others.count { it.int8 }
            val raw = others.associate { v ->
                v.id to when {
                    v.int8 && int8Count == 1 -> "int8"
                    v.id.contains('_') && v.id.substringAfterLast('_').isNotEmpty() -> v.id.substringAfterLast('_')
                    else -> v.id
                }
            }
            val dup = raw.values.groupingBy { it }.eachCount().filterValues { it > 1 }.keys
            val out = HashMap<String, String?>()
            out[reference.id] = null
            for ((id, t) in raw) out[id] = if (t in dup || t.isEmpty()) id else t
            return out
        }
    }
}

/** What happened to one candidate during a benchmark. [timesMs] null = it did not run ([error] says why). */
internal data class Measurement(
    val candidate: Candidate,
    val timesMs: DoubleArray? = null,
    val loadMs: Double? = null,
    val comparison: AccelRules.Comparison? = null,
    val error: String? = null,
)

internal object AccelRules {
    const val MIN_COSINE = 0.999
    const val MAX_REL_ERR = 0.01
    const val INT8_MIN_PATCH_COSINE = 0.99
    /** A non-CPU candidate must need at most this fraction of the CPU time (best runs) of the same variant. */
    const val MAX_TIME_VS_CPU = 0.9
    const val WARMUP_RUNS = 2
    const val TIMED_RUNS = 10

    // ------------------------------------------------------------------------------------------------ accuracy
    data class Comparison(val cosine: Double, val patchCosine: Double, val relErr: Double, val finite: Boolean)

    /** Compare [out] with the reference; [vec] = length of one feature vector for the per-patch cosine (0 = whole output). Float64 sums. */
    fun compare(out: FloatArray, ref: FloatArray, vec: Int = 0): Comparison {
        if (out.size != ref.size || out.isEmpty()) return Comparison(0.0, 0.0, Double.POSITIVE_INFINITY, false)
        var dot = 0.0
        var no = 0.0
        var nr = 0.0
        var diff = 0.0
        var finite = true
        for (i in out.indices) {
            val a = out[i].toDouble()
            val b = ref[i].toDouble()
            if (!a.isFinite()) finite = false
            dot += a * b
            no += a * a
            nr += b * b
            diff += (a - b) * (a - b)
        }
        val cosine = cos(dot, no, nr)
        var patchCosine = cosine
        if (vec > 0 && out.size % vec == 0) {
            val n = out.size / vec
            var sum = 0.0
            for (p in 0 until n) {
                var d = 0.0
                var x = 0.0
                var y = 0.0
                for (t in p * vec until (p + 1) * vec) {
                    val a = out[t].toDouble()
                    val b = ref[t].toDouble()
                    d += a * b
                    x += a * a
                    y += b * b
                }
                sum += cos(d, x, y)
            }
            patchCosine = sum / n
        }
        val relErr = if (nr == 0.0) sqrt(diff) else sqrt(diff / nr)
        return Comparison(
            cosine,
            patchCosine,
            relErr,
            finite && cosine.isFinite() && patchCosine.isFinite() && relErr.isFinite(),
        )
    }

    private fun cos(d: Double, x: Double, y: Double): Double {
        if (x == 0.0 && y == 0.0) return 1.0
        val p = x * y
        return d / sqrt(if (p == 0.0 || p.isNaN()) 1e-300 else p)
    }

    fun accurate(c: Comparison, int8: Boolean): Boolean {
        if (!c.finite) return false
        return if (int8) c.patchCosine >= INT8_MIN_PATCH_COSINE else c.cosine >= MIN_COSINE && c.relErr <= MAX_REL_ERR
    }

    /** Why a candidate failed its accuracy gate. */
    fun gateText(c: Comparison, int8: Boolean): String = when {
        !c.finite -> "non-finite output (NaN/Inf) or wrong output size"
        int8 -> "too far from the float CPU output (mean patch cosine ${f4(c.patchCosine)} < $INT8_MIN_PATCH_COSINE)"
        else -> "output differs from CPU (cosine ${f4(c.cosine)}, rel. error ${e1(c.relErr)}; need >= $MIN_COSINE and <= $MAX_REL_ERR)"
    }

    // --------------------------------------------------------------------------------------------------- timing
    data class Timing(val medianMs: Double, val minMs: Double, val p95Ms: Double)

    fun timing(timesMs: DoubleArray?): Timing? {
        if (timesMs == null || timesMs.isEmpty()) return null
        val s = timesMs.sortedArray()
        return Timing(percentileSorted(s, 50.0), s[0], percentileSorted(s, 95.0))
    }

    /** twin-spec §0 percentile (numpy "linear") of ascending [s]. */
    fun percentileSorted(s: DoubleArray, q: Double): Double {
        require(s.isNotEmpty()) { "percentile of nothing" }
        val h = (s.size - 1) * q / 100.0
        val lo = floor(h).toInt().coerceIn(0, s.size - 1)
        val hi = min(lo + 1, s.size - 1)
        return s[lo] + (h - lo) * (s[hi] - s[lo])
    }

    // ------------------------------------------------------------------------------------------------- decision
    data class Decision(
        val chosen: Candidate,
        val results: List<CandidateResult>,
        val speedupVsCpu: Double?,
        val badge: String,
    )

    /**
     * Apply the rules. [measurements]: one per candidate that was considered (order irrelevant); the reference CPU one must
     * have run. Results come back in [set] report order (candidates without a measurement are left out).
     */
    fun decide(set: CandidateSet, measurements: List<Measurement>): Decision {
        val by = measurements.associateBy { it.candidate.label }
        val refCand = set.referenceCpu
        val ref = requireNotNull(by[refCand.label]) { "reference CPU measurement missing" }
        val refT = requireNotNull(timing(ref.timesMs)) { "reference CPU run did not happen: ${ref.error}" }

        fun baselineTiming(c: Candidate): Timing? {
            if (c.tag == null) return refT
            val m = by[c.baselineLabel] ?: return null
            return if (m.error == null) timing(m.timesMs) else null
        }

        val errors = HashMap<String, String?>()
        val notes = HashMap<String, String?>()
        val oks = HashMap<String, Boolean>()
        val eligible = ArrayList<Pair<Candidate, Timing>>()
        eligible += refCand to refT
        oks[refCand.label] = true

        for (cand in set.all) {
            if (cand.isReference) continue
            val m = by[cand.label] ?: continue
            val t = timing(m.timesMs)
            val c = m.comparison
            var error = m.error
            var ok = false
            if (error == null) {
                when {
                    t == null || c == null -> error = "did not run"
                    !accurate(c, cand.int8) -> error = gateText(c, cand.int8)
                    else -> ok = true
                }
            }
            oks[cand.label] = ok
            errors[cand.label] = error
            if (!ok || t == null) continue
            if (cand.accel == Accel.CPU) {
                notes[cand.label] = "speed baseline for accelerators running ${cand.variant.id}; not selectable"
                continue
            }
            val base = baselineTiming(cand)
            when {
                base == null -> notes[cand.label] = "${cand.baselineLabel} baseline unavailable, so acceleration cannot be verified - not used"
                t.minMs > MAX_TIME_VS_CPU * base.minMs -> notes[cand.label] =
                    "${noSpeedupText(cand)}: best ${fmtMs(t.minMs)} vs ${fmtMs(base.minMs)} ms on ${cand.baselineLabel}"
                t.medianMs > MAX_TIME_VS_CPU * base.medianMs -> notes[cand.label] =
                    "${noSpeedupText(cand)}: median ${fmtMs(t.medianMs)} vs ${fmtMs(base.medianMs)} ms on ${cand.baselineLabel}"
                else -> eligible += cand to t
            }
        }

        var best = eligible[0]
        for (e in eligible) if (e.second.medianMs < best.second.medianMs) best = e
        for ((cand, _) in eligible) {
            if (cand.label != best.first.label && notes[cand.label] == null) notes[cand.label] = "slower than ${best.first.label}"
        }
        if (!best.first.isReference) notes[refCand.label] = "${best.first.label} is faster"

        val speedup = if (best.first.accel == Accel.CPU) null else baselineTiming(best.first)?.let { it.medianMs / best.second.medianMs }

        val results = ArrayList<CandidateResult>()
        for (cand in set.all) {
            val m = by[cand.label] ?: continue
            val t = timing(m.timesMs)
            val c = m.comparison
            val isRef = cand.isReference
            results += CandidateResult(
                name = cand.label,
                modelId = cand.variant.id,
                ok = oks[cand.label] == true,
                medianMs = t?.medianMs,
                minMs = t?.minMs,
                p95Ms = t?.p95Ms,
                cosine = if (isRef) 1.0 else c?.cosine,
                patchCosine = if (isRef) 1.0 else c?.patchCosine,
                relErr = if (isRef) 0.0 else c?.relErr,
                loadMs = m.loadMs,
                error = if (isRef) m.error else errors[cand.label],
                note = notes[cand.label],
            )
        }
        return Decision(best.first, results, speedup, badge(best.first, best.second.medianMs, speedup))
    }

    fun noSpeedupText(c: Candidate): String = "${c.short} requested - no speed-up (ran on CPU?)"

    // ---------------------------------------------------------------------------------------------------- badges
    /** "NPU 4.1 ms (7.9× CPU)", "NPU int8 3.0 ms (6.0× CPU int8)", "GPU 9.0 ms (2.1× CPU)", "CPU 38 ms". */
    fun badge(c: Candidate, medianMs: Double?, speedup: Double?): String {
        val sb = StringBuilder(c.short)
        if (medianMs != null && medianMs.isFinite()) sb.append(' ').append(fmtMs(medianMs)).append(" ms")
        if (c.accel != Accel.CPU && speedup != null && speedup.isFinite()) {
            sb.append(" (").append(fmtX(speedup)).append("× ").append(c.baselineShort).append(')')
        }
        return sb.toString()
    }

    /**
     * Badge for a forced (A/B) instance, honestly: an accurate accelerator that was not faster than the CPU keeps the
     * "requested - no speed-up" wording. [result] = the candidate's benchmark numbers, [baseline] = CPU on the same variant.
     */
    fun forcedBadge(c: Candidate, result: CandidateResult?, baseline: CandidateResult?): String {
        val median = result?.medianMs
        val noSpeedup = c.accel != Accel.CPU && result?.minMs != null && baseline?.minMs != null &&
            result.minMs > MAX_TIME_VS_CPU * baseline.minMs
        if (noSpeedup) return (noSpeedupText(c) + (median?.let { " ${fmtMs(it)} ms" } ?: "")) + " (forced)"
        val speedup = if (c.accel != Accel.CPU && median != null && baseline?.medianMs != null) baseline.medianMs / median else null
        return badge(c, median, speedup) + " (forced)"
    }

    fun fmtMs(ms: Double): String = if (ms < 10.0) String.format(Locale.US, "%.1f", ms) else String.format(Locale.US, "%.0f", ms)

    fun fmtX(x: Double): String = String.format(Locale.US, "%.1f", x)

    private fun f4(x: Double): String = String.format(Locale.US, "%.4f", x)

    private fun e1(x: Double): String = String.format(Locale.US, "%.1e", x)

    /** One line for logs, e.g. "[r18_320_f32] NPU 4.1 ms (7.9× CPU) :: NPU 4.1 ms (chosen) · GPU: failed (...) · CPU: 38 ms, NPU is faster". */
    fun describe(r: AcceleratorReport): String {
        val parts = r.results.map { x ->
            val t = x.medianMs?.let { "${fmtMs(it)} ms" } ?: ""
            val q = if (x.name.contains("int8") && x.patchCosine != null) ", patch cos ${f4(x.patchCosine)}" else ""
            when {
                x.name == r.chosen -> "${x.name} $t$q (chosen)"
                x.error != null -> "${x.name}: failed (${x.error})"
                else -> "${x.name}: $t$q${x.note?.let { ", $it" } ?: ""}"
            }
        }
        return "[${r.modelId}] ${r.badge} :: " + parts.joinToString(" · ") +
            (if (r.fromCache) " [cached]" else "") + (if (r.forced) " [forced]" else "") + (r.note?.let { " - $it" } ?: "")
    }
}
