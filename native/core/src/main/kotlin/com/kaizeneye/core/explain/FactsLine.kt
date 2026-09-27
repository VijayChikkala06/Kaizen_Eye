package com.kaizeneye.core.explain

import com.kaizeneye.core.model.Verdict
import java.util.Locale

/** Number and reason formatting shared by the explanation texts (locale-independent). */
internal object Fmt {
    fun pct(v: Double): String = String.format(Locale.ROOT, "%.1f", v)
    fun times(v: Double): String = String.format(Locale.ROOT, "%.2f", v)
    fun reject(n: Int): String = String.format(Locale.ROOT, "#%03d", n)

    /** "TOUCHES_BORDER" → "touches border", "identity" → "identity". */
    fun reason(r: String): String = r.lowercase(Locale.ROOT).replace('_', ' ')

    fun verdict(v: Verdict): String = when (v) {
        Verdict.PASS -> "PASS"
        Verdict.DEFECT -> "DEFECT"
        Verdict.NOT_ENROLLED -> "NOT ENROLLED"
        Verdict.REFRAME -> "REFRAME"
    }
}

/**
 * One compact facts line for the result card and the log, e.g.
 * `M8 nut · DEFECT · upper-left (row 7/40, col 9/40) · 3.1 % of part · 1.34× limit`.
 * Rows and columns are shown 1-based; the accelerator label, when known, is appended.
 */
object FactsLine {

    const val SEP = " · "

    fun render(facts: Facts): String {
        val parts = ArrayList<String>(6)
        parts.add(facts.partName)
        parts.add(Fmt.verdict(facts.verdict))
        when (facts.verdict) {
            Verdict.DEFECT -> {
                val region = facts.region
                if (region != null) {
                    parts.add("$region (row ${facts.peakRow!! + 1}/${facts.gh}, col ${facts.peakCol!! + 1}/${facts.gw})")
                }
                facts.areaPct?.let { parts.add("${Fmt.pct(it)} % of part") }
                facts.score?.let { parts.add("${Fmt.times(it)}× limit") }
            }
            Verdict.PASS -> facts.score?.let { parts.add("${Fmt.times(it)}× limit") }
            Verdict.NOT_ENROLLED -> {
                facts.reason?.let { parts.add(Fmt.reason(it)) }
                if (facts.reason.equals("coverage", ignoreCase = true)) facts.areaPct?.let { parts.add("${Fmt.pct(it)} % of part") }
            }
            Verdict.REFRAME -> facts.reason?.let { parts.add(Fmt.reason(it)) }
        }
        facts.accelLabel?.let { parts.add(it) }
        return parts.joinToString(SEP)
    }
}
