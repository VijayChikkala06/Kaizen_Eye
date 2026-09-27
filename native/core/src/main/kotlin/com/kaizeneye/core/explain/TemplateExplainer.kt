package com.kaizeneye.core.explain

import com.kaizeneye.core.model.SanityReason
import com.kaizeneye.core.model.Verdict
import java.util.Locale

/**
 * Deterministic one-sentence explanation of a verdict — the fallback whenever the VLM is paused, slow, or rejected by
 * [LocationGuard]. Example: `Reject #042: anomaly covering 3.1 % of the part at the upper-left, 1.34× the learned limit.`
 */
object TemplateExplainer {

    fun sentence(facts: Facts): String = when (facts.verdict) {
        Verdict.DEFECT -> defect(facts)
        Verdict.PASS -> facts.score?.let { "Pass: within the learned limit (${Fmt.times(it)}×)." } ?: "Pass: within the learned limit."
        Verdict.NOT_ENROLLED -> notEnrolled(facts)
        Verdict.REFRAME -> reframe(facts.reason)
    }

    private fun defect(f: Facts): String {
        val sb = StringBuilder()
        sb.append(f.rejectNumber?.let { "Reject ${Fmt.reject(it)}" } ?: "Reject")
        sb.append(": anomaly")
        f.areaPct?.let { sb.append(" covering ").append(Fmt.pct(it)).append(" % of the part") }
        f.region?.let { sb.append(" at the ").append(it) }
        f.score?.let { sb.append(", ").append(Fmt.times(it)).append("× the learned limit") }
        return sb.append('.').toString()
    }

    private fun notEnrolled(f: Facts): String = when (f.reason?.lowercase(Locale.ROOT)) {
        "identity" -> "Not the enrolled part: it does not look like the taught ${f.partName}."
        "shape" -> "Not the enrolled part: its shape does not match the taught ${f.partName}."
        "fit" -> "Not the enrolled part: overall it looks different from the taught ${f.partName} — a similar object, not this one."
        "coverage" -> "Not the enrolled part: most of its surface differs from the taught ${f.partName}" +
            (f.areaPct?.let { " (${Fmt.pct(it)} % of the part)" } ?: "") + "."
        else -> "Not the enrolled part."
    }

    private fun reframe(reason: String?): String {
        val r = reason?.let { name -> SanityReason.entries.firstOrNull { it.name.equals(name, ignoreCase = true) } }
        return when (r) {
            SanityReason.NO_OBJECT -> "Reframe: no part in view."
            SanityReason.TOUCHES_BORDER -> "Reframe: the part touches the edge of the view — move it fully into the frame."
            SanityReason.TOO_SMALL -> "Reframe: the part looks too small — move the camera closer."
            SanityReason.TOO_LARGE -> "Reframe: the part fills too much of the view — move the camera back."
            SanityReason.MULTIPLE -> "Reframe: more than one object — show one part at a time."
            SanityReason.NO_CORE -> "Reframe: the part could not be separated from the background."
            SanityReason.OK, null -> "Reframe: show the part again."
        }
    }
}

/** The short instruction sent with the crop to the offline VLM (≤ [MAX_WORDS]-word, one-sentence reply). */
object VlmPrompt {

    const val MAX_WORDS = 25

    fun build(facts: Facts): String {
        val region = facts.region
        return when {
            facts.verdict == Verdict.DEFECT && region != null -> {
                val area = facts.areaPct?.let { " and covers ${Fmt.pct(it)} % of it" } ?: ""
                "This is a close-up photo of a ${facts.partName} that an inspection system rejected. " +
                    "The anomaly is at the $region of the part$area. " +
                    "In one sentence of at most $MAX_WORDS words, describe the visible defect at the $region. " +
                    "Do not mention any other location and do not guess."
            }
            facts.verdict == Verdict.DEFECT ->
                "This is a close-up photo of a ${facts.partName} that an inspection system rejected. " +
                    "In one sentence of at most $MAX_WORDS words, describe the visible defect. " +
                    "Do not mention any location and do not guess."
            else ->
                "This is a close-up photo of a ${facts.partName}. " +
                    "In one sentence of at most $MAX_WORDS words, describe what is visible. " +
                    "Do not mention any location and do not guess."
        }
    }

    /** Whitespace-separated word count of a reply. */
    fun wordCount(reply: String): Int = reply.trim().split(Regex("\\s+")).count { it.isNotEmpty() }

    /** The reply trimmed to its first sentence and at most [MAX_WORDS] words (ending with a full stop). */
    fun clip(reply: String): String {
        val text = reply.trim().replace(Regex("\\s+"), " ")
        val end = text.indexOfFirst { it == '.' || it == '!' || it == '?' }
        val first = if (end >= 0) text.substring(0, end + 1) else text
        val words = first.split(' ').filter { it.isNotEmpty() }
        if (words.size <= MAX_WORDS) return first
        return words.take(MAX_WORDS).joinToString(" ").trimEnd(',', ';', ':') + "."
    }
}
