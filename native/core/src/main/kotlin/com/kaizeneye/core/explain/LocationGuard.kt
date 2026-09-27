package com.kaizeneye.core.explain

import java.util.Locale

/** Outcome of [LocationGuard.check]. */
enum class GuardDecision { ACCEPT, REJECT }

/** [decision] plus, for a rejection, the first location word that contradicts the facts. */
data class GuardResult(val decision: GuardDecision, val offendingWord: String? = null) {
    val accepted: Boolean get() = decision == GuardDecision.ACCEPT
}

/**
 * Keeps the offline VLM from contradicting the verdict's location. A reply is ACCEPTED if it names no location, or only
 * locations consistent with the peak's region ([GridNaming]); otherwise it is REJECTED (the app then shows the
 * [TemplateExplainer] sentence instead).
 *
 * Location words (whole words, case-insensitive; hyphenated forms such as "upper-left" are split):
 * top / upper / above (+ topmost, uppermost), bottom / lower / below (+ bottommost, lowermost), left (+ leftmost),
 * right (+ rightmost), centre / center / middle (+ central, centred, centered).
 * - A side word is consistent iff the region lies on that side: a corner region accepts either of its two sides
 *   ("upper-left" accepts top and left), an edge region only its own side.
 * - A centre word is consistent for the `centre` region, and for an edge region ("top", "left", …) only together with
 *   that region's side word ("top centre"); never for a corner.
 * - Facts without a peak (PASS, NOT_ENROLLED, REFRAME) accept no location word at all.
 */
object LocationGuard {

    private enum class Loc { TOP, BOTTOM, LEFT, RIGHT, CENTRE }

    private val WORDS: Map<String, Loc> = mapOf(
        "top" to Loc.TOP, "upper" to Loc.TOP, "above" to Loc.TOP, "topmost" to Loc.TOP, "uppermost" to Loc.TOP,
        "bottom" to Loc.BOTTOM, "lower" to Loc.BOTTOM, "below" to Loc.BOTTOM, "bottommost" to Loc.BOTTOM,
        "lowermost" to Loc.BOTTOM,
        "left" to Loc.LEFT, "leftmost" to Loc.LEFT,
        "right" to Loc.RIGHT, "rightmost" to Loc.RIGHT,
        "centre" to Loc.CENTRE, "center" to Loc.CENTRE, "middle" to Loc.CENTRE, "central" to Loc.CENTRE,
        "centred" to Loc.CENTRE, "centered" to Loc.CENTRE,
    )

    private val SPLIT = Regex("[^\\p{L}]+")

    fun check(sentence: String, facts: Facts): GuardResult {
        val found = ArrayList<Pair<String, Loc>>()
        for (token in sentence.lowercase(Locale.ROOT).split(SPLIT)) {
            val loc = WORDS[token] ?: continue
            found.add(token to loc)
        }
        if (found.isEmpty()) return GuardResult(GuardDecision.ACCEPT)
        val region = facts.region ?: return GuardResult(GuardDecision.REJECT, found.first().first)
        val vb = GridNaming.verticalBand(region)
        val hb = GridNaming.horizontalBand(region)
        fun sideOk(loc: Loc): Boolean = when (loc) {
            Loc.TOP -> vb == 0
            Loc.BOTTOM -> vb == 2
            Loc.LEFT -> hb == 0
            Loc.RIGHT -> hb == 2
            Loc.CENTRE -> false
        }
        val isCentre = vb == 1 && hb == 1
        val isEdge = (vb == 1) != (hb == 1)
        val namesOwnSide = found.any { sideOk(it.second) }
        for ((word, loc) in found) {
            val ok = if (loc == Loc.CENTRE) isCentre || (isEdge && namesOwnSide) else sideOk(loc)
            if (!ok) return GuardResult(GuardDecision.REJECT, word)
        }
        return GuardResult(GuardDecision.ACCEPT)
    }
}
