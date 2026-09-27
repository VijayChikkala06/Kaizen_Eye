package com.kaizeneye.core.power

// SPEC-QUESTION: (§12 "it falls only after the lower target has held continuously for cooldownMs"): the cooldown clock
// starts at the first sample whose target is below the current level and keeps running while every sample's target stays
// below the level (L1 and L0 both count while at L2); when it reaches cooldownMs the level drops straight to the target
// of that sample. A sample whose target equals or exceeds the level stops the clock.
// SPEC-QUESTION: (§12 thresholds): headroom is compared in float32 against float32 thresholds (0.85f, 0.70f), so a reading
// of exactly 0.85 is not "> 0.85" and exactly 0.70 is ">= 0.70", as the decimal reading intends.

/** Governor level (spec §12). */
enum class PowerLevel { L0, L1, L2 }

/** How the offline VLM may be used at a level (spec §12). */
enum class VlmMode { AUTO, ON_TAP, PAUSED }

/**
 * Thermal governor parameters (spec §12; defaults = the spec's). Android `PowerManager` thermal status codes:
 * NONE 0, LIGHT 1, MODERATE 2, SEVERE 3, CRITICAL 4, EMERGENCY 5, SHUTDOWN 6.
 */
data class GovernorParams(
    val l2Status: Int = 3,
    val l1Status: Int = 2,
    val l2Headroom: Float = 0.85f,
    val l1Headroom: Float = 0.70f,
    val cooldownMs: Long = 30_000L,
    val fpsL0: Int = 30,
    val fpsL1: Int = 24,
    val fpsL2: Int = 15,
    val fpsIdle: Int = 2,
    val l2Banner: String = "Phone is hot — slowed to 15 fps, explanations paused",
)

/**
 * Governor output (spec §12). [changed] = any actionable output (level, fps, voting, VLM mode, banner) differs from the
 * previous sample; [levelChanged] = the level itself changed (these are the changes to log, [reason] carries the inputs).
 */
data class GovernorState(
    val level: PowerLevel,
    val fps: Int,
    val votingEnabled: Boolean,
    val vlmMode: VlmMode,
    val banner: String?,
    val changed: Boolean,
    val reason: String,
    val levelChanged: Boolean = false,
)

/**
 * Thermal governor (spec §12). It changes only fps, borderline voting and VLM use — never the backbone (that would
 * break the Twin fingerprint) and never the HTP performance mode.
 *
 * Target level: `L2` if `status ≥ 3` or `headroom > 0.85`; else `L1` if `status == 2` or `headroom ≥ 0.70`; else `L0`
 * (NaN headroom = unknown = ignored). The level rises immediately and falls only after the target has stayed below it
 * for `cooldownMs` (then straight to the target). `L0` → 30 fps, voting on, VLM AUTO; `L1` → 24 fps, voting off, VLM
 * ON_TAP; `L2` → 15 fps, voting off, VLM PAUSED + banner; `idle` overrides the fps only (→ 2).
 */
class Governor(val params: GovernorParams = GovernorParams()) {

    private var level = PowerLevel.L0
    private var lowSince = NO_TIME
    private var last: GovernorState? = null

    /** The latest state (L0 defaults before the first sample). */
    val state: GovernorState
        get() = last ?: output(PowerLevel.L0, idle = false, changed = false, levelChanged = false, reason = "initial")

    /** Spec §12 target level for one reading. */
    fun targetLevel(status: Int, headroom: Float): PowerLevel {
        val known = !headroom.isNaN()
        return when {
            status >= params.l2Status || (known && headroom > params.l2Headroom) -> PowerLevel.L2
            status >= params.l1Status || (known && headroom >= params.l1Headroom) -> PowerLevel.L1
            else -> PowerLevel.L0
        }
    }

    /**
     * One sample at frame/wall time [tMs] (monotonic ms): thermal [status], [headroom] (NaN = unknown; the caller polls
     * it at most once per 10 s and passes the last value), [idle] = no component for `idleMs` (see [IdleDetector]).
     */
    fun update(tMs: Long, status: Int, headroom: Float, idle: Boolean): GovernorState {
        val target = targetLevel(status, headroom)
        val from = level
        val why = when {
            target > level -> {
                level = target
                lowSince = NO_TIME
                "rise"
            }
            target < level -> {
                if (lowSince == NO_TIME) lowSince = tMs
                val held = tMs - lowSince
                if (held >= params.cooldownMs) {
                    level = target
                    lowSince = NO_TIME
                    "cooled ${held / 1000} s"
                } else {
                    "cooling ${held / 1000}/${params.cooldownMs / 1000} s"
                }
            }
            else -> {
                lowSince = NO_TIME
                "steady"
            }
        }
        val levelChanged = level != from
        val inputs = "status=${statusName(status)}($status) headroom=${if (headroom.isNaN()) "NaN" else headroom} idle=$idle"
        val reason = if (levelChanged) "$from→$level ($why): $inputs" else "$level $why target=$target: $inputs"
        val prev = last
        val candidate = output(level, idle, changed = false, levelChanged = levelChanged, reason = reason)
        val changed = prev == null ||
            prev.level != candidate.level || prev.fps != candidate.fps || prev.votingEnabled != candidate.votingEnabled ||
            prev.vlmMode != candidate.vlmMode || prev.banner != candidate.banner
        val out = candidate.copy(changed = changed)
        last = out
        return out
    }

    /** Back to L0 with no history (e.g. a new session). */
    fun reset() {
        level = PowerLevel.L0
        lowSince = NO_TIME
        last = null
    }

    private fun output(lv: PowerLevel, idle: Boolean, changed: Boolean, levelChanged: Boolean, reason: String): GovernorState {
        val fps = if (idle) params.fpsIdle else when (lv) {
            PowerLevel.L0 -> params.fpsL0
            PowerLevel.L1 -> params.fpsL1
            PowerLevel.L2 -> params.fpsL2
        }
        return GovernorState(
            level = lv,
            fps = fps,
            votingEnabled = lv == PowerLevel.L0,
            vlmMode = when (lv) {
                PowerLevel.L0 -> VlmMode.AUTO
                PowerLevel.L1 -> VlmMode.ON_TAP
                PowerLevel.L2 -> VlmMode.PAUSED
            },
            banner = if (lv == PowerLevel.L2) params.l2Banner else null,
            changed = changed,
            reason = reason,
            levelChanged = levelChanged,
        )
    }

    companion object {
        private const val NO_TIME = Long.MIN_VALUE

        /** Android `PowerManager.THERMAL_STATUS_*` names. */
        fun statusName(status: Int): String = when (status) {
            0 -> "NONE"
            1 -> "LIGHT"
            2 -> "MODERATE"
            3 -> "SEVERE"
            4 -> "CRITICAL"
            5 -> "EMERGENCY"
            6 -> "SHUTDOWN"
            else -> "UNKNOWN"
        }
    }
}

/**
 * Spec §12 `idle` input: true once no component has been seen for [idleMs] (default 3000 ms). Before the first
 * component, idle counts from the first sample.
 */
class IdleDetector(val idleMs: Long = 3_000L) {
    private var lastSeen = Long.MIN_VALUE

    fun update(tMs: Long, hasComponent: Boolean): Boolean {
        if (hasComponent || lastSeen == Long.MIN_VALUE) lastSeen = tMs
        return !hasComponent && tMs - lastSeen >= idleMs
    }

    fun reset() {
        lastSeen = Long.MIN_VALUE
    }
}
