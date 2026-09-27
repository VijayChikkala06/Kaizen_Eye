package com.kaizeneye.runtime

import android.app.ApplicationExitInfo

/*
 * Crash-guard policy (pure; the file I/O lives in DecisionStore / VlmService).
 *
 * Before an accelerator candidate is tried, a marker {key, candidate, phase, pid, token} is written (filesDir/accel/trying.json,
 * filesDir/vlm/trying.json). Outcomes:
 *  - the ":probe" process dies while running the candidate (binderDied / onServiceDisconnected) or does not answer within
 *    60 s: the candidate gets strikes according to why the process died (below); a timeout always counts as a crash;
 *  - the process that wrote the marker dies before clearing it ("stale" marker, noticed on the next start): phase "main"
 *    (the app itself was creating/warming the accelerated model) or "vlm" (engine init) is judged by that process's exit
 *    reason; phase "probe" means the app died while the separate probe ran, which does not implicate the candidate unless
 *    nothing is known about the death (e.g. the phone rebooted) - one strike, so a reboot loop stops after two tries.
 * A candidate with CRASHED (2) strikes is skipped for that cache key.
 * Death reasons come from ActivityManager.getHistoricalProcessExitReasons (API 30+).
 */
internal object CrashGuard {
    const val CRASHED = 2

    enum class Death { CRASH, EXTERNAL, UNKNOWN }

    object Phase {
        const val PROBE = "probe"
        const val MAIN = "main"
        const val VLM = "vlm"
    }

    /** [reason] = ApplicationExitInfo.getReason() of the dead process, null when no record was found. */
    fun classify(reason: Int?): Death = when (reason) {
        ApplicationExitInfo.REASON_CRASH,
        ApplicationExitInfo.REASON_CRASH_NATIVE,
        ApplicationExitInfo.REASON_ANR,
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE,
        -> Death.CRASH

        ApplicationExitInfo.REASON_LOW_MEMORY,
        ApplicationExitInfo.REASON_USER_REQUESTED,
        ApplicationExitInfo.REASON_USER_STOPPED,
        ApplicationExitInfo.REASON_PERMISSION_CHANGE,
        ApplicationExitInfo.REASON_DEPENDENCY_DIED,
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
        ApplicationExitInfo.REASON_FREEZER,
        ApplicationExitInfo.REASON_PACKAGE_STATE_CHANGE,
        ApplicationExitInfo.REASON_PACKAGE_UPDATED,
        ApplicationExitInfo.REASON_EXIT_SELF,
        -> Death.EXTERNAL

        else -> Death.UNKNOWN   // REASON_UNKNOWN, REASON_SIGNALED, REASON_OTHER, no record
    }

    fun reasonName(reason: Int?): String = when (reason) {
        null -> "no exit record"
        ApplicationExitInfo.REASON_EXIT_SELF -> "exit self"
        ApplicationExitInfo.REASON_SIGNALED -> "signaled"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "low memory"
        ApplicationExitInfo.REASON_CRASH -> "crash (java)"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "native crash"
        ApplicationExitInfo.REASON_ANR -> "ANR"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "initialization failure"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "killed by user"
        ApplicationExitInfo.REASON_USER_STOPPED -> "stopped by user"
        else -> "reason $reason"
    }

    /** Strikes after the probe process died while running a candidate (observed live). */
    fun strikesAfterProbeDeath(current: Int, death: Death, timedOut: Boolean): Int = when {
        timedOut || death == Death.CRASH || death == Death.UNKNOWN -> CRASHED
        else -> minOf(CRASHED, current + 1)                           // killed by the system/user: retry once more
    }

    /** Strikes after a stale marker of [phase] was found; [death] = how the process that wrote it died. */
    fun strikesAfterStaleMarker(current: Int, phase: String, death: Death): Int = when (phase) {
        Phase.PROBE -> if (death == Death.UNKNOWN) minOf(CRASHED, current + 1) else current
        else -> when (death) {                                         // MAIN / VLM: that process was running the candidate
            Death.CRASH -> CRASHED
            Death.UNKNOWN -> minOf(CRASHED, current + 1)
            Death.EXTERNAL -> current
        }
    }

    fun isCrashed(strikes: Int?): Boolean = (strikes ?: 0) >= CRASHED
}

/** The persisted crash-guard marker. [token] identifies the writing process instance (pids are reused). */
internal data class TryingMarker(
    val key: String,
    val candidate: String,
    val phase: String,
    val pid: Int,
    val token: String,
    val atMs: Long,
)

internal object CacheKeys {
    /** Bump when candidate options (threads, precision, HTP mode) or the selection rules change: invalidates every decision. */
    const val DECISION_VERSION = 1
    /** QNN runtime the app bundles (plan: qnn-runtime 2.49.0); part of every decision key. */
    const val QNN_VERSION = "qnn2.49.0"

    /**
     * model name + variant SHA-256s + app versionCode + Build.FINGERPRINT + QNN version + DECISION_VERSION, plus
     * [nativeStamp] (sizes of the bundled dispatch/QNN libraries, so swapping those libraries re-benchmarks).
     */
    fun decisionKey(
        model: String,
        variants: List<Pair<String, String>>,
        versionCode: Long,
        fingerprint: String,
        nativeStamp: String,
        qnn: String = QNN_VERSION,
    ): String {
        val v = variants.joinToString("+") { (id, sha) -> "$id:${sha.lowercase()}" }
        return "$model#$v@v$versionCode|$fingerprint|$qnn|$nativeStamp|d$DECISION_VERSION"
    }
}
