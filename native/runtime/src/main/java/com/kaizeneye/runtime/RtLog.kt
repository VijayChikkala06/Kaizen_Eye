package com.kaizeneye.runtime

import android.util.Log

/** Logcat tag "KaizenRuntime" (`adb logcat -s KaizenRuntime`). Never throws (also usable from JVM unit tests). */
internal object RtLog {
    const val TAG = "KaizenRuntime"

    fun i(msg: String) {
        try {
            Log.i(TAG, msg)
        } catch (_: Throwable) {
            println("I/$TAG: $msg")
        }
    }

    fun w(msg: String, t: Throwable? = null) {
        try {
            if (t != null) Log.w(TAG, msg, t) else Log.w(TAG, msg)
        } catch (_: Throwable) {
            println("W/$TAG: $msg ${t?.brief() ?: ""}")
        }
    }

    fun e(msg: String, t: Throwable? = null) {
        try {
            if (t != null) Log.e(TAG, msg, t) else Log.e(TAG, msg)
        } catch (_: Throwable) {
            println("E/$TAG: $msg ${t?.brief() ?: ""}")
        }
    }
}

/** Short structured error text for reports: "ExceptionClass: message" (cause appended), at most 400 chars. */
internal fun Throwable.brief(): String {
    val own = "${javaClass.simpleName}: ${message ?: ""}".trim()
    val c = cause
    val full = if (c != null && c !== this) "$own (cause ${c.javaClass.simpleName}: ${c.message ?: ""})" else own
    return if (full.length > 400) full.substring(0, 400) + "…" else full
}

/** Monotonic clock in ms (double precision) for latency measurements. */
internal fun nowMs(): Double = System.nanoTime() / 1e6
