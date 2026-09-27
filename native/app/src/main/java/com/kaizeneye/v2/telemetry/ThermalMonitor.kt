package com.kaizeneye.v2.telemetry

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import kotlin.math.abs

/**
 * One thermal/power reading. [headroom] is PowerManager.getThermalHeadroom(10) (NaN when unavailable). The platform throttles
 * callers that poll it more than about once per 10 s, so it is re-polled at most that often ([headroomAgeMs] says how old).
 * Whole-phone power = |current| × voltage; the SIGN convention of BATTERY_PROPERTY_CURRENT_NOW differs between vendors and
 * must be checked on the demo phone before any watt figure is quoted (plan B7) — never an "NPU watts" claim.
 */
data class ThermalSample(
    val elapsedMs: Long,
    val status: Int,
    val headroom: Float,
    val headroomAgeMs: Long,
    val batteryTempC: Double?,
    val currentUa: Int?,
    val voltageMv: Int?,
    val watts: Double?,
    val charging: Boolean,
    val capacityPct: Int?,
) {
    val statusName: String get() = statusName(status)

    companion object {
        fun statusName(s: Int): String = when (s) {
            PowerManager.THERMAL_STATUS_NONE -> "NONE"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
            else -> "?($s)"
        }
    }
}

class ThermalMonitor(private val context: Context) {
    private val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private var lastHeadroom = Float.NaN
    private var lastHeadroomAt = Long.MIN_VALUE
    private val _latest = MutableStateFlow(read(Float.NaN, 0L))
    val latest: StateFlow<ThermalSample> = _latest
    private var job: Job? = null
    private val listener = PowerManager.OnThermalStatusChangedListener { refresh() }
    private val direct = Executor { it.run() }

    fun start(scope: CoroutineScope, periodMs: Long = 2000L) {
        if (job != null) return
        try {
            pm.addThermalStatusListener(direct, listener)
        } catch (t: Throwable) {
            Log.w(TAG, "thermal listener unavailable", t)
        }
        job = scope.launch {
            while (isActive) {
                refresh()
                delay(periodMs)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
        try {
            pm.removeThermalStatusListener(listener)
        } catch (_: Throwable) {
        }
    }

    @Synchronized
    fun refresh(): ThermalSample {
        val now = SystemClock.elapsedRealtime()
        if (lastHeadroomAt == Long.MIN_VALUE || now - lastHeadroomAt >= HEADROOM_MIN_INTERVAL_MS) {
            lastHeadroom = try {
                pm.getThermalHeadroom(10)
            } catch (t: Throwable) {
                Float.NaN
            }
            lastHeadroomAt = now
        }
        val s = read(lastHeadroom, now - lastHeadroomAt)
        _latest.value = s
        return s
    }

    private fun read(headroom: Float, age: Long): ThermalSample {
        val i: Intent? = try {
            context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        } catch (t: Throwable) {
            null
        }
        val temp = i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)?.takeIf { it != Int.MIN_VALUE }?.let { it / 10.0 }
        val volt = i?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, Int.MIN_VALUE)?.takeIf { it != Int.MIN_VALUE }
        val plugged = (i?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val cur = try {
            bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW).takeIf { it != Int.MIN_VALUE }
        } catch (t: Throwable) {
            null
        }
        val cap = try {
            bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY).takeIf { it != Int.MIN_VALUE }
        } catch (t: Throwable) {
            null
        }
        val watts = if (cur != null && volt != null) abs(cur.toDouble()) * volt / 1e9 else null
        return ThermalSample(
            elapsedMs = SystemClock.elapsedRealtime(),
            status = pm.currentThermalStatus,
            headroom = headroom,
            headroomAgeMs = age,
            batteryTempC = temp,
            currentUa = cur,
            voltageMv = volt,
            watts = watts,
            charging = plugged,
            capacityPct = cap,
        )
    }

    companion object {
        private const val TAG = "KaizenThermal"
        const val HEADROOM_MIN_INTERVAL_MS = 10_000L
    }
}
