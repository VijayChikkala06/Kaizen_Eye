package com.kaizeneye.core.telemetry

import kotlin.math.abs

/** Whole-phone electrical power from one battery reading: absolute [watts] and whether the battery is [charging]. */
data class PowerSample(val watts: Double, val charging: Boolean)

/**
 * Whole-phone power from `BatteryManager` readings (telemetry only — never an "NPU watts" claim).
 *
 * **Units and sign must be checked on the actual phone before any number is reported:**
 * - `BATTERY_PROPERTY_CURRENT_NOW` is documented in microamperes, but some OEM kernels report milliamperes (a phone
 *   drawing ~1 W at 3.9 V reads ≈ 250 000 µA; a reading of ≈ 250 means mA — scale it by 1000 before calling);
 * - its sign convention varies: most devices report **negative** while discharging, some positive. Unplug the phone,
 *   keep the screen on, read a few samples and pass that sign as `dischargeSign` (−1 or +1);
 * - the voltage (`ACTION_BATTERY_CHANGED` / `EXTRA_VOLTAGE`) is in millivolts on most devices (≈ 3500–4500); a
 *   reading below 100 means volts.
 */
object PowerMath {

    /**
     * `|currentNowUa| · 1e-6 A · voltageMv · 1e-3 V` in watts; `charging` = current flowing against [dischargeSign]
     * (a zero current is "not charging").
     */
    fun wattsFrom(currentNowUa: Int, voltageMv: Int, dischargeSign: Int): PowerSample {
        require(dischargeSign == 1 || dischargeSign == -1) { "dischargeSign must be +1 or -1, got $dischargeSign" }
        val watts = abs(currentNowUa.toDouble()) * 1e-6 * (voltageMv.toDouble() * 1e-3)
        val charging = currentNowUa != 0 && Integer.signum(currentNowUa) != dischargeSign
        return PowerSample(watts, charging)
    }
}
