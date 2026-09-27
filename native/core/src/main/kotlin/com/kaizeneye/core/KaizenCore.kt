package com.kaizeneye.core

import kotlin.math.pow

/** Entry point of the pure-JVM core. Real algorithms are added under this package (patchcore, calibration, tracker...). */
object KaizenCore {
    const val VERSION = "0.1.0-b0"

    /**
     * Distribution-free false-alarm bound for a threshold set as the MAX of [m] normal scores from independent good units:
     * Pr[FPR <= alpha] >= [confidence] when (1 - alpha)^m <= 1 - confidence, i.e. alpha = 1 - (1 - confidence)^(1/m).
     * m = 20 -> 13.9 %, m = 29 -> 9.8 %, m = 59 -> 4.95 %, m = 299 -> 1.0 % at 95 % confidence.
     */
    fun orderStatisticAlpha(m: Int, confidence: Double = 0.95): Double {
        require(m >= 1) { "m must be >= 1" }
        require(confidence > 0.0 && confidence < 1.0) { "confidence must be in (0,1)" }
        return 1.0 - (1.0 - confidence).pow(1.0 / m)
    }
}
