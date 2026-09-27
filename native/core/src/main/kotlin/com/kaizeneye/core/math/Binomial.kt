package com.kaizeneye.core.math

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

/**
 * Distribution-free bounds of the calibration certificate (spec §10.2): order-statistic α, Clopper–Pearson bounds,
 * the regularised incomplete beta function and its inverse. All float64.
 */
object Binomial {

    /** Bisection stops when the bracket is narrower than this (spec §10.2: `|Δx| < 1e-12`). */
    const val BETA_INV_TOLERANCE: Double = 1e-12

    /**
     * Order-statistic false-alarm bound of a threshold set as the max of [m] valid normal scores (spec §10.2):
     * `α = 1 − (1 − conf)^(1/m)`. m = 20 → 13.9 %, 29 → 9.8 %, 59 → 4.95 %, 299 → 1.0 % at 95 %.
     */
    fun orderStatisticAlpha(m: Int, conf: Double = 0.95): Double {
        require(m >= 1) { "m must be >= 1, got $m" }
        requireConf(conf)
        return 1.0 - (1.0 - conf).pow(1.0 / m)
    }

    /**
     * One-sided Clopper–Pearson upper bound on a rate with [r] events (rejections) out of [n] (spec §10.2):
     * `BetaInv(conf; r+1, n−r)` for `r < n`, `1.0` for `r = n`; for `r = 0` the closed form `1 − (1−conf)^(1/n)`.
     */
    fun clopperPearsonUpper(r: Int, n: Int, conf: Double = 0.95): Double {
        require(n >= 0 && r >= 0 && r <= n) { "need 0 <= r <= n, got r=$r n=$n" }
        requireConf(conf)
        if (r == n) return 1.0
        if (r == 0) return 1.0 - (1.0 - conf).pow(1.0 / n)
        return betaInv(conf, (r + 1).toDouble(), (n - r).toDouble())
    }

    /** Two-sided interval (spec §10.2, G3 reporting). */
    data class Interval(val lower: Double, val upper: Double)

    /**
     * Two-sided [conf] Clopper–Pearson interval for [k] successes out of [n] (spec §10.2):
     * `lower = BetaInv((1−conf)/2; k, n−k+1)` (0 if k = 0), `upper = BetaInv((1+conf)/2; k+1, n−k)` (1 if k = n).
     */
    fun clopperPearsonInterval(k: Int, n: Int, conf: Double = 0.95): Interval {
        require(n >= 0 && k >= 0 && k <= n) { "need 0 <= k <= n, got k=$k n=$n" }
        requireConf(conf)
        val lower = if (k == 0) 0.0 else betaInv((1.0 - conf) / 2.0, k.toDouble(), (n - k + 1).toDouble())
        val upper = if (k == n) 1.0 else betaInv((1.0 + conf) / 2.0, (k + 1).toDouble(), (n - k).toDouble())
        return Interval(lower, upper)
    }

    /**
     * `BetaInv(p; a, b)`: the x in [0, 1] with `I_x(a, b) = p`, by bisection until the bracket is < 1e-12 (spec §10.2).
     * Returns the bracket midpoint; 0 for p <= 0 and 1 for p >= 1.
     */
    fun betaInv(p: Double, a: Double, b: Double): Double {
        require(a > 0 && b > 0) { "a and b must be > 0" }
        require(!p.isNaN()) { "p is NaN" }
        if (p <= 0.0) return 0.0
        if (p >= 1.0) return 1.0
        var lo = 0.0
        var hi = 1.0
        while (hi - lo >= BETA_INV_TOLERANCE) {
            val mid = 0.5 * (lo + hi)
            if (mid <= lo || mid >= hi) break
            if (regularizedIncompleteBeta(mid, a, b) < p) lo = mid else hi = mid
        }
        return 0.5 * (lo + hi)
    }

    /**
     * Regularised incomplete beta `I_x(a, b)` (spec §10.2): Numerical Recipes `betai` with the Lentz continued fraction
     * `betacf`, using `I_x(a,b) = 1 − I_{1−x}(b,a)` when `x > (a+1)/(a+b+2)`.
     */
    fun regularizedIncompleteBeta(x: Double, a: Double, b: Double): Double {
        require(a > 0 && b > 0) { "a and b must be > 0" }
        require(!x.isNaN()) { "x is NaN" }
        if (x <= 0.0) return 0.0
        if (x >= 1.0) return 1.0
        val lnBt = lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * ln(x) + b * ln(1.0 - x)
        val bt = exp(lnBt)
        return if (x > (a + 1.0) / (a + b + 2.0)) {
            1.0 - bt * betaContinuedFraction(b, a, 1.0 - x) / b
        } else {
            bt * betaContinuedFraction(a, b, x) / a
        }
    }

    private const val CF_EPS = 1e-15
    private const val CF_FPMIN = 1e-300
    private const val CF_MAX_ITER = 100_000

    /** Continued fraction for the incomplete beta (Numerical Recipes `betacf`, modified Lentz). */
    private fun betaContinuedFraction(a: Double, b: Double, x: Double): Double {
        val qab = a + b
        val qap = a + 1.0
        val qam = a - 1.0
        var c = 1.0
        var d = 1.0 - qab * x / qap
        if (abs(d) < CF_FPMIN) d = CF_FPMIN
        d = 1.0 / d
        var h = d
        for (m in 1..CF_MAX_ITER) {
            val m2 = 2.0 * m
            var aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if (abs(d) < CF_FPMIN) d = CF_FPMIN
            c = 1.0 + aa / c
            if (abs(c) < CF_FPMIN) c = CF_FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if (abs(d) < CF_FPMIN) d = CF_FPMIN
            c = 1.0 + aa / c
            if (abs(c) < CF_FPMIN) c = CF_FPMIN
            d = 1.0 / d
            val del = d * c
            h *= del
            if (abs(del - 1.0) <= CF_EPS) return h
        }
        return h
    }

    private val HALF_LN_2PI = 0.5 * ln(2.0 * Math.PI)

    /** Arguments below this are shifted up by the recurrence before the Stirling series is used. */
    private const val STIRLING_MIN = 15.0

    /**
     * ln Γ(x) for x > 0: the Stirling series with Bernoulli terms up to B16 for x >= 15 (truncation < 1e-20), and the
     * recurrence `ln Γ(x) = ln Γ(x + n) − ln(x (x+1) … (x+n−1))` below. Absolute error ~1e-14 for moderate x.
     */
    fun lnGamma(x: Double): Double {
        require(x > 0) { "lnGamma needs x > 0, got $x" }
        var z = x
        var prod = 1.0
        while (z < STIRLING_MIN) {
            prod *= z
            z += 1.0
        }
        val inv = 1.0 / z
        val inv2 = inv * inv
        // B2/(1·2)x^-1 + B4/(3·4)x^-3 + ... + B16/(15·16)x^-15
        val series = inv * (1.0 / 12 + inv2 * (-1.0 / 360 + inv2 * (1.0 / 1260 + inv2 * (-1.0 / 1680 +
            inv2 * (1.0 / 1188 + inv2 * (-691.0 / 360360 + inv2 * (1.0 / 156 + inv2 * (-3617.0 / 122400))))))))
        val lg = (z - 0.5) * ln(z) - z + HALF_LN_2PI + series
        return if (prod == 1.0) lg else lg - ln(prod)
    }

    private fun requireConf(conf: Double) = require(conf > 0.0 && conf < 1.0) { "conf must be in (0, 1), got $conf" }
}
