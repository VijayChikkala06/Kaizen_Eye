package com.kaizeneye.core.math

import com.kaizeneye.core.KaizenCore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

class BinomialTest {

    /** Exact-ish binomial CDF P(X <= r), X ~ Bin(n, p), by summing log-space terms. */
    private fun binomCdf(r: Int, n: Int, p: Double): Double {
        var s = 0.0
        for (i in 0..r) s += exp(lnChoose(n, i) + i * ln(p) + (n - i) * ln(1 - p))
        return s
    }

    private fun lnChoose(n: Int, k: Int): Double {
        var s = 0.0
        for (i in 1..k) s += ln((n - k + i).toDouble()) - ln(i.toDouble())
        return s
    }

    @Test
    fun orderStatisticAlphaMatchesTheResearchTable() {
        assertEquals(0.1391, Binomial.orderStatisticAlpha(20), 5e-4)
        assertEquals(0.0981, Binomial.orderStatisticAlpha(29), 5e-4)
        assertEquals(0.0495, Binomial.orderStatisticAlpha(59), 5e-4)
        assertEquals(0.0100, Binomial.orderStatisticAlpha(299), 2e-4)
        assertEquals(0.95, Binomial.orderStatisticAlpha(1), 1e-15)
        for (m in listOf(1, 20, 29, 59, 299)) {
            assertEquals(KaizenCore.orderStatisticAlpha(m), Binomial.orderStatisticAlpha(m), 0.0)
        }
    }

    @Test
    fun clopperPearsonUpperWithZeroRejectionsIsTheOrderStatisticBound() {
        for (n in listOf(1, 5, 20, 29, 40, 59, 299)) {
            assertEquals(Binomial.orderStatisticAlpha(n), Binomial.clopperPearsonUpper(0, n), 0.0)
            // ...and the bisection path agrees with the closed form.
            assertEquals(Binomial.orderStatisticAlpha(n), Binomial.betaInv(0.95, 1.0, n.toDouble()), 1e-11)
        }
        assertEquals(1.0, Binomial.clopperPearsonUpper(7, 7), 0.0)
        assertEquals(1.0, Binomial.clopperPearsonUpper(0, 0), 0.0)
    }

    @Test
    fun clopperPearsonUpperSatisfiesItsDefinition() {
        for ((r, n) in listOf(1 to 20, 2 to 29, 3 to 40, 5 to 59, 10 to 299, 19 to 20)) {
            val u = Binomial.clopperPearsonUpper(r, n)
            // P(X <= r | p = U) = 1 - conf
            assertEquals("r=$r n=$n", 0.05, binomCdf(r, n, u), 1e-9)
        }
    }

    @Test
    fun twoSidedIntervalSatisfiesItsDefinition() {
        for ((k, n) in listOf(1 to 20, 10 to 20, 19 to 20, 38 to 40, 29 to 29, 0 to 29, 150 to 299)) {
            val iv = Binomial.clopperPearsonInterval(k, n)
            if (k == 0) assertEquals(0.0, iv.lower, 0.0) else assertEquals(0.025, 1 - binomCdf(k - 1, n, iv.lower), 1e-9)
            if (k == n) assertEquals(1.0, iv.upper, 0.0) else assertEquals(0.025, binomCdf(k, n, iv.upper), 1e-9)
            assertTrue(iv.lower < iv.upper)
        }
    }

    @Test
    fun incompleteBetaIsTheBinomialTail() {
        // I_x(a, b) = P(X >= a), X ~ Bin(a + b - 1, x), for integer a, b.
        for ((a, b) in listOf(1 to 1, 1 to 30, 3 to 7, 10 to 10, 25 to 4, 100 to 200)) {
            for (x in listOf(0.001, 0.05, 0.3, 0.5, 0.77, 0.999)) {
                val want = 1 - binomCdf(a - 1, a + b - 1, x)
                assertEquals("I_$x($a,$b)", want, Binomial.regularizedIncompleteBeta(x, a.toDouble(), b.toDouble()), 1e-12)
            }
        }
        assertEquals(0.0, Binomial.regularizedIncompleteBeta(0.0, 2.0, 3.0), 0.0)
        assertEquals(1.0, Binomial.regularizedIncompleteBeta(1.0, 2.0, 3.0), 0.0)
        // Non-integer: I_x(1/2, 1/2) = (2/π) asin(sqrt x)
        for (x in listOf(0.1, 0.4, 0.9)) {
            assertEquals(2 / Math.PI * Math.asin(Math.sqrt(x)), Binomial.regularizedIncompleteBeta(x, 0.5, 0.5), 1e-12)
        }
    }

    @Test
    fun betaInvAndIncompleteBetaRoundTrip() {
        for ((a, b) in listOf(1.0 to 1.0, 2.0 to 5.0, 0.5 to 3.0, 30.0 to 11.0, 1.0 to 299.0, 300.0 to 2.0)) {
            for (p in listOf(1e-6, 0.025, 0.05, 0.5, 0.95, 0.975, 0.999999)) {
                val x = Binomial.betaInv(p, a, b)
                // The spec resolves x to 1e-12 (bisection): the root lies within that bracket.
                val lo = Binomial.regularizedIncompleteBeta(maxOf(0.0, x - 1e-12), a, b)
                val hi = Binomial.regularizedIncompleteBeta(minOf(1.0, x + 1e-12), a, b)
                assertTrue("p=$p a=$a b=$b x=$x", lo <= p + 1e-15 && p <= hi + 1e-15)
                if (x > 1e-6 && x < 1 - 1e-6) assertEquals("p=$p a=$a b=$b", p, Binomial.regularizedIncompleteBeta(x, a, b), 1e-9)
            }
            for (x in listOf(0.01, 0.2, 0.5, 0.8, 0.99)) {
                val p = Binomial.regularizedIncompleteBeta(x, a, b)
                if (p > 1e-12 && p < 1 - 1e-12) assertEquals(x, Binomial.betaInv(p, a, b), 1e-9)
            }
        }
        assertEquals(0.0, Binomial.betaInv(0.0, 2.0, 2.0), 0.0)
        assertEquals(1.0, Binomial.betaInv(1.0, 2.0, 2.0), 0.0)
    }

    @Test
    fun lnGammaMatchesFactorials() {
        var lf = 0.0
        for (n in 1..170) {
            if (n > 1) lf += ln((n - 1).toDouble())
            assertEquals("lnGamma($n)", lf, Binomial.lnGamma(n.toDouble()), 1e-12 * maxOf(1.0, lf))
        }
        assertEquals(0.5 * ln(Math.PI), Binomial.lnGamma(0.5), 1e-14)
        assertEquals(ln(Math.PI.pow(0.5) / 2), Binomial.lnGamma(1.5), 1e-14)
        assertEquals(ln(3.625609908221908), Binomial.lnGamma(0.25), 1e-13)
    }
}
