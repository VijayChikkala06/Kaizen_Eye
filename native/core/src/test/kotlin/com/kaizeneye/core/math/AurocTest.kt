package com.kaizeneye.core.math

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class AurocTest {

    private fun brute(pos: DoubleArray, neg: DoubleArray): Double {
        var u = 0.0
        for (p in pos) for (n in neg) u += if (p > n) 1.0 else if (p == n) 0.5 else 0.0
        return u / (pos.size * neg.size)
    }

    @Test
    fun handCases() {
        assertEquals(1.0, Auroc.auroc(doubleArrayOf(3.0, 4.0), doubleArrayOf(1.0, 2.0)), 0.0)
        assertEquals(0.0, Auroc.auroc(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)), 0.0)
        assertEquals(0.5, Auroc.auroc(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0)), 0.0)
        // pos {1, 3}, neg {2, 3}: (1>2 no, 1>3 no, 3>2 yes, 3=3 half) = 1.5 / 4
        assertEquals(0.375, Auroc.auroc(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 3.0)), 0.0)
        assertTrue(Auroc.auroc(DoubleArray(0), doubleArrayOf(1.0)).isNaN())
    }

    @Test
    fun equalsBruteForceWithTies() {
        val rnd = Random(3)
        repeat(200) {
            val pos = DoubleArray(1 + rnd.nextInt(40)) { rnd.nextInt(12).toDouble() }
            val neg = DoubleArray(1 + rnd.nextInt(40)) { rnd.nextInt(12).toDouble() / 1.5 }
            assertEquals(brute(pos, neg), Auroc.auroc(pos, neg), 1e-15)
        }
    }
}
