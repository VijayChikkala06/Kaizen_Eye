package com.kaizeneye.core.math

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class HalfTest {

    private fun hex(v: Int) = "0x%04X".format(v)

    private fun enc(f: Float, want: Int) {
        assertEquals("fromFloat($f)", hex(want), hex(Half.fromFloat(f)))
        assertEquals("fromDouble($f)", hex(want), hex(Half.fromDouble(f.toDouble())))
    }

    @Test
    fun knownValues() {
        enc(1.0f, 0x3C00)
        enc(-2.0f, 0xC000)
        enc(0.5f, 0x3800)
        enc(0.0f, 0x0000)
        enc(-0.0f, 0x8000)
        enc(65504.0f, 0x7BFF)                       // max finite
        enc(65519.996f, 0x7BFF)                     // just below the overflow tie
        enc(65520.0f, 0x7C00)                       // tie between 65504 and 65536 → even → +inf
        enc(-65520.0f, 0xFC00)
        enc(1e6f, 0x7C00)
        enc(Float.POSITIVE_INFINITY, 0x7C00)
        enc(Float.NEGATIVE_INFINITY, 0xFC00)
        enc(Math.scalb(1.0f, -24), 0x0001)          // smallest subnormal
        enc(Math.scalb(1.0f, -14), 0x0400)          // smallest normal
        enc(Math.scalb(1023.0f, -24), 0x03FF)       // largest subnormal
    }

    @Test
    fun roundHalfToEvenTies() {
        // 1 + 2^-11 lies exactly between 0x3C00 (even) and 0x3C01: ties to even (down).
        enc(1.0f + Math.scalb(1.0f, -11), 0x3C00)
        // 1 + 3·2^-11 lies between 0x3C01 and 0x3C02 (even): up.
        enc(1.0f + 3 * Math.scalb(1.0f, -11), 0x3C02)
        // Just above / below a tie rounds to the nearest.
        enc(Math.nextUp(1.0f + Math.scalb(1.0f, -11)), 0x3C01)
        enc(Math.nextDown(1.0f + 3 * Math.scalb(1.0f, -11)), 0x3C01)
        // Subnormal ties: 2^-25 (between 0 and 2^-24) → 0; 3·2^-25 (between 2^-24 and 2^-23) → 2^-23.
        enc(Math.scalb(1.0f, -25), 0x0000)
        enc(Math.nextUp(Math.scalb(1.0f, -25)), 0x0001)
        enc(Math.scalb(3.0f, -25), 0x0002)
        enc(Math.scalb(5.0f, -25), 0x0002)          // between 2 and 3 (×2^-24) → even 2
        enc(Math.scalb(1.0f, -26), 0x0000)          // underflow to zero
        enc(-Math.scalb(1.0f, -26), 0x8000)         // to negative zero
        // Rounding up into the next binade / from the largest subnormal to the smallest normal.
        enc(Math.scalb(2047.0f, -24) / 2f, 0x0400)  // 1023.5 × 2^-24 → tie → even 1024 = smallest normal
    }

    @Test
    fun nanIsPreserved() {
        assertEquals(hex(0x7E00), hex(Half.fromFloat(Float.fromBits(0x7FC00000))))
        assertEquals(0x7FC00000, Half.toFloat(0x7E00).toRawBits())
        // A NaN whose payload lives only in low bits must stay a NaN (not become inf).
        val h = Half.fromFloat(Float.fromBits(0x7F800001))
        assertTrue((h and 0x7C00) == 0x7C00 && (h and 0x3FF) != 0)
        assertTrue(Half.toFloat(h).isNaN())
        assertTrue(Half.toFloat(Half.fromDouble(Double.NaN)).isNaN())
    }

    @Test
    fun everyHalfRoundTripsThroughFloatAndDouble() {
        for (h in 0 until 65536) {
            val f = Half.toFloat(h)
            if (f.isNaN()) {
                assertTrue(Half.toFloat(Half.fromFloat(f)).isNaN())
                continue
            }
            assertEquals(hex(h), hex(Half.fromFloat(f)))
            assertEquals(hex(h), hex(Half.fromDouble(f.toDouble())))
            assertEquals(f.toDouble(), Half.toDouble(h), 0.0)
        }
    }

    /** Midpoints between consecutive halves go to the even neighbour; one float ulp away goes to the nearer one. */
    @Test
    fun allMidpointsRoundToEven() {
        for (h in 0 until 0x7BFF) {                  // positive finite pairs (h, h+1)
            val a = Half.toDouble(h)
            val b = Half.toDouble(h + 1)
            val mid = ((a + b) / 2).toFloat()
            assertEquals(mid.toDouble(), (a + b) / 2, 0.0)   // midpoints are exact float32 values
            val even = if (h % 2 == 0) h else h + 1
            assertEquals("mid of ${hex(h)}", hex(even), hex(Half.fromFloat(mid)))
            assertEquals(hex(even), hex(Half.fromDouble((a + b) / 2)))
            assertEquals(hex(h), hex(Half.fromFloat(Math.nextDown(mid))))
            assertEquals(hex(h + 1), hex(Half.fromFloat(Math.nextUp(mid))))
            assertEquals(hex(h or 0x8000), hex(Half.fromFloat(-Math.nextDown(mid))))
        }
    }

    @Test
    fun floatPathEqualsDoublePathOnRandomBits() {
        val rnd = Random(7)
        repeat(2_000_000) {
            val f = Float.fromBits(rnd.nextInt())
            if (f.isNaN()) return@repeat
            val a = Half.fromFloat(f)
            val b = Half.fromDouble(f.toDouble())
            if (a != b) throw AssertionError("bits ${Integer.toHexString(f.toRawBits())}: float ${hex(a)} double ${hex(b)}")
        }
        // Dense sweep around the subnormal/normal and overflow boundaries.
        for (base in intArrayOf(0x33000000, 0x38000000, 0x387FC000, 0x477FE000)) {
            for (d in -20000..20000) {
                val f = Float.fromBits(base + d)
                assertEquals(hex(Half.fromDouble(f.toDouble())), hex(Half.fromFloat(f)))
            }
        }
    }

    @Test
    fun littleEndianEncoding() {
        val v = floatArrayOf(1.0f, -2.0f, 65504f, Math.scalb(1.0f, -24))
        val bytes = Half.encodeLE(v)
        assertArrayEquals(
            byteArrayOf(0x00, 0x3C, 0x00, 0xC0.toByte(), 0xFF.toByte(), 0x7B, 0x01, 0x00),
            bytes,
        )
        assertArrayEquals(v, Half.decodeLE(bytes), 0f)
    }

    @Test
    fun roundingHelpers() {
        val x = floatArrayOf(0.1f, 1.0f / 3, 123.456f)
        val r = Half.roundedCopy(x)
        for (i in x.indices) {
            assertEquals(Half.toFloat(Half.fromFloat(x[i])), r[i], 0f)
            assertEquals(r[i], Half.round(r[i]), 0f)           // idempotent
        }
        assertEquals(Half.toFloat(Half.fromDouble(0.1)), Half.round(0.1), 0f)
    }
}
