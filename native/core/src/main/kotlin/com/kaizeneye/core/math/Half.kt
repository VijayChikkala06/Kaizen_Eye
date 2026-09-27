package com.kaizeneye.core.math

/**
 * IEEE-754 binary16 ("half", f16) conversions (spec §0): round-to-nearest-even exactly like numpy `astype(float16)`,
 * including subnormals, ±inf on overflow and NaN preserved (the top 10 significand bits are kept, never turned into inf).
 * Stored little-endian. Half values are passed around as the low 16 bits of an [Int].
 *
 * [fromFloat] is the fast bit-level path (numpy's `npy_floatbits_to_halfbits`); [fromDouble] is an independent
 * arithmetic path used for float64 values (single rounding, no detour through float32). Both are cross-checked by tests.
 */
object Half {
    const val POSITIVE_INFINITY: Int = 0x7C00
    const val NEGATIVE_INFINITY: Int = 0xFC00
    /** Bits of the largest finite half, 65504. */
    const val MAX_VALUE_BITS: Int = 0x7BFF
    const val MAX_VALUE: Float = 65504.0f
    /** Bits of the smallest positive subnormal half, 2^-24. */
    const val MIN_SUBNORMAL_BITS: Int = 0x0001

    private val TWO_POW_M24: Float = Math.scalb(1.0f, -24)

    /** Decoding table: every half pattern → its exact float32 value. */
    private val TABLE: FloatArray = FloatArray(65536) { decode(it) }

    /** Rounds a float32 to binary16 (ties to even) and returns its bits. */
    fun fromFloat(f: Float): Int {
        val x = f.toRawBits()
        val hSgn = (x ushr 16) and 0x8000
        val fExp = x and 0x7F800000
        if (fExp >= 0x47800000) {                     // |f| >= 65536, inf or NaN
            val fSig = x and 0x007FFFFF
            if (fExp == 0x7F800000 && fSig != 0) {    // NaN: keep the top payload bits, make sure it stays a NaN
                val p = fSig ushr 13
                return hSgn or 0x7C00 or (if (p == 0) 1 else p)
            }
            return hSgn or 0x7C00                     // inf, or overflow to inf
        }
        if (fExp <= 0x38000000) {                     // |f| < 2^-14: half subnormal or zero
            if (fExp < 0x33000000) return hSgn        // |f| < 2^-25: rounds to (signed) zero
            val e = fExp ushr 23
            var fSig = 0x00800000 + (x and 0x007FFFFF)
            fSig = fSig ushr (113 - e)
            // Round half to even: add half an ulp unless it is an exact tie with an even last bit (lost bits count too).
            if ((fSig and 0x3FFF) != 0x1000 || (x and 0x7FF) != 0) fSig += 0x1000
            return hSgn + (fSig ushr 13)              // a carry into the exponent gives the smallest normal: correct
        }
        val hExp = (fExp - 0x38000000) ushr 13
        var fSig = x and 0x007FFFFF
        if ((fSig and 0x3FFF) != 0x1000) fSig += 0x1000
        return hSgn + hExp + (fSig ushr 13)           // a carry may reach 0x7C00 = inf: correct
    }

    /** Rounds a float64 directly to binary16 (one rounding, ties to even) and returns its bits. */
    fun fromDouble(v: Double): Int {
        val raw = v.toRawBits()
        val sign = ((raw ushr 48) and 0x8000L).toInt()
        if (v.isNaN()) {
            val p = ((raw ushr 42) and 0x3FFL).toInt()
            return sign or 0x7C00 or (if (p == 0) 1 else p)
        }
        val a = Math.abs(v)
        // 65520 is the midpoint between 65504 (odd significand) and 65536 (overflow): ties to even → inf.
        if (a >= 65520.0) return sign or 0x7C00
        if (a < 6.103515625E-5) {                     // < 2^-14: subnormal; a * 2^24 is exact, rint = ties to even
            return sign or Math.rint(a * 16777216.0).toInt()
        }
        var e = Math.getExponent(a)                   // -14..15
        var r = Math.rint((Math.scalb(a, -e) - 1.0) * 1024.0).toInt()   // exact scaling, ties to even
        if (r == 1024) {
            r = 0
            e += 1
        }
        if (e > 15) return sign or 0x7C00
        return sign or ((e + 15) shl 10) or r
    }

    /** Exact float32 value of half bits [h] (low 16 bits used). */
    fun toFloat(h: Int): Float = TABLE[h and 0xFFFF]

    fun toDouble(h: Int): Double = TABLE[h and 0xFFFF].toDouble()

    /** [v] rounded through binary16 (spec §7 step 6: "all stored feature rows are rounded through binary16"). */
    fun round(v: Float): Float = TABLE[fromFloat(v)]

    /** float64 [v] rounded through binary16 (single rounding), as float32 (exact: every half is a float32). */
    fun round(v: Double): Float = TABLE[fromDouble(v)]

    /** Rounds every element of [a] through binary16 in place; returns [a]. */
    fun roundInPlace(a: FloatArray): FloatArray {
        for (i in a.indices) a[i] = TABLE[fromFloat(a[i])]
        return a
    }

    /** New float32 array with every element of [a] rounded through binary16. */
    fun roundedCopy(a: FloatArray): FloatArray = roundInPlace(a.copyOf())

    /** float64 values rounded through binary16 into a new float32 array. */
    fun roundedCopy(a: DoubleArray): FloatArray = FloatArray(a.size) { TABLE[fromDouble(a[it])] }

    /** Encodes [count] values of [src] from [srcOff] as little-endian halves into [dst] at [dstOff] (2 bytes each). */
    fun encodeLE(src: FloatArray, srcOff: Int, count: Int, dst: ByteArray, dstOff: Int) {
        var o = dstOff
        for (i in srcOff until srcOff + count) {
            val h = fromFloat(src[i])
            dst[o] = h.toByte()
            dst[o + 1] = (h ushr 8).toByte()
            o += 2
        }
    }

    /** Decodes [count] little-endian halves from [src] at [srcOff] into [dst] at [dstOff]. */
    fun decodeLE(src: ByteArray, srcOff: Int, count: Int, dst: FloatArray, dstOff: Int) {
        var o = srcOff
        for (i in dstOff until dstOff + count) {
            val h = (src[o].toInt() and 0xFF) or ((src[o + 1].toInt() and 0xFF) shl 8)
            dst[i] = TABLE[h]
            o += 2
        }
    }

    /** Little-endian bytes of the halves of [values] (convenience for small arrays). */
    fun encodeLE(values: FloatArray): ByteArray =
        ByteArray(values.size * 2).also { encodeLE(values, 0, values.size, it, 0) }

    fun decodeLE(bytes: ByteArray): FloatArray {
        require(bytes.size % 2 == 0) { "odd byte count ${bytes.size}" }
        return FloatArray(bytes.size / 2).also { decodeLE(bytes, 0, it.size, it, 0) }
    }

    private fun decode(h: Int): Float {
        val s = (h and 0x8000) shl 16
        val e = (h ushr 10) and 0x1F
        val m = h and 0x3FF
        val bits = when (e) {
            0 -> if (m == 0) 0 else (m.toFloat() * TWO_POW_M24).toRawBits()
            31 -> 0x7F800000 or (m shl 13)
            else -> ((e + 112) shl 23) or (m shl 13)
        }
        return Float.fromBits(s or bits)
    }
}
