package com.kaizeneye.runtime

import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

/**
 * Deterministic benchmark / self-test inputs, bit-compatible ports of the legacy JavaScript generators
 * (mobile/src/ml/backbone.ts benchmarkImage, mobile/src/ml/knn.ts lcg + fakeFeatures). Pure Kotlin.
 */
internal object BenchInputs {
    /** Fixed synthetic photo-like RGB image (smooth gradients + a bright block + noise), NHWC float32 0..255, [size*size*3]. */
    fun image(size: Int): FloatArray {
        val a = FloatArray(size * size * 3)
        var s = 20240926
        for (y in 0 until size) {
            for (x in 0 until size) {
                s = s * 1664525 + 1013904223                      // == (Math.imul(s, 1664525) + 1013904223) >>> 0 (mod 2^32)
                val n = (s.toLong() and 0xFFFFFFFFL).toDouble() / 4294967296.0 - 0.5
                val i = (y * size + x) * 3
                val edge = if (x > size * 0.3 && x < size * 0.7 && y > size * 0.35 && y < size * 0.65) 60.0 else 0.0
                a[i] = clamp255(120 + 70 * sin(x / 11.0 + y / 29.0) + edge + 25 * n)
                a[i + 1] = clamp255(110 + 60 * cos(x / 23.0 - y / 7.0) + edge * 0.5 + 25 * n)
                a[i + 2] = clamp255(100 + 50 * sin((x * y) / 1500.0) - edge * 0.3 + 25 * n)
            }
        }
        return a
    }

    /** The benchmark image rounded to crop bytes (twin-spec §1.4 rounding: floor(v + 0.5), clamped). */
    fun imageBytes(size: Int): ByteArray {
        val f = image(size)
        return ByteArray(f.size) { i -> floor(f[i] + 0.5).toInt().coerceIn(0, 255).toByte() }
    }

    private fun clamp255(v: Double): Float = v.toFloat().coerceIn(0f, 255f)

    /** Uniform [0, 1) generator: s = s * 1664525 + 1013904223 (mod 2^32), value = s / 2^32. */
    class Lcg(seed: Int) {
        private var s = seed

        fun next(): Double {
            s = s * 1664525 + 1013904223
            return (s.toLong() and 0xFFFFFFFFL).toDouble() / 4294967296.0
        }
    }

    /** Rows that look like backbone features: non-negative, skewed, from 32 clusters ([rows*dim], row-major). */
    fun fakeFeatures(rnd: Lcg, rows: Int, dim: Int): FloatArray {
        val centres = FloatArray(32 * dim)
        for (i in centres.indices) centres[i] = (0.35 * rnd.next() * rnd.next()).toFloat()
        val x = FloatArray(rows * dim)
        for (r in 0 until rows) {
            val c = floor(rnd.next() * 32).toInt() * dim
            for (d in 0 until dim) {
                x[r * dim + d] = (0.8 * centres[c + d] + 0.2 * 0.35 * rnd.next() * rnd.next() * 2).toFloat()
            }
        }
        return x
    }
}
