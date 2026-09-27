package com.kaizeneye.v2.ml

import com.kaizeneye.core.twin.Knn
import com.kaizeneye.runtime.Accel
import com.kaizeneye.runtime.AcceleratorReport
import com.kaizeneye.runtime.BackboneSpec
import com.kaizeneye.runtime.FeatureBackbone
import com.kaizeneye.runtime.KnnEngine
import com.kaizeneye.runtime.ModelAsset
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * DEBUG / EMULATOR ONLY — never used for any claim. A pure-Kotlin "backbone" producing locally aware colour + texture
 * statistics on the same 40×40 patch grid as ResNet18@320, so the whole app (teach, live line, replay, calibration,
 * export, self-test mechanics) can run where LiteRT cannot (the x86_64 emulator's ARM translation crashes inside XNNPACK).
 * Features per cell (over its 3×3-cell neighbourhood): mean RGB, std RGB, 8-bin gradient-orientation histogram, Laplacian
 * energy → 15 numbers, expanded to 128 dims by a fixed seeded random projection (distances are roughly preserved).
 */
class DebugKotlinBackbone(override val spec: BackboneSpec) : FeatureBackbone {
    override val modelId: String = MODEL_ID
    override val sha256: String = "0".repeat(64)
    override val report = AcceleratorReport(
        model = spec.name, chosen = "CPU (Kotlin debug)", accel = Accel.CPU, modelId = MODEL_ID,
        badge = "DEBUG Kotlin features (no LiteRT)", speedupVsCpu = null, results = emptyList(), fromCache = false,
        measuredAtMs = System.currentTimeMillis(), cacheKey = "debug-kotlin",
    )
    private val proj: FloatArray = FloatArray(spec.dim * RAW).also { w ->
        var s = 0x2545F4914F6CDD1DL
        for (i in w.indices) {
            s = s xor (s shl 13); s = s xor (s ushr 7); s = s xor (s shl 17)
            w[i] = ((s ushr 11).toDouble() / (1L shl 53).toDouble() * 2 - 1).toFloat()
        }
    }

    @Synchronized
    override fun embed(rgb: ByteArray): FloatArray {
        val n = spec.inputSize
        val gh = spec.gh
        val gw = spec.gw
        val cell = n / gw
        val grey = FloatArray(n * n) { i -> 0.299f * (rgb[i * 3].toInt() and 0xFF) + 0.587f * (rgb[i * 3 + 1].toInt() and 0xFF) + 0.114f * (rgb[i * 3 + 2].toInt() and 0xFF) }
        val out = FloatArray(gh * gw * spec.dim)
        val raw = FloatArray(RAW)
        for (r in 0 until gh) for (c in 0 until gw) {
            raw.fill(0f)
            val y0 = maxOf(0, (r - 1) * cell)
            val y1 = minOf(n, (r + 2) * cell)
            val x0 = maxOf(0, (c - 1) * cell)
            val x1 = minOf(n, (c + 2) * cell)
            var cnt = 0
            val sum = DoubleArray(3)
            val sq = DoubleArray(3)
            var lap = 0.0
            for (y in y0 until y1) for (x in x0 until x1) {
                val i = (y * n + x) * 3
                for (ch in 0..2) {
                    val v = (rgb[i + ch].toInt() and 0xFF) / 255.0
                    sum[ch] += v; sq[ch] += v * v
                }
                cnt++
                if (x in 1 until n - 1 && y in 1 until n - 1) {
                    val g = grey[y * n + x]
                    val gx = grey[y * n + x + 1] - grey[y * n + x - 1]
                    val gy = grey[(y + 1) * n + x] - grey[(y - 1) * n + x]
                    val mag = sqrt(gx * gx + gy * gy) / 255f
                    val bin = (((atan2(gy, gx) + Math.PI) / (2 * Math.PI) * 8).toInt()).coerceIn(0, 7)
                    raw[6 + bin] += mag
                    val l = grey[y * n + x - 1] + grey[y * n + x + 1] + grey[(y - 1) * n + x] + grey[(y + 1) * n + x] - 4 * g
                    lap += (l / 255.0) * (l / 255.0)
                }
            }
            for (ch in 0..2) {
                val m = sum[ch] / cnt
                raw[ch] = m.toFloat()
                raw[3 + ch] = sqrt(maxOf(0.0, sq[ch] / cnt - m * m)).toFloat()
            }
            for (b in 0 until 8) raw[6 + b] = raw[6 + b] / cnt * 4f
            raw[14] = sqrt(lap / cnt).toFloat()
            val o = (r * gw + c) * spec.dim
            for (d in 0 until spec.dim) {
                var acc = 0f
                val wo = d * RAW
                for (k in 0 until RAW) acc += proj[wo + k] * raw[k]
                out[o + d] = acc
            }
        }
        return out
    }

    override fun close() {}

    companion object {
        const val MODEL_ID = "debug_kotlin_stats_320"
        private const val RAW = 15
        val ASSET = ModelAsset(MODEL_ID, "(built in)", "0".repeat(64))
        val SPEC = BackboneSpec("debug_kotlin_stats_320", listOf(ASSET), inputSize = 320, gh = 40, gw = 40, dim = 128, l2NormalizePatches = false)
    }
}

/** Exact CPU k-NN through the core (no LiteRT) for the debug backbone. */
class KotlinKnnEngine : KnnEngine {
    override val label: String = "exact CPU (Kotlin, debug)"
    override fun minSq(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?): FloatArray {
        val out = DoubleArray(nq)
        Knn.CPU.minSq(feats, nq, bank, nb, dim, extra, out)
        return FloatArray(nq) { out[it].toFloat() }
    }

    override fun close() {}
}
