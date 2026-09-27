package com.kaizeneye.v2.pipeline

import com.kaizeneye.core.twin.KnnBackend
import com.kaizeneye.runtime.KnnEngine

/**
 * Plugs the runtime's LiteRT matmul k-NN graph (or its exact CPU fallback) into the core's [KnnBackend]. The graph works in
 * float32; results agree with the float64 CPU search to ≲ 1e-4 relative (twin-spec §5), and teach + judge always use the
 * same backend so τ and live scores are computed the same way.
 */
class KnnAdapter(private val engine: KnnEngine) : KnnBackend {
    override fun minSq(feats: FloatArray, nq: Int, bank: FloatArray, nb: Int, dim: Int, extra: FloatArray?, out: DoubleArray) {
        val r = engine.minSq(feats, nq, bank, nb, dim, extra)
        for (i in 0 until nq) out[i] = r[i].toDouble()
    }

    val label: String get() = engine.label
}
