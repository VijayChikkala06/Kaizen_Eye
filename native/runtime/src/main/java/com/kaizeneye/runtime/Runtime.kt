package com.kaizeneye.runtime

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Entry points of the vision runtime (app process). All loads run on Dispatchers.IO and are serialised (one benchmark /
 * model creation at a time). NOTE: this object shadows java.lang.Runtime for star-importers of com.kaizeneye.runtime.*;
 * import it explicitly (or alias it) in files that also use java.lang.Runtime.
 */
object Runtime {
    private val mutex = Mutex()

    /**
     * Load [spec] on the best accelerator for this device (cached decision, else benchmark in the ":probe" process), warmed up.
     * [forceAccel] = A/B toggle: run the decision's model variant on that accelerator instead (only if it was benchmarked
     * without crash or accuracy failure; otherwise the CPU runs it and the report says why) - the decision itself is kept.
     * [rebenchmark] = ignore the cached decision (crash records are kept; see [forgetCrashes]).
     * Throws IllegalStateException when no float variant of the model can be found / verified or it cannot run even on the CPU.
     */
    suspend fun loadBackbone(context: Context, spec: BackboneSpec, forceAccel: Accel? = null, rebenchmark: Boolean = false): FeatureBackbone =
        withContext(Dispatchers.IO) {
            mutex.withLock { BackboneLoader.load(context.applicationContext ?: context, spec, forceAccel, rebenchmark) }
        }

    /**
     * k-NN engine over the graphs knn_p{P}_d{dim}_k{K}.tflite for K in [buckets] (fp32 on the CPU by default: the 1e30 mask
     * overflows fp16), each verified against the exact search at load; falls back to the exact Kotlin search when no graph
     * is usable. Never throws for missing graphs.
     */
    suspend fun loadKnn(context: Context, P: Int, dim: Int, buckets: List<Int>, forceAccel: Accel? = null): KnnEngine =
        withContext(Dispatchers.IO) {
            mutex.withLock { KnnLoader.load(context.applicationContext ?: context, P, dim, buckets, forceAccel) }
        }

    /** Forget the benchmark decisions (next load benchmarks again). Crash records are kept. */
    fun forgetDecisions(context: Context) {
        val app = context.applicationContext ?: context
        DecisionStore.get(app).forgetDecisions()
        File(app.filesDir, "accel/probe").listFiles()?.forEach { it.delete() }
        RtLog.i("accelerator decisions forgotten")
    }

    /** Forget which candidates crashed (they will be tried again - in the probe first). */
    fun forgetCrashes(context: Context) {
        DecisionStore.get(context.applicationContext ?: context).forgetCrashes()
        RtLog.i("accelerator crash records forgotten")
    }
}
