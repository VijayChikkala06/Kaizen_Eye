package com.kaizeneye.v2.ml

import android.content.Context
import android.util.Log
import com.kaizeneye.runtime.Accel
import com.kaizeneye.runtime.FeatureBackbone
import com.kaizeneye.runtime.KnnEngine
import com.kaizeneye.runtime.Runtime
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

/**
 * Owns the loaded backbone + k-NN engine for the main process. Loading benchmarks the accelerators once (in the :probe
 * process, crash-guarded) and afterwards reuses the cached decision. [forceAccel] = CPU implements the CPU↔accelerator A/B
 * toggle of the telemetry screen (same model variant, so a Twin stays valid).
 */
class EngineHolder(private val context: Context, private val scope: CoroutineScope) {

    sealed interface State {
        data object Idle : State
        data class Loading(val message: String) : State
        data class Ready(val choice: BackboneChoice, val backbone: FeatureBackbone, val knn: KnnEngine, val forced: Accel?) : State
        data class Failed(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state
    private val mutex = Mutex()
    @Volatile var choice: BackboneChoice = Models.R18_CHOICE
        private set

    fun ready(): State.Ready? = _state.value as? State.Ready

    /** Present while a model load is in progress; left behind only if the load killed the process (native crash). */
    private val loadMarker = File(context.filesDir, "engine_loading.marker")

    /**
     * Start-up load that can never become a crash loop: if the previous load died natively (marker still present), do not
     * try again automatically — report it and wait for a manual retry (Telemetry → Accelerator / Re-benchmark).
     */
    fun autoLoad(choice: BackboneChoice) {
        if (loadMarker.isFile) {
            val what = runCatching { loadMarker.readText() }.getOrDefault("?")
            loadMarker.delete()
            _state.value = State.Failed("The last model load ($what) crashed the app natively — not retried automatically. Retry from Telemetry.")
            return
        }
        load(choice)
    }

    fun load(choice: BackboneChoice = this.choice, forceAccel: Accel? = null, rebenchmark: Boolean = false) {
        scope.launch(Dispatchers.Default) { loadNow(choice, forceAccel, rebenchmark) }
    }

    /** Loads (or returns the already loaded) engines for [choice]; suspends until done. */
    suspend fun ensure(choice: BackboneChoice = this.choice): State.Ready? {
        ready()?.let { if (it.choice.id == choice.id) return it }
        return loadNow(choice) as? State.Ready
    }

    suspend fun loadNow(choice: BackboneChoice = this.choice, forceAccel: Accel? = null, rebenchmark: Boolean = false): State = mutex.withLock {
        val old = _state.value as? State.Ready
        this.choice = choice
        _state.value = State.Loading(if (rebenchmark) "benchmarking accelerators…" else "loading ${choice.label}…")
        val next = try {
            if (choice.id == Models.DEBUG_CHOICE.id) {
                State.Ready(choice, DebugKotlinBackbone(choice.spec), KotlinKnnEngine(), null)
            } else {
                runCatching { loadMarker.writeText("${choice.id} ${forceAccel ?: "auto"}") }
                val bb = Runtime.loadBackbone(context, choice.spec, forceAccel, rebenchmark)
                val knn = Runtime.loadKnn(context, choice.knnP, choice.knnD, choice.knnBuckets, forceAccel)
                State.Ready(choice, bb, knn, forceAccel)
            }
        } catch (t: Throwable) {
            Log.e("KaizenEngine", "model load failed", t)
            State.Failed("${t.javaClass.simpleName}: ${t.message}")
        } finally {
            loadMarker.delete()
        }
        if (old != null && next is State.Ready) {
            try {
                old.backbone.close(); old.knn.close()
            } catch (_: Throwable) {
            }
        }
        _state.value = next
        next
    }

    fun badge(): String = when (val s = _state.value) {
        is State.Ready -> s.backbone.report.badge + if (s.forced != null) " [forced]" else ""
        is State.Loading -> "loading…"
        is State.Failed -> "model error"
        State.Idle -> "not loaded"
    }
}
