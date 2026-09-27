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

    /** The last failed (re)load, kept while the previous engine stays in use; null when the last load succeeded. */
    @Volatile var lastError: String? = null
        private set

    /** Present while a model load is in progress; left behind only if the load killed the process (native crash). */
    private val loadMarker = File(context.filesDir, "engine_loading.marker")

    /**
     * Start-up load that can never become a crash loop: if the previous load died natively (marker still present), do not
     * try again automatically — report it and wait for a manual retry (Telemetry → Accelerator / Re-benchmark).
     */
    fun autoLoad(choice: BackboneChoice) {
        // A marker left behind means the previous load did not finish (native crash, but also a swipe-away or a low-memory
        // kill). The runtime's own crash guard already strikes the accelerator that died, so loading again is safe: it
        // lands on the next candidate (CPU at worst) instead of leaving the app without a model.
        if (loadMarker.isFile) {
            Log.w("KaizenEngine", "previous model load did not finish (${runCatching { loadMarker.readText() }.getOrDefault("?")}) — loading again")
            loadMarker.delete()
        }
        load(choice)
    }

    fun load(choice: BackboneChoice = this.choice, forceAccel: Accel? = null, rebenchmark: Boolean = false) {
        scope.launch(Dispatchers.Default) { loadNow(choice, forceAccel, rebenchmark) }
    }

    /** Loads (or returns the already loaded) engines for [choice]; suspends until done. */
    suspend fun ensure(choice: BackboneChoice = this.choice): State.Ready? {
        ready()?.let { if (it.choice.id == choice.id) return it }
        return loadNow(choice, ifMissing = true) as? State.Ready
    }

    suspend fun loadNow(choice: BackboneChoice = this.choice, forceAccel: Accel? = null, rebenchmark: Boolean = false, ifMissing: Boolean = false): State = mutex.withLock {
        val old = _state.value as? State.Ready
        // A load that was queued behind another load of the same choice is already satisfied (never load twice).
        if (ifMissing && old != null && old.choice.id == choice.id) return@withLock old
        this.choice = choice
        _state.value = State.Loading(if (rebenchmark) "benchmarking accelerators…" else "loading ${choice.label}…")
        val next = try {
            if (choice.id == Models.DEBUG_CHOICE.id) {
                State.Ready(choice, DebugKotlinBackbone(choice.spec), KotlinKnnEngine(), null)
            } else {
                runCatching { loadMarker.writeText("${choice.id} ${forceAccel ?: "auto"}") }
                val bb = Runtime.loadBackbone(context, choice.spec, forceAccel, rebenchmark)
                val knn = try {
                    Runtime.loadKnn(context, choice.knnP, choice.knnD, choice.knnBuckets, forceAccel)
                } catch (t: Throwable) {
                    runCatching { bb.close() }
                    throw t
                }
                State.Ready(choice, bb, knn, forceAccel)
            }
        } catch (t: kotlinx.coroutines.CancellationException) {
            throw t
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
        // A failed reload keeps the working engine (the hub has already stopped its sessions; they restart on it).
        _state.value = if (next is State.Failed && old != null) old else next
        if (next is State.Failed && old != null) {
            this.choice = old.choice
            lastError = next.message
            Log.w("KaizenEngine", "keeping the previous engine after a failed load: ${next.message}")
        } else if (next is State.Ready) lastError = null
        _state.value
    }

    fun badge(): String = when (val s = _state.value) {
        is State.Ready -> s.backbone.report.badge + if (s.forced != null) " [forced]" else ""
        is State.Loading -> "loading…"
        is State.Failed -> "model error"
        State.Idle -> "not loaded"
    }
}
