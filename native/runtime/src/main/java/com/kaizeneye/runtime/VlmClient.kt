package com.kaizeneye.runtime

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import android.os.SystemClock
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger

/**
 * App-side handle of the ":vlm" process (see [VlmService]). Nothing here blocks: [bind] returns at once ([state] goes
 * Loading -> Ready/Failed), [explain] suspends without blocking a thread. All IPC callbacks run on a private HandlerThread.
 * Never loads LiteRT-LM in the app process.
 *
 *   val vlm = VlmClient(context).also { it.bind() }           // at app start (engine init ~10 s in the background)
 *   val r = vlm.explain(cropJpegInCacheDir, prompt)            // one sentence, or r.error = "timeout" / "busy" / ...
 *   vlm.close()                                                // unbinds: the engine closes and ":vlm" ends
 */
class VlmClient(context: Context) : AutoCloseable {
    private val app: Context = context.applicationContext ?: context
    private val _state = MutableStateFlow<VlmState>(VlmState.Idle)
    val state: StateFlow<VlmState> = _state.asStateFlow()

    private val thread = HandlerThread("kz-vlm-client").also { it.start() }
    private val handler = Handler(thread.looper) { m -> onMessage(m); true }
    private val replyMessenger = Messenger(handler)
    private val executor = Executor { r -> handler.post(r) }
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<VlmResult>>()
    private val nextId = AtomicInteger(1)
    @Volatile private var service: Messenger? = null
    @Volatile private var binder: IBinder? = null
    @Volatile private var bound = false
    @Volatile private var closed = false
    @Volatile private var rebinds = 0

    private val death = IBinder.DeathRecipient { onDied("VLM process died") }

    private val conn = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, b: IBinder?) {
            if (b == null) {
                _state.value = VlmState.Failed("VLM service returned no binder")
                return
            }
            binder = b
            try {
                b.linkToDeath(death, 0)
            } catch (e: RemoteException) {
                onDied("VLM process died while connecting")
                return
            }
            val m = Messenger(b)
            service = m
            try {
                m.send(Message.obtain(null, VlmProtocol.MSG_HELLO).apply { replyTo = replyMessenger })
            } catch (e: RemoteException) {
                onDied("VLM hello failed")
            }
        }

        // The ":vlm" process died; the system restarts it while we stay bound (state follows via MSG_STATE).
        override fun onServiceDisconnected(name: ComponentName?) = onDied("VLM process disconnected")

        // This binding will never reconnect (e.g. app update): bind again once.
        override fun onBindingDied(name: ComponentName?) {
            onDied("VLM binding died")
            handler.post {
                if (!closed && rebinds < 1) {
                    rebinds++
                    unbindQuietly()
                    bind()
                }
            }
        }

        override fun onNullBinding(name: ComponentName?) {
            _state.value = VlmState.Failed("VLM service returned a null binding")
        }
    }

    /** Start (or connect to) the ":vlm" process. Idempotent; returns immediately. */
    fun bind() {
        if (closed || bound) return
        _state.value = VlmState.Loading
        val ok = try {
            app.bindService(Intent(app, VlmService::class.java), Context.BIND_AUTO_CREATE, executor, conn)
        } catch (t: Throwable) {
            RtLog.e("bindService(VlmService) failed: ${t.brief()}")
            false
        }
        bound = ok
        if (!ok) {
            unbindQuietly()
            _state.value = VlmState.Failed("cannot bind VlmService - is com.kaizeneye.runtime.VlmService in the merged manifest?")
        }
    }

    /**
     * Explain one reject. [imageFile] = a JPEG the app wrote into its cacheDir (the ":vlm" process reads it by absolute
     * path). One request runs at a time in the service; at most two wait (then "busy"). After [timeoutMs] the answer is
     * VlmResult(null, backend, ms, "timeout") and the running generation is cancelled.
     */
    suspend fun explain(imageFile: File, prompt: String, timeoutMs: Long = 4000): VlmResult {
        val st = _state.value
        val s = service
        if (st !is VlmState.Ready || s == null) return VlmResult(null, (st as? VlmState.Ready)?.backend ?: "none", 0, "VLM not ready ($st)")
        val id = nextId.getAndIncrement()
        val d = CompletableDeferred<VlmResult>()
        pending[id] = d
        val t0 = SystemClock.elapsedRealtime()
        try {
            s.send(Message.obtain(null, VlmProtocol.MSG_EXPLAIN).apply {
                data = Bundle().apply {
                    putInt(VlmProtocol.K_ID, id)
                    putString(VlmProtocol.K_IMAGE, imageFile.absolutePath)
                    putString(VlmProtocol.K_PROMPT, prompt)
                    putLong(VlmProtocol.K_TIMEOUT, timeoutMs)
                    putInt(VlmProtocol.K_MAX_TOKENS, VlmProtocol.DEFAULT_MAX_TOKENS)
                }
                replyTo = replyMessenger
            })
        } catch (e: RemoteException) {
            pending.remove(id)
            return VlmResult(null, st.backend, SystemClock.elapsedRealtime() - t0, "send failed: ${e.brief()}")
        }
        // The service answers "timeout" itself at timeoutMs; the grace covers IPC delay or a dead service.
        val r = withTimeoutOrNull(timeoutMs + CLIENT_GRACE_MS) { d.await() }
        pending.remove(id)
        return r ?: VlmResult(null, st.backend, SystemClock.elapsedRealtime() - t0, "timeout")
    }

    private fun onMessage(m: Message) {
        when (m.what) {
            VlmProtocol.MSG_STATE -> {
                val b = m.data
                if (!closed) _state.value = when (b.getString(VlmProtocol.K_STATE)) {
                    VlmProtocol.READY -> VlmState.Ready(
                        backend = b.getString(VlmProtocol.K_BACKEND) ?: "?",
                        initMs = b.getLong(VlmProtocol.K_INIT_MS),
                        model = b.getString(VlmProtocol.K_MODEL) ?: "",
                        detail = b.getString(VlmProtocol.K_DETAIL) ?: "",
                    )
                    VlmProtocol.LOADING -> VlmState.Loading
                    VlmProtocol.FAILED -> VlmState.Failed(b.getString(VlmProtocol.K_MESSAGE) ?: "VLM failed")
                    else -> VlmState.Idle
                }
            }
            VlmProtocol.MSG_RESULT -> {
                val b = m.data
                val id = b.getInt(VlmProtocol.K_ID)
                pending.remove(id)?.complete(
                    VlmResult(
                        text = b.getString(VlmProtocol.K_TEXT),
                        backend = b.getString(VlmProtocol.K_BACKEND) ?: "none",
                        ms = b.getLong(VlmProtocol.K_MS),
                        error = b.getString(VlmProtocol.K_ERROR),
                    ),
                )
            }
        }
    }

    private fun onDied(why: String) {
        service = null
        failPending(why)
        if (!closed) _state.value = VlmState.Failed(why)
    }

    private fun failPending(why: String) {
        for ((id, d) in pending) d.complete(VlmResult(null, "none", 0, why))
        pending.clear()
    }

    private fun unbindQuietly() {
        try { binder?.unlinkToDeath(death, 0) } catch (_: Throwable) { }
        try { app.unbindService(conn) } catch (_: Throwable) { }
        bound = false
        service = null
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            service?.send(Message.obtain(null, VlmProtocol.MSG_BYE).apply { replyTo = replyMessenger })
        } catch (_: Throwable) {
        }
        unbindQuietly()
        failPending("closed")
        _state.value = VlmState.Idle
        thread.quitSafely()
    }

    companion object {
        const val CLIENT_GRACE_MS = 1500L
    }
}
