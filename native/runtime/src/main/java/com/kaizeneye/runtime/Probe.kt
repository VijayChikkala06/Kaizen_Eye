package com.kaizeneye.runtime

import android.app.Service
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
import android.os.Process
import android.os.RemoteException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/*
 * Accelerator probe: a bound service in its own process (android:process=":probe", exported=false, Messenger IPC).
 * The app asks it to benchmark ONE candidate per request (create model on the accelerator, fixed input, 2 warm-ups + 10 timed
 * runs); the patch output comes back through a float file in filesDir/accel/probe/. A native crash or hang of an NPU/GPU
 * driver therefore kills only this process; the app notices (binderDied / onServiceDisconnected / timeout) and records the
 * candidate in the crash guard (DecisionStore). The process ends itself when unbound (onDestroy), releasing DSP/GPU state.
 */
internal object ProbeProtocol {
    const val MSG_HELLO = 1
    const val MSG_RUN = 2
    const val MSG_RESULT = 3

    const val K_ID = "id"
    const val K_SOURCE = "source"
    const val K_PATH = "path"
    const val K_ACCEL = "accel"
    const val K_SIZE = "size"
    const val K_PATCH = "patchLen"
    const val K_CLS = "clsLen"
    const val K_WARMUP = "warmup"
    const val K_RUNS = "runs"
    const val K_OUT = "out"
    const val K_OK = "ok"
    const val K_ERROR = "error"
    const val K_LOAD = "loadMs"
    const val K_TIMES = "times"
    const val K_OUTPUTS = "outputs"
    const val K_PI = "patchIndex"
    const val K_CI = "clsIndex"
    const val K_PID = "pid"
    const val K_AVAIL = "available"
    const val K_SOC = "soc"
    const val K_ADSP = "adsp"
}

internal data class ProbeRequest(
    val id: Int,
    val source: String,
    val path: String,
    val accel: Accel,
    val inputSize: Int,
    val patchLen: Int,
    val clsLen: Int,
    val warmup: Int,
    val runs: Int,
    val outFile: String,
) {
    fun toBundle(): Bundle = Bundle().apply {
        putInt(ProbeProtocol.K_ID, id)
        putString(ProbeProtocol.K_SOURCE, source)
        putString(ProbeProtocol.K_PATH, path)
        putString(ProbeProtocol.K_ACCEL, accel.name)
        putInt(ProbeProtocol.K_SIZE, inputSize)
        putInt(ProbeProtocol.K_PATCH, patchLen)
        putInt(ProbeProtocol.K_CLS, clsLen)
        putInt(ProbeProtocol.K_WARMUP, warmup)
        putInt(ProbeProtocol.K_RUNS, runs)
        putString(ProbeProtocol.K_OUT, outFile)
    }

    companion object {
        fun from(b: Bundle) = ProbeRequest(
            id = b.getInt(ProbeProtocol.K_ID),
            source = b.getString(ProbeProtocol.K_SOURCE) ?: "",
            path = b.getString(ProbeProtocol.K_PATH) ?: "",
            accel = Accel.valueOf(b.getString(ProbeProtocol.K_ACCEL) ?: Accel.CPU.name),
            inputSize = b.getInt(ProbeProtocol.K_SIZE),
            patchLen = b.getInt(ProbeProtocol.K_PATCH),
            clsLen = b.getInt(ProbeProtocol.K_CLS),
            warmup = b.getInt(ProbeProtocol.K_WARMUP),
            runs = b.getInt(ProbeProtocol.K_RUNS),
            outFile = b.getString(ProbeProtocol.K_OUT) ?: "",
        )
    }
}

internal data class ProbeReply(
    val id: Int,
    val ok: Boolean,
    val error: String?,
    val loadMs: Double,
    val timesMs: DoubleArray,
    val outputs: Int,
    val patchIndex: Int,
    val clsIndex: Int,
    val pid: Int,
    val available: String,
) {
    fun toBundle(): Bundle = Bundle().apply {
        putInt(ProbeProtocol.K_ID, id)
        putBoolean(ProbeProtocol.K_OK, ok)
        putString(ProbeProtocol.K_ERROR, error)
        putDouble(ProbeProtocol.K_LOAD, loadMs)
        putDoubleArray(ProbeProtocol.K_TIMES, timesMs)
        putInt(ProbeProtocol.K_OUTPUTS, outputs)
        putInt(ProbeProtocol.K_PI, patchIndex)
        putInt(ProbeProtocol.K_CI, clsIndex)
        putInt(ProbeProtocol.K_PID, pid)
        putString(ProbeProtocol.K_AVAIL, available)
    }

    companion object {
        fun from(b: Bundle) = ProbeReply(
            id = b.getInt(ProbeProtocol.K_ID),
            ok = b.getBoolean(ProbeProtocol.K_OK),
            error = b.getString(ProbeProtocol.K_ERROR),
            loadMs = b.getDouble(ProbeProtocol.K_LOAD),
            timesMs = b.getDoubleArray(ProbeProtocol.K_TIMES) ?: DoubleArray(0),
            outputs = b.getInt(ProbeProtocol.K_OUTPUTS),
            patchIndex = b.getInt(ProbeProtocol.K_PI),
            clsIndex = b.getInt(ProbeProtocol.K_CI),
            pid = b.getInt(ProbeProtocol.K_PID),
            available = b.getString(ProbeProtocol.K_AVAIL) ?: "",
        )
    }
}

/** Runs one probe request in THIS process (the ":probe" process). Never throws: failures come back as ok=false. */
internal object ProbeWorker {
    fun run(context: Context, req: ProbeRequest): ProbeReply = try {
        RtLog.i("probe: ${req.accel} on ${req.path} ...")
        val r = BackboneBench.measure(
            context, req.source, req.path, req.accel, req.inputSize, req.patchLen, req.clsLen, req.warmup, req.runs,
        )
        FloatFiles.write(File(req.outFile), r.patches)
        val t = AccelRules.timing(r.timesMs)
        RtLog.i("probe: ${req.accel} on ${req.path}: load ${AccelRules.fmtMs(r.loadMs)} ms, median ${t?.medianMs?.let { AccelRules.fmtMs(it) }} ms (available: ${r.available})")
        ProbeReply(req.id, true, null, r.loadMs, r.timesMs, r.outputs, r.patchIndex, r.clsIndex, Process.myPid(), r.available)
    } catch (t: Throwable) {
        RtLog.w("probe: ${req.accel} on ${req.path} failed: ${t.brief()}")
        ProbeReply(req.id, false, t.brief(), 0.0, DoubleArray(0), 0, -1, -1, Process.myPid(), "")
    }
}

/** The ":probe" process service. Public only because the manifest instantiates it; not part of the app-facing API. */
class ProbeService : Service() {
    private var thread: HandlerThread? = null
    private var messenger: Messenger? = null

    override fun onCreate() {
        super.onCreate()
        NpuEnv.ensure(this)
        val t = HandlerThread("kz-probe-worker").also { it.start() }
        thread = t
        messenger = Messenger(Handler(t.looper) { msg -> handle(msg); true })
        RtLog.i("probe service up (pid ${Process.myPid()}): ${NpuEnv.describe()}")
    }

    override fun onBind(intent: Intent?): IBinder? = messenger?.binder

    override fun onDestroy() {
        RtLog.i("probe service down (pid ${Process.myPid()}), ending the probe process")
        thread?.quitSafely()
        super.onDestroy()
        // Nothing else lives in ":probe": end it so accelerator state (DSP sessions, GPU contexts, compiled graphs) is released.
        Process.killProcess(Process.myPid())
    }

    private fun handle(msg: Message) {
        val to = msg.replyTo ?: return
        val reply = when (msg.what) {
            ProbeProtocol.MSG_HELLO -> Message.obtain(null, ProbeProtocol.MSG_HELLO).apply {
                data = Bundle().apply {
                    putInt(ProbeProtocol.K_PID, Process.myPid())
                    putString(ProbeProtocol.K_SOC, NpuEnv.socModel)
                    putString(ProbeProtocol.K_ADSP, NpuEnv.adspLibraryPath)
                }
            }
            ProbeProtocol.MSG_RUN -> {
                val req = try { ProbeRequest.from(msg.data) } catch (t: Throwable) { null }
                Message.obtain(null, ProbeProtocol.MSG_RESULT).apply {
                    data = if (req == null) {
                        ProbeReply(-1, false, "bad request", 0.0, DoubleArray(0), 0, -1, -1, Process.myPid(), "").toBundle()
                    } else {
                        ProbeWorker.run(this@ProbeService, req).toBundle()
                    }
                }
            }
            else -> return
        }
        try {
            to.send(reply)
        } catch (e: RemoteException) {
            RtLog.w("probe: app went away before the reply: ${e.brief()}")
        }
    }
}

/**
 * App-side client of the ":probe" service. One request at a time (Runtime serialises benchmarks). All callbacks run on a
 * private HandlerThread (bindService with an executor), never on the app's main thread.
 */
internal class ProbeClient(context: Context) : AutoCloseable {
    sealed class Outcome {
        class Ok(val reply: ProbeReply) : Outcome()
        class Died(val pid: Int, val how: String) : Outcome()
        object TimedOut : Outcome()
        class Unavailable(val reason: String) : Outcome()
    }

    private val app: Context = context.applicationContext ?: context
    private val thread = HandlerThread("kz-probe-client").also { it.start() }
    private val handler = Handler(thread.looper) { msg -> onReply(msg); true }
    private val replyMessenger = Messenger(handler)
    private val executor = Executor { r -> handler.post(r) }
    private val nextId = AtomicInteger(1)
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<Outcome>>()
    @Volatile private var conn: Conn? = null
    @Volatile private var hello: CompletableDeferred<Bundle>? = null

    /** pid of the current probe process (0 = unknown). */
    @Volatile var pid: Int = 0
        private set

    fun nextRequestId(): Int = nextId.getAndIncrement()

    private inner class Conn : ServiceConnection, IBinder.DeathRecipient {
        val connected = CompletableDeferred<Messenger>()
        val dead = AtomicBoolean(false)
        @Volatile var binder: IBinder? = null
        @Volatile var messenger: Messenger? = null

        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            if (service == null) {
                connected.completeExceptionally(IllegalStateException("probe returned a null binder"))
                return
            }
            binder = service
            try {
                service.linkToDeath(this, 0)
            } catch (e: RemoteException) {
                died("died while connecting")
                return
            }
            connected.complete(Messenger(service))
        }

        override fun onServiceDisconnected(name: ComponentName?) = died("disconnected")

        override fun binderDied() = died("binder died")

        override fun onBindingDied(name: ComponentName?) = died("binding died")

        override fun onNullBinding(name: ComponentName?) {
            connected.completeExceptionally(IllegalStateException("probe returned a null binding"))
        }

        fun died(how: String) {
            if (!dead.compareAndSet(false, true)) return
            connected.completeExceptionally(IllegalStateException("probe $how"))
            hello?.completeExceptionally(IllegalStateException("probe $how"))
            val p = pid
            for (d in pending.values) d.complete(Outcome.Died(p, how))
            pending.clear()
        }
    }

    private fun onReply(msg: Message) {
        when (msg.what) {
            ProbeProtocol.MSG_HELLO -> hello?.complete(Bundle(msg.data))
            ProbeProtocol.MSG_RESULT -> {
                val r = try { ProbeReply.from(msg.data) } catch (_: Throwable) { return }
                pending.remove(r.id)?.complete(Outcome.Ok(r))
            }
        }
    }

    private suspend fun connect(): Conn {
        conn?.let { if (!it.dead.get() && it.messenger != null) return it }
        disconnect()
        val c = Conn()
        val bound = try {
            app.bindService(Intent(app, ProbeService::class.java), Context.BIND_AUTO_CREATE, executor, c)
        } catch (t: Throwable) {
            throw IllegalStateException("bindService(ProbeService) failed: ${t.brief()}")
        }
        if (!bound) {
            try { app.unbindService(c) } catch (_: Throwable) { }
            throw IllegalStateException("ProbeService not found - is com.kaizeneye.runtime.ProbeService in the merged manifest?")
        }
        conn = c
        val m = try {
            withTimeoutOrNull(CONNECT_TIMEOUT_MS) { c.connected.await() }
        } catch (t: Throwable) {
            disconnect()
            throw IllegalStateException("probe did not connect: ${t.message}")
        } ?: run {
            disconnect()
            throw IllegalStateException("probe did not connect within ${CONNECT_TIMEOUT_MS / 1000} s")
        }
        c.messenger = m
        val h = CompletableDeferred<Bundle>()
        hello = h
        try {
            m.send(Message.obtain(null, ProbeProtocol.MSG_HELLO).apply { replyTo = replyMessenger })
        } catch (e: RemoteException) {
            disconnect()
            throw IllegalStateException("probe hello failed: ${e.brief()}")
        }
        val hb = try {
            withTimeoutOrNull(CONNECT_TIMEOUT_MS) { h.await() }
        } catch (t: Throwable) {
            null
        } ?: run {
            disconnect()
            throw IllegalStateException("probe did not answer hello")
        }
        pid = hb.getInt(ProbeProtocol.K_PID)
        RtLog.i("probe connected: pid $pid, SoC '${hb.getString(ProbeProtocol.K_SOC)}', ADSP ${hb.getString(ProbeProtocol.K_ADSP)}")
        return c
    }

    /** Run one request. Unavailable = could not reach the probe at all (nothing was tried). */
    suspend fun run(req: ProbeRequest, timeoutMs: Long): Outcome {
        val c = try {
            connect()
        } catch (t: Throwable) {
            return Outcome.Unavailable(t.message ?: t.brief())
        }
        val m = c.messenger ?: return Outcome.Unavailable("no messenger")
        val d = CompletableDeferred<Outcome>()
        pending[req.id] = d
        if (c.dead.get()) {
            pending.remove(req.id)
            disconnect()
            return Outcome.Died(pid, "dead before the request")
        }
        try {
            m.send(Message.obtain(null, ProbeProtocol.MSG_RUN).apply {
                data = req.toBundle()
                replyTo = replyMessenger
            })
        } catch (e: RemoteException) {
            pending.remove(req.id)
            disconnect()
            return Outcome.Died(pid, "send failed: ${e.brief()}")
        }
        val out = withTimeoutOrNull(timeoutMs) { d.await() }
        pending.remove(req.id)
        if (out == null) {
            RtLog.e("probe: no answer within ${timeoutMs / 1000} s for ${req.accel} on ${req.path}; killing probe pid $pid")
            killProbe()
            disconnect()
            return Outcome.TimedOut
        }
        if (out is Outcome.Died) disconnect()
        return out
    }

    private fun killProbe() {
        val p = pid
        if (p > 0 && p != Process.myPid()) {
            try { Process.killProcess(p) } catch (_: Throwable) { }
        }
    }

    private fun disconnect() {
        val c = conn ?: return
        conn = null
        try { c.binder?.unlinkToDeath(c, 0) } catch (_: Throwable) { }
        try { app.unbindService(c) } catch (_: Throwable) { }
    }

    override fun close() {
        disconnect()
        for (d in pending.values) d.complete(Outcome.Unavailable("client closed"))
        pending.clear()
        thread.quitSafely()
    }

    companion object {
        const val CONNECT_TIMEOUT_MS = 15_000L
    }
}
