package com.kaizeneye.runtime

import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.os.RemoteException
import android.os.SystemClock
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

internal object VlmProtocol {
    const val MSG_HELLO = 1      // client -> service (replyTo = client); service answers MSG_STATE and keeps sending state changes
    const val MSG_STATE = 2      // service -> client
    const val MSG_EXPLAIN = 3    // client -> service
    const val MSG_RESULT = 4     // service -> client
    const val MSG_BYE = 5        // client -> service: stop sending state

    const val K_STATE = "state"
    const val K_BACKEND = "backend"
    const val K_INIT_MS = "initMs"
    const val K_MESSAGE = "message"
    const val K_MODEL = "model"
    const val K_DETAIL = "detail"
    const val K_ID = "id"
    const val K_IMAGE = "image"
    const val K_PROMPT = "prompt"
    const val K_TIMEOUT = "timeoutMs"
    const val K_MAX_TOKENS = "maxTokens"
    const val K_TEXT = "text"
    const val K_MS = "ms"
    const val K_ERROR = "error"

    const val IDLE = "idle"
    const val LOADING = "loading"
    const val READY = "ready"
    const val FAILED = "failed"

    const val DEFAULT_MAX_TOKENS = 64
    /** At most this many explanations wait behind the running one; more are answered "busy" at once. */
    const val MAX_WAITING = 2
    /** Conversation settings levels: 0 greedy sampler + output cap, 1 output cap only, 2 library defaults. */
    const val MAX_CONFIG_LEVEL = 2
}

/** Crash guard of the ":vlm" process: filesDir/vlm/{trying.json, strikes.json} (CrashGuard policy, phase "vlm"). */
internal class VlmGuard(private val ctx: Context) {
    private val dir = File(ctx.filesDir, "vlm")
    private val tryingFile = File(dir, "trying.json")
    private val strikesFile = File(dir, "strikes.json")
    private val token = DecisionStore.PROCESS_TOKEN

    private fun strikes(): JSONObject = try {
        if (strikesFile.isFile) JSONObject(strikesFile.readText()) else JSONObject()
    } catch (_: Throwable) {
        JSONObject()
    }

    @Synchronized
    fun strikesOf(id: String): Int = strikes().optInt(id, 0)

    @Synchronized
    fun isCrashed(id: String): Boolean = CrashGuard.isCrashed(strikesOf(id))

    @Synchronized
    private fun setStrikes(id: String, n: Int) {
        try {
            writeTextAtomic(strikesFile, strikes().put(id, n).toString())
        } catch (t: Throwable) {
            RtLog.w("vlm guard: cannot write strikes: ${t.brief()}")
        }
    }

    @Synchronized
    fun markTrying(id: String) {
        try {
            writeTextAtomic(
                tryingFile,
                JSONObject().put("id", id).put("pid", Process.myPid()).put("token", token).put("atMs", System.currentTimeMillis()).toString(),
            )
        } catch (t: Throwable) {
            RtLog.w("vlm guard: cannot write marker: ${t.brief()}")
        }
    }

    @Synchronized
    fun clearTrying() {
        try {
            if (tryingFile.exists()) tryingFile.delete()
        } catch (_: Throwable) {
        }
    }

    /** A marker left by a dead ":vlm" process: judge the death (native crash -> skip that model/backend from now on). */
    @Synchronized
    fun judgeStale() {
        val m = try {
            if (tryingFile.isFile) JSONObject(tryingFile.readText()) else null
        } catch (_: Throwable) {
            null
        } ?: return
        if (m.optString("token") == token) return
        val id = m.optString("id")
        val pid = m.optInt("pid")
        val reason = DecisionStore.exitReason(ctx, pid, m.optLong("atMs"))
        val before = strikesOf(id)
        val after = CrashGuard.strikesAfterStaleMarker(before, CrashGuard.Phase.VLM, CrashGuard.classify(reason))
        if (after != before) setStrikes(id, after)
        RtLog.w("vlm guard: process $pid died (${CrashGuard.reasonName(reason)}) initialising $id; strikes $before -> $after")
        clearTrying()
    }

    fun clearAll() {
        clearTrying()
        try { strikesFile.delete() } catch (_: Throwable) { }
    }
}

/**
 * Offline explanation engine, alone in the ":vlm" process (liblitertlm_jni.so embeds its own LiteRT; it must never share a
 * process with the vision runtime). Bound service + Messenger. On first bind: NpuEnv.ensure, locate the model
 * (FastVLM-0.5B sm8850, SHA-256 pinned; else gemma-4-E2B-it-gpu), then try LiteRT-LM backends (VlmPlan: NPU -> GPU -> CPU),
 * each under the crash guard. Explanations run one at a time on a worker thread; a deadline answers "timeout" and cancels
 * the running conversation. Unbinding closes the engine and ends the process (frees ~1 GB).
 * Public only because the manifest instantiates it; use [VlmClient].
 */
class VlmService : Service() {
    private lateinit var ipcThread: HandlerThread
    private lateinit var workThread: HandlerThread
    private lateinit var ipc: Handler
    private lateinit var work: Handler
    private lateinit var messenger: Messenger
    private val clients = CopyOnWriteArrayList<Messenger>()
    @Volatile private var stateBundle: Bundle = stateOf(VlmProtocol.IDLE)
    @Volatile private var initStarted = false
    private val waiting = AtomicInteger(0)

    // Engine objects: touched on the work thread only (except cancelProcess under convLock).
    private var engine: Engine? = null
    @Volatile private var backend: String = "none"
    private val convLock = Any()
    private var runningConv: Conversation? = null
    private var runningJob: Job? = null

    private class Job(
        val id: Int,
        val path: String,
        val prompt: String,
        val timeoutMs: Long,
        val maxTokens: Int,
        val replyTo: Messenger?,
        val receivedAt: Long,
    ) {
        val answered = AtomicBoolean(false)
    }

    override fun onCreate() {
        super.onCreate()
        NpuEnv.ensure(this)
        ipcThread = HandlerThread("kz-vlm-ipc").also { it.start() }
        workThread = HandlerThread("kz-vlm-work").also { it.start() }
        ipc = Handler(ipcThread.looper) { m -> onMessage(m); true }
        work = Handler(workThread.looper)
        messenger = Messenger(ipc)
        RtLog.i("vlm service up (pid ${Process.myPid()}): ${NpuEnv.describe()}")
    }

    override fun onBind(intent: Intent?): IBinder {
        if (!initStarted) {
            initStarted = true
            setState(stateOf(VlmProtocol.LOADING))
            work.post { initEngine() }
        }
        return messenger.binder
    }

    override fun onUnbind(intent: Intent?): Boolean {
        work.post { closeEngine() }
        return false
    }

    override fun onDestroy() {
        RtLog.i("vlm service down (pid ${Process.myPid()}): closing the engine and ending the process")
        // Close on the worker (after the running job); hard stop after 3 s if the worker is stuck in native code.
        work.post {
            closeEngine()
            VlmGuard(this).clearTrying()
            Process.killProcess(Process.myPid())
        }
        ipc.postDelayed({
            VlmGuard(this).clearTrying()     // our own kill is not a crash of the candidate being initialised
            Process.killProcess(Process.myPid())
        }, 3000)
        super.onDestroy()
    }

    // ------------------------------------------------------------------------------------------------------ state
    private fun stateOf(s: String, backend: String? = null, initMs: Long = 0, message: String? = null, model: String? = null, detail: String? = null) =
        Bundle().apply {
            putString(VlmProtocol.K_STATE, s)
            putString(VlmProtocol.K_BACKEND, backend)
            putLong(VlmProtocol.K_INIT_MS, initMs)
            putString(VlmProtocol.K_MESSAGE, message)
            putString(VlmProtocol.K_MODEL, model)
            putString(VlmProtocol.K_DETAIL, detail)
        }

    private fun setState(b: Bundle) {
        stateBundle = b
        for (c in clients) sendState(c, b)
    }

    private fun sendState(to: Messenger, b: Bundle) {
        try {
            to.send(Message.obtain(null, VlmProtocol.MSG_STATE).apply { data = Bundle(b) })
        } catch (e: RemoteException) {
            clients.remove(to)
        }
    }

    // --------------------------------------------------------------------------------------------------------- ipc
    private fun onMessage(m: Message) {
        when (m.what) {
            VlmProtocol.MSG_HELLO -> m.replyTo?.let { c ->
                if (clients.none { it.binder == c.binder }) clients += c
                sendState(c, stateBundle)
            }
            VlmProtocol.MSG_BYE -> m.replyTo?.let { c -> clients.removeAll { it.binder == c.binder } }
            VlmProtocol.MSG_EXPLAIN -> onExplain(m)
        }
    }

    private fun onExplain(m: Message) {
        val d = m.data
        val job = Job(
            id = d.getInt(VlmProtocol.K_ID),
            path = d.getString(VlmProtocol.K_IMAGE) ?: "",
            prompt = d.getString(VlmProtocol.K_PROMPT) ?: "",
            timeoutMs = d.getLong(VlmProtocol.K_TIMEOUT, 4000L).coerceIn(100L, 120_000L),
            maxTokens = d.getInt(VlmProtocol.K_MAX_TOKENS, VlmProtocol.DEFAULT_MAX_TOKENS).coerceIn(1, 1024),
            replyTo = m.replyTo,
            receivedAt = SystemClock.elapsedRealtime(),
        )
        val st = stateBundle.getString(VlmProtocol.K_STATE)
        if (st != VlmProtocol.READY) {
            answer(job, null, "VLM not ready ($st${stateBundle.getString(VlmProtocol.K_MESSAGE)?.let { ": $it" } ?: ""})")
            return
        }
        if (waiting.get() >= VlmProtocol.MAX_WAITING) {
            answer(job, null, "busy")
            return
        }
        waiting.incrementAndGet()
        work.post { runJob(job) }
        ipc.postDelayed({ onDeadline(job) }, job.timeoutMs)
    }

    private fun onDeadline(job: Job) {
        if (answer(job, null, "timeout")) {
            synchronized(convLock) {
                if (runningJob === job) {
                    try {
                        runningConv?.cancelProcess()
                    } catch (t: Throwable) {
                        RtLog.w("vlm: cancelProcess failed: ${t.brief()}")
                    }
                }
            }
        }
    }

    /** Sends the result once; false if the job was already answered (timeout vs result race). */
    private fun answer(job: Job, text: String?, error: String?): Boolean {
        if (!job.answered.compareAndSet(false, true)) return false
        val ms = SystemClock.elapsedRealtime() - job.receivedAt
        val to = job.replyTo ?: return true
        try {
            to.send(Message.obtain(null, VlmProtocol.MSG_RESULT).apply {
                data = Bundle().apply {
                    putInt(VlmProtocol.K_ID, job.id)
                    putString(VlmProtocol.K_TEXT, text)
                    putString(VlmProtocol.K_BACKEND, backend)
                    putLong(VlmProtocol.K_MS, ms)
                    putString(VlmProtocol.K_ERROR, error)
                }
            })
        } catch (e: RemoteException) {
            RtLog.w("vlm: client went away before the answer")
        }
        return true
    }

    // ------------------------------------------------------------------------------------------------ work thread
    private fun runJob(job: Job) {
        waiting.decrementAndGet()
        if (job.answered.get()) return
        val e = engine
        if (e == null) {
            answer(job, null, "engine not available")
            return
        }
        val img = File(job.path)
        if (!img.isFile || !img.canRead()) {
            answer(job, null, "image not readable: ${job.path}")
            return
        }
        // Conversation settings, most specific first: greedy sampler + output cap, output cap only, library defaults.
        // A simpler level is kept for later requests only once it has produced an answer.
        var firstError: String? = null
        for (level in configLevel..VlmProtocol.MAX_CONFIG_LEVEL) {
            if (job.answered.get()) return
            var conv: Conversation? = null
            try {
                conv = e.createConversation(conversationConfig(level, job.maxTokens))
                synchronized(convLock) {
                    runningConv = conv
                    runningJob = job
                }
                if (job.answered.get()) return
                val reply = conv.sendMessage(Contents.of(Content.ImageFile(img.absolutePath), Content.Text(job.prompt)))
                val text = VlmPlan.joinText(reply.contents.contents.filterIsInstance<Content.Text>().map { it.text })
                if (level != configLevel) {
                    RtLog.w("vlm: conversation settings level $level works (level $configLevel failed: $firstError); keeping it")
                    configLevel = level
                }
                if (text.isEmpty()) answer(job, null, "empty answer") else answer(job, text, null)
                return
            } catch (t: Throwable) {
                if (job.answered.get()) return                     // timed out (cancelled) - already answered
                if (firstError == null) firstError = t.brief()
                RtLog.w("vlm: request failed with conversation settings level $level: ${t.brief()}")
            } finally {
                synchronized(convLock) {
                    runningConv = null
                    runningJob = null
                }
                try { conv?.close() } catch (_: Throwable) { }
            }
        }
        answer(job, null, firstError ?: "failed")
    }

    @Volatile private var configLevel = 0

    private fun conversationConfig(level: Int, maxTokens: Int): ConversationConfig = when (level) {
        0 -> ConversationConfig(samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 1.0, seed = 0), maxOutputToken = maxTokens)
        1 -> ConversationConfig(maxOutputToken = maxTokens)
        else -> ConversationConfig()
    }

    private fun initEngine() {
        val t0 = SystemClock.elapsedRealtime()
        try {
            val guard = VlmGuard(this)
            guard.judgeStale()
            val versionCode = AppInfo.versionCode(this)
            val fp = AppInfo.fingerprint()
            val libDir = applicationInfo.nativeLibraryDir
            val errors = ArrayList<String>()
            var anyModel = false
            // Models in preference order; the (unpinned, 2 GB) fallback is only located/hashed if the first one fails.
            for (asset in listOf(KnownModels.FASTVLM_05B_SM8850, KnownModels.GEMMA4_E2B_GPU)) {
                val lm = try {
                    ModelLocator.locate(this, asset)
                } catch (t: Throwable) {
                    errors += "${asset.fileName}: ${t.brief()}"
                    null
                }
                if (lm == null) {
                    errors += "${asset.fileName}: not found"
                    continue
                }
                val path = lm.path
                if (!lm.sha256Ok || lm.source == ModelLocator.SOURCE_BUNDLED || path == null) {
                    errors += "${asset.fileName}: ${lm.note ?: "bundled copies are not supported"}"
                    continue
                }
                anyModel = true
                for (c in VlmPlan.plan(listOf(asset.fileName to path))) {
                    val id = c.id + "|" + VlmPlan.stamp(versionCode, fp, lm.bytes)
                    if (guard.isCrashed(id)) {
                        errors += "${c.fileName} ${c.backend}: crashed earlier - skipped"
                        continue
                    }
                    if (tryEngine(c, id, guard, libDir, lm, t0, errors)) return
                }
            }
            val msg = if (!anyModel) "no VLM model (${errors.joinToString("; ")})" else "VLM init failed: ${errors.joinToString(" | ")}"
            setState(stateOf(VlmProtocol.FAILED, message = msg.take(1500)))
        } catch (t: Throwable) {
            RtLog.e("vlm: init crashed: ${t.brief()}", t)
            setState(stateOf(VlmProtocol.FAILED, message = t.brief()))
        }
    }

    /** One (model, backend) attempt under the crash guard. True = engine ready (state set). */
    private fun tryEngine(c: VlmCandidate, id: String, guard: VlmGuard, libDir: String, lm: LocatedModel, t0: Long, errors: MutableList<String>): Boolean {
        guard.markTrying(id)
        var e: Engine? = null
        val ts = SystemClock.elapsedRealtime()
        try {
            val b: Backend = when (c.backend) {
                VlmPlan.NPU -> Backend.NPU(nativeLibraryDir = libDir)
                VlmPlan.GPU -> Backend.GPU()
                else -> Backend.CPU()
            }
            RtLog.i("vlm: initialising ${c.fileName} on ${c.backend} ...")
            val eng = Engine(
                EngineConfig(
                    modelPath = c.path,
                    backend = b,
                    visionBackend = b,
                    audioBackend = null,
                    maxNumTokens = null,
                    maxNumImages = 1,
                    cacheDir = cacheDir.absolutePath,
                ),
            )
            e = eng
            eng.initialize()
            guard.clearTrying()
            engine = eng
            backend = c.backend
            val initMs = SystemClock.elapsedRealtime() - ts
            val detail = listOfNotNull(
                "sha256 ${lm.actualSha256 ?: "?"}${if (lm.asset.sha256.isBlank()) " (not pinned)" else " (pinned, OK)"}",
                "source ${lm.source}",
                "total ${SystemClock.elapsedRealtime() - t0} ms incl. model check",
                errors.takeIf { it.isNotEmpty() }?.joinToString(" | ", prefix = "tried before: "),
            ).joinToString("; ")
            RtLog.i("vlm: ready on ${c.backend} (${c.fileName}) in $initMs ms; $detail")
            setState(stateOf(VlmProtocol.READY, backend = c.backend, initMs = initMs, model = c.fileName, detail = detail))
            return true
        } catch (t: Throwable) {
            guard.clearTrying()
            try { e?.close() } catch (_: Throwable) { }
            errors += "${c.fileName} ${c.backend}: ${t.brief()}"
            RtLog.w("vlm: ${c.fileName} on ${c.backend} failed: ${t.brief()}")
            return false
        }
    }

    private fun closeEngine() {
        val e = engine ?: return
        engine = null
        backend = "none"
        try {
            e.close()
            RtLog.i("vlm: engine closed")
        } catch (t: Throwable) {
            RtLog.w("vlm: engine close failed: ${t.brief()}")
        }
        setState(stateOf(VlmProtocol.IDLE))
    }
}
