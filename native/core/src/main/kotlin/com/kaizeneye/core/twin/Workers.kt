package com.kaizeneye.core.twin

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.ThreadFactory

/**
 * A small fork/join helper for the exact CPU loops of teach. Work is split into fixed, index-ordered parts whose
 * results never depend on the split, so every result is identical for any thread count (determinism, spec §7).
 * `threads <= 1` runs everything on the calling thread.
 */
internal class Workers(threads: Int) : AutoCloseable {
    val threads: Int = maxOf(1, threads)

    private val pool: ExecutorService? = if (this.threads > 1) {
        Executors.newFixedThreadPool(this.threads - 1, DaemonFactory)
    } else null

    /** Runs `body(part)` for every part in `0 until parts`; part 0 on the calling thread. Rethrows the first failure. */
    fun run(parts: Int, body: (Int) -> Unit) {
        val p = pool
        if (p == null || parts <= 1) {
            for (i in 0 until parts) body(i)
            return
        }
        val futures = ArrayList<Future<*>>(parts - 1)
        for (i in 1 until parts) futures.add(p.submit { body(i) })
        var error: Throwable? = null
        try {
            body(0)
        } catch (e: Throwable) {
            error = e
        }
        for (f in futures) {
            try {
                f.get()
            } catch (e: java.util.concurrent.ExecutionException) {
                if (error == null) error = e.cause ?: e
            }
        }
        if (error != null) throw error
    }

    override fun close() {
        pool?.shutdown()
    }

    private object DaemonFactory : ThreadFactory {
        override fun newThread(r: Runnable): Thread = Thread(r, "kaizen-teach").apply { isDaemon = true }
    }
}
