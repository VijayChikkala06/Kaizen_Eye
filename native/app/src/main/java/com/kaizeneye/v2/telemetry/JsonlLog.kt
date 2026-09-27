package com.kaizeneye.v2.telemetry

import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Append-only JSON-lines session log (one object per line), written off the hot path on a single background thread and
 * flushed every second. It lives in getExternalFilesDir("logs"), so
 * `adb pull /sdcard/Android/data/com.kaizeneye.v2/files/logs` retrieves it. Every record carries "type", "wallMs" and
 * "elapsedMs".
 */
class JsonlLog(private val dir: File) {
    private val io = Executors.newSingleThreadScheduledExecutor { r -> Thread(r, "kz-log").apply { priority = Thread.MIN_PRIORITY } }
    private var writer: BufferedWriter? = null

    @Volatile var file: File? = null
        private set

    init {
        io.scheduleWithFixedDelay({
            try {
                writer?.flush()
            } catch (_: Throwable) {
            }
        }, 1, 1, TimeUnit.SECONDS)
    }

    @Synchronized
    fun openSession(name: String): File {
        closeSession()
        dir.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val f = File(dir, "${stamp}_${name.replace(Regex("[^A-Za-z0-9_-]"), "_")}.jsonl")
        io.execute { writer = BufferedWriter(OutputStreamWriter(FileOutputStream(f, true), Charsets.UTF_8), 1 shl 16) }
        file = f
        return f
    }

    /** Thread-safe; the JSON object must not be modified after the call. No-op when no session is open. */
    fun write(type: String, fields: JSONObject = JSONObject()) {
        if (file == null) return
        fields.put("type", type)
        fields.put("wallMs", System.currentTimeMillis())
        fields.put("elapsedMs", SystemClock.elapsedRealtime())
        val line = fields.toString()
        io.execute {
            try {
                writer?.apply {
                    write(line)
                    newLine()
                }
            } catch (t: Throwable) {
                Log.w("KaizenLog", "log write failed", t)
            }
        }
    }

    @Synchronized
    fun closeSession() {
        io.execute {
            try {
                writer?.flush()
                writer?.close()
            } catch (_: Throwable) {
            }
            writer = null
        }
        file = null
    }
}
