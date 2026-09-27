package com.kaizeneye.v2.telemetry

import java.io.File
import java.io.FileWriter

/** Tiny CSV appender for the soak / energy telemetry (plan B7 evidence). Call from one coroutine at a time. */
class CsvLog(val file: File, header: List<String>) : AutoCloseable {
    private val w: FileWriter

    init {
        file.parentFile?.mkdirs()
        val fresh = !file.exists() || file.length() == 0L
        w = FileWriter(file, true)
        if (fresh) {
            w.write(header.joinToString(","))
            w.write("\n")
        }
    }

    fun row(values: List<Any?>) {
        w.write(values.joinToString(",") { it?.toString()?.replace(",", ";") ?: "" })
        w.write("\n")
        w.flush()
    }

    override fun close() = w.close()
}
