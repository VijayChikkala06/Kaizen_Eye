package com.kaizeneye.runtime

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.sqrt

/** Streaming SHA-256 (1 MB buffer; a 942 MB model takes ~1-2 s on the phone, so verdicts are cached by ModelLocator). */
internal object Sha256 {
    class Result(val hex: String, val bytes: Long)

    fun of(input: InputStream, bufferSize: Int = 1 shl 20): Result {
        val md = MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(bufferSize)
        var total = 0L
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            md.update(buf, 0, n)
            total += n
        }
        return Result(hex(md.digest()), total)
    }

    fun of(file: File): Result = file.inputStream().use { of(it) }

    fun of(bytes: ByteArray): String = hex(MessageDigest.getInstance("SHA-256").digest(bytes))

    fun hex(d: ByteArray): String {
        val chars = "0123456789abcdef"
        val sb = StringBuilder(d.size * 2)
        for (b in d) {
            val v = b.toInt() and 0xff
            sb.append(chars[v ushr 4]).append(chars[v and 15])
        }
        return sb.toString()
    }

    /** Pinned digest check; a blank pin means "not pinned" (accepted, digest reported). */
    fun matches(pinned: String, actual: String): Boolean = pinned.isBlank() || pinned.trim().equals(actual.trim(), ignoreCase = true)
}

/** Raw little-endian float32 files: how the ":probe" process hands model outputs back to the main process. */
internal object FloatFiles {
    fun write(file: File, a: FloatArray) {
        val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(a)
        file.parentFile?.mkdirs()
        val tmp = File(file.path + ".tmp")
        tmp.writeBytes(bb.array())
        if (!tmp.renameTo(file)) {
            file.delete()
            if (!tmp.renameTo(file)) throw java.io.IOException("cannot rename $tmp to $file")
        }
    }

    fun read(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size % 4 == 0) { "$file: ${bytes.size} bytes is not a whole number of floats" }
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }
}

/** Write text atomically (temp file + rename), so a crash mid-write never leaves a truncated JSON file behind. */
internal fun writeTextAtomic(file: File, text: String) {
    file.parentFile?.mkdirs()
    val tmp = File(file.path + ".tmp")
    tmp.writeText(text)
    if (!tmp.renameTo(file)) {
        file.delete()
        if (!tmp.renameTo(file)) throw java.io.IOException("cannot rename $tmp to $file")
    }
}

internal object FeatureMath {
    /** twin-spec §0 L2 normalisation of each [dim]-row in place (float64 norm; an all-zero row is left unchanged). */
    fun l2NormalizeRows(v: FloatArray, dim: Int) {
        require(dim > 0 && v.size % dim == 0) { "length ${v.size} is not a multiple of dim $dim" }
        var o = 0
        while (o < v.size) {
            var s = 0.0
            for (t in o until o + dim) s += v[t].toDouble() * v[t].toDouble()
            if (s > 0.0) {
                val n = sqrt(s)
                for (t in o until o + dim) v[t] = (v[t].toDouble() / n).toFloat()
            }
            o += dim
        }
    }
}
