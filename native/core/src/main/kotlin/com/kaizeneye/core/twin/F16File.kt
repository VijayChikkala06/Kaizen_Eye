package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Half
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream

/** Thrown when a `.f16` file is malformed (magic, header or size). */
class F16FormatException(message: String) : IOException(message)

/**
 * The `.f16` matrix format of spec §8: 8-byte ASCII magic `KZF16v1\n`, int32 LE `rows`, int32 LE `cols`, then
 * `rows·cols` binary16 LE values (round-to-nearest-even).
 */
object F16File {
    val MAGIC: ByteArray = "KZF16v1\n".toByteArray(Charsets.US_ASCII)
    const val HEADER_BYTES = 16
    private const val CHUNK = 32768

    /** A decoded matrix `[rows, cols]` (float32 values, each exactly a binary16). */
    class Matrix(val rows: Int, val cols: Int, val data: FloatArray)

    /** Whole-file bytes (small matrices / tests). */
    fun encode(rows: Int, cols: Int, values: FloatArray): ByteArray {
        val out = java.io.ByteArrayOutputStream(HEADER_BYTES + 2 * rows * cols)
        write(out, rows, cols, values)
        return out.toByteArray()
    }

    fun decode(bytes: ByteArray): Matrix = read(bytes.inputStream(), bytes.size.toLong())

    fun write(file: File, rows: Int, cols: Int, values: FloatArray) {
        BufferedOutputStream(FileOutputStream(file), 1 shl 16).use { write(it, rows, cols, values) }
    }

    fun read(file: File): Matrix = BufferedInputStream(FileInputStream(file), 1 shl 16).use { read(it, file.length()) }

    /** Streams the header and values of `[rows, cols]` [values] to [out] (not closed). */
    fun write(out: OutputStream, rows: Int, cols: Int, values: FloatArray) {
        require(rows >= 0 && cols >= 0) { "negative shape" }
        require(values.size.toLong() == rows.toLong() * cols) { "values has ${values.size} floats, shape is $rows x $cols" }
        out.write(MAGIC)
        out.write(le32(rows))
        out.write(le32(cols))
        val buf = ByteArray(2 * CHUNK)
        var i = 0
        while (i < values.size) {
            val n = minOf(CHUNK, values.size - i)
            Half.encodeLE(values, i, n, buf, 0)
            out.write(buf, 0, 2 * n)
            i += n
        }
    }

    /**
     * Reads a matrix from [input]; [totalBytes] (the file length, or -1 if unknown) is checked against the header so a
     * truncated or padded file is refused.
     */
    fun read(input: InputStream, totalBytes: Long = -1): Matrix {
        val din = DataInputStream(input)
        val head = ByteArray(HEADER_BYTES)
        try {
            din.readFully(head)
        } catch (e: java.io.EOFException) {
            throw F16FormatException("file shorter than the 16-byte header")
        }
        for (i in MAGIC.indices) if (head[i] != MAGIC[i]) throw F16FormatException("bad magic")
        val rows = readLe32(head, 8)
        val cols = readLe32(head, 12)
        if (rows < 0 || cols < 0) throw F16FormatException("negative shape $rows x $cols")
        val count = rows.toLong() * cols
        if (count > Int.MAX_VALUE) throw F16FormatException("matrix too large: $rows x $cols")
        if (totalBytes >= 0 && totalBytes != HEADER_BYTES + 2 * count) {
            throw F16FormatException("size $totalBytes bytes, header says ${HEADER_BYTES + 2 * count} ($rows x $cols)")
        }
        val data = FloatArray(count.toInt())
        val buf = ByteArray(2 * CHUNK)
        var i = 0
        try {
            while (i < data.size) {
                val n = minOf(CHUNK, data.size - i)
                din.readFully(buf, 0, 2 * n)
                Half.decodeLE(buf, 0, n, data, i)
                i += n
            }
        } catch (e: java.io.EOFException) {
            throw F16FormatException("file truncated after $i of ${data.size} values")
        }
        if (totalBytes < 0 && din.read() != -1) throw F16FormatException("trailing bytes after $rows x $cols values")
        return Matrix(rows, cols, data)
    }

    private fun le32(v: Int) = byteArrayOf(v.toByte(), (v ushr 8).toByte(), (v ushr 16).toByte(), (v ushr 24).toByte())

    private fun readLe32(b: ByteArray, o: Int): Int =
        (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or ((b[o + 2].toInt() and 0xFF) shl 16) or
            ((b[o + 3].toInt() and 0xFF) shl 24)
}
