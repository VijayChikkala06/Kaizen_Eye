package com.kaizeneye.runtime

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files
import kotlin.math.sqrt

class UtilTest {
    @Test
    fun sha256KnownVectorsAndPins() {
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", Sha256.of("abc".toByteArray()))
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Sha256.of(ByteArray(0)))
        val streamed = Sha256.of(ByteArray(3_000_001) { (it % 251).toByte() }.inputStream(), bufferSize = 4096)
        assertEquals(Sha256.of(ByteArray(3_000_001) { (it % 251).toByte() }), streamed.hex)
        assertEquals(3_000_001L, streamed.bytes)
        assertTrue(Sha256.matches("0EB6860D5C3800DC02C86515A54387F0D8289E5096DB291FF6E4A993E312B482", "0eb6860d5c3800dc02c86515a54387f0d8289e5096db291ff6e4a993e312b482"))
        assertTrue(Sha256.matches("", "anything"))                 // not pinned
        assertFalse(Sha256.matches("abc", "abd"))
    }

    @Test
    fun floatFilesRoundTrip() {
        val dir = Files.createTempDirectory("kz-ff").toFile()
        try {
            val a = FloatArray(1000) { it * 0.5f - 7f }.also { it[3] = Float.NaN; it[4] = -0f; it[5] = 1e30f }
            val f = File(dir, "sub/x.f32")
            FloatFiles.write(f, a)
            val b = FloatFiles.read(f)
            assertEquals(a.size, b.size)
            for (i in a.indices) assertEquals(a[i].toRawBits(), b[i].toRawBits())
            assertEquals(4000L, f.length())
            FloatFiles.write(f, floatArrayOf(1f))                  // replace
            assertArrayEquals(floatArrayOf(1f), FloatFiles.read(f), 0f)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun atomicTextWrite() {
        val dir = Files.createTempDirectory("kz-at").toFile()
        try {
            val f = File(dir, "a/b.json")
            writeTextAtomic(f, "{\"x\":1}")
            writeTextAtomic(f, "{\"x\":2}")
            assertEquals("{\"x\":2}", f.readText())
            assertFalse(File(dir, "a/b.json.tmp").exists())
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun l2NormalisationPerPatch() {
        val v = floatArrayOf(3f, 4f, 0f, 0f, 0f, 0f, 1f, 1f, 1f)
        FeatureMath.l2NormalizeRows(v, 3)
        assertArrayEquals(floatArrayOf(0.6f, 0.8f, 0f), v.copyOfRange(0, 3), 1e-7f)
        assertArrayEquals(floatArrayOf(0f, 0f, 0f), v.copyOfRange(3, 6), 0f)       // zero row unchanged
        val r = (1.0 / sqrt(3.0)).toFloat()
        assertArrayEquals(floatArrayOf(r, r, r), v.copyOfRange(6, 9), 0f)
    }

    @Test
    fun briefErrorText() {
        val e = IllegalStateException("outer", RuntimeException("inner"))
        assertEquals("IllegalStateException: outer (cause RuntimeException: inner)", e.brief())
        assertTrue(RuntimeException("x".repeat(1000)).brief().length <= 401)
        assertEquals("a; b", joinNotes("a", null, " ", "b"))
        assertEquals(null, joinNotes(null, ""))
    }
}
