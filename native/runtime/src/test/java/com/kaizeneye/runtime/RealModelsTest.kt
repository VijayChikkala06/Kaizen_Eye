package com.kaizeneye.runtime

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile
import java.nio.channels.FileChannel

/**
 * Reads the real model files (bundled app assets: -Dmodels.dir; DINOv2 export: -Ddinov2.dir). Verifies the flatbuffer
 * reader against the layouts reported by ai_edge_litert and the SHA-256 pins in KnownModels.
 */
class RealModelsTest {
    private fun file(prop: String, name: String): File {
        val dir = System.getProperty(prop).orEmpty()
        val f = File(dir, name)
        assumeTrue("$name not available in '$dir'", dir.isNotEmpty() && f.isFile)
        return f
    }

    private fun parse(f: File): TfliteInfo = RandomAccessFile(f, "r").use { raf ->
        TfliteInfo.parse(raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length()))
    }

    @Test
    fun knnGraphLayout() {
        for (k in listOf(800, 1600, 2400)) {
            val info = parse(file("models.dir", "knn_p1600_d128_k$k.tflite"))
            assertEquals(listOf("feats", "bankT", "bankNorm"), info.inputs.map { it.name })
            assertArrayEquals(intArrayOf(1600, 128), info.inputs[0].shape)
            assertArrayEquals(intArrayOf(128, k), info.inputs[1].shape)
            assertArrayEquals(intArrayOf(k), info.inputs[2].shape)
            assertEquals(1, info.outputs.size)
            assertEquals("Identity", info.outputs[0].name)
            assertArrayEquals(intArrayOf(1600), info.outputs[0].shape)
            assertTrue(info.signatures.isEmpty())
            assertTrue((info.inputs + info.outputs).all { it.type == TfliteInfo.FLOAT32 })
            // identification by shape (k = 1600 makes feats and bankT the same SIZE but not the same shape)
            assertEquals("feats", info.inputWithShape(1600, 128)?.name)
            assertEquals("bankT", info.inputWithShape(128, k)?.name)
            assertEquals("bankNorm", info.inputWithShape(k)?.name)
            assertEquals("bankT", info.signatureInputName(info.inputWithShape(128, k)!!))   // no signature -> tensor name
        }
    }

    @Test
    fun backboneLayoutAndSignature() {
        for (name in listOf("backbone_r18_320.tflite", "backbone_r18_320_int8.tflite")) {
            val info = parse(file("models.dir", name))
            assertEquals(1, info.inputs.size)
            assertArrayEquals(intArrayOf(1, 320, 320, 3), info.inputs[0].shape)
            assertEquals(TfliteInfo.FLOAT32, info.inputs[0].type)            // float I/O also for the int8 model
            assertEquals("serving_default_image:0", info.inputs[0].name)
            assertArrayEquals(intArrayOf(1, 40, 40, 128), info.outputs[0].shape)
            val sig = info.signatures.single()
            assertEquals("serving_default", sig.key)
            assertEquals("image", info.signatureInputName(info.inputs[0]))
        }
    }

    @Test
    fun dinov2OutputsAreFoundBySize() {
        for (name in listOf("dinov2_s14_448_fp16w.tflite", "dinov2_s14_448_fp32.tflite")) {
            val info = parse(file("dinov2.dir", name))
            assertArrayEquals(intArrayOf(1, 448, 448, 3), info.inputs.single().shape)
            val sizes = info.outputs.map { it.elements.toInt() }
            assertEquals(2, sizes.size)
            assertNotNull(info.outputWithShape(1, 32, 32, 384))
            assertNotNull(info.outputWithShape(1, 384))
            val (pi, ci) = OutputPick.pick(sizes, 32 * 32 * 384, 384)
            assertEquals(32 * 32 * 384, sizes[pi])
            assertEquals(384, sizes[ci])
        }
        val knn = parse(file("dinov2.dir", "knn_p1024_d384_k2400.tflite"))
        assertNotNull(knn.inputWithShape(1024, 384))
        assertNotNull(knn.inputWithShape(384, 2400))
        assertNotNull(knn.inputWithShape(2400))
        assertNotNull(knn.outputWithShape(1024))
    }

    @Test
    fun shaPinsMatchTheRealFiles() {
        val checked = ArrayList<String>()
        fun check(prop: String, a: ModelAsset) {
            val f = File(System.getProperty(prop).orEmpty(), a.fileName)
            if (!f.isFile) return
            assertTrue("${a.fileName} pin", Sha256.matches(a.sha256, Sha256.of(f).hex))
            checked += a.fileName
        }
        assumeTrue(File(System.getProperty("models.dir").orEmpty()).isDirectory)
        check("models.dir", KnownModels.R18_320_F32)
        check("models.dir", KnownModels.R18_320_INT8)
        for (k in listOf(800, 1600, 2400)) check("models.dir", KnownModels.knnGraph(1600, 128, k))
        assertEquals("all bundled R18 files present", 5, checked.size)
        if (File(System.getProperty("dinov2.dir").orEmpty()).isDirectory) {
            check("dinov2.dir", KnownModels.DINOV2_S14_448_FP16W)
            check("dinov2.dir", KnownModels.DINOV2_S14_448_FP32)
            check("dinov2.dir", KnownModels.knnGraph(1024, 384, 2400))
        }
        println("SHA-256 pins verified: $checked")
        assertEquals("", KnownModels.knnGraph(999, 3, 5).sha256)              // unknown graph: unpinned
    }

    @Test(expected = IllegalArgumentException::class)
    fun garbageIsRejectedCleanly() {
        TfliteInfo.parse(ByteArray(64) { (it * 37).toByte() })
    }

    @Test(expected = IllegalArgumentException::class)
    fun truncatedModelIsRejectedCleanly() {
        val f = file("models.dir", "knn_p1600_d128_k800.tflite")
        TfliteInfo.parse(f.readBytes().copyOf(16))
    }
}
