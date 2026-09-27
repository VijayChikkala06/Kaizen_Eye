package com.kaizeneye.core.twin

import com.kaizeneye.core.math.Half
import com.kaizeneye.core.model.FeatureMap
import com.kaizeneye.core.model.GeometryFeatures
import com.kaizeneye.core.model.SanityReason
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.Random

/** Pipeline fingerprint (spec §8), `.f16` codec, TwinStore and NegativesStore. */
class StoreTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private val specPipeline = PipelineInfo(
        backboneId = "r18_320_f32", backboneSha256 = "abc", inputSize = 320, gh = 40, gw = 40, dim = 128,
        knnGraphId = "knn_p1600_d128",
    )

    @Test
    fun canonicalPipelineJson() {
        assertEquals(
            """{"backboneId":"r18_320_f32","backboneSha256":"abc","bankRule":"CORESET","dim":128,"gh":40,"gw":40,""" +
                """"inputSize":320,"knnGraphId":"knn_p1600_d128","l2NormalizePatches":false,"mask":{"analysisFactor":4,""" +
                """"borderMargin":2,"coreThreshold":0.5,"cropMargin":0.1,"kSigma":4.0,"minBlobPx":30,"sigmaMin":3.0},""" +
                """"precision":"fp32","preprocessVersion":1,"rotation":"NONE","scoreRule":"SMOOTHED_MAX"}""",
            specPipeline.canonicalJson(),
        )
        assertEquals(CanonicalJson.sha256Hex(specPipeline.canonicalJson()), specPipeline.fingerprint())
        assertEquals(64, specPipeline.fingerprint().length)
        assertTrue(specPipeline.fingerprint() != specPipeline.copy(dim = 384).fingerprint())
        // Known SHA-256 vector.
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", CanonicalJson.sha256Hex("abc"))
    }

    @Test
    fun shortestRoundTripNumbers() {
        val cases = mapOf(
            0.1 to "0.1", 4.0 to "4.0", 0.5 to "0.5", -2.5 to "-2.5", 1e-5 to "1e-05", 0.0001 to "0.0001",
            1e16 to "1e+16", 1e15 to "1000000000000000.0", 123456.789 to "123456.789", 0.30000000000000004 to "0.30000000000000004",
            1.5e300 to "1.5e+300", 5e-324 to "5e-324", 0.0 to "0.0", 2.0 / 3 to "0.6666666666666666",
        )
        for ((v, s) in cases) assertEquals("$v", s, CanonicalJson.formatDouble(v))
        assertEquals("4", CanonicalJson.formatNumber("4"))
        assertEquals("4.0", CanonicalJson.formatNumber("4.0"))
        assertEquals("0.1", CanonicalJson.formatNumber("1.0E-1"))
    }

    @Test
    fun f16FileLayout() {
        val bytes = F16File.encode(1, 2, floatArrayOf(1.0f, -2.0f))
        val hex = bytes.joinToString("") { "%02x".format(it) }
        assertEquals("4b5a46313676310a" + "01000000" + "02000000" + "003c" + "00c0", hex)
        val m = F16File.decode(bytes)
        assertEquals(1, m.rows)
        assertEquals(2, m.cols)
        assertArrayEquals(floatArrayOf(1.0f, -2.0f), m.data, 0f)
        // Refused: wrong magic, truncated, padded.
        for (bad in listOf(bytes.copyOf().also { it[0] = 'X'.code.toByte() }, bytes.copyOf(bytes.size - 1), bytes + byteArrayOf(0))) {
            try {
                F16File.decode(bad)
                fail("accepted a malformed file")
            } catch (e: F16FormatException) {
                // expected
            }
        }
        // Zero rows are fine.
        val empty = F16File.decode(F16File.encode(0, 128, FloatArray(0)))
        assertEquals(0, empty.rows)
        assertEquals(128, empty.cols)
    }

    private val gh = 8
    private val gw = 8
    private val dim = 16
    private val pipeline = Synth.pipeline(gh, gw, dim)

    private fun twin(): TwinModel {
        val frames = Synth.teachFrames(40, gh, gw, dim, seed = 31)
        val negs = List(3) { i -> Descriptor.l2NormalizeInPlace(DoubleArray(dim) { (it * (i + 1) % 7).toDouble() }).map { it.toFloat() }.toFloatArray() }
        return (TeachBuilder.build(frames, pipeline, Synth.meta("M8 nut"), negatives = negs) as TeachResult.Success).twin
    }

    private fun assertSameTwin(a: TwinModel, b: TwinModel) {
        assertEquals(a.id, b.id)
        assertEquals(a.name, b.name)
        assertEquals(a.createdAtMs, b.createdAtMs)
        assertEquals(a.pipeline, b.pipeline)
        assertEquals(a.fingerprint, b.fingerprint)
        assertEquals(a.teach, b.teach)
        assertEquals(a.keyframes, b.keyframes)
        assertArrayEquals(a.globals, b.globals, 0f)
        assertArrayEquals(a.bank, b.bank, 0f)
        assertArrayEquals(a.bankKf, b.bankKf)
        assertArrayEquals(a.bankSeg, b.bankSeg)
        assertArrayEquals(a.kfFeats, b.kfFeats, 0f)
        assertArrayEquals(a.kfCov, b.kfCov, 0f)
        assertArrayEquals(a.lso, b.lso, 0.0)
        assertArrayEquals(a.positives, b.positives, 0.0)
        assertArrayEquals(a.negativeSims, b.negativeSims, 0.0)
        assertArrayEquals(a.negatives, b.negatives, 0f)
        assertEquals(a.thresholds, b.thresholds)
        assertEquals(a.certificate, b.certificate)
    }

    @Test
    fun saveLoadRoundTripIsExact() {
        val store = TwinStore(tmp.newFolder("twins"))
        val t = twin()
        val jpegs = List(t.keyframeCount) { byteArrayOf(0xFF.toByte(), 0xD8.toByte(), it.toByte()) }
        val dir = store.save(t, jpegs)
        assertTrue(File(dir, "keyframes/kf_00.jpg").isFile)
        assertArrayEquals(jpegs[1], File(dir, "keyframes/kf_01.jpg").readBytes())
        val back = store.load(t.id, pipeline).getOrThrow()
        assertSameTwin(t, back)
        // Same judgement from the reloaded Twin.
        val rnd = Random(32)
        val q = FeatureMap(gh, gw, dim, FloatArray(gh * gw * dim) { rnd.nextGaussian().toFloat() })
        val cov = Synth.discCov(gh, gw, 4.0, 4.0, 2.0)
        val geo = GeometryFeatures(1500.0, 0.62, 0.9, 0.16, 0.95)
        val a = t.judge(q, cov, geo, SanityReason.OK)
        val b = back.judge(q, cov, geo, SanityReason.OK)
        assertEquals(a.copy(smoothed = null, dmap = null), b.copy(smoothed = null, dmap = null))
        assertArrayEquals(a.dmap, b.dmap, 0.0)

        // Calibrate, re-save without JPEGs (kept), reload: certificate and thresholds survive.
        val (cal, _) = t.calibrate(List(5) { CalibrationSample(true, true, true, 0.5 + it, 0.95, geo) }, latenciesMs = doubleArrayOf(9.0, 11.0), accelerator = "CPU")
        store.save(cal.withSensitivity(1.3), null)
        val back2 = store.load(t.id).getOrThrow()
        assertSameTwin(cal.withSensitivity(1.3), back2)
        assertTrue(File(store.rootDir, t.id + "/keyframes/kf_01.jpg").isFile)
        assertFalse(File(store.rootDir, t.id + ".tmp").exists())
        assertFalse(File(store.rootDir, t.id + ".old").exists())
    }

    @Test
    fun twinJsonMatchesTheSchemaKeyForKey() {
        val store = TwinStore(tmp.newFolder("twins"))
        val t = twin()
        val dir = store.save(t, List(t.keyframeCount) { ByteArray(1) })
        val j = Json.parseToJsonElement(File(dir, "twin.json").readText()).jsonObject
        fun keys(o: JsonObject) = o.keys.toList()
        assertEquals(
            listOf("schema", "id", "name", "createdAtMs", "pipeline", "fingerprint", "teach", "keyframes", "bankKf", "lso",
                "positives", "negatives", "thresholds", "certificate"),
            keys(j),
        )
        assertEquals(
            listOf("backboneId", "backboneSha256", "inputSize", "gh", "gw", "dim", "precision", "l2NormalizePatches",
                "preprocessVersion", "mask", "knnGraphId", "scoreRule", "bankRule", "rotation"),
            keys(j.getValue("pipeline").jsonObject),
        )
        assertEquals(
            listOf("analysisFactor", "kSigma", "sigmaMin", "minBlobPx", "borderMargin", "cropMargin", "coreThreshold"),
            keys(j.getValue("pipeline").jsonObject.getValue("mask").jsonObject),
        )
        assertEquals(
            listOf("framesSeen", "framesAccepted", "framesKept", "keyframes", "segmentsUsed", "durationMs", "bankRows", "pooledRows"),
            keys(j.getValue("teach").jsonObject),
        )
        assertEquals(listOf("index", "tMs", "segment", "sharpness", "file"), keys(j.getValue("keyframes").jsonArray[0].jsonObject))
        assertEquals(listOf("count", "similarities"), keys(j.getValue("negatives").jsonObject))
        assertEquals(
            listOf("tau", "tauTeach", "tauFactor", "calibrated", "tauId", "tauIdRule", "identityMargin", "coverageCut",
                "sensitivity", "geometry"),
            keys(j.getValue("thresholds").jsonObject),
        )
        assertEquals(
            listOf("kGeo", "meanArea", "fill", "aspect", "hu1", "solidity"),
            keys(j.getValue("thresholds").jsonObject.getValue("geometry").jsonObject),
        )
        assertEquals("null", j.getValue("certificate").toString())
        assertEquals(t.fingerprint, (j.getValue("fingerprint") as kotlinx.serialization.json.JsonPrimitive).content)
    }

    @Test
    fun fingerprintMismatchIsRefused() {
        val store = TwinStore(tmp.newFolder("twins"))
        val t = twin()
        store.save(t, List(t.keyframeCount) { ByteArray(1) })
        val other = pipeline.copy(backboneSha256 = "1".repeat(64))
        val r = store.load(t.id, other)
        assertTrue(r is TwinLoadResult.FingerprintMismatch)
        r as TwinLoadResult.FingerprintMismatch
        assertEquals(t.fingerprint, r.stored)
        assertEquals(other.fingerprint(), r.expected)
        try {
            r.getOrThrow()
            fail("mismatch not refused")
        } catch (e: TwinFingerprintMismatchException) {
            assertEquals(t.id, e.id)
        }
        // An edited pipeline inside twin.json (fingerprint no longer matches it) is refused as invalid.
        val f = File(store.rootDir, t.id + "/twin.json")
        f.writeText(f.readText().replace("\"precision\": \"fp32\"", "\"precision\": \"fp16\""))
        assertTrue(store.load(t.id) is TwinLoadResult.Invalid)
    }

    @Test
    fun corruptOrMissingFilesAreRefused() {
        val store = TwinStore(tmp.newFolder("twins"))
        val t = twin()
        store.save(t, List(t.keyframeCount) { ByteArray(1) })
        val bank = File(store.rootDir, t.id + "/bank.f16")
        val good = bank.readBytes()
        bank.writeBytes(good.copyOf(good.size - 2))
        assertTrue(store.load(t.id) is TwinLoadResult.Invalid)
        bank.writeBytes(F16File.encode(t.bankRows - 1, t.dim, t.bank.copyOf((t.bankRows - 1) * t.dim)))
        val r = store.load(t.id)
        assertTrue(r is TwinLoadResult.Invalid && r.message.contains("bank.f16"))
        bank.writeBytes(good)
        assertTrue(store.load(t.id) is TwinLoadResult.Loaded)
        File(store.rootDir, t.id + "/kf_cov.f16").delete()
        assertTrue(store.load(t.id) is TwinLoadResult.Invalid)
        assertTrue(store.load("nope") is TwinLoadResult.Invalid)
        assertTrue(store.load("..") is TwinLoadResult.Invalid)
    }

    @Test
    fun listDeleteExportAndTempCleanup() {
        val root = tmp.newFolder("twins")
        val store = TwinStore(root)
        val t = twin()
        val t2 = t.copy(id = "20260101-000000-0001", name = "washer")
        store.save(t, List(t.keyframeCount) { ByteArray(1) })
        store.save(t2, List(t.keyframeCount) { ByteArray(1) })
        File(root, "20260101-000000-dead.tmp/keyframes").mkdirs()        // an interrupted save
        File(root, "junk").mkdirs()                                         // not a Twin
        val list = store.list()
        assertEquals(setOf(t.id, t2.id), list.map { it.id }.toSet())
        assertFalse(File(root, "20260101-000000-dead.tmp").exists())
        assertTrue(list.all { !it.calibrated })
        assertEquals("M8 nut", list.first { it.id == t.id }.name)
        val exported = store.exportTo(t.id, tmp.newFolder("backup"))
        assertTrue(File(exported, "twin.json").isFile)
        assertTrue(TwinStore(exported.parentFile).load(t.id).getOrThrow().bankRows == t.bankRows)
        assertTrue(store.delete(t2.id))
        assertFalse(store.delete(t2.id))
        assertEquals(listOf(t.id), store.list().map { it.id })
    }

    @Test
    fun negativesStore() {
        val store = NegativesStore(tmp.newFolder("negatives"))
        val g = floatArrayOf(0.1f, 0.2f, 0.3f, 0.927f)
        val a = store.add("fpA", g, byteArrayOf(9, 9), createdAtMs = 1000, id = "neg-a")
        store.add("fpB", g, null, createdAtMs = 2000, id = "neg-b")
        store.add("fpA", g, null, createdAtMs = 3000, id = "neg-c")
        assertArrayEquals(Half.roundedCopy(g), a.global, 0f)
        assertTrue(a.hasCrop)
        assertEquals(listOf("neg-a", "neg-c"), store.list("fpA").map { it.id })
        assertEquals(3, store.list().size)
        assertEquals(2, store.globals("fpA").size)
        assertArrayEquals(byteArrayOf(9, 9), store.crop("neg-a"))
        assertNull(store.crop("neg-c"))
        assertTrue(store.delete("neg-a"))
        assertFalse(store.delete("neg-a"))
        assertEquals(listOf("neg-c"), store.list("fpA").map { it.id })
    }
}
