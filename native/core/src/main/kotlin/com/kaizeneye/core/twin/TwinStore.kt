package com.kaizeneye.core.twin

/*
 * SPEC-QUESTION: the §8 twin.json schema has no key for the geometry area factor (§7.4, 2.5); twin.json follows the
 * schema key-for-key, so a loaded Twin uses the default area factor (GeometryParams().areaFactor).
 */

import kotlinx.serialization.SerializationException
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

// ------------------------------------------------------------------------------------------------ twin.json (spec §8)

/** twin.json, key-for-key the schema of spec §8. */
@Serializable
data class TwinJson(
    val schema: Int,
    val id: String,
    val name: String,
    val createdAtMs: Long,
    val pipeline: PipelineInfo,
    val fingerprint: String,
    val teach: TeachStats,
    val keyframes: List<KeyframeInfo>,
    val bankKf: List<Int>,
    val lso: List<Double>,
    val positives: List<Double>,
    val negatives: NegativesJson,
    val thresholds: ThresholdsJson,
    val certificate: CertificateJson?,
)

@Serializable
data class NegativesJson(val count: Int, val similarities: List<Double>)

@Serializable
data class ThresholdsJson(
    val tau: Double,
    val tauTeach: Double,
    val tauFactor: Double,
    val calibrated: Boolean,
    val tauId: Double,
    val tauIdRule: String,
    val identityMargin: Double?,
    val coverageCut: Double,
    val sensitivity: Double,
    val geometry: GeometryJson,
)

/** `geometry.<φ>` = `[mean, sigma-after-floor]`. */
@Serializable
data class GeometryJson(
    val kGeo: Double,
    val meanArea: Double,
    val fill: List<Double>,
    val aspect: List<Double>,
    val hu1: List<Double>,
    val solidity: List<Double>,
)

@Serializable
data class GateCountJson(val k: Int, val n: Int, val rejections: Int, val upper: Double)

/** Calibrated certificate (layout not fixed by spec §8; see the SPEC-QUESTION in Calibration.kt). */
@Serializable
data class CertificateJson(
    val conf: Double,
    val n: Int,
    val nSanityOk: Int,
    val m: Int,
    val alpha: Double?,
    val sanity: GateCountJson,
    val identity: GateCountJson,
    val geometry: GateCountJson,
    val identityMargin: Double?,
    val bUsed: Int,
    val withinPart: Double,
    val latencyP50Ms: Double?,
    val latencyP95Ms: Double?,
    val accelerator: String?,
    val calMax: Double?,
    val tauCal: Double,
)

/** Conversions between [TwinModel] parts and the twin.json DTOs. */
object TwinJsonCodec {
    const val SCHEMA = 1

    internal val json: Json = Json {
        encodeDefaults = true
        explicitNulls = true
        prettyPrint = true
        ignoreUnknownKeys = false
    }

    fun toJson(t: TwinModel): TwinJson = TwinJson(
        schema = SCHEMA,
        id = t.id,
        name = t.name,
        createdAtMs = t.createdAtMs,
        pipeline = t.pipeline,
        fingerprint = t.fingerprint,
        teach = t.teach,
        keyframes = t.keyframes,
        bankKf = t.bankKf.toList(),
        lso = t.lso.toList(),
        positives = t.positives.toList(),
        negatives = NegativesJson(t.negativeSims.size, t.negativeSims.toList()),
        thresholds = thresholdsToJson(t.thresholds),
        certificate = t.certificate?.let(::certificateToJson),
    )

    fun encode(t: TwinModel): String = json.encodeToString(TwinJson.serializer(), toJson(t))

    fun decode(text: String): TwinJson = json.decodeFromString(TwinJson.serializer(), text)

    fun thresholdsToJson(th: Thresholds): ThresholdsJson = ThresholdsJson(
        tau = th.tau, tauTeach = th.tauTeach, tauFactor = th.tauFactor, calibrated = th.calibrated, tauId = th.tauId,
        tauIdRule = th.tauIdRule, identityMargin = th.identityMargin, coverageCut = th.coverageCut,
        sensitivity = th.sensitivity,
        geometry = GeometryJson(
            kGeo = th.geometry.kGeo, meanArea = th.geometry.meanArea,
            fill = band(th.geometry.fill), aspect = band(th.geometry.aspect), hu1 = band(th.geometry.hu1),
            solidity = band(th.geometry.solidity),
        ),
    )

    fun thresholdsFromJson(j: ThresholdsJson, areaFactor: Double = GeometryParams().areaFactor): Thresholds = Thresholds(
        tau = j.tau, tauTeach = j.tauTeach, tauFactor = j.tauFactor, calibrated = j.calibrated, tauId = j.tauId,
        tauIdRule = j.tauIdRule, identityMargin = j.identityMargin, coverageCut = j.coverageCut,
        sensitivity = j.sensitivity,
        geometry = GeometryModel(
            kGeo = j.geometry.kGeo, meanArea = j.geometry.meanArea, fill = band(j.geometry.fill, "fill"),
            aspect = band(j.geometry.aspect, "aspect"), hu1 = band(j.geometry.hu1, "hu1"),
            solidity = band(j.geometry.solidity, "solidity"), areaFactor = areaFactor,
        ),
    )

    fun certificateToJson(c: Certificate): CertificateJson = CertificateJson(
        conf = c.conf, n = c.n, nSanityOk = c.nSanityOk, m = c.m, alpha = c.alpha, sanity = gate(c.sanity),
        identity = gate(c.identity), geometry = gate(c.geometry), identityMargin = c.identityMargin, bUsed = c.bUsed,
        withinPart = c.withinPart, latencyP50Ms = c.latencyP50Ms, latencyP95Ms = c.latencyP95Ms,
        accelerator = c.accelerator, calMax = c.calMax, tauCal = c.tauCal,
    )

    fun certificateFromJson(j: CertificateJson): Certificate = Certificate(
        conf = j.conf, n = j.n, nSanityOk = j.nSanityOk, m = j.m, alpha = j.alpha, sanity = gate(j.sanity),
        identity = gate(j.identity), geometry = gate(j.geometry), identityMargin = j.identityMargin, bUsed = j.bUsed,
        withinPart = j.withinPart, latencyP50Ms = j.latencyP50Ms, latencyP95Ms = j.latencyP95Ms,
        accelerator = j.accelerator, calMax = j.calMax, tauCal = j.tauCal,
    )

    private fun band(b: FeatureBand) = listOf(b.mean, b.sigma)

    private fun band(v: List<Double>, name: String): FeatureBand {
        if (v.size != 2) throw TwinFormatException("geometry.$name must be [mean, sigma], got $v")
        return FeatureBand(v[0], v[1])
    }

    private fun gate(g: GateCount) = GateCountJson(g.k, g.n, g.rejections, g.upper)
    private fun gate(g: GateCountJson) = GateCount(g.k, g.n, g.rejections, g.upper)
}

// ------------------------------------------------------------------------------------------------------ TwinStore

/** A stored Twin is malformed (bad twin.json, a missing or mis-sized `.f16` file, ...). */
class TwinFormatException(message: String, cause: Throwable? = null) : IOException(message, cause)

/** A stored Twin was built with a different pipeline than the running one (spec §8: "Rebuild from stored crops"). */
class TwinFingerprintMismatchException(val id: String, val stored: String, val expected: String) :
    IOException("Twin $id was built with pipeline $stored, the app runs $expected — rebuild from stored crops")

/** Row of [TwinStore.list]. */
data class TwinSummary(
    val id: String,
    val name: String,
    val createdAtMs: Long,
    val calibrated: Boolean,
    val fingerprint: String,
    val keyframes: Int,
)

/** Result of [TwinStore.load]. */
sealed class TwinLoadResult {
    class Loaded(val twin: TwinModel) : TwinLoadResult()

    /** Refused: the Twin's fingerprint differs from the running pipeline's (spec §8). */
    class FingerprintMismatch(val id: String, val name: String, val stored: String, val expected: String) : TwinLoadResult()

    /** Missing or malformed files. */
    class Invalid(val id: String, val message: String) : TwinLoadResult()

    /** The Twin, or a typed exception ([TwinFingerprintMismatchException], [TwinFormatException]). */
    fun getOrThrow(): TwinModel = when (this) {
        is Loaded -> twin
        is FingerprintMismatch -> throw TwinFingerprintMismatchException(id, stored, expected)
        is Invalid -> throw TwinFormatException("Twin $id: $message")
    }
}

/**
 * Twin folders `rootDir/<id>/` (spec §8): `twin.json`, `bank.f16`, `globals.f16`, `kf_feats.f16`, `kf_cov.f16`,
 * `negatives.f16`, `keyframes/kf_NN.jpg`. A save writes `<id>.tmp/` and then renames it, so a crash never leaves a
 * half-written Twin under its real name. JPEG encoding happens on Android; this class only stores the bytes.
 */
class TwinStore(val rootDir: File) {

    companion object {
        const val TWIN_JSON = "twin.json"
        const val BANK = "bank.f16"
        const val GLOBALS = "globals.f16"
        const val KF_FEATS = "kf_feats.f16"
        const val KF_COV = "kf_cov.f16"
        const val NEGATIVES = "negatives.f16"
        const val KEYFRAMES_DIR = "keyframes"
        private const val TMP = ".tmp"
        private const val OLD = ".old"
        /** Folder-safe ids: no path separators, never "." / ".." (first character is not a dot). */
        private val ID_OK = Regex("[A-Za-z0-9_-][A-Za-z0-9._-]*")
    }

    /**
     * Saves [twin] (atomically: `<id>.tmp/` then rename; an existing Twin with the same id is replaced).
     * [keyframeJpegs] are the `N×N` keyframe crops as JPEG bytes, one per keyframe; null keeps the JPEGs of an already
     * stored Twin with this id (e.g. re-saving after calibration). Returns the Twin folder.
     */
    fun save(twin: TwinModel, keyframeJpegs: List<ByteArray>?): File {
        checkId(twin.id)
        rootDir.mkdirs()
        val dst = File(rootDir, twin.id)
        val tmp = File(rootDir, twin.id + TMP)
        if (tmp.exists()) tmp.deleteRecursively()
        if (!tmp.mkdirs()) throw IOException("cannot create $tmp")
        try {
            File(tmp, TWIN_JSON).writeText(TwinJsonCodec.encode(twin), Charsets.UTF_8)
            val k = twin.keyframeCount
            val d = twin.dim
            F16File.write(File(tmp, BANK), twin.bankRows, d, twin.bank)
            F16File.write(File(tmp, GLOBALS), k, d, twin.globals)
            F16File.write(File(tmp, KF_FEATS), k, twin.patches * d, twin.kfFeats)
            F16File.write(File(tmp, KF_COV), k, twin.patches, twin.kfCov)
            F16File.write(File(tmp, NEGATIVES), twin.negativeCount, d, twin.negatives)
            val kfDir = File(tmp, KEYFRAMES_DIR).also { it.mkdirs() }
            if (keyframeJpegs != null) {
                require(keyframeJpegs.size == k) { "need $k keyframe JPEGs, got ${keyframeJpegs.size}" }
                twin.keyframes.forEachIndexed { i, kf -> File(tmp, kf.file).writeBytes(keyframeJpegs[i]) }
            } else {
                val old = File(dst, KEYFRAMES_DIR)
                if (old.isDirectory) old.copyRecursively(kfDir, overwrite = true)
            }
            if (dst.exists()) {
                val old = File(rootDir, twin.id + OLD)
                if (old.exists()) old.deleteRecursively()
                move(dst, old)
                move(tmp, dst)
                old.deleteRecursively()
            } else {
                move(tmp, dst)
            }
        } catch (e: Throwable) {
            tmp.deleteRecursively()
            throw e
        }
        return dst
    }

    /**
     * Loads Twin [id]. When [running] is given, a Twin built with a different pipeline fingerprint is refused
     * ([TwinLoadResult.FingerprintMismatch]). Every `.f16` file is checked (magic, header, size, shape vs twin.json).
     */
    fun load(id: String, running: PipelineInfo? = null): TwinLoadResult {
        if (!ID_OK.matches(id)) return TwinLoadResult.Invalid(id, "bad id")
        val dir = File(rootDir, id)
        if (!dir.isDirectory) return TwinLoadResult.Invalid(id, "no such Twin")
        val j = try {
            readJson(dir)
        } catch (e: IOException) {
            return TwinLoadResult.Invalid(id, e.message ?: "unreadable twin.json")
        }
        if (running != null) {
            val expected = running.fingerprint()
            if (expected != j.fingerprint) return TwinLoadResult.FingerprintMismatch(id, j.name, j.fingerprint, expected)
        }
        return try {
            TwinLoadResult.Loaded(assemble(dir, j))
        } catch (e: IOException) {
            TwinLoadResult.Invalid(id, e.message ?: e.toString())
        } catch (e: IllegalArgumentException) {
            TwinLoadResult.Invalid(id, e.message ?: e.toString())
        }
    }

    /** Stored Twins, newest first (stale `.tmp` folders are removed first). Unreadable folders are skipped. */
    fun list(): List<TwinSummary> {
        cleanTemp()
        val dirs = rootDir.listFiles { f -> f.isDirectory && ID_OK.matches(f.name) } ?: return emptyList()
        val out = ArrayList<TwinSummary>()
        for (d in dirs) {
            val j = try {
                readJson(d)
            } catch (e: IOException) {
                continue
            }
            out.add(TwinSummary(j.id, j.name, j.createdAtMs, j.thresholds.calibrated, j.fingerprint, j.keyframes.size))
        }
        return out.sortedWith(compareByDescending<TwinSummary> { it.createdAtMs }.thenBy { it.id })
    }

    /** Deletes Twin [id]; returns false when it did not exist. */
    fun delete(id: String): Boolean {
        checkId(id)
        val dir = File(rootDir, id)
        if (!dir.exists()) return false
        return dir.deleteRecursively()
    }

    /** Copies Twin [id]'s folder to `destDir/<id>/` (backup / export); returns the copy. */
    fun exportTo(id: String, destDir: File): File {
        checkId(id)
        val src = File(rootDir, id)
        if (!src.isDirectory) throw IOException("no such Twin $id")
        val dst = File(destDir, id)
        if (dst.exists()) dst.deleteRecursively()
        src.copyRecursively(dst, overwrite = true)
        return dst
    }

    /** Removes leftover `<id>.tmp` / `<id>.old` folders of interrupted saves; returns how many were removed. */
    fun cleanTemp(): Int {
        val stale = rootDir.listFiles { f -> f.isDirectory && (f.name.endsWith(TMP) || f.name.endsWith(OLD)) } ?: return 0
        var n = 0
        for (f in stale) if (f.deleteRecursively()) n++
        return n
    }

    /** Path of a keyframe JPEG of a stored Twin (for "rebuild from stored crops"). */
    fun keyframeFile(id: String, index: Int): File = File(File(rootDir, id), KeyframeInfo.fileName(index))

    private fun readJson(dir: File): TwinJson {
        val f = File(dir, TWIN_JSON)
        if (!f.isFile) throw TwinFormatException("missing $TWIN_JSON")
        val j = try {
            TwinJsonCodec.decode(f.readText(Charsets.UTF_8))
        } catch (e: SerializationException) {
            throw TwinFormatException("bad $TWIN_JSON: ${e.message}", e)
        } catch (e: IllegalArgumentException) {
            throw TwinFormatException("bad $TWIN_JSON: ${e.message}", e)
        }
        if (j.schema != TwinJsonCodec.SCHEMA) throw TwinFormatException("unsupported schema ${j.schema}")
        if (j.id != dir.name) throw TwinFormatException("twin.json id ${j.id} does not match folder ${dir.name}")
        if (j.pipeline.fingerprint() != j.fingerprint) {
            throw TwinFormatException("stored fingerprint does not match the stored pipeline (file edited?)")
        }
        return j
    }

    private fun assemble(dir: File, j: TwinJson): TwinModel {
        val p = j.pipeline
        val d = p.dim
        val np = p.gh * p.gw
        val k = j.keyframes.size
        val m = j.bankKf.size
        if (j.negatives.count != j.negatives.similarities.size) throw TwinFormatException("negatives.count mismatch")
        j.keyframes.forEachIndexed { i, kf -> if (kf.index != i) throw TwinFormatException("keyframe $i has index ${kf.index}") }
        if (j.bankKf.any { it !in 0 until k }) throw TwinFormatException("bankKf out of range")
        val bank = readF16(dir, BANK, m, d)
        val globals = readF16(dir, GLOBALS, k, d)
        val kfFeats = readF16(dir, KF_FEATS, k, np * d)
        val kfCov = readF16(dir, KF_COV, k, np)
        val negatives = readF16(dir, NEGATIVES, j.negatives.count, d)
        val bankKf = j.bankKf.toIntArray()
        return TwinModel(
            id = j.id,
            name = j.name,
            createdAtMs = j.createdAtMs,
            pipeline = p,
            fingerprint = j.fingerprint,
            teach = j.teach,
            keyframes = j.keyframes,
            globals = globals,
            bank = bank,
            bankKf = bankKf,
            bankSeg = IntArray(m) { j.keyframes[bankKf[it]].segment },
            kfFeats = kfFeats,
            kfCov = kfCov,
            lso = j.lso.toDoubleArray(),
            positives = j.positives.toDoubleArray(),
            negativeSims = j.negatives.similarities.toDoubleArray(),
            negatives = negatives,
            thresholds = TwinJsonCodec.thresholdsFromJson(j.thresholds),
            certificate = j.certificate?.let(TwinJsonCodec::certificateFromJson),
        )
    }

    private fun readF16(dir: File, name: String, rows: Int, cols: Int): FloatArray {
        val f = File(dir, name)
        if (!f.isFile) throw TwinFormatException("missing $name")
        val mtx = F16File.read(f)
        if (mtx.rows != rows || mtx.cols != cols) {
            throw TwinFormatException("$name is ${mtx.rows} x ${mtx.cols}, twin.json needs $rows x $cols")
        }
        return mtx.data
    }

    private fun checkId(id: String) = require(ID_OK.matches(id) && !id.endsWith(TMP) && !id.endsWith(OLD)) { "bad Twin id '$id'" }

    private fun move(from: File, to: File) {
        try {
            Files.move(from.toPath(), to.toPath(), StandardCopyOption.ATOMIC_MOVE)
        } catch (e: AtomicMoveNotSupportedException) {
            Files.move(from.toPath(), to.toPath())
        }
    }
}
