package com.kaizeneye.core.twin

import kotlinx.serialization.SerializationException
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Random

/**
 * One captured wrong object of the negatives library (spec §7.3): its global descriptor (binary16 values) under the
 * pipeline [fingerprint], optionally with the crop JPEG it came from.
 */
class NegativeEntry(
    val id: String,
    val createdAtMs: Long,
    val fingerprint: String,
    /** `[dim]`, every value exactly a binary16. */
    val global: FloatArray,
    val hasCrop: Boolean,
)

/**
 * A negative's backbone feature map (`[patches · dim]`, binary16 values) with its patch coverage (`[patches]`): what the FIT gate
 * needs to score the negative against any Twin's bank (one map per orientation of a canonical crop).
 */
class NegativeMap(val features: FloatArray, val cov: FloatArray)

@Serializable
private data class NegativeJson(
    val id: String,
    val createdAtMs: Long,
    val fingerprint: String,
    val dim: Int,
    val crop: String?,
)

/**
 * The negatives library: `rootDir/<id>/negative.json` + `global.f16` (`[1, dim]`) + optional `crop.jpg`. Globals are
 * per pipeline fingerprint (a different backbone makes them meaningless). Writes go to `<id>.tmp/` then rename.
 */
class NegativesStore(val rootDir: File) {

    companion object {
        private const val META = "negative.json"
        private const val GLOBAL = "global.f16"
        private const val CROP = "crop.jpg"
        private const val TMP = ".tmp"
        private val ID_OK = Regex("[A-Za-z0-9_-][A-Za-z0-9._-]*")
        private val json = Json { encodeDefaults = true; explicitNulls = true; prettyPrint = true }
    }

    /** Adds a negative (its [global] is stored as binary16) and returns the stored entry. */
    fun add(
        fingerprint: String,
        global: FloatArray,
        cropJpeg: ByteArray? = null,
        createdAtMs: Long = System.currentTimeMillis(),
        id: String = "neg-" + TwinMeta.newId(createdAtMs, Random()),
    ): NegativeEntry {
        require(ID_OK.matches(id) && !id.endsWith(TMP)) { "bad negative id '$id'" }
        require(global.isNotEmpty()) { "empty global" }
        rootDir.mkdirs()
        val dst = File(rootDir, id)
        require(!dst.exists()) { "negative $id already exists" }
        val tmp = File(rootDir, id + TMP)
        if (tmp.exists()) tmp.deleteRecursively()
        if (!tmp.mkdirs()) throw IOException("cannot create $tmp")
        try {
            F16File.write(File(tmp, GLOBAL), 1, global.size, global)
            if (cropJpeg != null) File(tmp, CROP).writeBytes(cropJpeg)
            val meta = NegativeJson(id, createdAtMs, fingerprint, global.size, if (cropJpeg != null) CROP else null)
            File(tmp, META).writeText(json.encodeToString(NegativeJson.serializer(), meta), Charsets.UTF_8)
            try {
                Files.move(tmp.toPath(), dst.toPath(), StandardCopyOption.ATOMIC_MOVE)
            } catch (e: AtomicMoveNotSupportedException) {
                Files.move(tmp.toPath(), dst.toPath())
            }
        } catch (e: Throwable) {
            tmp.deleteRecursively()
            throw e
        }
        return read(dst) ?: throw IOException("negative $id unreadable after save")
    }

    /** Stored negatives (optionally only those of [fingerprint]), oldest first; unreadable entries are skipped. */
    fun list(fingerprint: String? = null): List<NegativeEntry> {
        val dirs = rootDir.listFiles { f -> f.isDirectory } ?: return emptyList()
        val out = ArrayList<NegativeEntry>()
        for (d in dirs) {
            if (d.name.endsWith(TMP)) {
                d.deleteRecursively()
                continue
            }
            val e = read(d) ?: continue
            if (fingerprint == null || e.fingerprint == fingerprint) out.add(e)
        }
        return out.sortedWith(compareBy<NegativeEntry> { it.createdAtMs }.thenBy { it.id })
    }

    /** The globals of [fingerprint]'s negatives (input of teach / τ_id). */
    fun globals(fingerprint: String): List<FloatArray> = list(fingerprint).map { it.global }

    /** Stores the feature maps of negative [id] (replacing older ones): `map_<i>.f16` `[patches, dim]` and `mapcov_<i>.f16` `[1, patches]`. */
    fun addMaps(id: String, maps: List<NegativeMap>, patches: Int, dim: Int) {
        require(ID_OK.matches(id)) { "bad negative id '$id'" }
        val dir = File(rootDir, id)
        require(dir.isDirectory) { "no such negative $id" }
        maps.forEachIndexed { i, m -> require(m.features.size == patches * dim && m.cov.size == patches) { "map $i has the wrong size" } }
        // Write under temp names first, then swap, so a crash never leaves a half-written set.
        maps.forEachIndexed { i, m ->
            F16File.write(File(dir, "map_$i.f16$TMP"), patches, dim, m.features)
            F16File.write(File(dir, "mapcov_$i.f16$TMP"), 1, patches, m.cov)
        }
        dir.listFiles { f -> (f.name.startsWith("map_") || f.name.startsWith("mapcov_")) && !f.name.endsWith(TMP) }?.forEach { it.delete() }
        maps.indices.forEach { i ->
            File(dir, "map_$i.f16$TMP").renameTo(File(dir, "map_$i.f16"))
            File(dir, "mapcov_$i.f16$TMP").renameTo(File(dir, "mapcov_$i.f16"))
        }
    }

    /** The feature maps stored for negative [id] that match `patches × dim` (others are skipped); empty when it has none. */
    fun maps(id: String, patches: Int, dim: Int): List<NegativeMap> {
        require(ID_OK.matches(id)) { "bad negative id '$id'" }
        val dir = File(rootDir, id)
        val out = ArrayList<NegativeMap>()
        var i = 0
        while (true) {
            val f = File(dir, "map_$i.f16")
            val c = File(dir, "mapcov_$i.f16")
            if (!f.isFile || !c.isFile) break
            try {
                val fm = F16File.read(f)
                val cm = F16File.read(c)
                if (fm.rows == patches && fm.cols == dim && cm.rows * cm.cols == patches) out += NegativeMap(fm.data, cm.data)
            } catch (e: IOException) {
                // damaged entry: skip it
            }
            i++
        }
        return out
    }

    /** The crop JPEG of negative [id], if it was stored. */
    fun crop(id: String): ByteArray? {
        require(ID_OK.matches(id)) { "bad negative id '$id'" }
        val f = File(File(rootDir, id), CROP)
        return if (f.isFile) f.readBytes() else null
    }

    /** Deletes negative [id]; false when it did not exist. */
    fun delete(id: String): Boolean {
        require(ID_OK.matches(id)) { "bad negative id '$id'" }
        val d = File(rootDir, id)
        return d.exists() && d.deleteRecursively()
    }

    private fun read(dir: File): NegativeEntry? {
        return try {
            val meta = json.decodeFromString(NegativeJson.serializer(), File(dir, META).readText(Charsets.UTF_8))
            if (meta.id != dir.name) return null
            val m = F16File.read(File(dir, GLOBAL))
            if (m.rows != 1 || m.cols != meta.dim) return null
            NegativeEntry(meta.id, meta.createdAtMs, meta.fingerprint, m.data, meta.crop != null && File(dir, CROP).isFile)
        } catch (e: IOException) {
            null
        } catch (e: SerializationException) {
            null
        } catch (e: IllegalArgumentException) {
            null
        }
    }
}
