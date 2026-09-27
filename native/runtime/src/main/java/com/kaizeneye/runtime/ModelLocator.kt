package com.kaizeneye.runtime

import android.content.Context
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.security.MessageDigest

/**
 * Finds model files and verifies their SHA-256. Lookup order: getExternalFilesDir("models") (adb push) -> filesDir/models
 * (SAF import) -> app asset "models/<file>" (bundled). The first copy whose digest matches the pin wins; mismatching copies
 * are skipped (and named in [LocatedModel.note]); if no copy matches, the first mismatching one is returned with
 * sha256Ok = false (not usable, and the note says why); null = no copy anywhere.
 * Digests are streamed and cached in filesDir/models/.sha-cache.json keyed by path + size + mtime (assets: by app
 * versionCode + install time), so a 942 MB model is hashed once.
 */
object ModelLocator {
    const val SOURCE_EXTERNAL = "external"
    const val SOURCE_IMPORTED = "imported"
    const val SOURCE_BUNDLED = "bundled"
    const val ASSET_DIR = "models"

    fun externalDir(context: Context): File? = try { context.getExternalFilesDir(ASSET_DIR) } catch (_: Throwable) { null }

    fun importedDir(context: Context): File = File(context.filesDir, ASSET_DIR)

    fun assetName(asset: ModelAsset): String = "$ASSET_DIR/${asset.fileName}"

    fun locate(context: Context, asset: ModelAsset): LocatedModel? {
        val rejected = ArrayList<LocatedModel>()
        for ((source, dir) in listOf(SOURCE_EXTERNAL to externalDir(context), SOURCE_IMPORTED to importedDir(context))) {
            if (dir == null) continue
            val f = File(dir, asset.fileName)
            if (!f.isFile) continue
            val lm = try {
                checkFile(context, asset, f, source)
            } catch (t: Throwable) {
                LocatedModel(asset, source, f.absolutePath, false, f.length(), null, "cannot read ${f.absolutePath}: ${t.brief()}")
            }
            if (lm.sha256Ok) return withRejected(lm, rejected)
            RtLog.w("model ${asset.fileName}: ${lm.note}")
            rejected += lm
        }
        if (assetExists(context, asset)) {
            val lm = try {
                checkAsset(context, asset)
            } catch (t: Throwable) {
                LocatedModel(asset, SOURCE_BUNDLED, assetName(asset), false, -1, null, "cannot read asset ${assetName(asset)}: ${t.brief()}")
            }
            if (lm.sha256Ok) return withRejected(lm, rejected)
            RtLog.w("model ${asset.fileName}: ${lm.note}")
            rejected += lm
        }
        if (rejected.isEmpty()) return null
        return rejected[0].copy(note = rejected.joinToString("; ") { it.note ?: "${it.source}: unusable" })
    }

    private fun withRejected(lm: LocatedModel, rejected: List<LocatedModel>): LocatedModel =
        if (rejected.isEmpty()) lm else lm.copy(note = (listOfNotNull(lm.note) + rejected.map { "ignored ${it.source} copy: ${it.note}" }).joinToString("; "))

    private fun checkFile(context: Context, asset: ModelAsset, f: File, source: String): LocatedModel {
        val size = f.length()
        val stamp = "$size|${f.lastModified()}"
        val key = f.absolutePath
        val sha = ShaCache.lookup(context, key, stamp)?.sha256 ?: run {
            val t0 = nowMs()
            val r = Sha256.of(f)
            RtLog.i("sha256 ${f.name} (${r.bytes} bytes) in ${AccelRules.fmtMs(nowMs() - t0)} ms")
            ShaCache.store(context, key, stamp, r.hex, r.bytes)
            r.hex
        }
        val ok = Sha256.matches(asset.sha256, sha)
        return LocatedModel(
            asset, source, f.absolutePath, ok, size, sha,
            if (ok) null else "SHA-256 mismatch at ${f.absolutePath}: expected ${asset.sha256.lowercase()}, got $sha",
        )
    }

    private fun checkAsset(context: Context, asset: ModelAsset): LocatedModel {
        val name = assetName(asset)
        val key = "asset:$name"
        val stamp = AppInfo.stamp(context)
        val entry = ShaCache.lookup(context, key, stamp) ?: run {
            val r = context.assets.open(name).use { Sha256.of(it) }
            ShaCache.store(context, key, stamp, r.hex, r.bytes)
            ShaCache.Entry(r.hex, r.bytes)
        }
        val ok = Sha256.matches(asset.sha256, entry.sha256)
        return LocatedModel(
            asset, SOURCE_BUNDLED, name, ok, entry.bytes, entry.sha256,
            if (ok) null else "SHA-256 mismatch of bundled asset $name: expected ${asset.sha256.lowercase()}, got ${entry.sha256}",
        )
    }

    @Volatile private var assetList: Set<String>? = null

    private fun assetExists(context: Context, asset: ModelAsset): Boolean {
        val list = assetList ?: (try { context.assets.list(ASSET_DIR)?.toSet() } catch (_: Throwable) { null } ?: emptySet()).also { assetList = it }
        return asset.fileName in list
    }

    /**
     * Copy a model picked with the Storage Access Framework into filesDir/models/<fileName>, hashing while copying.
     * Blocking (a 942 MB file takes a while): call it off the main thread. A digest mismatch deletes the copy and returns
     * sha256Ok = false with the reason; I/O failures (no space, unreadable URI) throw IOException.
     */
    fun importFromUri(context: Context, uri: Uri, asset: ModelAsset): LocatedModel {
        val dir = importedDir(context)
        if (!dir.isDirectory && !dir.mkdirs()) throw IOException("cannot create $dir")
        val dest = File(dir, asset.fileName)
        val tmp = File(dir, asset.fileName + ".part")
        val expected = querySize(context, uri)
        if (expected > 0 && dir.usableSpace < expected + (64L shl 20)) {
            throw IOException("not enough space for ${asset.fileName}: need ${expected shr 20} MB (+64 MB), have ${dir.usableSpace shr 20} MB")
        }
        val md = MessageDigest.getInstance("SHA-256")
        var total = 0L
        val t0 = nowMs()
        try {
            val input = context.contentResolver.openInputStream(uri) ?: throw IOException("cannot open $uri")
            input.use { ins ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(1 shl 20)
                    while (true) {
                        val n = ins.read(buf)
                        if (n < 0) break
                        md.update(buf, 0, n)
                        out.write(buf, 0, n)
                        total += n
                    }
                    out.fd.sync()
                }
            }
        } catch (t: Throwable) {
            tmp.delete()
            throw if (t is IOException) t else IOException("import of ${asset.fileName} failed: ${t.brief()}", t)
        }
        val sha = Sha256.hex(md.digest())
        RtLog.i("imported ${asset.fileName}: $total bytes in ${AccelRules.fmtMs(nowMs() - t0)} ms, sha256 $sha")
        if (!Sha256.matches(asset.sha256, sha)) {
            tmp.delete()
            return LocatedModel(
                asset, SOURCE_IMPORTED, null, false, total, sha,
                "SHA-256 mismatch of the picked file: expected ${asset.sha256.lowercase()}, got $sha - not imported",
            )
        }
        if (dest.exists() && !dest.delete()) throw IOException("cannot replace $dest")
        if (!tmp.renameTo(dest)) throw IOException("cannot rename $tmp to $dest")
        ShaCache.store(context, dest.absolutePath, "${dest.length()}|${dest.lastModified()}", sha, total)
        return LocatedModel(asset, SOURCE_IMPORTED, dest.absolutePath, true, total, sha, null)
    }

    private fun querySize(context: Context, uri: Uri): Long = try {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else -1L
        } ?: -1L
    } catch (_: Throwable) {
        -1L
    }
}

/** Cross-process cache of SHA-256 verdicts: filesDir/models/.sha-cache.json (read-merge-write under a file lock). */
internal object ShaCache {
    class Entry(val sha256: String, val bytes: Long)

    private fun dir(ctx: Context) = File(ctx.filesDir, ModelLocator.ASSET_DIR)
    private fun file(ctx: Context) = File(dir(ctx), ".sha-cache.json")
    private fun lockFile(ctx: Context) = File(dir(ctx), ".sha-cache.lock")

    @Synchronized
    fun lookup(ctx: Context, key: String, stamp: String): Entry? = try {
        val e = read(ctx).optJSONObject("entries")?.optJSONObject(key)
        if (e == null || e.optString("stamp") != stamp) null else Entry(e.getString("sha256"), e.optLong("bytes", -1L))
    } catch (_: Throwable) {
        null
    }

    @Synchronized
    fun store(ctx: Context, key: String, stamp: String, sha256: String, bytes: Long) {
        try {
            dir(ctx).mkdirs()
            RandomAccessFile(lockFile(ctx), "rw").use { raf ->
                raf.channel.lock().use {
                    val o = read(ctx)
                    val entries = o.optJSONObject("entries") ?: JSONObject().also { o.put("entries", it) }
                    entries.put(
                        key,
                        JSONObject().put("stamp", stamp).put("sha256", sha256).put("bytes", bytes).put("atMs", System.currentTimeMillis()),
                    )
                    o.put("version", 1)
                    writeTextAtomic(file(ctx), o.toString())
                }
            }
        } catch (t: Throwable) {
            RtLog.w("sha cache write failed: ${t.brief()}")
        }
    }

    private fun read(ctx: Context): JSONObject = try {
        val f = file(ctx)
        if (f.isFile) JSONObject(f.readText()) else JSONObject()
    } catch (_: Throwable) {
        JSONObject()
    }
}

/** App identity facts used in cache keys. */
internal object AppInfo {
    private fun info(context: Context): PackageInfo? = try {
        if (Build.VERSION.SDK_INT >= 33) {
            context.packageManager.getPackageInfo(context.packageName, PackageManager.PackageInfoFlags.of(0))
        } else {
            @Suppress("DEPRECATION")
            context.packageManager.getPackageInfo(context.packageName, 0)
        }
    } catch (_: Throwable) {
        null
    }

    fun versionCode(context: Context): Long = info(context)?.longVersionCode ?: -1L

    /** "v<versionCode>|<lastUpdateTime>": changes with every (re)install, so a rebuilt APK with the same versionCode re-hashes its assets. */
    fun stamp(context: Context): String {
        val i = info(context)
        return "v${i?.longVersionCode ?: -1}|${i?.lastUpdateTime ?: -1}"
    }

    fun fingerprint(): String = try { Build.FINGERPRINT ?: "" } catch (_: Throwable) { "" }

    /** Sizes of the bundled NPU runtime libraries (part of the decision key: swapping QNN/dispatch builds re-benchmarks). */
    fun nativeStamp(context: Context): String = try {
        val dir = File(context.applicationInfo.nativeLibraryDir)
        listOf("libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so", "libQnnHtp.so", "libQnnHtpV81Skel.so")
            .joinToString(",") { n -> File(dir, n).let { if (it.isFile) "${n.removePrefix("lib").substringBefore('.')}=${it.length()}" else "${n.removePrefix("lib").substringBefore('.')}=none" } }
    } catch (_: Throwable) {
        "native=?"
    }
}
