package com.kaizeneye.v2

import android.content.Context
import java.io.File

/** Where everything lives. External-files dirs are adb-pullable without root (`/sdcard/Android/data/com.kaizeneye.v2/files/...`). */
class AppDirs(ctx: Context) {
    val twins = File(ctx.filesDir, "twins")
    val negatives = File(ctx.filesDir, "negatives")
    val accel = File(ctx.filesDir, "accel")
    private val ext: File = ctx.getExternalFilesDir(null) ?: File(ctx.filesDir, "ext")
    val logs = File(ext, "logs")
    val clips = File(ext, "clips")
    val exports = File(ext, "exports")
    val selftest = File(ext, "selftest")
    val models = File(ext, "models")
    val vlmCache = File(ctx.cacheDir, "vlm")

    fun ensure(): AppDirs {
        listOf(twins, negatives, logs, clips, exports, selftest, models, vlmCache).forEach { it.mkdirs() }
        return this
    }
}
