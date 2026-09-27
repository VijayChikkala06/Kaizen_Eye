package com.kaizeneye.runtime

import android.app.Application
import android.content.Context
import android.os.Build
import android.system.Os

/**
 * Per-process NPU environment. [ensure] MUST run before the first LiteRT or LiteRT-LM call in EACH process (":main",
 * ":probe", ":vlm"): it points the Hexagon FastRPC loader (ADSP_LIBRARY_PATH) at the app's nativeLibraryDir, where the
 * app ships libLiteRtDispatch_Qualcomm.so, libLiteRtCompilerPlugin_Qualcomm.so and the QNN v81 skel/stub libraries
 * (useLegacyPackaging = true, so they are real files). Runtime, ProbeService and VlmService call it themselves.
 * Holds no LiteRT/LiteRT-LM classes, so the ":vlm" process never loads the vision LiteRT and vice versa.
 */
object NpuEnv {
    @Volatile private var ensured = false

    /** The value set for ADSP_LIBRARY_PATH in this process (null before [ensure] or if setenv failed). */
    @Volatile var adspLibraryPath: String? = null
        private set

    /** setenv failure, if any (structured, for the device passport / self-test). */
    @Volatile var setenvError: String? = null
        private set

    fun ensure(context: Context) {
        if (ensured) return
        synchronized(this) {
            if (ensured) return
            val dir = try { context.applicationInfo.nativeLibraryDir } catch (_: Throwable) { null }
            val path = NpuSoc.adspPath(dir)
            try {
                Os.setenv("ADSP_LIBRARY_PATH", path, true)
                adspLibraryPath = path
                RtLog.i("ADSP_LIBRARY_PATH=$path (process ${processName()}, SoC '$socModel')")
            } catch (t: Throwable) {
                setenvError = t.brief()
                RtLog.e("setenv(ADSP_LIBRARY_PATH) failed: ${t.brief()}")
            }
            ensured = true
        }
    }

    /** Build.SOC_MODEL as reported by this phone (e.g. "SM8850"; the real string goes into the device passport). */
    val socModel: String
        get() = try { Build.SOC_MODEL ?: "" } catch (_: Throwable) { "" }

    val socManufacturer: String
        get() = try { Build.SOC_MANUFACTURER ?: "" } catch (_: Throwable) { "" }

    /** Our NPU allow-list: SOC_MODEL starts with "SM8850" (LiteRT's default checker demands exactly "SM8850"). */
    fun npuSocSupported(): Boolean = NpuSoc.supported(socModel)

    /** e.g. "com.kaizeneye.v2:probe". */
    fun processName(): String = try { Application.getProcessName() ?: "?" } catch (_: Throwable) { "?" }

    /** One line for logs / self-tests. */
    fun describe(): String =
        "SoC ${socManufacturer.ifEmpty { "?" }}/${socModel.ifEmpty { "?" }} (NPU ${if (npuSocSupported()) "allowed" else "not allowed"}), " +
            "process ${processName()}, ADSP_LIBRARY_PATH=${adspLibraryPath ?: "unset"}${setenvError?.let { " (setenv failed: $it)" } ?: ""}"
}

/** Pure helpers behind [NpuEnv] (JVM-tested). */
internal object NpuSoc {
    const val SOC_PREFIX = "SM8850"

    fun supported(soc: String?): Boolean = soc != null && soc.trim().uppercase().startsWith(SOC_PREFIX)

    fun adspPath(nativeLibDir: String?): String = listOfNotNull(
        nativeLibDir?.takeIf { it.isNotBlank() },
        "/odm/lib/rfsa/adsp",
        "/vendor/lib/rfsa/adsp",
        "/system/lib/rfsa/adsp",
        "/system/vendor/lib/rfsa/adsp",
        "/dsp",
    ).joinToString(";")
}
