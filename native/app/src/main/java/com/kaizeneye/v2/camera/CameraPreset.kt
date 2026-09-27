package com.kaizeneye.v2.camera

/**
 * Capture settings for inspection (plan: "30 fps, AE-off 0.5–1 ms — 10 ms if flicker is detected — or AE-lock, AF/AWB lock,
 * no stabilisation"; the torch is never switched on). Exact values are tuned on the phone (device passport + flicker check); these are the defaults.
 */
data class CameraPreset(
    val fps: Int = 30,
    /** true = manual exposure (AE off) with [exposureNs] / [iso]; false = auto exposure (lockable with [aeLock]). */
    val manualExposure: Boolean = false,
    val exposureNs: Long = 1_000_000L,
    val iso: Int = 800,
    val aeLock: Boolean = false,
    val awbLock: Boolean = false,
    /** Torch is never used (operator request 2026-09-27: no flash at all); kept only for saved-preset compatibility. */
    val torch: Boolean = false,
    /** Fixed focus distance in diopters (from a converged AF result); null = continuous AF. */
    val focusDiopters: Float? = null,
    val analysisWidth: Int = 1280,
    val analysisHeight: Int = 720,
) {
    /** Exposure used when mains flicker banding is detected (a whole number of 100 Hz / 120 Hz half-cycles). */
    fun flickerSafe(): CameraPreset = copy(manualExposure = true, exposureNs = 10_000_000L)

    fun describe(): String {
        val exp = if (manualExposure) "AE off ${"%.2f".format(exposureNs / 1e6)} ms ISO $iso" else "AE auto${if (aeLock) " (locked)" else ""}"
        val af = focusDiopters?.let { "AF locked ${"%.2f".format(it)} D" } ?: "AF continuous"
        return "$fps fps · $exp · AWB ${if (awbLock) "locked" else "auto"} · $af · torch ${if (torch) "on" else "off"}"
    }
}
