package com.kaizeneye.runtime

/**
 * Which (VLM model, LiteRT-LM backend) pairs the ":vlm" process tries, in order (pure; VlmPlanTest).
 *  - FastVLM-0.5B qualcomm.sm8850 (AOT-compiled for the SM8850 NPU): NPU -> GPU -> CPU.
 *  - gemma-4-E2B-it-gpu (GPU build, fallback): GPU -> CPU (an NPU attempt makes no sense for a GPU build).
 * A model whose file name contains ".qualcomm." gets the NPU attempt; every model gets GPU and CPU.
 * Candidates with crash-guard strikes >= CrashGuard.CRASHED are skipped.
 */
internal data class VlmCandidate(val fileName: String, val path: String, val backend: String) {
    /** Crash-guard id (the stamp adds app version / build fingerprint / file size). */
    val id: String get() = "$fileName|$backend"
}

internal object VlmPlan {
    const val NPU = "NPU"
    const val GPU = "GPU"
    const val CPU = "CPU"

    /** [models]: (file name, absolute path) in preference order. */
    fun plan(models: List<Pair<String, String>>, isCrashed: (VlmCandidate) -> Boolean = { false }): List<VlmCandidate> {
        val out = ArrayList<VlmCandidate>()
        for ((name, path) in models) {
            val backends = if (name.contains(".qualcomm.", ignoreCase = true)) listOf(NPU, GPU, CPU) else listOf(GPU, CPU)
            for (b in backends) {
                val c = VlmCandidate(name, path, b)
                if (!isCrashed(c)) out += c
            }
        }
        return out
    }

    fun stamp(versionCode: Long, fingerprint: String, fileBytes: Long): String = "v$versionCode|$fingerprint|$fileBytes"

    /** The answer text: all text parts of the model message, concatenated (no separator: parts are token chunks). */
    fun joinText(parts: List<String>): String = parts.joinToString("").trim()
}
