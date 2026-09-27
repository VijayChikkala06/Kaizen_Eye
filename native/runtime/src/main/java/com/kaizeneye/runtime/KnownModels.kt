package com.kaizeneye.runtime

/**
 * Pinned model files (SHA-256 verified on this laptop against the real files; see KnownModelsTest). Convenience only: the
 * app may build its own [BackboneSpec]s / [ModelAsset]s.
 */
object KnownModels {
    val R18_320_F32 = ModelAsset("r18_320_f32", "backbone_r18_320.tflite", "e881924d3498291f33a7c5e21c01e4f4489d334a0c5c19151c354c38ca2619ba")
    val R18_320_INT8 = ModelAsset("r18_320_int8", "backbone_r18_320_int8.tflite", "882778161abee20f91c65e7dd4c092843d197f97f73f0a2cbfcbbf3a405bf1cb", int8 = true)

    /** ResNet18 @ 320 px (bundled): [1,320,320,3] RGB 0..255 -> [1,40,40,128], not L2-normalised. */
    val BACKBONE_R18_320 = BackboneSpec(
        name = "backbone_r18_320",
        variants = listOf(R18_320_F32, R18_320_INT8),
        inputSize = 320, gh = 40, gw = 40, dim = 128,
        l2NormalizePatches = false,
    )

    val DINOV2_S14_448_FP16W = ModelAsset("dinov2_s14_448_fp16w", "dinov2_s14_448_fp16w.tflite", "58c86e0b5e1c6ccdaee4fb329b1d6e0e1e8b04577d53c2afd9f9f18d504ec223")
    val DINOV2_S14_448_FP32 = ModelAsset("dinov2_s14_448_fp32", "dinov2_s14_448_fp32.tflite", "6c44ca64acd23b38bc4e8f813f8792e1baf8c44c1ae2bf825ca15890481dc290")

    /**
     * DINOv2-S/14 @ 448 px (adb-pushed to getExternalFilesDir("models"), not bundled): outputs patch tokens [1,32,32,384]
     * and CLS [1,384] (identified by element count); two FLOAT variants (fp16 weights / fp32), float accuracy gate for both.
     */
    val BACKBONE_DINOV2_S14_448 = BackboneSpec(
        name = "backbone_dinov2_s14_448",
        variants = listOf(DINOV2_S14_448_FP16W, DINOV2_S14_448_FP32),
        inputSize = 448, gh = 32, gw = 32, dim = 384,
        l2NormalizePatches = true,
    )

    /** SHA-256 of the k-NN graphs knn_p{P}_d{D}_k{K}.tflite built by tools/make_knn_model.py. */
    val KNN_SHA256: Map<String, String> = mapOf(
        "knn_p1600_d128_k800.tflite" to "5705d1a3e8602cc851cc794ed03cc5bc1fdd1a551fec0fbbc7913b732a8d636e",
        "knn_p1600_d128_k1600.tflite" to "c71cb5f44b8cc79482aca4d582d6e24d772a788c25ed8060491f01dc8fc10621",
        "knn_p1600_d128_k2400.tflite" to "8fcecc40dd5c8868da37f82d82fd90fd666c1bbbfba7c6f37024e92d4b3df9fc",
        "knn_p1024_d384_k2400.tflite" to "41ffa92cd06e3b296048cfb866c358c535d735253795a4bfea2a5afc01159fb8",
    )

    /** The k-NN graph for (P, D, K); unknown combinations are unpinned (digest computed and reported). */
    fun knnGraph(p: Int, d: Int, k: Int): ModelAsset {
        val file = "knn_p${p}_d${d}_k$k.tflite"
        return ModelAsset("knn_p${p}_d${d}_k$k", file, KNN_SHA256[file] ?: "")
    }

    /** FastVLM-0.5B, AOT-compiled for the SM8850 NPU (942,374,912 bytes; Apple AMLR licence - research/demo only). */
    val FASTVLM_05B_SM8850 = ModelAsset("fastvlm_0.5b_sm8850", "FastVLM-0.5B.qualcomm.sm8850.litertlm", "0EB6860D5C3800DC02C86515A54387F0D8289E5096DB291FF6E4A993E312B482")
    const val FASTVLM_05B_SM8850_BYTES = 942374912L

    /** Gemma 4 E2B GPU build (Apache-2.0), fallback VLM; not pinned yet (its digest is reported in VlmState.Ready.detail). */
    val GEMMA4_E2B_GPU = ModelAsset("gemma4_e2b_it_gpu", "gemma-4-E2B-it-gpu.litertlm", "")
}
