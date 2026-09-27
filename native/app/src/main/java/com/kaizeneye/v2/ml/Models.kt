package com.kaizeneye.v2.ml

import com.kaizeneye.runtime.BackboneSpec
import com.kaizeneye.runtime.ModelAsset

/** A backbone plus the k-NN graph family that matches its patch grid. */
data class BackboneChoice(
    val id: String,
    val label: String,
    val spec: BackboneSpec,
    val knnP: Int,
    val knnD: Int,
    val knnBuckets: List<Int>,
    val knnFiles: List<ModelAsset>,
) {
    val knnGraphId: String get() = Models.knnGraphId(knnP, knnD)
}

/**
 * Every model the app can use, with its pinned SHA-256 (plan guardrail 5: the app refuses mismatches). Bundled models live in
 * assets/models; big ones are pushed to /sdcard/Android/data/com.kaizeneye.v2/files/models (or SAF-imported).
 */
object Models {
    val R18_F32 = ModelAsset("r18_320_f32", "backbone_r18_320.tflite", "e881924d3498291f33a7c5e21c01e4f4489d334a0c5c19151c354c38ca2619ba")
    val R18_INT8 = ModelAsset("r18_320_int8", "backbone_r18_320_int8.tflite", "882778161abee20f91c65e7dd4c092843d197f97f73f0a2cbfcbbf3a405bf1cb", int8 = true)

    /** ResNet18 patch-feature backbone @ 320 px → 40×40×128 (normalisation baked into the graph). The default. */
    val R18 = BackboneSpec(
        name = "backbone_r18_320",
        variants = listOf(R18_F32, R18_INT8),
        inputSize = 320, gh = 40, gw = 40, dim = 128,
        l2NormalizePatches = false,
    )

    /** LiteRT matmul k-NN graphs for the R18 grid (P = 1600, D = 128), one per bank-size bucket. */
    const val KNN_R18_P = 1600
    const val KNN_R18_D = 128
    val KNN_R18_BUCKETS = listOf(800, 1600, 2400)
    val KNN_R18_FILES = listOf(
        ModelAsset("knn_p1600_d128_k800", "knn_p1600_d128_k800.tflite", "5705d1a3e8602cc851cc794ed03cc5bc1fdd1a551fec0fbbc7913b732a8d636e"),
        ModelAsset("knn_p1600_d128_k1600", "knn_p1600_d128_k1600.tflite", "c71cb5f44b8cc79482aca4d582d6e24d772a788c25ed8060491f01dc8fc10621"),
        ModelAsset("knn_p1600_d128_k2400", "knn_p1600_d128_k2400.tflite", "8fcecc40dd5c8868da37f82d82fd90fd666c1bbbfba7c6f37024e92d4b3df9fc"),
    )

    /**
     * DINOv2-S/14 @ 448 px → 32×32×384 (tools/dinov2, Apache-2.0; parity vs PyTorch: cosine 1.0 fp32, 0.99999 fp16 weights).
     * Optional challenger (plan B5): adb-pushed, not bundled. Patch vectors are L2-normalised (twin-spec §5).
     */
    val DINO_F16W = ModelAsset("dinov2_s14_448_fp16w", "dinov2_s14_448_fp16w.tflite", "58c86e0b5e1c6ccdaee4fb329b1d6e0e1e8b04577d53c2afd9f9f18d504ec223")
    val DINO_F32 = ModelAsset("dinov2_s14_448_fp32", "dinov2_s14_448_fp32.tflite", "6c44ca64acd23b38bc4e8f813f8792e1baf8c44c1ae2bf825ca15890481dc290")
    val DINO = BackboneSpec(
        name = "backbone_dinov2_s14_448",
        // fp32 first (the CPU accuracy reference; plain float ops, the form the NPU / GPU compilers accept), the 44 MB
        // fp16-weight file as the fallback when only it is on the phone. Both are adb-pushed, never bundled.
        variants = listOf(DINO_F32, DINO_F16W),
        inputSize = 448, gh = 32, gw = 32, dim = 384,
        l2NormalizePatches = true,
    )
    val KNN_DINO_FILES = listOf(
        ModelAsset("knn_p1024_d384_k2400", "knn_p1024_d384_k2400.tflite", "41ffa92cd06e3b296048cfb866c358c535d735253795a4bfea2a5afc01159fb8"),
    )

    val R18_CHOICE = BackboneChoice("r18", "ResNet18 @ 320", R18, KNN_R18_P, KNN_R18_D, KNN_R18_BUCKETS, KNN_R18_FILES)
    val DINO_CHOICE = BackboneChoice("dinov2", "DINOv2-S/14 @ 448", DINO, 1024, 384, listOf(2400), KNN_DINO_FILES)
    val choices = listOf(R18_CHOICE, DINO_CHOICE)

    /** DEBUG / emulator only: pure-Kotlin features, no LiteRT (see [DebugKotlinBackbone]). Never for claims. */
    val DEBUG_CHOICE = BackboneChoice("debug", "DEBUG Kotlin (emulator)", DebugKotlinBackbone.SPEC, KNN_R18_P, KNN_R18_D, KNN_R18_BUCKETS, emptyList())

    fun choice(id: String?): BackboneChoice = (choices + DEBUG_CHOICE).firstOrNull { it.id == id } ?: R18_CHOICE

    /** Offline VLM for reject explanations (Apple AMLR licence: research/demo only). ~942 MB, adb-pushed. */
    val FASTVLM = ModelAsset(
        "fastvlm_0.5b_sm8850", "FastVLM-0.5B.qualcomm.sm8850.litertlm",
        "0EB6860D5C3800DC02C86515A54387F0D8289E5096DB291FF6E4A993E312B482",
    )

    /** Everything the readiness panel lists. */
    val all: List<ModelAsset> = listOf(R18_F32, R18_INT8) + KNN_R18_FILES + listOf(DINO_F16W, DINO_F32) + KNN_DINO_FILES + FASTVLM

    /** Short id of the k-NN graph family, recorded in the Twin pipeline. */
    fun knnGraphId(P: Int, D: Int) = "knn_p${P}_d$D"
}
