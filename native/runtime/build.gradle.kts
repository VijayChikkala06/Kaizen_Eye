// :runtime - LiteRT model runtime (honest NPU/GPU/CPU labels, CPU fallback), accelerator probe (":probe" process) with crash
// guard, k-NN graph runner, model locator, offline VLM explanation service (":vlm" process). See README.md.
plugins {
    alias(libs.plugins.android.library)   // AGP 9 built-in Kotlin: do NOT apply org.jetbrains.kotlin.android
}

android {
    namespace = "com.kaizeneye.runtime"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        minSdk = 31            // LiteRT NPU path + ApplicationExitInfo + bindService(executor) need API 31+
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        // Pure-Kotlin logic is unit-tested on the JVM; any stray android.util.Log call returns a default instead of throwing.
        unitTests.isReturnDefaultValues = true
    }
}

configurations.configureEach {
    // litert -> litert-api -> ai-delivery -> asset-delivery -> androidx.work (Play AI-pack delivery): unused, and it would add
    // network permissions + a startup initializer. Same exclusion as :app.
    exclude(group = "androidx.work")
}

dependencies {
    implementation(libs.litert)                    // LiteRT 2.2.0 (Kotlin API lives in the transitive litert-api)
    implementation(libs.litertlm.android)          // LiteRT-LM 0.16.1 - used ONLY inside the ":vlm" process (VlmService)
    api(libs.kotlinx.coroutines.android)           // VlmClient exposes StateFlow; Runtime API is suspend
    implementation(libs.androidx.core.ktx)

    testImplementation(libs.junit)
}

// Real model files for the JVM tests (flatbuffer reader, SHA pins). Tests skip themselves when the folder is missing.
val kzModelsDir: String = listOf(
    rootProject.file("app/src/main/assets/models"),                       // native/ layout (bundled app assets)
    rootProject.file("../mobile/assets/models"),                          // legacy Expo assets next to native/
    file("D:/Projects/Kaizen_Eye/native/app/src/main/assets/models"),     // helper scratch copies of the project
).firstOrNull { it.isDirectory }?.absolutePath ?: ""
// DINOv2 export (H2): dinov2_s14_448_{fp16w,fp32}.tflite + knn_p1024_d384_k2400.tflite (adb-pushed/bundled later).
val kzDinoDir: String = listOf(
    rootProject.file("../tools/dinov2/out"),
    file("D:/Projects/Kaizen_Eye/tools/dinov2/out"),
).firstOrNull { it.isDirectory }?.absolutePath ?: ""

tasks.withType<Test>().configureEach {
    systemProperty("models.dir", kzModelsDir)
    systemProperty("dinov2.dir", kzDinoDir)
    maxHeapSize = "1g"
    testLogging {
        events("failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
