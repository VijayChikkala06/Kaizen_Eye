plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.kaizeneye.v2"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.kaizeneye.v2"
        minSdk = 31            // LiteRT NPU path requires API 31+
        targetSdk = 36
        versionCode = 1
        versionName = "2.0.0-dev"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            optimization { enable = false }          // no R8: JNI-heavy runtimes, nothing to gain for a sideloaded demo
            signingConfig = signingConfigs.getByName("debug")   // same stable debug key as the legacy app (reinstall with -r keeps data)
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    androidResources {
        noCompress += "tflite"
    }

    packaging {
        jniLibs {
            // Hexagon skel/stub libraries must be real files on disk (FastRPC opens them by path).
            useLegacyPackaging = true
            // qnn-runtime ships every Hexagon generation (200 MB). SM8850 needs v81 only.
            excludes += listOf(
                "**/libQnnDsp.so", "**/libQnnDspV66Skel.so", "**/libQnnDspV66Stub.so", "**/libQnnGpu.so",
                "**/libQnnHtpV68Skel.so", "**/libQnnHtpV68Stub.so",
                "**/libQnnHtpV69Skel.so", "**/libQnnHtpV69Stub.so",
                "**/libQnnHtpV73Skel.so", "**/libQnnHtpV73Stub.so",
                "**/libQnnHtpV75Skel.so", "**/libQnnHtpV75Stub.so",
                "**/libQnnHtpV79Skel.so", "**/libQnnHtpV79Stub.so",
            )
        }
    }
}

configurations.configureEach {
    // litert -> litert-api -> ai-delivery -> asset-delivery -> androidx.work:work-runtime arrives transitively (Play AI-pack delivery).
    // It is unused here (models are sideloaded) and would add ACCESS_NETWORK_STATE / WAKE_LOCK / RECEIVE_BOOT_COMPLETED,
    // a startup initializer and background components to an app that must be provably offline. Found by tools/apk-audit.ps1.
    exclude(group = "androidx.work")
}

dependencies {
    implementation(project(":core"))
    implementation(project(":runtime"))

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)

    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.camera.video)

    // ML runtime family "A" (LiteRT 2.x CompiledModel) + VLM + Qualcomm QNN runtime libraries.
    implementation(libs.litert)
    implementation(libs.litertlm.android)
    implementation(libs.qnn.runtime)

    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)

    testImplementation(libs.junit)
}
