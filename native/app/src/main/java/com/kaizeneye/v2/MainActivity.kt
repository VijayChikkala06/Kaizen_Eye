package com.kaizeneye.v2

import android.Manifest
import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import com.kaizeneye.v2.telemetry.Readiness
import com.kaizeneye.v2.ui.CertificateScreen
import com.kaizeneye.v2.ui.ClipsScreen
import com.kaizeneye.v2.ui.HomeScreen
import com.kaizeneye.v2.ui.InspectScreen
import com.kaizeneye.v2.ui.KaizenTheme
import com.kaizeneye.v2.ui.Kz
import com.kaizeneye.v2.ui.Navigator
import com.kaizeneye.v2.ui.NegativesScreen
import com.kaizeneye.v2.ui.ReadinessScreen
import com.kaizeneye.v2.ui.Screen
import com.kaizeneye.v2.ui.SelfTestScreen
import com.kaizeneye.v2.ui.TeachScreen
import com.kaizeneye.v2.ui.TelemetryScreen

/**
 * Single activity, portrait-locked (manifest). Self-test without touching the screen:
 *   adb shell am start -n com.kaizeneye.v2/.MainActivity --es selftest all        (or quick | accel | replay | vlm)
 * Results: /sdcard/Android/data/com.kaizeneye.v2/files/selftest/selftest.json
 */
class MainActivity : ComponentActivity() {
    private val nav = Navigator()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val g = AppGraph.get(this)
        if (savedInstanceState == null) handleIntent(intent)
        setContent { KaizenTheme { AppRoot(g, nav) } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(i: Intent?) {
        // Scripted runs: --es backbone r18|dinov2|debug switches the backbone (debug = pure-Kotlin features for the emulator).
        i?.getStringExtra("backbone")?.let { id ->
            val g = AppGraph.get(this)
            g.prefs.backbone = id
            g.engines.load(com.kaizeneye.v2.ml.Models.choice(id))
        }
        val st = i?.getStringExtra("selftest") ?: return
        nav.go(Screen.SelfTest(autoStart = st))
    }
}

@Composable
private fun AppRoot(g: AppGraph, nav: Navigator) {
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        g.cameraGranted.value = granted
    }
    LaunchedEffect(Unit) {
        g.cameraGranted.value = Readiness.cameraGranted(g.context)
        if (!g.cameraGranted.value) permission.launch(Manifest.permission.CAMERA)
        g.thermal.start(g.scope)
        if (g.engines.ready() == null) g.engines.autoLoad(com.kaizeneye.v2.ml.Models.choice(g.prefs.backbone))
    }
    BackHandler(enabled = nav.stack.size > 1) { nav.back() }
    Box(
        Modifier
            .fillMaxSize()
            .background(Kz.Bg)
            .systemBarsPadding()
            .imePadding(),
    ) {
        val s = nav.current
        // Keyed on the screen value: a screen replaced by another instance of itself is fully recomposed (camera re-bound,
        // its dispose effects run) instead of silently keeping the old composition.
        androidx.compose.runtime.key(s, nav.stack.size) {
            when (s) {
                Screen.Home -> HomeScreen(g, nav)
                Screen.Teach -> TeachScreen(g, nav)
                is Screen.Inspect -> InspectScreen(g, nav, s)
                Screen.Certificate -> CertificateScreen(g, nav)
                Screen.Telemetry -> TelemetryScreen(g, nav)
                Screen.Readiness -> ReadinessScreen(g, nav)
                Screen.Clips -> ClipsScreen(g, nav)
                Screen.Negatives -> NegativesScreen(g, nav)
                is Screen.SelfTest -> SelfTestScreen(g, nav, s.autoStart)
            }
        }
    }
}
