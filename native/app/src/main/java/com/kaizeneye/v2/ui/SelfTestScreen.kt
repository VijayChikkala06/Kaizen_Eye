package com.kaizeneye.v2.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph

/**
 * One-run device self-test (plan "Verification — Device"): accelerator parity + latency, k-NN graph accuracy, on-device
 * golden checks of the core maths, REPLAY regression on the bundled synthetic clip, offline/permission proof, optional VLM.
 * Also reachable without touching the phone: adb shell am start -n com.kaizeneye.v2/.MainActivity --es selftest all
 */
@Composable
fun SelfTestScreen(g: AppGraph, nav: Navigator, autoStart: String?) {
    val ui by g.hub.selfTest.collectAsStateWithLifecycle()
    LaunchedEffect(autoStart) { if (autoStart != null && !ui.running) g.hub.runSelfTest(autoStart) }
    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Self-test", onBack = { nav.back() })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PrimaryButton("RUN ALL", Modifier.weight(1f), enabled = !ui.running) { g.hub.runSelfTest("all") }
                SecondaryButton("Quick", Modifier.weight(1f), enabled = !ui.running) { g.hub.runSelfTest("quick") }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SecondaryButton("Accelerators", Modifier.weight(1f), enabled = !ui.running) { g.hub.runSelfTest("accel") }
                SecondaryButton("Replay", Modifier.weight(1f), enabled = !ui.running) { g.hub.runSelfTest("replay") }
                SecondaryButton("VLM", Modifier.weight(1f), enabled = !ui.running) { g.hub.runSelfTest("vlm") }
            }
            if (ui.running) Text("running…", color = Kz.Warn)
            ui.pass?.let { Text(if (it) "SELF-TEST PASS" else "SELF-TEST FAIL", color = if (it) Kz.Pass else Kz.Defect, fontWeight = FontWeight.Bold, fontSize = 20.sp) }
            Card {
                if (ui.steps.isEmpty()) Note("No results yet.")
                ui.steps.forEach { (k, v) ->
                    val c = when {
                        v.startsWith("PASS") -> Kz.Pass
                        v.startsWith("FAIL") -> Kz.Defect
                        v.startsWith("SKIP") -> Kz.TextDim
                        else -> Kz.Text
                    }
                    Column(Modifier.fillMaxWidth()) {
                        Text(k, color = Kz.Text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                        Text(v, color = c, fontSize = 12.sp, fontFamily = Kz.Mono)
                    }
                }
            }
            ui.resultFile?.let { Note("results: $it") }
        }
    }
}
