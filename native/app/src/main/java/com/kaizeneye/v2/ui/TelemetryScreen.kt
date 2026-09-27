package com.kaizeneye.v2.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.runtime.Accel
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.ml.EngineHolder
import kotlinx.coroutines.delay

/** Per-stage timers, honest accelerator badges + CPU↔accelerator A/B, thermal/power readings, governor, soak logging. */
@Composable
fun TelemetryScreen(g: AppGraph, nav: Navigator) {
    val thermal by g.thermal.latest.collectAsStateWithLifecycle()
    val engine by g.engines.state.collectAsStateWithLifecycle()
    val inspect by g.hub.inspect.collectAsStateWithLifecycle()
    val soakFile by g.soak.file.collectAsStateWithLifecycle()
    var stages by remember { mutableStateOf(g.hub.stageSummary()) }
    var soakOn by remember { mutableStateOf(g.soak.running) }
    LaunchedEffect(Unit) {
        while (true) {
            stages = g.hub.stageSummary(); delay(1000)
        }
    }
    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Telemetry", onBack = { nav.back() })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Card {
                Text("Thermal & power", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                KeyValue("thermal status", thermal.statusName, if (thermal.status >= 3) Kz.Defect else if (thermal.status == 2) Kz.Warn else Kz.Text)
                KeyValue("headroom (10 s)", if (thermal.headroom.isNaN()) "n/a" else "%.2f (%.0f s old)".format(thermal.headroom, thermal.headroomAgeMs / 1000.0))
                KeyValue("battery", "${thermal.batteryTempC?.let { "%.1f °C".format(it) } ?: "?"} · ${thermal.capacityPct ?: "?"} %${if (thermal.charging) " · charging" else ""}")
                KeyValue("whole phone", thermal.watts?.let { "%.2f W  (|I| %d µA × %d mV)".format(it, thermal.currentUa ?: 0, thermal.voltageMv ?: 0) } ?: "n/a")
                KeyValue("governor", inspect.governor.ifBlank { "idle (no line running)" })
                Note("Whole-phone W = |current| × voltage; the current sign convention is checked on the phone before any W figure is used.")
            }
            Card {
                Text("Accelerators", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                when (val e = engine) {
                    is EngineHolder.State.Ready -> {
                        KeyValue("backbone", e.backbone.report.badge)
                        KeyValue("model", "${e.backbone.modelId} · sha ${e.backbone.sha256.take(12)}…")
                        KeyValue("k-NN", e.knn.label)
                        for (r in e.backbone.report.results) {
                            Text(
                                "${r.name.padEnd(10)} ${if (r.ok) "ok " else "—  "} ${r.medianMs?.let { "%6.1f".format(it) } ?: "     -"} ms p95 ${r.p95Ms?.let { "%6.1f".format(it) } ?: "-"}" +
                                    (r.cosine?.let { " cos %.4f".format(it) } ?: "") + (r.error?.let { " ✗ $it" } ?: r.note?.let { " · $it" } ?: ""),
                                color = if (r.name == e.backbone.report.chosen) Kz.Accent else Kz.TextDim, fontSize = 11.sp, fontFamily = Kz.Mono,
                            )
                        }
                        Note(if (e.backbone.report.fromCache) "decision from cache (measured ${java.util.Date(e.backbone.report.measuredAtMs)})" else "measured just now")
                    }
                    is EngineHolder.State.Loading -> Note(e.message)
                    is EngineHolder.State.Failed -> Text(e.message, color = Kz.Defect, fontSize = 13.sp)
                    EngineHolder.State.Idle -> Note("not loaded")
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SecondaryButton("Accelerator", Modifier.weight(1f)) { g.engines.load(forceAccel = null) }
                    SecondaryButton("Force CPU (A/B)", Modifier.weight(1f)) { g.engines.load(forceAccel = Accel.CPU) }
                }
                SecondaryButton("Re-benchmark all") { g.engines.load(rebenchmark = true) }
            }
            Card {
                Text("Pipeline stage timers (p50 / p95 ms)", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                if (stages.isEmpty()) Note("Run Inspect or a replay to collect timings.")
                stages.forEach { (k, v) -> KeyValue(k, v) }
            }
            Card {
                Text("Soak log", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                Note("One CSV row every 2 s (thermal, headroom, battery °C, W, governor, fps, p95). Start it, then run Inspect for 10 min.")
                soakFile?.let { Note(it.absolutePath) }
                PrimaryButton(if (soakOn) "STOP SOAK LOG" else "START SOAK LOG", color = if (soakOn) Kz.Defect else Kz.Accent) {
                    if (g.soak.running) g.soak.stop() else g.soak.start("soak")
                    soakOn = g.soak.running
                }
            }
        }
    }
}
