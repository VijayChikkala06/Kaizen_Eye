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
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.telemetry.Readiness
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HomeScreen(g: AppGraph, nav: Navigator) {
    val twins by g.hub.twins.collectAsStateWithLifecycle()
    val active by g.hub.active.collectAsStateWithLifecycle()
    val engine by g.engines.state.collectAsStateWithLifecycle()
    val thermal by g.thermal.latest.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { g.hub.refreshTwins() }
    val ctx = g.context
    val airplane = Readiness.airplaneMode(ctx)
    val noNet = !Readiness.internetDeclared(ctx)

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Kaizen Eye 2", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = Kz.Text)
        Note("Visual Twin inspection · learns a part from one short video · runs fully offline")
        @OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
        androidx.compose.foundation.layout.FlowRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Chip(if (engine is com.kaizeneye.v2.ml.EngineHolder.State.Failed) "model error — see Telemetry" else g.engines.badge(), if (engine is com.kaizeneye.v2.ml.EngineHolder.State.Failed) Kz.Defect else Kz.Accent)
            Chip(if (airplane) "AIRPLANE ✓" else "airplane off", if (airplane) Kz.Pass else Kz.TextDim)
            Chip(if (noNet) "NO INTERNET PERMISSION ✓" else "INTERNET DECLARED ✗", if (noNet) Kz.Pass else Kz.Defect)
            Chip("thermal ${thermal.statusName}${thermal.batteryTempC?.let { " · %.1f °C".format(it) } ?: ""}")
        }

        val a = active
        if (a != null) {
            Card(border = Kz.Accent) {
                Text(a.name, fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = Kz.Text)
                Note("taught ${fmt(a.createdAtMs)} · ${a.keyframes} keyframes · τ ${"%.3f".format(a.tau)} · τ_id ${"%.3f".format(a.tauId)}")
                Text(a.headline, color = if (a.calibrated) Kz.Pass else Kz.Warn, fontSize = 13.sp)
                if (!a.loadable) Text(a.problem ?: "Twin cannot be loaded", color = Kz.Defect, fontSize = 13.sp)
                PrimaryButton("INSPECT", Modifier.fillMaxWidth(), enabled = a.loadable) { nav.go(Screen.Inspect(LineMode.INSPECT)) }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SecondaryButton("Calibrate", Modifier.weight(1f), enabled = a.loadable) { nav.go(Screen.Inspect(LineMode.CALIBRATE)) }
                    SecondaryButton("Certificate", Modifier.weight(1f)) { nav.go(Screen.Certificate) }
                    SecondaryButton("Negatives", Modifier.weight(1f), enabled = a.loadable) { nav.go(Screen.Negatives) }
                }
            }
        } else {
            Card { Note("No Twin yet. Teach a part: put the phone on a stand over a plain contrasting sheet, then record the part for 12 s.") }
        }

        PrimaryButton("TEACH A NEW PART", Modifier.fillMaxWidth(), color = Kz.SurfaceHi) { nav.go(Screen.Teach) }

        val others = twins.filter { it.id != a?.id }
        if (others.isNotEmpty()) {
            Text("Other Twins", color = Kz.TextDim, fontSize = 13.sp)
            for (t in others) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(t.name, color = Kz.Text, fontSize = 15.sp)
                        Note("${fmt(t.createdAtMs)} · ${if (t.calibrated) "calibrated" else "not calibrated"}${if (!t.loadable) " · ${t.problem}" else ""}")
                    }
                    TextButton(onClick = { g.hub.selectTwin(t.id) }, enabled = t.loadable) { Text("Use") }
                    TextButton(onClick = { g.hub.deleteTwin(t.id) }) { Text("Delete", color = Kz.Defect) }
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SecondaryButton("Clips & replay", Modifier.weight(1f)) { nav.go(Screen.Clips) }
            SecondaryButton("Telemetry", Modifier.weight(1f)) { nav.go(Screen.Telemetry) }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SecondaryButton("Readiness", Modifier.weight(1f)) { nav.go(Screen.Readiness) }
            SecondaryButton("Self-test", Modifier.weight(1f)) { nav.go(Screen.SelfTest()) }
        }
    }
}

private fun fmt(ms: Long): String = SimpleDateFormat("d MMM HH:mm", Locale.US).format(Date(ms))
