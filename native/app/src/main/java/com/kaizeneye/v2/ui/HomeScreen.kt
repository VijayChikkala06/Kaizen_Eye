package com.kaizeneye.v2.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.telemetry.Readiness
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Operator home: the active part (inspect / wrong objects / calibrate / certificate), teach a new part, other parts.
 * Everything a developer or evaluator needs (clips, telemetry, readiness, self-test) sits behind one "Advanced" toggle.
 */
@Composable
fun HomeScreen(g: AppGraph, nav: Navigator) {
    val twins by g.hub.twins.collectAsStateWithLifecycle()
    val active by g.hub.active.collectAsStateWithLifecycle()
    val engine by g.engines.state.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { g.hub.refreshTwins() }
    val ctx = g.context
    val noNet = remember { !Readiness.internetDeclared(ctx) }
    val airplane = Readiness.airplaneMode(ctx)
    var showAdvanced by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf<TwinSummaryRef?>(null) }
    val modelReady = engine is EngineHolder.State.Ready

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Kaizen Eye 2", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = Kz.Text)
        Note("Visual Twin inspection · learns a part from one short video · runs fully offline")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            when (val e = engine) {
                is EngineHolder.State.Ready -> Chip("Model ready", Kz.Pass)
                is EngineHolder.State.Loading -> Chip("Loading model…", Kz.Warn)
                is EngineHolder.State.Failed -> Chip("Model error", Kz.Defect)
                EngineHolder.State.Idle -> Chip("Model not loaded", Kz.TextDim)
            }
            Chip(
                if (noNet) "Offline ✓${if (airplane) " · airplane" else ""}" else "INTERNET permission ✗",
                if (noNet) Kz.Pass else Kz.Defect,
            )
        }
        ((engine as? EngineHolder.State.Failed)?.message ?: g.engines.lastError)?.let { Text(it, color = Kz.Defect, fontSize = 12.sp) }

        val a = active
        if (a != null) {
            Card(border = Kz.Accent) {
                Text(a.name, fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = Kz.Text)
                Note("Taught ${fmt(a.createdAtMs)}")
                Text(a.headline, color = if (a.calibrated) Kz.Pass else Kz.Warn, fontSize = 13.sp)
                a.fitLine?.let { Note(it, if (it.startsWith("⚠")) Kz.Warn else Kz.TextDim) }
                PrimaryButton(if (modelReady) "INSPECT" else "LOADING MODEL…", Modifier.fillMaxWidth(), enabled = a.loadable && modelReady) {
                    nav.go(Screen.Inspect(LineMode.INSPECT))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SecondaryButton("Wrong objects", Modifier.weight(1f), enabled = a.loadable && modelReady) { nav.go(Screen.Negatives) }
                    SecondaryButton("Calibrate", Modifier.weight(1f), enabled = a.loadable && modelReady) { nav.go(Screen.Inspect(LineMode.CALIBRATE)) }
                }
                SecondaryButton("Certificate", Modifier.fillMaxWidth()) { nav.go(Screen.Certificate) }
            }
        } else {
            Card { Note("No part yet. Put the phone on a stand over a plain sheet, then teach a part: circle it and record it for 12 s.") }
        }

        PrimaryButton(
            if (modelReady) "TEACH A NEW PART" else "LOADING MODEL…", Modifier.fillMaxWidth(),
            enabled = modelReady, color = if (a == null) Kz.Accent else Kz.SurfaceHi,
        ) { nav.go(Screen.Teach) }

        val others = twins.filter { it.id != a?.id }
        if (others.isNotEmpty()) {
            Text("Other parts", color = Kz.TextDim, fontSize = 13.sp)
            for (t in others) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(t.name, color = Kz.Text, fontSize = 15.sp)
                        Note("${fmt(t.createdAtMs)} · ${if (t.calibrated) "calibrated" else "not calibrated"}${if (!t.loadable) " · built with a different model" else ""}")
                    }
                    TextButton(onClick = { g.hub.selectTwin(t.id) }, enabled = t.loadable) { Text("Use") }
                    TextButton(onClick = { confirmDelete = TwinSummaryRef(t.id, t.name) }) { Text("Delete", color = Kz.Defect) }
                }
            }
        }

        TextButton(onClick = { showAdvanced = !showAdvanced }) { Text(if (showAdvanced) "Advanced ▴" else "Advanced ▾", color = Kz.TextDim) }
        if (showAdvanced) {
            Note("Developer and evaluation tools: recorded clips + replay, timings and accelerators, offline proof, the device self-test.")
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

    confirmDelete?.let { t ->
        AlertDialog(
            onDismissRequest = { confirmDelete = null },
            title = { Text("Delete ${t.name}?") },
            text = { Text("The taught part and its calibration are removed for good.") },
            confirmButton = { TextButton(onClick = { g.hub.deleteTwin(t.id); confirmDelete = null }) { Text("Delete", color = Kz.Defect) } },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Keep") } },
        )
    }
}

private class TwinSummaryRef(val id: String, val name: String)

private fun fmt(ms: Long): String = SimpleDateFormat("d MMM HH:mm", Locale.US).format(Date(ms))
