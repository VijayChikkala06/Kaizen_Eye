package com.kaizeneye.v2.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraController
import com.kaizeneye.v2.pipeline.InspectUi
import com.kaizeneye.v2.pipeline.RejectCard

/** Live line (or REPLAY / CALIBRATE): overlay riding each part, HUD, sensitivity slider, reject card with the explanation. */
@Composable
fun InspectScreen(g: AppGraph, nav: Navigator, s: Screen.Inspect) {
    val ui by g.hub.inspect.collectAsStateWithLifecycle()
    val sheet by g.hub.sheet.collectAsStateWithLifecycle()
    LaunchedEffect(s) {
        // Camera: nothing runs until the operator clears the sheet and taps LEARN SHEET (never learned silently).
        if (s.replayClip != null) g.hub.startReplay(s.replayClip, s.paced, s.mode) else g.hub.openInspect(s.mode)
    }
    DisposableEffect(Unit) { onDispose { g.hub.stopLine() } }

    Column(Modifier.fillMaxSize()) {
        ScreenHeader(
            title = when {
                s.replayClip != null -> "REPLAY · ${s.replayClip.name}"
                s.mode == LineMode.CALIBRATE -> "Calibrate · ${ui.twinName}"
                else -> "Inspect · ${ui.twinName}"
            },
            onBack = { nav.back() },
        ) { Chip(ui.accel.ifBlank { g.engines.badge() }, Kz.Accent) }
        Hud(ui)
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (s.replayClip != null) ReplayFrameView(ui, Modifier.fillMaxSize())
            else CameraPreview(g, CameraController.Mode.ANALYSIS, Modifier.fillMaxSize()) {
                InspectOverlay(ui)
                CircleLayer(g, ui, enabled = !ui.needsSheet)
            }
            Column(Modifier.align(Alignment.TopCenter).padding(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                if (s.replayClip != null) Chip("REPLAY — identical pipeline, file frames", Kz.Warn)
                if (ui.lineTooFast) Banner("LINE TOO FAST — judge queue is growing", Kz.Defect)
                ui.banner?.let { Banner(it, Kz.Warn) }
                ui.message?.let { Banner(it, Kz.TextDim) }
                if (!ui.needsSheet) ui.hint?.let { Banner(it, Kz.Text) }
                if (ui.circled && s.replayClip == null) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Banner("Circled area only", Kz.Pass)
                        SecondaryButton("Whole view") { g.hub.clearCircle() }
                    }
                }
            }
            if (ui.needsSheet && s.replayClip == null) {
                Column(
                    Modifier.align(Alignment.Center).padding(20.dp).background(Kz.Bg.copy(alpha = 0.88f), androidx.compose.foundation.shape.RoundedCornerShape(12.dp)).padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Step 1 — learn the empty sheet", color = Kz.Text, fontWeight = FontWeight.Bold, fontSize = 17.sp)
                    Text(
                        "Phone fixed on its stand. Take the part and your hands OFF the sheet, then tap LEARN SHEET and wait 2 seconds. " +
                            "Exposure, colour and focus are then locked for this screen.",
                        color = Kz.TextDim, fontSize = 13.sp,
                    )
                    if (sheet.state == com.kaizeneye.v2.pipeline.SheetUi.State.CAPTURING) LinearProgressIndicator(progress = { sheet.progress }, modifier = Modifier.fillMaxWidth())
                    ui.hint?.let { Text(it, color = Kz.Warn, fontSize = 13.sp) }
                    Text("Zoom in until only the sheet is visible (use the same zoom as when teaching).", color = Kz.TextDim, fontSize = 12.sp)
                    ZoomControl(g, Modifier.fillMaxWidth())
                    PrimaryButton("LEARN SHEET", Modifier.fillMaxWidth(), enabled = sheet.state != com.kaizeneye.v2.pipeline.SheetUi.State.CAPTURING) {
                        g.hub.learnSheetAndStart()
                    }
                }
            }
            ui.replayProgress?.let { LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth().align(Alignment.BottomCenter)) }
        }
        Column(Modifier.fillMaxWidth().background(Kz.Surface).padding(10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            ui.calibration?.let { c ->
                Text(
                    "Calibration: ${c.valid}/${c.target} valid parts (presented ${c.presented}) · sanity ${c.sanityOk} · identity ${c.identityOk} · shape ${c.geometryOk}" +
                        (c.alphaPct?.let { " · bound ≤ ${"%.1f".format(it)} %" } ?: ""),
                    color = Kz.Text, fontSize = 13.sp,
                )
                Note("Present ≥ 40 DISTINCT good parts (not the taught one). The threshold is only ever raised.")
                PrimaryButton("FINISH CALIBRATION", Modifier.fillMaxWidth(), enabled = c.valid >= 1) {
                    g.hub.finishCalibration(); nav.replace(Screen.Certificate)
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Parts move", color = Kz.TextDim, fontSize = 12.sp)
                val current = if (g.prefs.triggerMode == "STEADY") "STEADY" else g.prefs.motion
                for ((key, label) in listOf("H" to "across", "V" to "down", "STEADY" to "held still")) {
                    androidx.compose.material3.FilterChip(
                        selected = current == key,
                        onClick = {
                            g.hub.setMotion(key) {
                                if (s.replayClip != null) g.hub.startReplay(s.replayClip, s.paced, s.mode) else g.hub.startLine(s.mode)
                            }
                        },
                        label = { Text(label, fontSize = 12.sp) },
                    )
                }
            }
            if (s.replayClip == null && !ui.needsSheet) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SecondaryButton("Re-learn sheet") { g.hub.learnSheetAndStart() }
                    SecondaryButton("Zoom") { g.hub.openInspect(s.mode); g.hub.releaseSheet() }
                    Text(sheet.detail, color = Kz.TextDim, fontSize = 11.sp, maxLines = 3, modifier = Modifier.weight(1f))
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Sensitivity ${"%.2f".format(ui.sensitivity)}×τ", color = Kz.TextDim, fontSize = 12.sp, modifier = Modifier.padding(end = 8.dp))
                Slider(value = ui.sensitivity, onValueChange = { g.hub.setSensitivity(it) }, valueRange = 0.8f..2.0f, modifier = Modifier.weight(1f))
            }
            ui.reject?.let { RejectCardView(it, onExplain = { g.hub.explainLast() }) }
        }
    }
}

@Composable
private fun Hud(ui: InspectUi) {
    val c = ui.counters
    Row(
        Modifier.fillMaxWidth().background(Kz.Bg).padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        Stat("judged", c.judged.toString(), Kz.Text)
        Stat("pass", c.pass.toString(), Kz.Pass)
        Stat("defect", c.defect.toString(), Kz.Defect)
        Stat("not part", c.notEnrolled.toString(), Kz.NotEnrolled)
        Stat("reframe", c.reframe.toString(), Kz.Reframe)
        Column(Modifier.weight(1f), horizontalAlignment = Alignment.End) {
            Text(
                "${"%.0f".format(c.ppm)} ppm · ${"%.0f".format(ui.fps)} fps · p95 ${ui.latencyP95?.let { "%.0f ms".format(it) } ?: "–"}",
                color = Kz.TextDim, fontSize = 11.sp, fontFamily = Kz.Mono,
            )
            Text(ui.governor, color = Kz.TextDim, fontSize = 11.sp, fontFamily = Kz.Mono)
        }
    }
}

@Composable
private fun Stat(label: String, value: String, color: androidx.compose.ui.graphics.Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, color = color, fontSize = 18.sp, fontWeight = FontWeight.Bold, fontFamily = Kz.Mono)
        Text(label, color = Kz.TextDim, fontSize = 10.sp)
    }
}

@Composable
private fun Banner(text: String, color: androidx.compose.ui.graphics.Color) {
    Text(
        text, color = color, fontSize = 14.sp, fontWeight = FontWeight.Bold,
        modifier = Modifier.background(Kz.Bg.copy(alpha = 0.8f), RoundedCornerShape(8.dp)).padding(horizontal = 10.dp, vertical = 4.dp),
    )
}

@Composable
private fun RejectCardView(r: RejectCard, onExplain: () -> Unit) {
    Card(border = Kz.Defect) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            val bmp = remember(r.cropFile, r.number) { r.cropFile?.takeIf { it.isFile }?.let { BitmapFactory.decodeFile(it.absolutePath) } }
            bmp?.let { Image(it.asImageBitmap(), null, Modifier.size(84.dp)) }
            Column(Modifier.weight(1f)) {
                Text(r.title, color = Kz.Defect, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                Text(r.facts, color = Kz.TextDim, fontSize = 11.sp, fontFamily = Kz.Mono)
                Text(r.sentence, color = Kz.Text, fontSize = 13.sp)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Note("explained by ${r.sentenceSource}")
                    if (r.sentenceSource == "tap to explain") TextButton(onClick = onExplain) { Text("Explain") }
                }
            }
        }
    }
}
