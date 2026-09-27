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
import androidx.compose.foundation.layout.height
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
    DisposableEffect(Unit) { onDispose { g.hub.closeCameraScreen() } }

    Column(Modifier.fillMaxSize()) {
        ScreenHeader(
            title = when {
                s.replayClip != null -> "REPLAY · ${s.replayClip.name}"
                s.mode == LineMode.CALIBRATE -> "Calibrate · ${ui.twinName}"
                else -> "Inspect · ${ui.twinName}"
            },
            onBack = { nav.back() },
        ) { if (s.replayClip != null) Chip(ui.accel.ifBlank { g.engines.badge() }, Kz.Accent) }
        Hud(ui)
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (s.replayClip != null) ReplayFrameView(ui, Modifier.fillMaxSize())
            else CameraPreview(g, CameraController.Mode.ANALYSIS, Modifier.fillMaxSize()) {
                InspectOverlay(ui)
                CircleLayer(g, ui, enabled = !ui.needsSheet)
            }
            Column(Modifier.align(Alignment.TopCenter).padding(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                if (s.replayClip != null) Chip("REPLAY — identical pipeline, file frames", Kz.Warn)
                if (ui.lineTooFast) Banner("Too fast — slow the line down", Kz.Defect)
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
                    if (sheet.state == com.kaizeneye.v2.pipeline.SheetUi.State.FAILED) Text(sheet.detail, color = Kz.Defect, fontSize = 13.sp)
                    else ui.hint?.let { Text(it, color = Kz.Warn, fontSize = 13.sp) }
                    Note("The zoom set while teaching is used again here.")
                    PrimaryButton("LEARN SHEET", Modifier.fillMaxWidth(), enabled = sheet.state != com.kaizeneye.v2.pipeline.SheetUi.State.CAPTURING) {
                        g.hub.learnSheetAndStart()
                    }
                }
            }
            ui.replayProgress?.let { LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth().align(Alignment.BottomCenter)) }
        }
        Column(Modifier.fillMaxWidth().background(Kz.Surface).padding(10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val calibrating = s.mode == LineMode.CALIBRATE
            ui.calibration?.let { c ->
                Text("Calibration: ${c.valid} / ${c.target} good parts", color = Kz.Text, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                LinearProgressIndicator(progress = { (c.valid.toFloat() / c.target).coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                Note("Show ${c.target} different GOOD parts, one at a time — not the one you taught. Nothing else changes while calibrating.")
                val enough = c.valid >= c.target
                PrimaryButton(
                    if (enough) "FINISH CALIBRATION" else "FINISH WITH ${c.valid} / ${c.target}", Modifier.fillMaxWidth(),
                    enabled = c.valid >= 1, color = if (enough) Kz.Accent else Kz.Warn,
                ) { g.hub.finishCalibration(); nav.replace(Screen.Certificate) }
            }
            if (!calibrating && s.replayClip == null && !ui.needsSheet) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SecondaryButton("Re-learn sheet", enabled = sheet.state != com.kaizeneye.v2.pipeline.SheetUi.State.CAPTURING) { g.hub.learnSheetAndStart() }
                    if (sheet.detail.contains("NOT plain")) Text("Background not plain", color = Kz.Warn, fontSize = 12.sp)
                }
            }
            if (!calibrating) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.padding(end = 8.dp)) {
                        Text("Tolerance ${"%.1f".format(ui.sensitivity)}×", color = Kz.TextDim, fontSize = 12.sp)
                        Text("◀ stricter · forgiving ▶", color = Kz.TextDim, fontSize = 9.sp)
                    }
                    Slider(
                        value = ui.sensitivity, onValueChange = { g.hub.setSensitivity(it) },
                        onValueChangeFinished = { g.hub.setSensitivity(g.hub.inspect.value.sensitivity, persist = true) },
                        valueRange = 0.5f..2.5f, modifier = Modifier.weight(1f),
                    )
                }
                if (g.hub.certificateVoidAtCurrentTolerance()) Text("Certificate not valid at this tolerance (move the slider back towards 1.0×)", color = Kz.Warn, fontSize = 11.sp)
                // A fixed slot for the last reject, so the camera view never jumps when the first reject appears.
                Box(Modifier.fillMaxWidth().height(104.dp)) {
                    ui.reject?.let { RejectCardView(it, onExplain = { g.hub.explainLast() }) }
                        ?: Note("The last REJECT / NOT THE PART is explained here.", Kz.TextDim)
                }
            }
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
        Stat("PASS", c.pass.toString(), Kz.Pass)
        Stat("REJECT", c.defect.toString(), Kz.Defect)
        Stat("NOT THE PART", c.notEnrolled.toString(), Kz.NotEnrolled)
        Column(Modifier.weight(1f), horizontalAlignment = Alignment.End) {
            Text("${c.judged} judged", color = Kz.TextDim, fontSize = 11.sp, fontFamily = Kz.Mono)
            ui.latencyP95?.let { Text("${"%.0f".format(it)} ms per part", color = Kz.TextDim, fontSize = 11.sp, fontFamily = Kz.Mono) }
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
                Text(r.sentence, color = Kz.Text, fontSize = 13.sp, maxLines = 3)
                if (r.sentenceSource == "tap to explain") TextButton(onClick = onExplain) { Text("Explain") }
            }
        }
    }
}
