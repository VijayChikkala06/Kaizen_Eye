package com.kaizeneye.v2.ui

import androidx.compose.foundation.Canvas
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
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraController
import com.kaizeneye.v2.pipeline.SheetUi
import com.kaizeneye.v2.pipeline.TeachUi

/**
 * Teach flow (plan): empty-sheet tap → record 12 s (coverage ring, "slow down"/"turn the part" hints) → build (≤ 15 s)
 * → Armed (target ≤ 30 s from the tap) → optional negatives / calibrate.
 */
@Composable
fun TeachScreen(g: AppGraph, nav: Navigator) {
    val sheet by g.hub.sheet.collectAsStateWithLifecycle()
    val teach by g.hub.teach.collectAsStateWithLifecycle()
    val live by g.hub.teachView.collectAsStateWithLifecycle()
    androidx.compose.runtime.LaunchedEffect(Unit) { g.hub.openTeach() }
    androidx.compose.runtime.DisposableEffect(Unit) { onDispose { g.hub.closeTeach() } }
    var name by remember { mutableStateOf("Part ${System.currentTimeMillis() % 1000}") }

    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Teach a part", onBack = { g.hub.resetTeach(); nav.back() })
        CameraPreview(g, CameraController.Mode.ANALYSIS, Modifier.weight(1f).fillMaxWidth()) {
            val canCircle = sheet.state == SheetUi.State.READY && (teach is TeachUi.Idle || teach is TeachUi.Recording)
            if (canCircle) InspectOverlay(live)
            CircleLayer(g, live, enabled = canCircle)
            val t = teach
            if (t is TeachUi.Recording) {
                CoverageRing(t.coverage, Modifier.align(Alignment.TopEnd).padding(12.dp).size(84.dp))
                t.hint?.let {
                    Text(
                        it, color = Kz.Warn, fontSize = 18.sp, fontWeight = FontWeight.Bold,
                        modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp)
                            .background(Kz.Bg.copy(alpha = 0.7f), RoundedCornerShape(8.dp)).padding(horizontal = 12.dp, vertical = 6.dp),
                    )
                }
            }
        }
        Column(Modifier.fillMaxWidth().background(Kz.Surface).padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            when (val t = teach) {
                TeachUi.Idle -> {
                    if (sheet.state != SheetUi.State.READY) {
                        Text("Step 1 — learn the empty sheet", color = Kz.Text, fontWeight = FontWeight.SemiBold)
                        Note("Phone fixed on a stand, looking down at a plain matte sheet that contrasts with the part. Take the part and your hands off the sheet, then tap LEARN SHEET (2 s).", Kz.Text)
                        Note(sheet.detail)
                        Note("Something other than the sheet in view (table, stand, floor)? Zoom in until only the sheet is visible — only the zoomed view is used.")
                        ZoomControl(g, Modifier.fillMaxWidth())
                        if (sheet.state == SheetUi.State.CAPTURING) LinearProgressIndicator(progress = { sheet.progress }, modifier = Modifier.fillMaxWidth())
                        PrimaryButton("LEARN SHEET", Modifier.fillMaxWidth(), enabled = sheet.state != SheetUi.State.CAPTURING) { g.hub.captureSheet() }
                    } else {
                        Text("Step 2 — circle the part, then record it for 12 s", color = Kz.Text, fontWeight = FontWeight.SemiBold)
                        CircleHint(g, live.circled)
                        live.hint?.let { Text(it, color = if (live.boxes.firstOrNull()?.state == com.kaizeneye.v2.pipeline.OverlayState.PASS) Kz.Pass else Kz.Warn, fontSize = 13.sp) }
                        Note("Sheet: ${sheet.detail}${sheet.flicker?.let { " · $it" } ?: ""}")
                        Note("Keep the part fully in view; turn it slowly so every side that matters is seen.")
                        OutlinedTextField(value = name, onValueChange = { name = it.take(40) }, label = { Text("Part name") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            PrimaryButton("START RECORDING", Modifier.weight(1f)) { g.hub.startTeach(name.ifBlank { "Part" }) }
                            SecondaryButton("Zoom / sheet") { g.hub.releaseSheet() }
                        }
                    }
                }
                is TeachUi.Recording -> {
                    Text("Recording… ${t.elapsedMs / 1000}/${t.totalMs / 1000} s", color = Kz.Text, fontWeight = FontWeight.SemiBold)
                    LinearProgressIndicator(progress = { (t.elapsedMs.toFloat() / t.totalMs).coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                    Note("frames ${t.accepted}/${t.seen} usable · coverage ${(t.coverage * 100).toInt()} % · ${t.sanity}")
                    live.hint?.let { Text(it, color = if (live.boxes.firstOrNull()?.state == com.kaizeneye.v2.pipeline.OverlayState.PASS) Kz.Pass else Kz.Warn, fontSize = 13.sp) }
                    if (!live.circled) Note("Tip: circle the part with your finger so only it is recorded.")
                    SecondaryButton("Stop early", enabled = t.elapsedMs >= 6000) { g.hub.stopTeach() }
                }
                is TeachUi.Building -> {
                    Text("Building the Twin — ${t.stage}", color = Kz.Text, fontWeight = FontWeight.SemiBold)
                    LinearProgressIndicator(progress = { t.progress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                }
                is TeachUi.Armed -> {
                    Text("ARMED — ${t.twin.name}", color = Kz.Pass, fontWeight = FontWeight.Bold, fontSize = 18.sp)
                    Note("tap → armed in ${"%.1f".format(t.tapToArmedMs / 1000.0)} s (target ≤ 30 s)")
                    t.lines.forEach { Text(it, color = if (it.startsWith("⚠")) Kz.Warn else Kz.Text, fontSize = 13.sp, fontFamily = Kz.Mono) }
                    PrimaryButton("INSPECT NOW", Modifier.fillMaxWidth()) { nav.replace(Screen.Inspect(LineMode.INSPECT)) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SecondaryButton("Add negatives", Modifier.weight(1f)) { nav.replace(Screen.Negatives) }
                        SecondaryButton("Calibrate", Modifier.weight(1f)) { nav.replace(Screen.Inspect(LineMode.CALIBRATE)) }
                    }
                }
                is TeachUi.Failed -> {
                    Text("Teach failed", color = Kz.Defect, fontWeight = FontWeight.Bold)
                    Note(t.reason, Kz.Text)
                    PrimaryButton("TRY AGAIN", Modifier.fillMaxWidth()) { g.hub.resetTeach() }
                }
            }
        }
    }
}

/** "Circle the part" instruction, or (once circled) what that means plus a way back to automatic selection. */
@Composable
fun CircleHint(g: AppGraph, circled: Boolean) {
    if (!circled) {
        Note("Draw a circle around the part with your finger (or tap it). Only what is inside is used — the table, stand or hands outside are ignored.", Kz.Text)
    } else {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("✓ Circled — only this part is used", color = Kz.Pass, fontSize = 13.sp, modifier = Modifier.weight(1f))
            SecondaryButton("Whole view") { g.hub.clearCircle() }
        }
    }
}

@Composable
fun CoverageRing(fraction: Float, modifier: Modifier = Modifier) {
    Box(modifier, contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            val stroke = 10f
            val d = size.minDimension - stroke
            val tl = Offset((size.width - d) / 2, (size.height - d) / 2)
            drawArc(Kz.Line, 0f, 360f, false, tl, Size(d, d), style = Stroke(stroke))
            drawArc(if (fraction >= 1f) Kz.Pass else Kz.Accent, -90f, 360f * fraction.coerceIn(0f, 1f), false, tl, Size(d, d), style = Stroke(stroke))
        }
        Text("${(fraction * 100).toInt()}%", color = Kz.Text, fontSize = 15.sp, fontWeight = FontWeight.Bold)
    }
}
