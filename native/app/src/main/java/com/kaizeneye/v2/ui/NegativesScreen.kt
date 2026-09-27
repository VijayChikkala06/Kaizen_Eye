package com.kaizeneye.v2.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraController
import com.kaizeneye.v2.pipeline.SheetUi

/**
 * Wrong objects (the negatives library): show 3–50 objects that are NOT the part, one by one; each steady hold adds one.
 * They tighten the identity and look-alike gates. Never reuse them as test "wrong objects" in an evaluation.
 */
@Composable
fun NegativesScreen(g: AppGraph, nav: Navigator) {
    val ui by g.hub.negatives.collectAsStateWithLifecycle()
    val sheet by g.hub.sheet.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { g.hub.openNegatives() }
    DisposableEffect(Unit) { onDispose { g.hub.closeCameraScreen() } }
    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Teach wrong objects", onBack = { nav.back() })
        CameraPreview(g, CameraController.Mode.ANALYSIS, Modifier.weight(1f).fillMaxWidth())
        Column(Modifier.fillMaxWidth().background(Kz.Surface).padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (sheet.state != SheetUi.State.READY) {
                Text("Step 1 — take everything off the sheet, then LEARN SHEET", color = Kz.Text, fontWeight = FontWeight.SemiBold)
                if (sheet.state == SheetUi.State.CAPTURING) LinearProgressIndicator(progress = { sheet.progress }, modifier = Modifier.fillMaxWidth())
                if (sheet.state == SheetUi.State.FAILED) Text(sheet.detail, color = Kz.Defect, fontSize = 13.sp)
                PrimaryButton("LEARN SHEET", Modifier.fillMaxWidth(), enabled = sheet.state != SheetUi.State.CAPTURING) {
                    g.hub.learnSheetForNegatives()
                }
            } else {
                Text("Step 2 — show one WRONG object and hold it still for ½ s", color = Kz.Text, fontWeight = FontWeight.SemiBold)
                Note("${ui.count} wrong object${if (ui.count == 1) "" else "s"} learned (${ui.sessionAdded} now). Look-alikes of the part help most.")
            }
            ui.lastMessage?.let { Text(it, color = Kz.Accent, fontSize = 13.sp) }
            Note("Afterwards, calibrate again: adding wrong objects changes the gates, so the certificate needs a fresh calibration.")
            PrimaryButton("DONE", Modifier.fillMaxWidth()) { nav.back() }
        }
    }
}
