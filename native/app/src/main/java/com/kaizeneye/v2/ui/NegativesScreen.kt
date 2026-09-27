package com.kaizeneye.v2.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
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

/**
 * Negatives library capture (plan): show 3–50 WRONG objects one by one; each steady hold adds its object descriptor.
 * τ_id is then placed midway between the good parts' 5th percentile and the negatives' 99th percentile.
 * Objects used here must never be used as test "wrong objects" in the evaluation.
 */
@Composable
fun NegativesScreen(g: AppGraph, nav: Navigator) {
    val ui by g.hub.negatives.collectAsStateWithLifecycle()
    val sheet by g.hub.sheet.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { g.hub.openNegatives() }
    DisposableEffect(Unit) { onDispose { g.hub.stopNegatives() } }
    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Negatives library", onBack = { nav.back() })
        CameraPreview(g, CameraController.Mode.ANALYSIS, Modifier.weight(1f).fillMaxWidth())
        Column(Modifier.fillMaxWidth().background(Kz.Surface).padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (sheet.state != com.kaizeneye.v2.pipeline.SheetUi.State.READY) {
                Text("Step 1 — take everything off the sheet, then LEARN SHEET", color = Kz.Text, fontWeight = FontWeight.SemiBold)
                if (sheet.state == com.kaizeneye.v2.pipeline.SheetUi.State.CAPTURING) {
                    androidx.compose.material3.LinearProgressIndicator(progress = { sheet.progress }, modifier = Modifier.fillMaxWidth())
                }
                ZoomControl(g, Modifier.fillMaxWidth())
                PrimaryButton("LEARN SHEET", Modifier.fillMaxWidth(), enabled = sheet.state != com.kaizeneye.v2.pipeline.SheetUi.State.CAPTURING) {
                    g.hub.learnSheetForNegatives()
                }
            }
            Text("Show a WRONG object and hold it still for ½ s", color = Kz.Text, fontWeight = FontWeight.SemiBold)
            Note("Each steady hold adds one negative (${ui.sessionAdded} this session, ${ui.count} in the library for this pipeline).")
            ui.lastMessage?.let { Text(it, color = Kz.Accent, fontSize = 13.sp) }
            ui.tauIdLine?.let { Text(it, color = Kz.Text, fontSize = 13.sp, fontFamily = Kz.Mono) }
            PrimaryButton("DONE", Modifier.fillMaxWidth()) { nav.back() }
        }
    }
}
