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
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph

/** The part's false-alarm certificate: one plain sentence first, the gate counts and bounds below it. */
@Composable
fun CertificateScreen(g: AppGraph, nav: Navigator) {
    val active by g.hub.active.collectAsStateWithLifecycle()
    val lines = g.hub.certificateLines()
    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Certificate${active?.let { " · ${it.name}" } ?: ""}", onBack = { nav.back() })
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Card(border = if (active?.calibrated == true) Kz.Pass else Kz.Warn) {
                Text(active?.headline ?: "No part selected.", color = if (active?.calibrated == true) Kz.Pass else Kz.Warn, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                if (g.hub.certificateVoidAtCurrentTolerance()) {
                    Text("Not valid at the current tolerance — move the Inspect slider back towards 1.0×.", color = Kz.Defect, fontSize = 13.sp)
                }
            }
            Text("Technical details", color = Kz.TextDim, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            Card {
                lines.forEach { Text(it, color = Kz.TextDim, fontSize = 12.sp, fontFamily = Kz.Mono) }
            }
            Note(
                "α is a distribution-free order-statistic bound on the score gate: with m valid good parts the threshold is the largest of " +
                    "their scores, and P[false-alarm rate ≤ α] ≥ 95 %. Gate pass counts carry Clopper–Pearson bounds. The taught part itself is never used for calibration.",
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SecondaryButton("Calibrate", Modifier.weight(1f), enabled = active?.loadable == true) { nav.replace(Screen.Inspect(LineMode.CALIBRATE)) }
                SecondaryButton("Home", Modifier.weight(1f)) { nav.home() }
            }
        }
    }
}
