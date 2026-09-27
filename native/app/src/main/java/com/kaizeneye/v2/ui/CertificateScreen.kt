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
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph

/** The Twin's false-alarm certificate (plan "Calibrate"): score-gate bound, gate counts with CP bounds, identity margin, latency. */
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
                lines.forEach { Text(it, color = Kz.Text, fontSize = 14.sp, fontFamily = Kz.Mono) }
            }
            Note("α is a distribution-free order-statistic bound on the SCORE gate only: with m valid good parts, the threshold is the largest of their scores, and P[false-alarm rate ≤ α] ≥ 95 %. Gate pass counts are reported separately with Clopper–Pearson bounds. The taught part itself is never used for calibration.")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SecondaryButton("Calibrate", Modifier.weight(1f), enabled = active?.loadable == true) { nav.replace(Screen.Inspect(LineMode.CALIBRATE)) }
                SecondaryButton("Home", Modifier.weight(1f)) { nav.home() }
            }
            Row(Modifier.fillMaxWidth()) { }
        }
    }
}
