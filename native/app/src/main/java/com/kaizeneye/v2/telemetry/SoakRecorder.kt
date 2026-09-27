package com.kaizeneye.v2.telemetry

import com.kaizeneye.v2.AppGraph
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Soak / energy evidence (plan B7): every 2 s one CSV row with thermal status, headroom, battery °C, whole-phone W, the
 * governor level, analysis fps and trigger→verdict p95. Runs on the app scope so it survives screen changes.
 */
class SoakRecorder(private val g: AppGraph) {
    private var job: Job? = null
    private val _file = MutableStateFlow<File?>(null)
    val file: StateFlow<File?> = _file

    val running: Boolean get() = job?.isActive == true

    fun start(label: String) {
        if (running) return
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val f = File(g.dirs.logs, "soak_${stamp}_$label.csv")
        _file.value = f
        job = g.scope.launch {
            CsvLog(f, listOf("elapsedMs", "thermalStatus", "headroom", "headroomAgeMs", "batteryC", "currentUa", "voltageMv", "watts", "charging", "capacityPct", "governor", "fps", "latencyP50", "latencyP95", "judged", "accel")).use { csv ->
                while (isActive) {
                    val s = g.thermal.refresh()
                    val ui = g.hub.inspect.value
                    csv.row(listOf(s.elapsedMs, s.statusName, s.headroom, s.headroomAgeMs, s.batteryTempC, s.currentUa, s.voltageMv, s.watts, s.charging, s.capacityPct, ui.governor, "%.1f".format(ui.fps), ui.latencyP50, ui.latencyP95, ui.counters.judged, g.engines.badge()))
                    delay(2000)
                }
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }
}
