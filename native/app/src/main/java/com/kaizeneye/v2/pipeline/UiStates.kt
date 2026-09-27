package com.kaizeneye.v2.pipeline

import android.graphics.Bitmap
import java.io.File

/** Empty-sheet model status (plan: "Empty-sheet tap" + flicker check). */
data class SheetUi(
    val state: State = State.NONE,
    val detail: String = "Clear the sheet and tap LEARN SHEET",
    val flicker: String? = null,
    val progress: Float = 0f,
) {
    enum class State { NONE, CAPTURING, READY, FAILED }
}

/** One stored Twin as the Home / list screens show it. */
data class TwinSummary(
    val id: String,
    val name: String,
    val createdAtMs: Long,
    val keyframes: Int,
    val calibrated: Boolean,
    val tau: Double,
    val tauId: Double,
    val headline: String,
    val loadable: Boolean,
    val problem: String? = null,
    /** FIT gate (look-alike protection) status line, null for a Twin without one. */
    val fitLine: String? = null,
)

sealed interface TeachUi {
    data object Idle : TeachUi
    data class Recording(
        val elapsedMs: Long,
        val totalMs: Long,
        val seen: Int,
        val accepted: Int,
        val coverage: Float,
        val hint: String?,
        val sanity: String,
    ) : TeachUi
    data class Building(val stage: String, val progress: Float) : TeachUi
    data class Armed(val twin: TwinSummary, val tapToArmedMs: Long, val lines: List<String>) : TeachUi
    data class Failed(val reason: String) : TeachUi
}

enum class OverlayState { TRACKING, JUDGING, PASS, DEFECT, NOT_ENROLLED, REFRAME }

/** Heat map riding a track: [values] in 0..1 on the backbone grid, placed on the crop square (analysis px, sensor orientation). */
class HeatOverlay(
    val gh: Int, val gw: Int, val values: FloatArray, val x0: Float, val y0: Float, val side: Float,
    /** Canonical crops: the heat grid is rotated by this angle (radians) about the centre of the (x0, y0, side) square. */
    val rot: Float = 0f,
)

/** A box to draw, in ANALYSIS pixels (sensor orientation); the overlay rotates/scales it to the view. */
data class OverlayBox(
    val trackId: Int,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val state: OverlayState,
    val label: String?,
    val heat: HeatOverlay?,
)

data class LineOverlay(val axis: String, val position: Float)

data class RejectCard(
    val number: Int,
    val title: String,
    val facts: String,
    val sentence: String,
    val sentenceSource: String,
    val cropFile: File?,
    val tMs: Long,
)

data class CountersUi(
    val judged: Int = 0,
    val pass: Int = 0,
    val defect: Int = 0,
    val notEnrolled: Int = 0,
    val reframe: Int = 0,
    val exitedUnjudged: Int = 0,
    val ppm: Double = 0.0,
)

data class CalibrationUi(
    val presented: Int,
    val valid: Int,
    val target: Int,
    val sanityOk: Int,
    val identityOk: Int,
    val geometryOk: Int,
    val alphaPct: Double?,
    val lastRawOverTau: Double?,
)

data class InspectUi(
    val running: Boolean = false,
    val mode: String = "",
    val source: String = "",
    val twinName: String = "",
    val analysisW: Int = 0,
    val analysisH: Int = 0,
    val rotation: Int = 0,
    val boxes: List<OverlayBox> = emptyList(),
    val line: LineOverlay? = null,
    val counters: CountersUi = CountersUi(),
    val fps: Double = 0.0,
    val loopMsP95: Double = 0.0,
    val latencyP50: Double? = null,
    val latencyP95: Double? = null,
    val accel: String = "",
    val governor: String = "",
    val lineTooFast: Boolean = false,
    val banner: String? = null,
    val reject: RejectCard? = null,
    val calibration: CalibrationUi? = null,
    val replayProgress: Float? = null,
    val message: String? = null,
    val sensitivity: Float = 1f,
    /** Replay only: a small copy of the current frame (there is no camera preview). */
    val replayFrame: Bitmap? = null,
    /** The empty sheet has not been learned on this camera session yet: show the LEARN SHEET step. */
    val needsSheet: Boolean = false,
    /** Plain-language next step for the operator ("slide it across the dashed line", "too many blobs → re-learn"). */
    val hint: String? = null,
    /** The operator circled a part / area: only what is inside the circle is analysed. */
    val circled: Boolean = false,
)

data class NegativesUi(val count: Int, val sessionAdded: Int, val lastMessage: String?, val tauIdLine: String?)

data class SelfTestUi(
    val running: Boolean = false,
    val steps: List<Pair<String, String>> = emptyList(),
    val resultFile: String? = null,
    val pass: Boolean? = null,
)
