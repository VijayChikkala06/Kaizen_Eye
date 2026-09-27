package com.kaizeneye.v2.ui

import android.util.Log
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraController
import com.kaizeneye.v2.pipeline.InspectUi
import com.kaizeneye.v2.pipeline.OverlayState
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.max
import kotlin.math.min

/** Live camera preview (FIT_CENTER, 16:9) bound to [mode]; frames go to [AppGraph.hub]'s router while [routeFrames]. */
@Composable
fun CameraPreview(g: AppGraph, mode: CameraController.Mode, modifier: Modifier = Modifier, routeFrames: Boolean = true, overlay: @Composable BoxScope.() -> Unit = {}) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(ctx).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
    }
    Box(modifier) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        overlay()
    }
    LaunchedEffect(mode) {
        try {
            g.camera.setConsumer(if (routeFrames && mode == CameraController.Mode.ANALYSIS) g.hub.router else null)
            g.camera.setZoom(g.prefs.zoomRatio)
            g.camera.bind(owner, previewView, mode, g.prefs.cameraPreset)
        } catch (t: Throwable) {
            Log.e("KaizenUi", "camera bind failed", t)
        }
    }
    DisposableEffect(Unit) {
        onDispose {
            g.camera.setConsumer(null)
            g.camera.unbind()
        }
    }
}

/**
 * Zoom slider for the "learn the sheet" step: zoom in until only the sheet fills the view (nothing around it is analysed).
 * The same zoom is used for Teach, Inspect, Calibrate and Negatives (saved in [com.kaizeneye.v2.Prefs.zoomRatio]).
 */
@Composable
fun ZoomControl(g: AppGraph, modifier: Modifier = Modifier) {
    val maxZoom by g.camera.maxZoom.collectAsStateWithLifecycle()
    var zoom by remember { mutableFloatStateOf(g.prefs.zoomRatio) }
    androidx.compose.foundation.layout.Row(modifier, verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
        androidx.compose.material3.Text(
            "Zoom ${"%.1f".format(zoom)}×", color = Kz.Text, fontSize = 13.sp,
            modifier = Modifier.padding(end = 8.dp),
        )
        androidx.compose.material3.Slider(
            value = zoom.coerceIn(1f, maxZoom),
            onValueChange = { zoom = it; g.camera.setZoom(it) },
            onValueChangeFinished = { g.prefs.zoomRatio = zoom },
            valueRange = 1f..maxOf(maxZoom, 1.01f),
            modifier = Modifier.weight(1f),
        )
    }
}

private enum class DoodlePhase { NONE, DRAWING, WAITING, DONE }

/**
 * "Circle to select" (like circle-to-search): draw around a part — or tap it — on the live preview. The stroke is drawn as
 * a marker doodle; the app looks for the part inside it on the next frame, the doodle turns green (found) or red (nothing
 * inside) with a short message, then fades out. [ui] supplies the frame size / rotation for mapping view → frame pixels.
 */
@Composable
fun CircleLayer(g: AppGraph, ui: InspectUi, enabled: Boolean, modifier: Modifier = Modifier) {
    val result by g.hub.circle.collectAsStateWithLifecycle()
    val points = remember { mutableStateListOf<Offset>() }
    var strokeId by remember { mutableLongStateOf(0L) }
    var phase by remember { mutableStateOf(DoodlePhase.NONE) }
    val alpha = remember { Animatable(0f) }
    val scope = rememberCoroutineScope()
    val latestUi by rememberUpdatedState(ui)
    val res = result?.takeIf { it.strokeId == strokeId }

    LaunchedEffect(strokeId, phase, res) {
        if (phase == DoodlePhase.WAITING) {
            if (res == null) delay(3000)
            phase = DoodlePhase.DONE
        } else if (phase == DoodlePhase.DONE) {
            delay(if (res?.ok == true) 1500 else 2200)
            alpha.animateTo(0f, tween(900))
            points.clear()
            phase = DoodlePhase.NONE
        }
    }

    val doodleColor = when {
        phase == DoodlePhase.DONE && res?.ok == true -> Kz.Pass
        phase == DoodlePhase.DONE && res != null -> Kz.Defect
        else -> Color.White
    }
    Box(
        modifier.fillMaxSize().pointerInput(enabled) {
            if (!enabled) return@pointerInput
            awaitEachGesture {
                val down = awaitFirstDown()
                down.consume()
                strokeId = System.nanoTime()
                points.clear()
                points.add(down.position)
                phase = DoodlePhase.DRAWING
                scope.launch { alpha.snapTo(1f) }
                while (true) {
                    val ev = awaitPointerEvent()
                    val ch = ev.changes.firstOrNull { it.id == down.id } ?: break
                    if (!ch.pressed) break
                    if ((ch.position - points.last()).getDistance() > 3f) points.add(ch.position)
                    ch.consume()
                }
                phase = DoodlePhase.WAITING
                val u = latestUi
                if (u.analysisW > 0 && u.analysisH > 0) {
                    val m = ViewMapper(u.analysisW, u.analysisH, u.rotation, size.width.toFloat(), size.height.toFloat())
                    val fr = points.map { m.unmap(it.x, it.y) }
                    g.hub.circle(FloatArray(fr.size) { fr[it].x }, FloatArray(fr.size) { fr[it].y }, strokeId)
                } else {
                    g.hub.circle(FloatArray(0), FloatArray(0), strokeId)
                }
            }
        },
    ) {
        Canvas(Modifier.fillMaxSize()) {
            if (points.isEmpty() || phase == DoodlePhase.NONE) return@Canvas
            val a = alpha.value
            if (points.size == 1) {
                drawCircle(doodleColor.copy(alpha = 0.35f * a), 26f, points[0])
                drawCircle(doodleColor.copy(alpha = a), 14f, points[0], style = Stroke(6f))
                return@Canvas
            }
            val path = androidx.compose.ui.graphics.Path().apply {
                moveTo(points[0].x, points[0].y)
                for (i in 1 until points.size) {
                    // Smooth marker look: quadratic curves through the midpoints.
                    val p0 = points[i - 1]
                    val p1 = points[i]
                    quadraticTo(p0.x, p0.y, (p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
                }
                lineTo(points.last().x, points.last().y)
            }
            val round = Stroke(width = 26f, cap = androidx.compose.ui.graphics.StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round)
            val core = Stroke(width = 9f, cap = androidx.compose.ui.graphics.StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round)
            drawPath(path, doodleColor.copy(alpha = 0.28f * a), style = round)
            drawPath(path, doodleColor.copy(alpha = a), style = core)
        }
        if (phase == DoodlePhase.DONE && res != null) {
            androidx.compose.material3.Text(
                res.message,
                color = if (res.ok) Kz.Pass else Kz.Defect, fontSize = 15.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.align(androidx.compose.ui.Alignment.BottomCenter).padding(12.dp)
                    .graphicsLayer { this.alpha = alpha.value }
                    .background(Kz.Bg.copy(alpha = 0.8f), androidx.compose.foundation.shape.RoundedCornerShape(10.dp))
                    .padding(horizontal = 12.dp, vertical = 6.dp),
            )
        }
    }
}

/**
 * Maps FRAME pixels (sensor orientation, frame W×H) to view pixels: rotate upright by [rotation] (clockwise degrees, the
 * CameraX imageInfo convention), then scale uniformly and centre (PreviewView FIT_CENTER).
 */
class ViewMapper(private val frameW: Int, private val frameH: Int, private val rotation: Int, viewW: Float, viewH: Float) {
    private val uprightW = if (rotation % 180 == 0) frameW else frameH
    private val uprightH = if (rotation % 180 == 0) frameH else frameW
    private val s = min(viewW / max(1, uprightW), viewH / max(1, uprightH))
    private val ox = (viewW - uprightW * s) / 2f
    private val oy = (viewH - uprightH * s) / 2f

    fun map(x: Float, y: Float): Offset {
        val (ux, uy) = when (((rotation % 360) + 360) % 360) {
            90 -> Pair(frameH - y, x)
            180 -> Pair(frameW - x, frameH - y)
            270 -> Pair(y, frameW - x)
            else -> Pair(x, y)
        }
        return Offset(ox + ux * s, oy + uy * s)
    }

    /** Inverse of [map]: view pixels → FRAME pixels (clamped to the frame). */
    fun unmap(vx: Float, vy: Float): Offset {
        val ux = (vx - ox) / s
        val uy = (vy - oy) / s
        val (x, y) = when (((rotation % 360) + 360) % 360) {
            90 -> Pair(uy, frameH - ux)
            180 -> Pair(frameW - ux, frameH - uy)
            270 -> Pair(frameW - uy, ux)
            else -> Pair(ux, uy)
        }
        return Offset(x.coerceIn(0f, frameW.toFloat()), y.coerceIn(0f, frameH.toFloat()))
    }

    fun mapRect(l: Float, t: Float, r: Float, b: Float): Rect {
        val a = map(l, t)
        val c = map(r, b)
        return Rect(min(a.x, c.x), min(a.y, c.y), max(a.x, c.x), max(a.y, c.y))
    }

    val scale: Float get() = s
}

fun colorFor(s: OverlayState): Color = when (s) {
    OverlayState.TRACKING -> Kz.Tracking
    OverlayState.JUDGING -> Kz.Judging
    OverlayState.PASS -> Kz.Pass
    OverlayState.DEFECT -> Kz.Defect
    OverlayState.NOT_ENROLLED -> Kz.NotEnrolled
    OverlayState.REFRAME -> Kz.Reframe
}

/** Tracks, heat maps and the photo-eye line on top of the preview (or the replay frame). */
@Composable
fun InspectOverlay(ui: InspectUi, modifier: Modifier = Modifier) {
    val measurer = rememberTextMeasurer()
    Canvas(modifier.fillMaxSize()) {
        if (ui.analysisW <= 0 || ui.analysisH <= 0) return@Canvas
        val m = ViewMapper(ui.analysisW, ui.analysisH, ui.rotation, size.width, size.height)
        ui.line?.let { line ->
            val (a, b) = if (line.axis == "X") m.map(line.position, 0f) to m.map(line.position, ui.analysisH.toFloat())
            else m.map(0f, line.position) to m.map(ui.analysisW.toFloat(), line.position)
            drawLine(Kz.Line, a, b, strokeWidth = 3f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(18f, 12f)))
        }
        for (box in ui.boxes) {
            val c = colorFor(box.state)
            box.heat?.let { h -> drawHeat(m, h.gh, h.gw, h.values, h.x0, h.y0, h.side) }
            val r = m.mapRect(box.left, box.top, box.right, box.bottom)
            val stroke = when (box.state) {
                OverlayState.DEFECT, OverlayState.NOT_ENROLLED -> 9f
                OverlayState.TRACKING -> 2f
                else -> 5f
            }
            val effect = if (box.state == OverlayState.REFRAME) PathEffect.dashPathEffect(floatArrayOf(14f, 10f)) else null
            drawRect(c, r.topLeft, r.size, style = Stroke(width = stroke, pathEffect = effect))
            box.label?.let { label ->
                drawText(
                    measurer, label,
                    topLeft = Offset(r.left, max(0f, r.top - 44f)),
                    style = TextStyle(color = c, fontSize = 15.sp, fontWeight = FontWeight.Bold),
                )
            }
        }
    }
}

/** Heat cells (values 0..1) over the crop square; only the hot part (> 0.35) is tinted so the part stays visible. */
private fun DrawScope.drawHeat(m: ViewMapper, gh: Int, gw: Int, v: FloatArray, x0: Float, y0: Float, side: Float) {
    val cw = side / gw
    val ch = side / gh
    for (r in 0 until gh) for (c in 0 until gw) {
        val a = v[r * gw + c]
        if (a <= 0.35f) continue
        val rect = m.mapRect(x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch)
        drawRect(Kz.Defect.copy(alpha = (0.15f + 0.5f * a).coerceAtMost(0.65f)), rect.topLeft, Size(rect.width + 0.5f, rect.height + 0.5f))
    }
}

/** Replay has no camera preview: show the frame the pipeline is processing (a small copy published by the pipeline). */
@Composable
fun ReplayFrameView(ui: InspectUi, modifier: Modifier = Modifier) {
    Box(modifier) {
        ui.replayFrame?.let { bmp ->
            val rot = ui.rotation.toFloat()
            Canvas(Modifier.fillMaxSize()) {
                val img = bmp.asImageBitmap()
                val m = ViewMapper(bmp.width, bmp.height, ui.rotation, size.width, size.height)
                val dst = m.mapRect(0f, 0f, bmp.width.toFloat(), bmp.height.toFloat())
                if (rot == 0f) {
                    drawImage(img, dstOffset = androidx.compose.ui.unit.IntOffset(dst.left.toInt(), dst.top.toInt()), dstSize = androidx.compose.ui.unit.IntSize(dst.width.toInt(), dst.height.toInt()))
                } else {
                    // Rotate about the view centre; for 90/270 the drawn (unrotated) rectangle has swapped sides.
                    val w = if (ui.rotation % 180 == 0) dst.width else dst.height
                    val h = if (ui.rotation % 180 == 0) dst.height else dst.width
                    val cx = dst.center.x
                    val cy = dst.center.y
                    rotate(rot, Offset(cx, cy)) {
                        drawImage(img, dstOffset = androidx.compose.ui.unit.IntOffset((cx - w / 2).toInt(), (cy - h / 2).toInt()), dstSize = androidx.compose.ui.unit.IntSize(w.toInt(), h.toInt()))
                    }
                }
            }
        }
        InspectOverlay(ui)
    }
}

@Suppress("unused")
@Composable
private fun StaticBitmap(bmp: android.graphics.Bitmap) = Image(bmp.asImageBitmap(), null, contentScale = ContentScale.Fit)
