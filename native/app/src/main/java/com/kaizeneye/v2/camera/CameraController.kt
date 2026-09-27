package com.kaizeneye.v2.camera

import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.min

/** What the camera actually delivers (from capture results), for the HUD, the device passport and focus locking. */
data class CameraStats(
    val analysisSize: String = "-",
    val rotationDegrees: Int = 0,
    val fps: Double = 0.0,
    val exposureNs: Long? = null,
    val iso: Int? = null,
    val frameDurationNs: Long? = null,
    val focusDiopters: Float? = null,
    val aeState: Int? = null,
    val preset: String = "-",
    val framesDelivered: Long = 0,
    val framesSkipped: Long = 0,
    val recording: Boolean = false,
)

/**
 * CameraX front end: Preview (+ ImageAnalysis RGBA_8888 1280×720 KEEP_ONLY_LATEST in [Mode.ANALYSIS], or VideoCapture in
 * [Mode.RECORD]) on the back camera, both 16:9 so the PreviewView (FIT_CENTER) and the analysis frames show the same field
 * of view and overlay coordinates map 1:1 after rotation. Capture settings ([CameraPreset]) are applied through
 * Camera2CameraControl so they can change at run time (governor fps, flicker fallback, locks).
 */
@OptIn(ExperimentalCamera2Interop::class)
class CameraController(private val context: Context) : FrameSource {

    enum class Mode { ANALYSIS, RECORD }

    override val kind = FrameSourceKind.CAMERA

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "kz-analysis").apply { priority = Thread.MAX_PRIORITY }
    }
    @Volatile private var consumer: FrameConsumer? = null
    private var provider: ProcessCameraProvider? = null
    var camera: Camera? = null
        private set
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var preset = CameraPreset()
    @Volatile private var analysisFpsCap = 30
    private var buffer = ByteArray(0)
    private var frameIndex = 0L
    private var lastProcessedMs = Long.MIN_VALUE
    private var skipped = 0L
    private var fpsWindowStart = 0L
    private var fpsWindowCount = 0
    @Volatile private var lastResult: TotalCaptureResult? = null

    /** Zoom ratio applied at every bind and by [setZoom]; the analysis frames see exactly the zoomed view. */
    @Volatile var zoomRatio: Float = 1f
        private set
    private val _maxZoom = MutableStateFlow(4f)
    /** Largest usable zoom of the bound camera (capped at [MAX_USEFUL_ZOOM]: beyond it the picture is too soft to judge). */
    val maxZoom: StateFlow<Float> = _maxZoom

    private val _stats = MutableStateFlow(CameraStats())
    val stats: StateFlow<CameraStats> = _stats

    override fun setConsumer(consumer: FrameConsumer?) {
        this.consumer = consumer
    }

    /** Software frame-rate cap for analysis (thermal governor / idle); the sensor keeps its preset rate. */
    fun setAnalysisFpsCap(fps: Int) {
        analysisFpsCap = fps.coerceIn(1, 60)
    }

    suspend fun bind(owner: LifecycleOwner, previewView: PreviewView, mode: Mode, preset: CameraPreset) {
        val p = provider ?: awaitProvider(context).also { provider = it }
        this.preset = preset
        p.unbindAll()
        recording?.stop()
        recording = null
        videoCapture = null

        val ratio16x9 = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .build()
        val previewBuilder = Preview.Builder().setResolutionSelector(ratio16x9)
        // The session capture callback sees every capture result (exposure, ISO, focus distance) for the HUD and focus lock.
        Camera2Interop.Extender(previewBuilder).setSessionCaptureCallback(object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                lastResult = result
            }
        })
        val preview = previewBuilder.build()
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val useCases = mutableListOf<UseCase>(preview)
        when (mode) {
            Mode.ANALYSIS -> {
                val sel = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(preset.analysisWidth, preset.analysisHeight),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                        ),
                    )
                    .build()
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(sel)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(analysisExecutor) { proxy -> onImage(proxy) }
                useCases += analysis
            }
            Mode.RECORD -> {
                val recorder = Recorder.Builder()
                    .setQualitySelector(QualitySelector.from(Quality.HD, FallbackStrategy.lowerQualityOrHigherThan(Quality.HD)))
                    .build()
                val vc = VideoCapture.withOutput(recorder)
                videoCapture = vc
                useCases += vc
            }
        }
        val cam = p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, *useCases.toTypedArray())
        camera = cam
        applyPreset(preset)
        cam.cameraInfo.zoomState.value?.let { _maxZoom.value = it.maxZoomRatio.coerceIn(1f, MAX_USEFUL_ZOOM) }
        setZoom(zoomRatio)
    }

    /** Zoom in so only the sheet is in view (what is outside the zoomed view is never analysed). */
    fun setZoom(ratio: Float) {
        zoomRatio = ratio.coerceIn(1f, if (camera == null) MAX_USEFUL_ZOOM else _maxZoom.value)
        val cam = camera ?: return
        try {
            cam.cameraControl.setZoomRatio(zoomRatio)
        } catch (t: Throwable) {
            Log.w(TAG, "setZoom failed", t)
        }
    }

    fun unbind() {
        recording?.stop()
        recording = null
        provider?.unbindAll()
        camera = null
    }

    /** Applies (or re-applies) capture settings on the running session. */
    fun applyPreset(preset: CameraPreset) {
        this.preset = preset
        val cam = camera ?: return
        try {
            Camera2CameraControl.from(cam.cameraControl).setCaptureRequestOptions(optionsFor(preset))
            _stats.value = _stats.value.copy(preset = preset.describe())
        } catch (t: Throwable) {
            Log.w(TAG, "applyPreset failed", t)
        }
    }

    fun currentPreset(): CameraPreset = preset

    /**
     * Before learning the empty sheet: let auto exposure, white balance and (unless the user fixed it) focus converge on the
     * sheet, starting from the user's base preset.
     */
    fun unlockForSheet(base: CameraPreset) {
        applyPreset(base.copy(aeLock = false, awbLock = false))
    }

    /**
     * After learning the empty sheet: freeze exposure, white balance and focus for the rest of this camera session, so a
     * part entering the view cannot shift the background brightness/colour away from the learned sheet model (the most
     * common cause of "boxes everywhere"). Manual-exposure presets are already fixed; AE lock applies to auto exposure.
     */
    fun lockForInspection(): CameraPreset {
        val p = preset
        val focus = p.focusDiopters ?: lastResult?.get(CaptureResult.LENS_FOCUS_DISTANCE)
        val next = p.copy(aeLock = !p.manualExposure, awbLock = true, focusDiopters = focus)
        applyPreset(next)
        return next
    }

    /** Freezes focus at the distance the continuous AF has converged to (read from the last capture result). */
    fun lockFocusAtCurrent(): CameraPreset {
        val d = lastResult?.get(CaptureResult.LENS_FOCUS_DISTANCE)
        val next = preset.copy(focusDiopters = d ?: preset.focusDiopters)
        applyPreset(next)
        return next
    }

    /** Characteristics helper (exposure/ISO ranges) for the camera tuning UI. */
    fun exposureRangeNs(): Range<Long>? = camera?.let {
        Camera2CameraInfo.from(it.cameraInfo).getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
    }

    fun isoRange(): Range<Int>? = camera?.let {
        Camera2CameraInfo.from(it.cameraInfo).getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
    }

    fun startRecording(file: File, onFinished: (File?, String?) -> Unit): Boolean {
        val vc = videoCapture ?: return false
        if (recording != null) return false
        file.parentFile?.mkdirs()
        recording = vc.output.prepareRecording(context, FileOutputOptions.Builder(file).build())
            .start(ContextCompat.getMainExecutor(context)) { ev ->
                if (ev is VideoRecordEvent.Finalize) {
                    recording = null
                    _stats.value = _stats.value.copy(recording = false)
                    if (ev.hasError()) onFinished(null, "recording error ${ev.error}: ${ev.cause?.message ?: ""}")
                    else onFinished(file, null)
                }
            }
        _stats.value = _stats.value.copy(recording = true)
        return true
    }

    fun stopRecording() {
        recording?.stop()
    }

    private fun onImage(proxy: ImageProxy) {
        try {
            val c = consumer ?: return
            val tMs = proxy.imageInfo.timestamp / 1_000_000L
            val minGap = 1000L / analysisFpsCap - 2
            if (lastProcessedMs != Long.MIN_VALUE && tMs - lastProcessedMs < minGap) {
                skipped++
                return
            }
            lastProcessedMs = tMs
            val plane = proxy.planes[0]
            val rowStride = plane.rowStride
            val h = proxy.height
            val need = rowStride * h
            if (buffer.size < need) buffer = ByteArray(need)
            val buf = plane.buffer
            buf.rewind()
            buf.get(buffer, 0, min(buf.remaining(), need))
            frameIndex++
            countFps(tMs, proxy)
            c.onFrame(CameraFrame(tMs, proxy.width, h, rowStride, buffer, proxy.imageInfo.rotationDegrees, kind, frameIndex))
        } catch (t: Throwable) {
            Log.e(TAG, "analysis frame failed", t)
        } finally {
            proxy.close()
        }
    }

    private fun countFps(tMs: Long, proxy: ImageProxy) {
        if (fpsWindowCount == 0) fpsWindowStart = tMs
        fpsWindowCount++
        val span = tMs - fpsWindowStart
        if (span >= 1000) {
            val r = lastResult
            _stats.value = _stats.value.copy(
                analysisSize = "${proxy.width}x${proxy.height}",
                rotationDegrees = proxy.imageInfo.rotationDegrees,
                fps = (fpsWindowCount - 1) * 1000.0 / span,
                exposureNs = r?.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                iso = r?.get(CaptureResult.SENSOR_SENSITIVITY),
                frameDurationNs = r?.get(CaptureResult.SENSOR_FRAME_DURATION),
                focusDiopters = r?.get(CaptureResult.LENS_FOCUS_DISTANCE),
                aeState = r?.get(CaptureResult.CONTROL_AE_STATE),
                framesDelivered = frameIndex,
                framesSkipped = skipped,
            )
            fpsWindowCount = 0
        }
    }

    fun shutdown() {
        unbind()
        analysisExecutor.shutdown()
    }

    companion object {
        private const val TAG = "KaizenCamera"

        const val MAX_USEFUL_ZOOM = 5f

        fun optionsFor(p: CameraPreset): CaptureRequestOptions {
            val b = CaptureRequestOptions.Builder()
            b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(p.fps, p.fps))
            b.setCaptureRequestOption(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            if (p.manualExposure) {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                b.setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, p.exposureNs)
                b.setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, p.iso)
                b.setCaptureRequestOption(CaptureRequest.SENSOR_FRAME_DURATION, 1_000_000_000L / p.fps)
            } else {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, p.aeLock)
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_ANTIBANDING_MODE, CaptureRequest.CONTROL_AE_ANTIBANDING_MODE_AUTO)
            }
            b.setCaptureRequestOption(CaptureRequest.CONTROL_AWB_LOCK, p.awbLock)
            b.setCaptureRequestOption(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF) // never the torch, whatever a saved preset says
            val focus = p.focusDiopters
            if (focus != null) {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                b.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, focus)
            } else {
                b.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
            }
            return b.build()
        }

        suspend fun awaitProvider(context: Context): ProcessCameraProvider = suspendCancellableCoroutine { cont ->
            val f = ProcessCameraProvider.getInstance(context)
            f.addListener({
                try {
                    cont.resume(f.get())
                } catch (t: Throwable) {
                    cont.resumeWithException(t)
                }
            }, ContextCompat.getMainExecutor(context))
        }
    }
}
