package com.kaizeneye.v2.camera

/**
 * One analysis frame as delivered by a [FrameSource]: RGBA_8888 (pixel stride 4) in SENSOR orientation, exactly as CameraX's
 * ImageAnalysis hands it over (replay sources produce the same layout). [rotationDegrees] rotates it upright for display.
 *
 * The [data] array is REUSED for the next frame: a consumer copies whatever it keeps (crops, thumbnails) before returning.
 * [tMs] is the frame's own timestamp (sensor clock or file timestamp) — the pipeline never reads the wall clock (twin-spec §11),
 * which is what makes REPLAY deterministic.
 */
class CameraFrame(
    val tMs: Long,
    val width: Int,
    val height: Int,
    val rowStride: Int,
    val data: ByteArray,
    val rotationDegrees: Int,
    val kind: FrameSourceKind,
    val index: Long,
)

enum class FrameSourceKind { CAMERA, REPLAY }

fun interface FrameConsumer {
    /** Called on the source's worker thread, one frame at a time. Must return quickly in live mode (KEEP_ONLY_LATEST drops the rest). */
    fun onFrame(frame: CameraFrame)
}

interface FrameSource {
    val kind: FrameSourceKind
    fun setConsumer(consumer: FrameConsumer?)
}
