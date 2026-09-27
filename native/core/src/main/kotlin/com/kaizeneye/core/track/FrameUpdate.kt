package com.kaizeneye.core.track

/** The two live triggers (spec §11.3–§11.4). */
enum class Trigger { LINE, STEADY }

/** Coarse track state for the overlay (grey tracking → amber holding/judging). */
enum class TrackState {
    /** Fewer than `minHits` matched frames. */
    TENTATIVE,

    /** Confirmed, visible this frame, not judged. */
    TRACKING,

    /** Confirmed and inside a steady-hold run (§11.4 clock running). */
    HOLDING,

    /** Confirmed, not judged, missed in the latest frame(s) (shown at its predicted position). */
    COASTING,

    /** A trigger has fired for this track (it is being / has been judged). */
    JUDGED,
}

/** A buffered frame of a track (spec §11.5): the caller's [frameRef], its quality `q = sharpness · centrality`, its time. */
data class BufferedFrame(val frameRef: Long, val quality: Double, val tMs: Long)

/**
 * A trigger that fired this frame (spec §11.3–§11.5). [frames] are in judging order: for `LINE` the buffered frames by
 * quality (best first, ties → earlier); for `STEADY` the firing frame first, then the other buffered frames by quality.
 * The judge uses `frames[0]`; the rest are the borderline-voting candidates (§9). An empty list means the track never had
 * a bufferable frame → judge it REFRAME (`TOUCHES_BORDER`, §11.5). [buffer] is the track's best-crop buffer at the
 * moment it fired (best first, ties → earlier), which equals [frames] for LINE.
 */
class FiredTrigger(
    val trackId: Int,
    val trigger: Trigger,
    val tMs: Long,
    val frames: List<BufferedFrame>,
    val buffer: List<BufferedFrame> = frames,
) {
    /** The frame to judge, or null → REFRAME (`TOUCHES_BORDER`). */
    val judgeFrame: BufferedFrame? get() = frames.firstOrNull()

    /** §11.5: fired with no buffered frame → REFRAME (`TOUCHES_BORDER`). */
    val needsReframe: Boolean get() = frames.isEmpty()

    override fun toString(): String = "FiredTrigger(track=$trackId, $trigger, t=$tMs, frames=$frames, buffer=$buffer)"
}

/**
 * The current frame ([frameRef]) entered track [trackId]'s best-crop buffer with [quality]: copy its crop bytes now (the
 * camera frame is recycled). [evictedFrameRef] is the buffered frame that dropped out (free its copy), if any.
 */
data class BestCropNotice(val trackId: Int, val frameRef: Long, val quality: Double, val evictedFrameRef: Long?)

/** A track was removed after more than `maxMissed` missed frames (spec §11.2 `EXITED`). */
data class TrackExit(val trackId: Int, val judged: Boolean, val confirmed: Boolean)

/** Overlay snapshot of one live track (analysis pixels; velocity in px/ms). */
data class TrackView(
    val id: Int,
    val cx: Double,
    val cy: Double,
    val vx: Double,
    val vy: Double,
    val minX: Int,
    val minY: Int,
    val maxX: Int,
    val maxY: Int,
    val area: Int,
    val hits: Int,
    val confirmed: Boolean,
    val judged: Boolean,
    val touchesBorder: Boolean,
    val missed: Int,
    val lastT: Long,
    val state: TrackState,
) {
    /** Spec §11.6 displayed box centre x: `pos + vel · latencyMs`. */
    fun displayCx(latencyMs: Double): Double = cx + vx * latencyMs

    /** Spec §11.6 displayed box centre y: `pos + vel · latencyMs`. */
    fun displayCy(latencyMs: Double): Double = cy + vy * latencyMs
}

/**
 * Everything the tracker decided for one frame (spec §11). Lists are empty (shared) when nothing happened.
 * [assignments]`[i]` is the id of the track that detection `i` (input order) was assigned to or opened.
 */
class FrameUpdate(
    val tMs: Long,
    val assignments: IntArray,
    val triggers: List<FiredTrigger>,
    val bestCrops: List<BestCropNotice>,
    val exits: List<TrackExit>,
    val tracks: List<TrackView>,
) {
    override fun toString(): String =
        "FrameUpdate(t=$tMs, triggers=$triggers, bestCrops=$bestCrops, exits=$exits, tracks=${tracks.size})"
}
