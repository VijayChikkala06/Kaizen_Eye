package com.kaizeneye.core.track

// SPEC-QUESTION: (§11.3 "crossFrames consecutive matched frames"): frames in which the track is missed are not matched
//   frames, so they neither count nor break the downstream run; only a matched frame on the upstream side resets it.
// SPEC-QUESTION: (§11.3/§11.4/§11.5 order within one frame): association and state update (§11.2) first, then for every
//   matched track in id order: best-crop buffer (§11.5, so the frame that fires can itself be the best crop), then LINE,
//   then STEADY. "If both triggers are enabled, the first one wins" = LINE is checked before STEADY in the same frame.
// SPEC-QUESTION: (§11.4 "its own maximum sharpness so far"): the maximum over every matched frame of the track up to
//   and including the current one (unconfirmed and border-touching frames included). Including or excluding the current
//   frame gives the same decision.
// SPEC-QUESTION: (§11.5): buffering stops once a track is judged (its frames were already handed to the judge), so no
//   best-crop notices are emitted for judged tracks.
// SPEC-QUESTION: (§11.5 STEADY frames): the judge uses the firing frame; the voting candidates (§9) are the other
//   buffered frames by quality, so FiredTrigger.frames = [firing frame] + best others.
// SPEC-QUESTION: (§11.2 order): matched tracks are updated, then unmatched tracks age (missed += 1) and are removed
//   (EXITED) when missed > maxMissed, then unmatched detections open new tracks in detection order.

import kotlin.math.sqrt

/**
 * The live-line centroid tracker with its two triggers (spec §11): greedy constant-velocity association with distance
 * gating (§11.2), the virtual photo-eye LINE trigger (§11.3), the STEADY hold trigger (§11.4) and the per-track
 * best-crop buffer (§11.5). Analysis-resolution coordinates (`width × height`), timestamps from the frames (ms).
 *
 * Call [update] once per analysis frame with all components of that frame (border-touching ones included). Tracks live
 * in a recycled pool; the per-frame work allocates only the returned [FrameUpdate] (and its small lists). Not
 * thread-safe: use it from the analyzer thread.
 */
class Tracker(val width: Int, val height: Int, val config: TriggerConfig = TriggerConfig()) {

    /** The line coordinate `L` (§11.3). */
    val linePosition: Double = config.position ?: if (config.axis == Axis.X) 0.5 * width else 0.5 * height

    private val centreX = width / 2.0
    private val centreY = height / 2.0
    private val halfDiagonal = 0.5 * sqrt(width.toDouble() * width + height.toDouble() * height)

    private val live = ArrayList<Track>()
    private val pool = ArrayList<Track>()
    private var nextId = 1

    // Association scratch (grown on demand, never shrunk).
    private var pairTrack = IntArray(64)
    private var pairDet = IntArray(64)
    private var pairDist = DoubleArray(64)
    private var order = IntArray(64)
    private var trackDet = IntArray(16)
    private var detTrack = IntArray(16)

    init {
        require(width > 0 && height > 0) { "tracker needs a non-empty frame, got ${width}x$height" }
    }

    /** Number of live tracks. */
    val liveTracks: Int get() = live.size

    /** Forgets every track and restarts ids at 1 (e.g. when leaving Inspect). */
    fun reset() {
        pool.addAll(live)
        live.clear()
        nextId = 1
    }

    /**
     * Marks live track [id] as judged by an operator trigger (the part was circled on screen), so LINE / STEADY never fire
     * for it afterwards. Returns false if no such live track exists.
     */
    fun markJudged(id: Int): Boolean {
        val tr = live.firstOrNull { it.id == id } ?: return false
        tr.judged = true
        return true
    }

    /** Processes one frame at time [tMs] (ms, from the frame) with its [detections]. */
    fun update(tMs: Long, detections: List<Detection>): FrameUpdate {
        val nT = live.size
        val nD = detections.size
        ensureCapacity(nT, nD)

        // §11.2 gating: predicted position and gate per live track; candidate pairs in (track id, detection) order.
        var np = 0
        for (k in 0 until nT) {
            val tr = live[k]
            trackDet[k] = -1
            val dt = (tMs - tr.lastT).toDouble()
            val px = tr.cx + tr.vx * dt
            val py = tr.cy + tr.vy * dt
            val speed = sqrt(tr.vx * tr.vx + tr.vy * tr.vy)
            val gate = config.gateBase + config.gateArea * sqrt(tr.area.toDouble()) + config.gateSpeed * speed * dt
            for (d in 0 until nD) {
                val c = detections[d].component
                val dx = c.cx - px
                val dy = c.cy - py
                val dist = sqrt(dx * dx + dy * dy)
                if (dist <= gate) {
                    pairTrack[np] = k
                    pairDet[np] = d
                    pairDist[np] = dist
                    order[np] = np
                    np++
                }
            }
        }
        for (d in 0 until nD) detTrack[d] = -1

        // Stable insertion sort by distance: equal distances keep generation order = lower track id, then detection.
        for (i in 1 until np) {
            val p = order[i]
            val dp = pairDist[p]
            var j = i - 1
            while (j >= 0 && pairDist[order[j]] > dp) {
                order[j + 1] = order[j]
                j--
            }
            order[j + 1] = p
        }
        for (i in 0 until np) {
            val p = order[i]
            val k = pairTrack[p]
            val d = pairDet[p]
            if (trackDet[k] < 0 && detTrack[d] < 0) {
                trackDet[k] = d
                detTrack[d] = k
            }
        }

        // Matched tracks update; unmatched ones age.
        val assignments = IntArray(nD)
        for (k in 0 until nT) {
            val tr = live[k]
            val d = trackDet[k]
            if (d >= 0) {
                applyMatch(tr, detections[d], tMs)
                assignments[d] = tr.id
            } else {
                tr.matched = false
                tr.justConfirmed = false
                tr.missed++
            }
        }

        // Removal (EXITED), preserving id order.
        var exits: ArrayList<TrackExit>? = null
        var w = 0
        for (k in 0 until nT) {
            val tr = live[k]
            if (tr.missed > config.maxMissed) {
                if (exits == null) exits = ArrayList(2)
                exits.add(TrackExit(tr.id, tr.judged, tr.confirmed))
                pool.add(tr)
            } else {
                live[w++] = tr
            }
        }
        while (live.size > w) live.removeAt(live.size - 1)

        // Unmatched detections open new tracks, in detection order.
        for (d in 0 until nD) {
            if (detTrack[d] >= 0) continue
            val tr = if (pool.isEmpty()) Track(config.bestFrames) else pool.removeAt(pool.size - 1)
            startTrack(tr, detections[d], tMs)
            live.add(tr)
            assignments[d] = tr.id
        }

        // Best crop and triggers for every matched track, in id order.
        var triggers: ArrayList<FiredTrigger>? = null
        var notices: ArrayList<BestCropNotice>? = null
        for (k in 0 until live.size) {
            val tr = live[k]
            if (!tr.matched || !tr.confirmed || tr.judged) continue
            val notice = offerBestCrop(tr, tMs)
            if (notice != null) {
                if (notices == null) notices = ArrayList(2)
                notices.add(notice)
            }
            var fired: FiredTrigger? = null
            if (config.lineEnabled) fired = checkLine(tr, tMs)
            if (fired == null && config.steadyEnabled) fired = checkSteady(tr, tMs)
            if (fired != null) {
                if (triggers == null) triggers = ArrayList(1)
                triggers.add(fired)
            }
        }

        val views = ArrayList<TrackView>(live.size)
        for (k in 0 until live.size) views.add(live[k].view())
        return FrameUpdate(tMs, assignments, triggers ?: emptyList(), notices ?: emptyList(), exits ?: emptyList(), views)
    }

    // ------------------------------------------------------------------------------------------------------------------

    /** §11.2 matched update. */
    private fun applyMatch(tr: Track, det: Detection, t: Long) {
        val c = det.component
        if (t != tr.lastT) {
            val dt = (t - tr.lastT).toDouble()
            if (tr.hits == 1) {
                tr.vx = (c.cx - tr.cx) / dt
                tr.vy = (c.cy - tr.cy) / dt
            } else {
                val a = config.velocityAlpha
                tr.vx = a * (c.cx - tr.cx) / dt + (1 - a) * tr.vx
                tr.vy = a * (c.cy - tr.cy) / dt + (1 - a) * tr.vy
            }
        }
        tr.setDetection(det)
        tr.hits++
        tr.missed = 0
        tr.lastT = t
        val was = tr.confirmed
        tr.confirmed = tr.hits >= config.minHits
        tr.justConfirmed = !was && tr.confirmed
        tr.matched = true
    }

    /** §11.2 new track for an unmatched detection. */
    private fun startTrack(tr: Track, det: Detection, t: Long) {
        tr.id = nextId++
        tr.vx = 0.0
        tr.vy = 0.0
        tr.maxSharp = det.sharpness
        tr.setDetection(det)
        tr.hits = 1
        tr.missed = 0
        tr.lastT = t
        tr.confirmed = 1 >= config.minHits
        tr.justConfirmed = tr.confirmed
        tr.judged = false
        tr.matched = true
        tr.upstream = when (config.direction) {
            Direction.POSITIVE -> NEGATIVE_SIDE
            Direction.NEGATIVE -> POSITIVE_SIDE
            Direction.ANY -> 0
        }
        tr.seenUpstream = false
        tr.lineBlocked = false
        tr.downFrames = 0
        tr.stillSince = NO_TIME
        tr.bestN = 0
    }

    /** §11.5: quality of the current frame, insert into the best-`bestFrames` buffer (ties → earlier). */
    private fun offerBestCrop(tr: Track, t: Long): BestCropNotice? {
        if (tr.touchesBorder) {
            tr.quality = Double.NaN
            return null
        }
        val dx = tr.cx - centreX
        val dy = tr.cy - centreY
        val centrality = 1 - minOf(1.0, sqrt(dx * dx + dy * dy) / halfDiagonal)
        val q = tr.sharpness * centrality
        tr.quality = q
        val cap = config.bestFrames
        var pos = tr.bestN
        for (i in 0 until tr.bestN) {
            if (q > tr.bestQ[i]) {
                pos = i
                break
            }
        }
        if (pos >= cap) return null
        val evicted: Long? = if (tr.bestN == cap) tr.bestRef[cap - 1] else null
        val last = minOf(tr.bestN, cap - 1)
        for (i in last downTo pos + 1) {
            tr.bestQ[i] = tr.bestQ[i - 1]
            tr.bestRef[i] = tr.bestRef[i - 1]
            tr.bestT[i] = tr.bestT[i - 1]
        }
        tr.bestQ[pos] = q
        tr.bestRef[pos] = tr.frameRef
        tr.bestT[pos] = t
        if (tr.bestN < cap) tr.bestN++
        return BestCropNotice(tr.id, tr.frameRef, q, evicted)
    }

    /** §11.3 virtual photo-eye. */
    private fun checkLine(tr: Track, t: Long): FiredTrigger? {
        val coord = if (config.axis == Axis.X) tr.cx else tr.cy
        val side = if (coord > linePosition) POSITIVE_SIDE else NEGATIVE_SIDE // exactly on the line = negative side
        if (tr.justConfirmed) {
            if (config.direction == Direction.ANY) tr.upstream = side
            else if (side != tr.upstream) tr.lineBlocked = true // first confirmed downstream: never fires
        }
        if (tr.lineBlocked) return null
        if (side == tr.upstream) {
            tr.seenUpstream = true
            tr.downFrames = 0
            return null
        }
        if (!tr.seenUpstream) return null
        tr.downFrames++
        if (tr.downFrames < config.crossFrames) return null
        tr.judged = true
        return FiredTrigger(tr.id, Trigger.LINE, t, tr.buffered(excludeT = NO_TIME, limit = config.bestFrames))
    }

    /** §11.4 steady hold. */
    private fun checkSteady(tr: Track, t: Long): FiredTrigger? {
        val speed = sqrt(tr.vx * tr.vx + tr.vy * tr.vy)
        val ok = !tr.touchesBorder && speed < config.vStill && tr.sharpness >= config.sharpRatio * tr.maxSharp
        if (!ok) {
            tr.stillSince = NO_TIME
            return null
        }
        if (tr.stillSince == NO_TIME) tr.stillSince = t
        if ((t - tr.stillSince).toDouble() < config.holdMs) return null
        tr.judged = true
        val frames = ArrayList<BufferedFrame>(config.bestFrames)
        frames.add(BufferedFrame(tr.frameRef, tr.quality, t))
        frames.addAll(tr.buffered(excludeT = t, limit = config.bestFrames - 1))
        return FiredTrigger(tr.id, Trigger.STEADY, t, frames, tr.buffered(excludeT = NO_TIME, limit = config.bestFrames))
    }

    private fun ensureCapacity(nT: Int, nD: Int) {
        val pairs = nT * nD
        if (pairs > pairTrack.size) {
            val n = maxOf(pairs, pairTrack.size * 2)
            pairTrack = IntArray(n)
            pairDet = IntArray(n)
            pairDist = DoubleArray(n)
            order = IntArray(n)
        }
        if (nT > trackDet.size) trackDet = IntArray(maxOf(nT, trackDet.size * 2))
        if (nD > detTrack.size) detTrack = IntArray(maxOf(nD, detTrack.size * 2))
    }

    /** Mutable per-track state, recycled through the pool. */
    private class Track(cap: Int) {
        var id = 0
        var cx = 0.0
        var cy = 0.0
        var vx = 0.0
        var vy = 0.0
        var area = 0
        var minX = 0
        var minY = 0
        var maxX = 0
        var maxY = 0
        var touchesBorder = false
        var hits = 0
        var missed = 0
        var lastT = 0L
        var confirmed = false
        var justConfirmed = false
        var judged = false
        var matched = false

        var sharpness = 0.0
        var maxSharp = 0.0
        var frameRef = 0L
        var quality = Double.NaN

        var upstream = 0
        var seenUpstream = false
        var lineBlocked = false
        var downFrames = 0

        var stillSince = NO_TIME

        val bestQ = DoubleArray(cap)
        val bestRef = LongArray(cap)
        val bestT = LongArray(cap)
        var bestN = 0

        fun setDetection(det: Detection) {
            val c = det.component
            cx = c.cx
            cy = c.cy
            area = c.area
            minX = c.minX
            minY = c.minY
            maxX = c.maxX
            maxY = c.maxY
            touchesBorder = c.touchesBorder
            sharpness = det.sharpness
            if (det.sharpness > maxSharp) maxSharp = det.sharpness
            frameRef = det.frameRef
        }

        /** Buffered frames by quality (best first), skipping the one taken at [excludeT], at most [limit]. */
        fun buffered(excludeT: Long, limit: Int): List<BufferedFrame> {
            if (bestN == 0 || limit <= 0) return emptyList()
            val out = ArrayList<BufferedFrame>(minOf(bestN, limit))
            for (i in 0 until bestN) {
                if (bestT[i] == excludeT) continue
                if (out.size >= limit) break
                out.add(BufferedFrame(bestRef[i], bestQ[i], bestT[i]))
            }
            return out
        }

        fun view(): TrackView = TrackView(
            id = id, cx = cx, cy = cy, vx = vx, vy = vy,
            minX = minX, minY = minY, maxX = maxX, maxY = maxY, area = area, hits = hits,
            confirmed = confirmed, judged = judged, touchesBorder = touchesBorder, missed = missed, lastT = lastT,
            state = when {
                judged -> TrackState.JUDGED
                !confirmed -> TrackState.TENTATIVE
                missed > 0 -> TrackState.COASTING
                stillSince != NO_TIME -> TrackState.HOLDING
                else -> TrackState.TRACKING
            },
        )
    }

    private companion object {
        const val NO_TIME = Long.MIN_VALUE
        const val POSITIVE_SIDE = 1
        const val NEGATIVE_SIDE = -1
    }
}
