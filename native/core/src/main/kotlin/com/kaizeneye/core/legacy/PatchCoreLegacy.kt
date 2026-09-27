package com.kaizeneye.core.legacy

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Legacy PatchCore scoring maths: a line-by-line port of `mobile/src/core/patchcore.ts` (itself a port of
 * `tools/lab/patchcore_ref.py`), checked against `testdata/golden_core.json` + `golden_app.json`. Not redefined by the
 * Twin spec (twin-spec.md, header) — keep it byte-for-byte faithful to the TypeScript, including its float32 storage:
 *   - squared distances accumulate in float64 over float32 inputs, 8 terms per partial-distance check;
 *   - the coreset's running min-distances and all distance maps are stored as float32 (JS `Float32Array`);
 *   - the global descriptor accumulates in float32 (JS `Float32Array` `+=`).
 *
 * Pipeline: `k = max(minK, floor(ratio·N))` greedy k-centre coreset (start 0, ties → lowest index), leave-one-frame-out
 * threshold `tau = max LOO · margin`, score = max patch distance / (tau · sensitivity).
 */
object PatchCoreLegacy {

    /** Minimum whole-image threshold, so near-identical enrolment frames do not make the check hair-trigger. */
    const val MIN_GLOBAL_TAU: Double = 0.02

    /**
     * Optional accelerated nearest-neighbour backend. Returns, for every patch of [feats] (P·dim), the squared distance
     * to the nearest bank row whose bankFrame != [skipFrame] (-1 = use all rows).
     */
    fun interface NnBackend {
        fun minSq(feats: FloatArray, bank: FloatArray, bankFrame: IntArray, dim: Int, skipFrame: Int): FloatArray
    }

    /** Enrolment options (patchcore.ts `EnrolOptions`). */
    data class EnrolOptions(
        val ratio: Double = 0.05,
        val minK: Int = 256,
        val margin: Double = 1.0,
        /** Ignore the outer `border` patch rows/columns in frame scores (0 = spec). */
        val border: Int = 0,
        /** Drop enrolment frames whose LOO score is an outlier (median + 3 MAD, max 20 %) and re-enrol without them. */
        val trimOutliers: Boolean = false,
        /** Frame score on the 3x3-smoothed distance map (patchcore_ref smooth=True). */
        val smooth: Boolean = false,
    )

    /** Enrolled profile (patchcore.ts `Profile`). */
    class Profile(
        val gh: Int,
        val gw: Int,
        val dim: Int,
        /** Memory bank, [k·dim] row-major. */
        val bank: FloatArray,
        /** Enrolment frame index each bank row came from, [k]. */
        val bankFrame: IntArray,
        /** Threshold = max LOO score · margin. */
        val tau: Double,
        val margin: Double,
        val looScores: List<Double>,
        val nFrames: Int,
        val createdAt: Long,
        /** L2-normalised mean feature vector per enrolment frame, [nFrames·dim]. */
        val globals: FloatArray?,
        val gTau: Double?,
        val gLoo: List<Double>?,
        val border: Int,
        val smooth: Boolean,
        /** Indices (into the frames passed to [enrol]) dropped as outliers; null when trimming was off. */
        val droppedFrames: List<Int>? = null,
    ) {
        fun withDropped(dropped: List<Int>) = Profile(
            gh, gw, dim, bank, bankFrame, tau, margin, looScores, nFrames, createdAt, globals, gTau, gLoo, border, smooth,
            dropped,
        )
    }

    /** Result of [score] (patchcore.ts `ScoreResult`). */
    class ScoreResult(
        /** Nearest-neighbour distance per patch, [gh·gw] (float32 like the TS Float32Array). */
        val dmap: FloatArray,
        /** Frame score (max of the optionally smoothed map, border excluded). */
        val raw: Double,
        /** max(patchScore, globalScore); > 1.0 = REJECT. */
        val score: Double,
        /** raw / (tau · sensitivity). */
        val patchScore: Double,
        /** Whole-image cosine distance / (gTau · sensitivity); 0 when the profile has no globals. */
        val globalScore: Double,
        /** "ok" | "defect" | "different". */
        val reason: String,
        val peakRow: Int,
        val peakCol: Int,
    )

    /** True when patch index p (row-major on a gh x gw grid) is inside the scored area. */
    fun inside(p: Int, gh: Int, gw: Int, border: Int): Boolean {
        if (border <= 0) return true
        val r = p / gw
        val c = p % gw
        return r >= border && r < gh - border && c >= border && c < gw - border
    }

    /** Squared distance between row a of x and row b of y (offsets in floats), abandoning once it reaches [limit]. */
    fun sqdistBounded(x: FloatArray, a: Int, y: FloatArray, b: Int, dim: Int, limit: Double): Double {
        var acc = 0.0
        var t = 0
        val end8 = dim - (dim % 8)
        while (t < end8) {
            val d0 = x[a + t].toDouble() - y[b + t].toDouble()
            val d1 = x[a + t + 1].toDouble() - y[b + t + 1].toDouble()
            val d2 = x[a + t + 2].toDouble() - y[b + t + 2].toDouble()
            val d3 = x[a + t + 3].toDouble() - y[b + t + 3].toDouble()
            val d4 = x[a + t + 4].toDouble() - y[b + t + 4].toDouble()
            val d5 = x[a + t + 5].toDouble() - y[b + t + 5].toDouble()
            val d6 = x[a + t + 6].toDouble() - y[b + t + 6].toDouble()
            val d7 = x[a + t + 7].toDouble() - y[b + t + 7].toDouble()
            acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7
            if (acc >= limit) return acc
            t += 8
        }
        while (t < dim) {
            val d = x[a + t].toDouble() - y[b + t].toDouble()
            acc += d * d
            t++
        }
        return acc
    }

    private fun prune(mind: Float): Double = 2 * sqrt(mind.toDouble()) * (1 + 1e-6) + 1e-12

    /**
     * Greedy k-centre coreset over the rows of [x] ([n·dim]). Deterministic: starts at [start], ties → lowest index
     * (like np.argmax). Uses the TS's exact triangle-inequality pruning and per-block maxima.
     */
    fun greedyCoreset(x: FloatArray, dim: Int, k: Int, start: Int = 0): IntArray {
        val n = x.size / dim
        val kk = min(k, n)
        if (kk <= 0) return IntArray(0)
        val sel = IntArray(kk)
        sel[0] = start
        val mind = FloatArray(n)
        // assign[i] = which selected centre (index into sel) is currently nearest to point i.
        val assign = IntArray(n)
        // cc[t] = distance from the newest centre to centre sel[t].
        val cc = DoubleArray(kk)
        // prune[i] = 2 * sqrt(mind[i]) with a tiny safety margin (see the triangle-inequality test below).
        val prune = DoubleArray(n)
        val s0 = start * dim
        for (i in 0 until n) {
            mind[i] = sqdistBounded(x, i * dim, x, s0, dim, Double.POSITIVE_INFINITY).toFloat()
            prune[i] = prune(mind[i])
        }
        // Farthest point via per-block maxima: only blocks whose points changed are rescanned.
        val bs = 128
        val nb = (n + bs - 1) / bs
        val blockMax = FloatArray(nb)
        val blockArg = IntArray(nb)
        val dirty = BooleanArray(nb) { true }
        for (s in 1 until kk) {
            for (b in 0 until nb) {
                if (!dirty[b]) continue
                val lo = b * bs
                val hi = min(n, lo + bs)
                var bm = mind[lo]
                var ba = lo
                for (i in lo + 1 until hi) {
                    if (mind[i] > bm) {
                        bm = mind[i]
                        ba = i
                    }
                }
                blockMax[b] = bm
                blockArg[b] = ba
                dirty[b] = false
            }
            // Strict '>' over blocks in order + lowest index inside each block = lowest index overall (np.argmax ties).
            var j = blockArg[0]
            var best = blockMax[0]
            for (b in 1 until nb) {
                if (blockMax[b] > best) {
                    best = blockMax[b]
                    j = blockArg[b]
                }
            }
            sel[s] = j
            val jo = j * dim
            for (t in 0 until s) cc[t] = sqrt(sqdistBounded(x, sel[t] * dim, x, jo, dim, Double.POSITIVE_INFINITY))
            for (i in 0 until n) {
                // Exact pruning (triangle inequality): d(i, new) >= d(new, a_i) - d(i, a_i). If the new centre is at least
                // twice as far from i's current centre a_i as i is, it cannot be closer to i - skip the distance entirely.
                if (cc[assign[i]] >= prune[i]) continue
                val m = mind[i].toDouble()
                if (m == 0.0) continue
                val d = sqdistBounded(x, i * dim, x, jo, dim, m)
                if (d < m) {
                    mind[i] = d.toFloat()
                    assign[i] = s
                    prune[i] = prune(mind[i])
                    dirty[i / bs] = true
                }
            }
        }
        return sel
    }

    /**
     * Nearest-neighbour squared distance of the patch at float offset [pOff] of [feats] to the bank rows allowed by
     * [skipFrame]. Stops early once the best found is <= [stopBelow] (max-of-min pruning; pass -1 to disable).
     */
    fun nnSq(
        feats: FloatArray,
        pOff: Int,
        bank: FloatArray,
        bankFrame: IntArray,
        dim: Int,
        skipFrame: Int,
        stopBelow: Double,
    ): Double {
        var best = Double.POSITIVE_INFINITY
        val k = bankFrame.size
        for (b in 0 until k) {
            if (bankFrame[b] == skipFrame) continue
            val d = sqdistBounded(feats, pOff, bank, b * dim, dim, best)
            if (d < best) {
                best = d
                if (best <= stopBelow) return best
            }
        }
        return best
    }

    /** 3x3 mean filter with edge replication (matches patchcore_ref.smooth3); float32 output like the TS. */
    fun smooth3(m: FloatArray, h: Int, w: Int): FloatArray {
        val out = FloatArray(h * w)
        for (r in 0 until h) {
            for (c in 0 until w) {
                var acc = 0.0
                for (dr in -1..1) {
                    val rr = min(h - 1, max(0, r + dr))
                    for (dc in -1..1) {
                        val cc = min(w - 1, max(0, c + dc))
                        acc += m[rr * w + cc]
                    }
                }
                out[r * w + c] = (acc / 9).toFloat()
            }
        }
        return out
    }

    /** L2-normalised mean of the patch vectors of one feature map [per·dim] → [dim] (float32 accumulation, as the TS). */
    fun globalDescriptor(feat: FloatArray, per: Int, dim: Int): FloatArray {
        val g = FloatArray(dim)
        for (p in 0 until per) {
            val o = p * dim
            for (t in 0 until dim) g[t] = (g[t].toDouble() + feat[o + t].toDouble()).toFloat()
        }
        var n = 0.0
        for (t in 0 until dim) n += g[t].toDouble() * g[t].toDouble()
        n = sqrt(n)
        if (n == 0.0 || n.isNaN()) n = 1.0
        for (t in 0 until dim) g[t] = (g[t].toDouble() / n).toFloat()
        return g
    }

    /** Cosine distance between unit vector g and row f of the globals matrix. */
    fun cosDist(g: FloatArray, globals: FloatArray, f: Int, dim: Int): Double {
        var s = 0.0
        val o = f * dim
        for (t in 0 until dim) s += g[t].toDouble() * globals[o + t].toDouble()
        return 1 - s
    }

    /**
     * Frame score (patchcore_ref.frame_score): max of the distance map - optionally 3x3-smoothed first (on the full map) -
     * ignoring the outer [border] patch rows/columns.
     */
    fun frameScore(dmap: FloatArray, gh: Int, gw: Int, smooth: Boolean = false, border: Int = 0): Double {
        val m = if (smooth) smooth3(dmap, gh, gw) else dmap
        var best = Double.NEGATIVE_INFINITY
        for (p in 0 until gh * gw) if (inside(p, gh, gw, border) && m[p] > best) best = m[p].toDouble()
        return best
    }

    /** Bank size rule: `max(minK, floor(ratio · nPatches))`. */
    fun bankSize(nPatches: Int, ratio: Double = 0.05, minK: Int = 256): Int =
        max(minK, floor(ratio * nPatches).toInt())

    /** Frames whose LOO score is far above the others: > median + 3 · MAD (scaled), at most 20 % of the frames. */
    fun outlierFrames(loo: List<Double>, maxFraction: Double = 0.2): List<Int> {
        val n = loo.size
        if (n < 6) return emptyList()
        fun med(v: List<Double>): Double {
            val s = v.sorted()
            return if (s.size % 2 == 1) s[(s.size - 1) / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
        }
        val m = med(loo)
        var mad = med(loo.map { kotlin.math.abs(it - m) }) * 1.4826
        if (mad == 0.0 || mad.isNaN()) mad = 1e-9
        return loo.withIndex()
            .filter { it.value > m + 3 * mad }
            .sortedByDescending { it.value }   // stable, like Array.prototype.sort
            .take(floor(maxFraction * n).toInt())
            .map { it.index }
    }

    /** Build the memory bank + threshold from good frames (each [gh·gw·dim]). */
    fun enrol(
        frames: List<FloatArray>,
        gh: Int,
        gw: Int,
        dim: Int,
        opts: EnrolOptions = EnrolOptions(),
        nn: NnBackend? = null,
        createdAt: Long = System.currentTimeMillis(),
    ): Profile {
        val prof = enrolOnce(frames, gh, gw, dim, opts, nn, createdAt)
        if (!opts.trimOutliers) return prof
        val drop = outlierFrames(prof.looScores)
        if (drop.isEmpty()) return prof.withDropped(emptyList())
        val keep = frames.filterIndexed { i, _ -> i !in drop }
        val trimmed = enrolOnce(keep, gh, gw, dim, opts, nn, createdAt)
        return trimmed.withDropped(drop.sorted())
    }

    private fun enrolOnce(
        frames: List<FloatArray>,
        gh: Int,
        gw: Int,
        dim: Int,
        opts: EnrolOptions,
        nn: NnBackend?,
        createdAt: Long,
    ): Profile {
        val n = frames.size
        require(n >= 2) { "Need at least 2 good frames to enrol (leave-one-out calibration)." }
        val per = gh * gw
        val x = FloatArray(n * per * dim)
        frames.forEachIndexed { i, f ->
            require(f.size == per * dim) { "Frame $i has ${f.size} values, expected ${per * dim}" }
            System.arraycopy(f, 0, x, i * per * dim, f.size)
        }
        val k = bankSize(n * per, opts.ratio, opts.minK)
        val sel = greedyCoreset(x, dim, k, 0)
        val kk = sel.size
        val bank = FloatArray(kk * dim)
        val bankFrame = IntArray(kk)
        for (i in 0 until kk) {
            System.arraycopy(x, sel[i] * dim, bank, i * dim, dim)
            bankFrame[i] = sel[i] / per
        }

        // Leave-one-frame-out calibration on the final coreset.
        val held = ArrayList<Double>()
        for (f in 0 until n) {
            if (bankFrame.none { it != f }) continue
            val feats = frames[f]
            if (nn != null || opts.smooth) {
                // Full distance map (needed for smoothing; cheap with an accelerated backend).
                val dm = FloatArray(per)
                if (nn != null) {
                    val d2 = nn.minSq(feats, bank, bankFrame, dim, f)
                    for (p in 0 until per) dm[p] = sqrt(max(0.0, d2[p].toDouble())).toFloat()
                } else {
                    for (p in 0 until per) dm[p] = sqrt(nnSq(feats, p * dim, bank, bankFrame, dim, f, -1.0)).toFloat()
                }
                held.add(frameScore(dm, gh, gw, opts.smooth, opts.border))
            } else {
                // Spec path: only the max is needed, so prune each patch's search at the frame's running max.
                var frameMaxSq = -1.0
                for (p in 0 until per) {
                    if (!inside(p, gh, gw, opts.border)) continue
                    val d = nnSq(feats, p * dim, bank, bankFrame, dim, f, frameMaxSq)
                    if (d > frameMaxSq) frameMaxSq = d
                }
                held.add(sqrt(max(0.0, frameMaxSq)))
            }
        }
        val tau = held.fold(Double.NEGATIVE_INFINITY) { a, b -> max(a, b) } * opts.margin

        // Whole-image descriptors + leave-one-out threshold (nearest OTHER enrolment frame).
        val globals = FloatArray(n * dim)
        frames.forEachIndexed { i, f -> System.arraycopy(globalDescriptor(f, per, dim), 0, globals, i * dim, dim) }
        val gLoo = ArrayList<Double>(n)
        for (f in 0 until n) {
            val g = globals.copyOfRange(f * dim, f * dim + dim)
            var best = Double.POSITIVE_INFINITY
            for (o in 0 until n) if (o != f) best = min(best, cosDist(g, globals, o, dim))
            gLoo.add(best)
        }
        val gTau = max(MIN_GLOBAL_TAU, gLoo.fold(Double.NEGATIVE_INFINITY) { a, b -> max(a, b) } * opts.margin)
        return Profile(
            gh = gh, gw = gw, dim = dim, bank = bank, bankFrame = bankFrame, tau = tau, margin = opts.margin,
            looScores = held, nFrames = n, createdAt = createdAt, globals = globals, gTau = gTau, gLoo = gLoo,
            border = opts.border, smooth = opts.smooth,
        )
    }

    /** Score one feature map [gh·gw·dim] against a profile. */
    fun score(feat: FloatArray, prof: Profile, sensitivity: Double = 1.0, nn: NnBackend? = null): ScoreResult {
        val gh = prof.gh
        val gw = prof.gw
        val dim = prof.dim
        val per = gh * gw
        require(feat.size == per * dim) { "Feature map has ${feat.size} values, expected ${per * dim}" }
        val border = prof.border
        val dmap = FloatArray(per)
        if (nn != null) {
            val d2 = nn.minSq(feat, prof.bank, prof.bankFrame, dim, -1)
            for (p in 0 until per) dmap[p] = sqrt(max(0.0, d2[p].toDouble())).toFloat()
        } else {
            for (p in 0 until per) {
                dmap[p] = sqrt(nnSq(feat, p * dim, prof.bank, prof.bankFrame, dim, -1, -1.0)).toFloat()
            }
        }
        val raw = frameScore(dmap, gh, gw, prof.smooth, border)
        val sm = smooth3(dmap, gh, gw)
        var pk = -1
        for (i in 0 until per) if (inside(i, gh, gw, border) && (pk < 0 || sm[i] > sm[pk])) pk = i
        if (pk < 0) pk = 0

        val patchScore = raw / (prof.tau * sensitivity)
        var globalScore = 0.0
        val globals = prof.globals
        val gTau = prof.gTau
        if (globals != null && gTau != null && gTau != 0.0) {
            val g = globalDescriptor(feat, per, dim)
            var best = Double.POSITIVE_INFINITY
            for (f in 0 until globals.size / dim) best = min(best, cosDist(g, globals, f, dim))
            globalScore = best / (gTau * sensitivity)
        }
        val s = max(patchScore, globalScore)
        return ScoreResult(
            dmap = dmap,
            raw = raw,
            score = s,
            patchScore = patchScore,
            globalScore = globalScore,
            reason = if (s <= 1.0) "ok" else if (globalScore > patchScore) "different" else "defect",
            peakRow = pk / gw,
            peakCol = pk % gw,
        )
    }
}
