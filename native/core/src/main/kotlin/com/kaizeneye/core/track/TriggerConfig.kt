package com.kaizeneye.core.track

/** Axis of the virtual photo-eye line (spec §11.3): `X` = a vertical line at `x = position`, `Y` = horizontal. */
enum class Axis { X, Y }

/** Which crossings fire the line trigger (spec §11.3). `POSITIVE` = from smaller to larger coordinate. */
enum class Direction { ANY, POSITIVE, NEGATIVE }

/**
 * Live-line tracker and trigger parameters (spec §11; defaults = the spec's). Coordinates are analysis pixels, times ms.
 *
 * @property axis line axis (§11.3).
 * @property position line coordinate `L`; `null` = the frame centre along [axis] (`0.5·W` for X, `0.5·H` for Y).
 * @property direction crossing direction (§11.3).
 * @property lineEnabled Trigger A (virtual photo-eye).
 * @property steadyEnabled Trigger B (steady hold).
 * @property gateBase association gate `G = gateBase + gateArea·sqrt(area) + gateSpeed·|vel|·(t − lastT)` (§11.2).
 * @property gateArea see [gateBase].
 * @property gateSpeed see [gateBase].
 * @property velocityAlpha EMA weight of the new velocity measurement (§11.2, `α`).
 * @property minHits a track is confirmed once it has this many matched frames (§11.2).
 * @property maxMissed a track is removed when it has missed more than this many consecutive frames (§11.2).
 * @property crossFrames consecutive matched frames on the downstream side needed to fire LINE (§11.3).
 * @property vStill steady-hold speed limit, px/ms (§11.4).
 * @property holdMs steady-hold duration (§11.4).
 * @property sharpRatio steady hold needs sharpness ≥ this × the track's maximum so far (§11.4).
 * @property bestFrames best-quality frames kept per track (§11.5).
 * @property maxQueuedJobs "line too fast" when the judge queue holds more jobs than this (§11.6).
 */
data class TriggerConfig(
    val axis: Axis = Axis.X,
    val position: Double? = null,
    val direction: Direction = Direction.ANY,
    val lineEnabled: Boolean = true,
    val steadyEnabled: Boolean = true,
    val gateBase: Double = 12.0,
    val gateArea: Double = 0.75,
    val gateSpeed: Double = 0.5,
    val velocityAlpha: Double = 0.5,
    val minHits: Int = 2,
    val maxMissed: Int = 5,
    val crossFrames: Int = 2,
    val vStill: Double = 0.02,
    val holdMs: Double = 500.0,
    val sharpRatio: Double = 0.7,
    val bestFrames: Int = 3,
    val maxQueuedJobs: Int = 3,
) {
    init {
        require(minHits >= 1) { "minHits must be >= 1" }
        require(maxMissed >= 0) { "maxMissed must be >= 0" }
        require(crossFrames >= 1) { "crossFrames must be >= 1" }
        require(bestFrames >= 1) { "bestFrames must be >= 1" }
        require(velocityAlpha in 0.0..1.0) { "velocityAlpha must be in [0, 1]" }
    }
}
