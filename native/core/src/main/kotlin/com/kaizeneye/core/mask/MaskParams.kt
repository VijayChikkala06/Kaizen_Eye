package com.kaizeneye.core.mask

/**
 * Mask / sanity / crop parameters (spec §2–§4; defaults = the spec's defaults). They are part of the Twin's pipeline
 * fingerprint (`pipeline.mask` in twin.json, §8), so a Twin is refused if any of them changes.
 *
 * @property kSigma foreground iff `d² > kSigma²` (§2.2).
 * @property sigmaMin floor of the sheet model's per-channel sigma (§2.1).
 * @property minBlobPx components with fewer pixels are removed (§2.4).
 * @property borderMargin a component touches the border if any pixel is within this many pixels of an edge (§2.4).
 * @property minAreaFrac `TOO_SMALL` below this fraction of the analysis frame (§2.6).
 * @property maxAreaFrac `TOO_LARGE` above this fraction (§2.6).
 * @property multipleRatio `MULTIPLE` if another component of at least this fraction of the chosen one's area has its
 *   centroid inside the chosen one's crop square (§2.6).
 * @property cropMargin crop square margin on each side, as a fraction of the longer bbox side (§3).
 * @property analysisFactor full-resolution → analysis downscale factor `f` (§1.1).
 * @property coreThreshold patch-coverage threshold of the core set (§4).
 */
data class MaskParams(
    val kSigma: Double = 4.0,
    val sigmaMin: Double = 3.0,
    val minBlobPx: Int = 30,
    val borderMargin: Int = 2,
    val minAreaFrac: Double = 0.01,
    val maxAreaFrac: Double = 0.8,
    val multipleRatio: Double = 0.25,
    val cropMargin: Double = 0.10,
    val analysisFactor: Int = 4,
    val coreThreshold: Double = 0.5,
) {
    init {
        require(kSigma > 0.0) { "kSigma must be > 0" }
        require(minBlobPx >= 1) { "minBlobPx must be >= 1" }
        require(borderMargin >= 0) { "borderMargin must be >= 0" }
        require(analysisFactor >= 1) { "analysisFactor must be >= 1" }
    }
}
