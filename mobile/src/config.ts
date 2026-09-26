/**
 * Detection tunables in one place. Values come from the handheld benchmark (tools/lab/handheld_eval.py):
 * ResNet18 @ 320 px, 3x3-smoothed distance map, outer 10% of patches unscored, sensitivity 1.00 ->
 * careful handheld capture: detection 55% -> 80% at 0% false rejects vs the README defaults.
 */
export const DETECTION = {
  /** Fraction of the patch grid ignored on each side by the frame score (0.1 of 40 = 4 patches). */
  borderFraction: 0.1,
  /** Frame score on the 3x3-smoothed distance map (patchcore_ref smooth=True). */
  smooth: true,
  /** Drop outlier enrolment photos before calibrating. Measured: +-1 point, so off. */
  trimOutliers: false,
  /** Sensitivity (margin) the app starts with. Measured: <= 1% false rejects at 1.00 with the settings above. */
  defaultSensitivity: 1.0,
  /** Enrolment threshold above which the photos are too inconsistent (clutter / blur / framing). */
  tauWarning: 0.7,
};

/** Patch rows/columns to ignore for a grid of `grid` patches per side. */
export function borderFor(grid: number): number {
  return Math.round(DETECTION.borderFraction * grid);
}
