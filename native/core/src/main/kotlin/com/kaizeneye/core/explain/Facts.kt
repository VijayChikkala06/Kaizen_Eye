package com.kaizeneye.core.explain

import com.kaizeneye.core.model.Verdict

/**
 * The verdict facts that surround an explanation (spec §9 outputs). The VLM only ever phrases these; it never changes
 * the verdict.
 *
 * @property partName the Twin's name ("M8 nut").
 * @property reason `NOT_ENROLLED`: "identity" | "shape" | "coverage"; `REFRAME`: a `SanityReason` name
 *   ("TOUCHES_BORDER", …); otherwise null.
 * @property peakRow `DEFECT` peak patch row (0-based, §6.4), null otherwise.
 * @property peakCol `DEFECT` peak patch column (0-based), null otherwise.
 * @property gh patch grid rows.
 * @property gw patch grid columns.
 * @property areaPct anomalous area in % of the part (§6.4 `100·a`), null if unknown.
 * @property score normalised score `s = raw / (τ · sensitivity)` (§6.3), null if not scored.
 * @property tau the Twin's `τ`, null if unknown.
 * @property rejectNumber running reject counter ("#042"), null if not a reject.
 * @property accelLabel accelerator badge ("NPU 4.1 ms"), null if unknown.
 */
data class Facts(
    val partName: String,
    val verdict: Verdict,
    val reason: String? = null,
    val peakRow: Int? = null,
    val peakCol: Int? = null,
    val gh: Int = 0,
    val gw: Int = 0,
    val areaPct: Double? = null,
    val score: Double? = null,
    val tau: Double? = null,
    val rejectNumber: Int? = null,
    val accelLabel: String? = null,
) {
    /** The §GridNaming region of the peak, or null when there is no peak. */
    val region: String?
        get() = if (peakRow != null && peakCol != null && gh > 0 && gw > 0) GridNaming.region(peakRow, peakCol, gh, gw) else null
}
