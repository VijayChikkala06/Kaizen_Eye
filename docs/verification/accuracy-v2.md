# Accuracy package v2 — telling look-alike parts apart (2026-09-27)

**Trigger.** After teaching one black earbud case, a *different* earbud case (and even other objects) got **PASS**.

## 1. Reproduced offline on the user's own captures
Twins pulled from the phone (`adb exec-out run-as com.kaizeneye.v2 tar -c -C files twins`): A = oval black boAt case
(Part 242), B = black case with a red seam (Part 251), C = folded jeans (Part 928). Their real bank / thresholds replay the phone's
numbers exactly (recomputed LSO = stored LSO). Old build (ResNet18, no rotation handling, τ = 1.4 × max LSO):

| enrolled A, old build | value |
|---|---|
| τ (score gate) | **1.718** |
| B (other case) score | 1.17–1.25 → **PASS 16/16** |
| jeans score | 1.12–1.24 → PASS (only the identity gate stopped it) |
| identity sim B vs τ_id | 0.976–0.985 vs 0.963 → **PASS 16/16** |

Why: (1) ResNet18 patch features are not rotation-invariant, so the same part re-presented scores ~1.0 and another case ~1.18 (AUROC
0.85–0.95); (2) τ = 1.4 × *max* leave-segment-out was inflated to 1.72 because both faces of the case sit in different time blocks
(LSO is bimodal: 0.5 / 1.2); (3) the score is the *max* over patches — a look-alike matches every local patch; (4) the identity
descriptor (mean of patch features) is nearly identical for all black plastic (0.98).

## 2. What was changed (all in the app pipeline; the frozen spec + goldens are untouched)
| change | where | why |
|---|---|---|
| **DINOv2-S/14 @448** used when its model is on the phone (else ResNet18) | `Prefs.backbone`, `TwinRepo.runningPipeline` | semantic features; far more robust to pose/lighting |
| **CANONICAL rotation** for teach and judge (long axis horizontal; judged at θ and θ+π; square fitted to the rotated extent, so a long part at 45° is never clipped) | `CanonicalCrop`, `FrameAnalysis.canonicalSnapshot`, `LineSession.judge` | same-part scores drop ~40 % (R18 1.02 → 0.60) |
| **FIT gate**: `fit = mean over the core of the nearest-patch distance`; NOT_ENROLLED "fit" when above τ_fit | `Fit.kt`, `Scoring`, `VerdictEngine.decide` | answers "does the WHOLE crop look like the part", which the max score cannot |
| τ_fit = 1.10 × max same-part fit (LSO fits ∪ calibration fits); with known other objects: midpoint to the nearest one | `FitThreshold.derive` | measured margins are 4–6 %, so real negatives matter |
| **Other Twins + the Negatives library act as known wrong objects automatically** | `FitNegatives`, `PipelineHub.refreshFitNegatives`, `NegativesStore.addMaps` | teaching the second earbud case teaches the first about it |
| score gate factor 1.4 → **1.25**, ≤ 24 keyframes | `TeachSession.teachParams` | tighter defect gate; capped for memory |
| Tolerance slider 0.5–2.5× (fit gate follows √) | `InspectScreen`, `Prefs` | the old 0.8× floor could not tighten below the inflated τ |
| circle-to-select ROI, weak-teach warning | see `decisions.md` | fewer bad teaches |

The fit gate only exists for Twins that carry `fit.json` (twin.json schema unchanged; spec Twins and the golden Twins have none).

## 3. Evidence (real captures, DINOv2 + canonical; `tools/lab/real_twin_accuracy.py`, seed 7, 5 augmented re-presentations per keyframe)
Same-part re-presentation = augmented keyframes (random rotation, ±8 px shift, ±4 % scale, brightness/contrast, blur). PASS rates in %:

| enrolled | query | old rules (spec τ, R18, no rotation) | new score gate only | + FIT gate |
|---|---|---|---|---|
| A oval case | A again | 100 | 100 | 100 |
| A oval case | **B other case** | **100 (wrongly passes)** | 100 | **0** |
| A oval case | jeans | 100 (identity gate only) | 82 | 0 |
| B red-seam case | B again | 100 | ≈75 (τ×1.15) | ≈71 (τ×1.15) — shipped τ×1.25 recovers most of this |
| B red-seam case | **A other case** | 55 | 0 | 0 |
| C jeans | C again / A / B | 100 / 62 / 99 | 100 / 0 / 0 | 100 / 0 / 0 |

AUROC(same vs other) of the **fit** statistic: 1.000 for all six ordered pairs with DINOv2+canonical, 0.48–1.00 with ResNet18 (B enrolled vs jeans 0.48).
Raw files: `accuracy-v2-results/` (`r18_none`, `dino_none`, `dino_canonical`). Kotlin = numpy on real twins to 1e-15
(`RealTwinFitTest`, env `KZ_REAL_TWINS` / `KZ_FIT_REF`).

**Margins are thin for the two black cases** (A: same-part fit ≤ 0.547 vs B ≥ 0.616; B: ≤ 0.535 vs A ≥ 0.567). That is why the
gate uses the other Twin as a known wrong object (midpoint) and why Calibrate (real good parts) and the Tolerance slider exist.

## 4. Honest limits
- Study = 3 real parts, keyframe-based re-presentations, not live captures. **Held-out live tests on the phone are still required**
  (≥ 20 good, ≥ 20 wrong incl. look-alikes, ≥ 10 defects, `claims.csv`).
- DINOv2 runs on the **CPU** (127 ms/embed; the GPU delegate cannot compile the ViT, the NPU showed no speed-up): teach +10 s,
  judging ≈ 0.25 s/part (two orientations). ResNet18 stays available (Readiness → Backbone) but separates look-alikes worse.
- Twins taught with the old build are refused ("built with another pipeline"): re-teach.
- Memory: DINOv2 maps are 1.5 MB each → `largeHeap`, ≤ 40 frames embedded, ≤ 24 keyframes (the user hit an OOM at 32 keyframes).
- Near-square parts have an unstable principal axis; both orientations are judged but a 90° ambiguity is not searched.
