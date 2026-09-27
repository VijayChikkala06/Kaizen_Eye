# Kaizen Eye 2 — Pitch Pack

*One page of story, one page of architecture, a word-for-word demo script, honest limits, and the questions you will be asked.*
*Open this file in VS Code (Ctrl+Shift+V) for the formatted view.*

---

## 0. The 30-second version (memorise this)

> **Kaizen Eye turns any Android phone into a quality inspector that learns a part from a single 12-second video, then checks every part on the line in about a tenth of a second — fully offline.**
> It tells you three things: **PASS**, **REJECT** (it marks *where* the defect is, beeps and vibrates), or **NOT THE PART** (a different object slipped in — even a look-alike).
> No cloud, no internet permission in the app at all, no photo dataset, no training run. Teach it at the start of the shift, inspect, and it hands you a **certificate** with a measured false-alarm bound.

**One-line hook:** *"A factory inspection system that fits in your pocket, learns in 12 seconds, and never sends a picture anywhere."*

---

## 1. Problem statement

### The problem, in one paragraph (say this)

> Small and medium manufacturers still inspect most parts **by eye**: it is slow, it depends on how tired the inspector is, and it leaves **no record of why** a part was passed or rejected. The alternative — an industrial machine-vision system — needs a special camera, a PC, an integrator and **hundreds of labelled photos of every part, including defective ones**, so it is out of reach for anyone without an AI team. Cloud AI services are cheaper to start but need **internet and upload product photos**, which many factories cannot or will not do.
> **There is no inspector that is cheap, learns a new part in minutes, works offline and can prove how reliable it is.**

### Problem statement (one sentence, for the slide)

**"Visual quality inspection is still manual or locked behind expensive, data-hungry, cloud-dependent systems — leaving small manufacturers with no affordable, private, easy-to-teach way to inspect parts."**

### The gaps we close

| Gap today | What that costs the user | How Kaizen Eye closes it |
|---|---|---|
| Manual inspection is inconsistent | Missed defects, customer returns, no audit trail | Same check on every part, every verdict logged with its scores |
| ML inspection needs hundreds of labelled images (and defect examples) | Weeks of setup, an AI engineer per part | **One 12-second video of a good part**; no defect examples needed |
| Industrial systems are expensive and fixed | Only large factories can afford them | An ordinary Android phone + a stand |
| Cloud tools need internet and upload photos | IP / privacy risk, no use on a plain shop floor | **Fully offline** — the app has no internet permission at all |
| Systems give a score but no reasons | Operators do not trust "black boxes" | Shows *where* the defect is and explains it in one plain sentence |
| No proof of reliability | Cannot show a customer or auditor how good the check is | Built-in **calibration + certificate** with a measured false-alarm bound |

## 1b. Who it is useful for

| Who | Their situation | What they get |
|---|---|---|
| **Small / medium manufacturers and workshops** (machining, moulding, packaging, electronics assembly) | Repeated visual checks, no budget for machine vision | Inspection station for the price of a phone and a stand; new part learned in minutes |
| **Quality / incoming-goods teams** | Check supplier parts against a golden sample | Teach the good sample once; "wrong part / defect / OK" with a record |
| **Packing and dispatch lines** | Wrong item or damaged item slips through | Live PASS / REJECT with beep and vibration at the station |
| **Contract manufacturers and job shops** | Many different parts, short runs | Re-teach per job in seconds; no training pipeline |
| **Quality managers and auditors** | Need evidence, not opinions | Certificate with a statistical false-alarm bound, per-part logs |
| **Places with no or restricted internet / strict data rules** (defence-adjacent, medical, IP-sensitive) | Cannot upload product images | Nothing leaves the device — provable |
| **Education / makerspaces / start-ups** | Want to prototype inspection cheaply | A working inspector on hardware they already own |

**The value in one line for each stakeholder:** operators get a clear verdict and a reason; managers get a measured reliability number; owners get a system that costs a fraction of a machine-vision cell and needs no cloud.

---

## 2. What it does (the 4 things a judge should remember)

1. **Learns from one video, no dataset.** Show one good part for 12 seconds (turn it slowly). That is the entire training.
2. **Judges live, per part.** Slide it under the camera or hold it still for half a second — verdict in ≈ 0.1 s.
3. **Catches two different kinds of "bad".**
   - **REJECT** = *this is the right part but it has a defect* — the defect is highlighted on screen, the phone beeps and vibrates, a card explains it in plain words.
   - **NOT THE PART** = *this is a different object* (wrong part, foreign object, or a look-alike).
4. **Proves itself.** After showing it ~40 good parts it computes a **statistical guarantee** ("false alarms ≤ x % at 95 % confidence") and shows it as a certificate.

**Fully offline, provably.** The app declares only CAMERA and VIBRATE permissions — **no INTERNET permission at all** (checked automatically by our APK audit script). Nothing *can* leave the phone.

---

## 3. How it works — the plain-English version

Say it like this (no jargon needed):

1. **Learn the empty table.** The phone looks at the plain sheet for two seconds so it knows what "background" looks like. Anything that is not background is an object.
2. **Learn the part (Teach).** During the 12-second video it keeps only the sharp frames, picks 16–24 that look most *different* from each other (front, back, turned…) and cuts each picture into a grid of small squares called **patches**. Each patch is turned into a list of numbers by an AI model (the "embedding" — think of it as a fingerprint of that little square). That collection of fingerprints is the part's **memory** — we call it the **Visual Twin**.
3. **Judge a new part.** The same steps are applied to the new part, and every one of its patches is compared with the memory: *"have I seen a patch like this before?"*
   - If **one small spot** is unlike anything seen → **defect** (that spot is the red heat-map).
   - If **the whole part** is a bit off everywhere → **not the same object** (look-alike).
   - If the outline/size/colour are wrong → **different part**.
4. **Decide, alert, explain.** The verdict appears on the live picture; a reject beeps, vibrates and can be explained in a sentence.

**Why this is smart:** we never need examples of defects. It only has to learn what *good* looks like — everything else is "unusual". That is why 12 seconds is enough.

---

## 4. What is new / what to be proud of (say these confidently)

| Claim | Why it is true (evidence you can point to) |
|---|---|
| **Learns from a single video** | Teach pipeline: sharp-frame filter → k-centre keyframes → coreset memory bank → thresholds derived *by the app itself* with leave-one-time-block-out testing |
| **Tells look-alikes apart** | We measured it on two real black earbud cases: the old scoring passed the wrong case 100 % of the time; the new design rejects it (0 % pass) while accepting the right one — see §8 |
| **Honest hardware use** | On start-up the app benchmarks NPU, GPU and CPU on the actual phone and only claims an accelerator if it is really faster (≥ 10 % gain). No fake "AI-accelerated" badge. |
| **Measured, not marketed, guarantees** | Calibration uses a distribution-free order-statistic bound (α = 1 − 0.05^(1/m)) plus Clopper–Pearson intervals — standard statistics, not a guess |
| **Independently verified maths** | The core algorithm was written twice — once in Kotlin, once from the spec in Python/numpy by separate code — and the two agree to 1e-15 on real captures; 244 automated tests |
| **Truly offline** | No INTERNET permission; enforced by an audit script; also works in airplane mode |
| **One phone, whole product** | Camera, AI, UI, alarms, storage, logging — all on-device |

---

## 5. Architecture (draw this on the whiteboard)

```
                    ┌─────────────────────────── ANDROID PHONE (offline) ───────────────────────────┐
  Plain sheet  ───► │  CameraX (30 fps, locked exposure / white-balance / focus, zoom)               │
  + part            │        │                                                                       │
                    │        ▼                                                                       │
                    │  FAST LOOP (every frame, ≈ ms)                                                 │
                    │   empty-sheet colour model → foreground mask → connected parts                 │
                    │   → tracker (photo-eye line + "held still ½ s" triggers, one verdict per part) │
                    │        │  best sharp crop of each part, turned so its long axis is horizontal  │
                    │        ▼                                                                       │
                    │  JUDGE (one worker thread)                                                     │
                    │   AI backbone (DINOv2 → patch fingerprints)  ──►  k-NN search vs the part's    │
                    │   memory  ──►  gates:  identity → shape → look-alike(FIT) → defect score       │
                    │        │                                                                       │
                    │        ▼                                                                       │
                    │  VERDICT  PASS / REJECT / NOT THE PART / REFRAME                               │
                    │   → overlay on live picture, beep + vibration, reject card, JSON log           │
                    │   → (optional) offline vision-language model writes a 1-sentence explanation   │
                    └────────────────────────────────────────────────────────────────────────────────┘
```

### Three modules (clean separation — good to mention)

| Module | Role | Notable |
|---|---|---|
| **`:core`** (pure Kotlin/JVM) | All the maths: segmentation, tracking, teach, scoring, calibration, storage | No Android code → testable on a laptop, verified against an independent numpy reference |
| **`:runtime`** | Runs AI models on NPU/GPU/CPU, the k-NN engine, model file checks (SHA-256), the optional on-device language model | Benchmarks accelerators in a **separate process** so a driver crash can never take the app down; remembers crashes and avoids them |
| **`:app`** | Camera, Compose UI, sessions, alarms, telemetry | Separate `:vlm` process for the language model |

### The four decision gates (in order)

1. **Sanity** — is there exactly one whole part in view? If not → *REFRAME* (asks you to reposition, never guesses).
2. **Identity + Shape** — does the global look and outline match the taught part? If not → **NOT THE PART**.
3. **Look-alike (FIT) gate** — does the *whole* part match its memory closely? A similar-but-different object fails here.
4. **Defect score** — is any small area unlike anything the part has ever looked like? → **REJECT** (with the location).

---

## 6. Tech stack

### In brief (30 seconds — say this)

> "It is a **native Android app written in Kotlin** with a Jetpack Compose interface and CameraX for the live camera. The AI runs **on-device through Google's LiteRT runtime**, which can target the CPU, GPU or the Snapdragon NPU; the app **benchmarks all three on the phone and only uses (and claims) the one that is really faster**. The vision model is Meta's **DINOv2**, which turns small squares of the image into numbers; the part's 'memory' is a set of those numbers, and judging is a **nearest-neighbour search** against it. An optional small **vision-language model (FastVLM)** writes a one-sentence explanation, also offline. The core maths is pure Kotlin and was **verified against an independent Python/numpy implementation**."

### One-line table (for the slide)

| What | Used for |
|---|---|
| **Kotlin + Jetpack Compose** | The whole native Android app and UI |
| **CameraX (+ Camera2)** | 30 fps live camera, locked exposure / focus / white-balance |
| **LiteRT (CPU / GPU / NPU) + Qualcomm QNN** | Running AI models on the phone's chips, with honest benchmarking |
| **DINOv2-S/14** (ResNet18 as fast alternative) | Turning image patches into "fingerprints" |
| **k-NN search (custom LiteRT graph)** | Finding the closest remembered patch = the anomaly score |
| **FastVLM-0.5B (LiteRT-LM)** | Optional offline explanation sentence |
| **Kotlin core module + numpy reference** | Testable maths, independently verified |
| **Gradle / JUnit** | Build and 244 automated tests |

### Detailed table

| Layer | Technology |
|---|---|
| Language / UI | **Kotlin**, **Jetpack Compose** (native Android, arm64, minSdk 31) |
| Camera | **CameraX 1.6** + Camera2 interop (locked exposure/WB/focus, 30 fps analysis stream) |
| AI runtime | **LiteRT 2.2 (CompiledModel)** — one API for **CPU / GPU / NPU**; **Qualcomm QNN 2.49** runtime for the Snapdragon NPU |
| Vision model | **DINOv2-S/14** (Meta, Apache-2.0) at 448 px → 32×32 patch fingerprints of 384 numbers; **ResNet18** kept as a fast alternative |
| Nearest-neighbour search | Custom matrix-multiply k-NN graph on LiteRT, exact CPU fallback (agree to 1e-3) |
| Language model (optional explanations) | **FastVLM-0.5B** via **LiteRT-LM 0.16**, in its own process; template sentences as fallback |
| Storage | Plain files: JSON + half-precision matrices; **SHA-256 pinned models** |
| Build | Gradle 9.5, AGP 9.3, Kotlin 2.2 |
| Verification | JUnit (244 tests), numpy reference implementation, on-device self-test (131 core checks + accelerator parity + replay regression), APK audit |
| Target phone | iQOO 15 (Snapdragon 8-series, SM8850) |

---

## 7. Live demo — word-for-word script (about 5 minutes)

**Before you go on stage (do this 15 minutes ahead — see checklist §11):** teach the part, teach the look-alike, add one wrong object.

| Time | Who / what | Say | Do |
|---|---|---|---|
| 0:00 | Speaker A | "This is a normal Android phone on a stand. No internet. Watch the top of the screen — it says *Offline ✓*." | Show Home: **Model ready**, **Offline ✓** |
| 0:20 | A | "We teach it a part in one video. No photos, no labelling." | Tap **TEACH A NEW PART** → LEARN SHEET (empty sheet, 2 s) |
| 0:45 | B | "I circle the part with my finger so it knows what to learn — like circle-to-search." | Place part, draw circle (doodle turns green) → START RECORDING, turn part slowly 12 s |
| 1:30 | A | "In about 20 seconds it has built the part's memory. It even tells us if the teach was good or weak." | Show **READY — Good teach** |
| 1:45 | A | "Now inspect. A good part…" | **INSPECT NOW** → LEARN SHEET → place a good part, hold still |
| 2:10 | B | "Green PASS." | Point to green box |
| 2:15 | B | "Now a part with a defect — I drew a dot on it." | Place defective copy → **red REJECT**, heat-map on the defect, beep + vibration |
| 2:45 | A | "The card explains it in plain language, generated offline on the phone." | Point to the reject card |
| 3:00 | A | "The hard case: a *different but similar* part — a look-alike. Older versions of our system passed this." | Place look-alike → **violet NOT THE PART** |
| 3:30 | B | "And something completely unrelated" | Place other object → NOT THE PART |
| 3:45 | A | "After ~40 good parts, it computes a measured false-alarm bound and shows a certificate." | Show Certificate screen (pre-calibrated) |
| 4:15 | A | "Everything ran on the phone. Judging takes about a tenth of a second; the AI runs on the phone's GPU, and the app benchmarks NPU, GPU and CPU itself, so any accelerator badge you see is measured, not claimed." | Home → Advanced → Telemetry (optional) |
| 4:45 | Both | Close with the 30-second version (§0). | |

**If something goes wrong on stage**
- *Boxes flicker / "too many blobs"*: "That's our safety check telling us the background changed" → **Re-learn sheet** (2 s).
- *No verdict*: keep the part **still** for a full second, hands out of view.
- *Model still loading*: wait for the green **Model ready** chip; talk through §3 meanwhile.
- *Total failure*: play the pre-recorded screen video (record one before the pitch!) and switch to the numbers slide.

---

## 8. Numbers you can quote (all measured by us)

| Fact | Value | Source |
|---|---|---|
| Teach time | ≈ 12 s video + ~10–15 s build | app log / phone |
| Judge time | ≈ 0.1–0.15 s per part (57 ms AI per orientation × 2, plus tiny search) | on-phone benchmark |
| AI on this phone | DINOv2: **GPU 57 ms** (2.1× faster than CPU 120 ms). ResNet18: **GPU 4.7 ms** (4.8× CPU), **NPU 10.9 ms** | on-phone benchmark, cached in the app |
| Look-alike test (real captures) | Old scoring: wrong case passed **100 %**. New: wrong case passed **0 %**, right part passed 100 % (oval case), jeans rejected 100 % | `docs/verification/accuracy-v2.md` |
| Accuracy of the "whole-part" check | AUROC **1.000** on all 6 pairs of 3 real parts (DINOv2) | same study |
| Tests | **244** automated tests, 0 failures; core maths identical to independent numpy version to 1e-15 | `native/…/test-results` |
| Privacy | 0 network permissions (CAMERA + VIBRATE only) | APK audit: PASS |
| App size | ≈ 132 MB (models included; debug build) | build |

**Do NOT quote:** "99 % accuracy", "certified", "NPU-accelerated DINOv2", or any number we did not measure. If asked for overall accuracy, use §10 Q4.

---

## 9. Honest remarks — say them yourselves before a judge finds them

Judges reward teams that know their limits. Deliver these calmly, each with its next step:

| Limitation | How to say it | Improvement path |
|---|---|---|
| **Look-alike results come from 3 real parts** and augmented re-presentations, not a big live test | "The design was validated on our own real captures; the next step is a held-out live test with 20+ good parts, 20+ wrong objects and 10+ defects, with confidence intervals." | Run the runbook (`phone-day-runbook.md`) — tooling for recording and offline evaluation already exists |
| **The NPU is not used for the vision transformer** | "We measured it: on this Snapdragon the transformer runs slower on the NPU than on the GPU, so our honest-badge rule keeps it on the GPU. The NPU is proven on the ResNet18 path (10.9 ms, identical outputs)." | Int8 quantisation + ahead-of-time QNN compilation, or a smaller ViT distilled from DINOv2 |
| **Needs a fixed phone and a plain contrasting sheet** | "It is designed for a fixed inspection station — a stand and a matte sheet, like a real line." | Automatic lighting/background check; support textured backgrounds with a learned mask |
| **One part at a time under the camera** | "One verdict per part; multi-part scenes report REFRAME." | Multi-object tracking already exists; per-part judging of several at once is next |
| **Very small defects** (smaller than one patch, ≈ 1/32 of the part width) can be missed | "Resolution-limited; zoom in or move the phone closer for smaller defects." | Higher-resolution tiling |
| **Thresholds are learned from one video** | "That is why calibration exists: it measures the real false-alarm rate on 40 good parts and only ever *raises* the limit." | Continual updating from operator confirmations |
| **Battery/heat over long shifts not yet soak-tested** | "A thermal governor throttles frame-rate and pauses explanations when the phone warms; a soak-log recorder is built and a 10-minute run is scheduled." | Publish the soak curve |
| **Offline language model is optional** | "If it is not available the card falls back to template sentences." (Only demo the LLM if you verified it on the phone.) | Ship the model file with the installer |

**Roadmap you can pitch as vision:** PLC / relay output for a physical reject arm • conveyor mode with speed estimation • fleet dashboard (still offline-first, sync when allowed) • multi-part recipes • quantised on-NPU model • operator feedback loop ("this was wrongly rejected") to refine thresholds.

---

## 10. Questions you will be asked — and good answers

**Q1. Why do this on a phone and not the cloud?**
Privacy (product photos never leave the device — provable, the app has no internet permission), zero latency, works in a factory with no network, no subscription. And the phone's chip is now powerful enough.

**Q2. How is this different from normal machine-learning inspection?**
Classic systems need hundreds of labelled photos *including defects* and a training run per part. Ours learns only what *good* looks like from one video — it flags anything unusual, so it also catches defects we never saw before.

**Q3. What algorithm is it based on?**
A patch-level nearest-neighbour anomaly method in the family of *PatchCore* (published, widely used in industry), on top of DINOv2 features. Our contributions are the engineering around it: single-video teach, tracker with per-part triggers, look-alike rejection, statistical calibration, and running it honestly on a phone.

**Q4. How accurate is it?**
"On our real captures it separated look-alike parts perfectly (AUROC 1.0) and rejected the wrong case in 100 % of trials while accepting the right part. We are explicit that this is 3 real parts; the app has a built-in calibration that measures the false-alarm bound on the customer's own parts, and a certificate to show it." Never claim a single accuracy number.

**Q5. What is the 'certificate'?**
After showing 40 good parts, the phone computes an upper bound on the false-alarm rate that holds with 95 % confidence (an order-statistic bound: with m good parts the limit is the largest of their scores). It is standard statistics — no assumptions about the data's distribution.

**Q6. What happens if the lighting changes?**
The camera exposure/white balance/focus are locked after the empty sheet is learned; a slow adaptation follows gentle drift; and a safety check says "too many blobs — re-learn the sheet" if the background changes, instead of guessing.

**Q7. Why not use the NPU? Isn't that the point of the chip?**
We *tried* it and measured: for this transformer the NPU was slower than the GPU (187 ms vs 57 ms), so our own rule refuses to claim it. The NPU path works on the ResNet18 model (10.9 ms). We report measured hardware use instead of a marketing badge; the next step is an int8 NPU-compiled model.

**Q8. Why DINOv2?**
It is a strong self-supervised vision model (Apache-2.0 licence, from Meta) whose features capture "same object / different object" much better than the older ResNet18 we started with — on our real captures the whole-part match scored AUROC 1.0 with DINOv2 versus 0.5–1.0 with ResNet18.

**Q9. Can it be fooled?**
Yes, like any vision system: a defect smaller than a patch, or a different part that looks identical under the camera. We handle look-alikes with a dedicated gate and let the operator add "wrong objects" to tighten it; and every verdict is logged with its scores so failures can be studied.

**Q10. How many parts can it hold? Does it scale?**
Each part is a small folder (a few tens of MB). You can teach several parts and switch between them; taught parts also help each other (they become each other's "wrong objects").

**Q11. What did you build vs. use off the shelf?**
Off the shelf: CameraX, LiteRT, DINOv2, FastVLM, Compose. Built by us: the whole pipeline — segmentation against the empty sheet, tracker and triggers, teach and threshold derivation, look-alike gate, calibration statistics, honest accelerator selection with crash isolation, alarm/overlay UI, replay and self-test tooling, and the independent verification.

**Q12. How do you know the code is correct?**
The core maths was implemented twice independently (Kotlin and a numpy reference written only from the written spec) and they match to 1e-15 on real data; 244 automated tests; a self-test on the phone (131 checks on the real Android runtime); a replay regression that re-judges a recorded clip; and reviews of every module.

**Q13. Is the on-device language model necessary?**
No — it is an optional, offline *explanation* layer ("scratch near the upper-left edge"). A location guard rejects any sentence that contradicts the measured defect location, and template text is the fallback. The inspection decision never depends on it.

**Q14. What about speed on a real conveyor?**
A verdict takes ≈ 0.1–0.15 s per part, so roughly 5–8 parts per second per phone in principle; the tracker gives one verdict per part and warns "too fast — slow the line" if the queue grows. A real conveyor deployment would add a light and encoder input (roadmap).

**Q15. Battery and heat?**
A thermal governor watches Android's thermal headroom and lowers frame-rate / pauses explanations before the phone throttles itself. Soak logging exists; we will publish a 10-minute curve.

**Q16. Business model / who pays?**
One-time per-station licence or per-device subscription for support; the target is SMEs who cannot afford integrator-installed vision systems. Low hardware cost (a phone, a stand, a light).

**Q17. Security / IP?**
Models are SHA-256 pinned; no network; parts and logs stay in the app's own storage. DINOv2 is Apache-2.0; FastVLM is used under Apple's research licence, so a commercial release would swap it or license it — we say this openly.

**Q18. What would you do with another month?**
(1) Held-out live evaluation with published confidence intervals; (2) int8 NPU-compiled model; (3) relay/PLC output; (4) operator feedback loop; (5) multi-part scenes.

**Q19. Why native Kotlin instead of a cross-platform framework?**
Direct access to the camera controls (locked exposure/focus), the LiteRT accelerator APIs and multi-process isolation; our first version was cross-platform and could not do live inspection or use the accelerators.

**Q20. What is 'REFRAME'?**
The system refuses to judge when it is not sure it is looking at exactly one whole part (touching the edge, two objects, too small) — it asks you to reposition instead of giving a wrong answer. That "know when you don't know" behaviour is intentional.

---

## 11. Preparation checklist (do these in order — ≈ 15 minutes)

1. Phone on the stand, straight down, ~25 cm, plain matte sheet (black for light parts, white for dark ones), **room light on** (no flash used).
2. **Alarm volume up** (the reject beep uses the alarm channel). Screen brightness up.
3. Open the app → wait for **Model ready** (10–30 s).
4. **Delete old parts** (they say "built with a different model") → **Teach** the demo part (circle it, 12 s) → note **Good teach**.
5. **Teach the look-alike** as its own part, then re-select the demo part (**Use**) — the two now know each other. (Or Home → **Wrong objects** and show the look-alike ½ s.)
6. Add 1–2 more **Wrong objects** (jeans, a coin…). Then **Calibrate** with ~40 good parts if you have them (do wrong objects *before* calibrating).
7. Dry-run: one good part → PASS, a defective copy (marker dot) → REJECT, look-alike → NOT THE PART. If the good part is rejected, move **Tolerance** slightly right.
8. **Record a 60-second screen video** of a perfect run — your safety net.
9. Charge the phone, disable notifications/DND popups, keep a second identical set of parts.

---

## 12. Glossary for the team (so you can answer follow-ups)

| Term | Plain meaning |
|---|---|
| **Visual Twin** | The phone's compact memory of what a good part looks like |
| **Patch / embedding** | A small square of the picture / the list of numbers (fingerprint) an AI model gives it |
| **k-NN (nearest neighbour)** | "Find the most similar remembered fingerprint" — the distance is the anomaly score |
| **Coreset** | A smaller, representative subset of the memory so search stays fast |
| **Threshold (τ)** | The distance beyond which we call something a defect; learned from the teach video |
| **Leave-one-time-block-out** | Test the memory on parts of the video it was *not* built from, to set an honest threshold |
| **False alarm** | A good part wrongly rejected |
| **Calibration / certificate** | Measuring the false-alarm rate on ~40 real good parts and showing a guaranteed upper bound |
| **AUROC** | 1.0 = the score perfectly separates "good" from "bad"; 0.5 = coin flip |
| **NPU / GPU / CPU** | AI accelerator chip / graphics chip / main processor — we benchmark all three and use the fastest one honestly |
| **DINOv2** | The pre-trained vision model that produces the patch fingerprints |
| **LiteRT** | Google's on-device AI runtime (successor of TensorFlow Lite) |
| **VLM** | A small on-device model that writes the one-sentence explanation |
| **REFRAME** | "I can't judge this view — please reposition" |

---

## 13. The closing line

> *"Quality inspection should not need a server room. Kaizen Eye puts a measured, explainable, private inspector in a phone — and we are honest about where it is today and exactly how it gets better."*
