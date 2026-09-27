# Kaizen Eye 2 — Live Demo Talk Track

*What to SAY while the app is on screen. Read it aloud twice before the pitch. Bold lines are word-for-word; everything else is guidance.*
*Two speakers: **A** = presenter (talks), **B** = operator (handles the phone and parts). If you are alone, do both — the script still works.*

Total time: **about 6 minutes** (a 4-minute cut is marked ✂).

---

## Before you start speaking

**On the table:** phone on stand (camera straight down, ~25 cm) · plain matte sheet · **Part** (the good one) · **Look-alike** (similar but different) · **Defective copy** (a marker dot or scratch) · **One unrelated object**.
**On the phone:** app open on Home · chip says **Model ready** and **Offline ✓** · your part already taught (and the look-alike taught as its own part, so they know each other) · alarm volume up.

**Golden rules while talking**
1. **Say what you are about to do, do it, then say what the audience just saw.** Never demo silently.
2. Keep your hands out of the picture when the phone is judging.
3. Never say "hopefully". Say "watch this".
4. If something misbehaves, use the recovery lines in §9 — stay calm, it reads as confidence.

---

## Scene 1 — Set the stage (0:00–0:45) · A

**Screen:** Home. Point at the chips.

> **"This is an ordinary Android phone. That's the entire inspection station — no PC, no special camera, no server."**
> **"Look at the top: 'Offline ✓'. This app has no internet permission at all — not switched off, *absent*. We check that automatically when we build it. So nothing you see here, no image and no result, can leave this device."**
> **"Three things we'll show: it learns a part from one video, it inspects live, and — the part most systems get wrong — it can tell a part that is *almost* right from one that *is* right."**

*Emphasise:* offline is a fact, not a promise.

---

## Scene 2 — Teach in 12 seconds (0:45–2:00) · A talks, B operates

**Screen:** Tap **TEACH A NEW PART** → Step 1 panel.

> **A:** "Step one: the phone learns what an *empty* table looks like. Two seconds. Everything that isn't table is an object."
> **B:** *(clear the sheet, tap **LEARN SHEET**)*

**Screen:** Step 2 — place the part.

> **A:** "Step two: we show it the part. I'll circle it with my finger — like circle-to-search — so it knows exactly which object to learn. The outline turns green when it has found the part."
> **B:** *(draw the circle; doodle turns green, "Got it")*
> **A:** "Now we record twelve seconds and turn the part slowly so it sees every side. There is no photo dataset, no labelling and no training run — this video is the training."
> **B:** *(START RECORDING, turn the part; watch the coverage ring)*

*While the ring fills (talk over it):*
> **"The phone is throwing away blurry frames, picking the views that look most different from each other, and cutting each into small squares. Each square becomes a list of numbers — a fingerprint — using a vision model running on the phone. That collection of fingerprints is the part's memory."**

**Screen:** "Building the part model…" then **READY — Good teach**.

> **"About twenty seconds from start to ready. And notice it grades its own teach — 'Good teach' means enough sharp, well-covered frames. If it were weak, it would tell us to redo it rather than pretend."**

✂ *(4-minute cut: skip the circle explanation, teach in advance and start at Scene 3.)*

---

## Scene 3 — A good part passes (2:00–2:40) · B operates, A talks

**Screen:** Tap **INSPECT NOW** → LEARN SHEET → place the good part, hold still.

> **A:** "Now inspection. Same first step — learn the empty sheet — because lighting changes every day and we never assume it."
> **B:** *(LEARN SHEET, then place the good part and keep it still about one second)*
> **A:** **"A part is judged when it stays still for half a second or slides past the dashed line. Green — PASS. Roughly a tenth of a second per part."**

*Emphasise:* no button pressed for the verdict; it triggers itself.

---

## Scene 4 — A defect is found and *located* (2:40–3:30) · A

**Screen:** Place the defective copy.

> **A:** "Now the same part, but with a defect — I put a mark on it."
> **B:** *(place it; wait for the red box)*
> **A:** **"Red — REJECT. Listen: beep and vibration, so an operator who isn't looking still knows. And it doesn't just say 'bad' — it highlights *where*."** *(point to the red heat-map)*
> **"The card at the bottom explains it in a plain sentence, generated on the phone. If the on-device language model isn't available it falls back to a template — the decision never depends on it."**

*Emphasise:* it never saw a defective example. It only learned what *good* looks like; anything unusual is flagged. **"That's why one video is enough — we don't need defect samples, which factories rarely have."**

---

## Scene 5 — The hard case: look-alike (3:30–4:30) · A (the star moment)

**Screen:** Place the look-alike.

> **A:** **"Now the hard one. This is a *different* part that looks very similar — same colour, same material. Earlier versions of our system, and most anomaly detectors, would pass this, because every small area of it looks familiar."**
> **B:** *(place the look-alike; violet box appears)*
> **A:** **"Violet — NOT THE PART. It's not a defect in *our* part; it's a different object. We built a separate check for this: it doesn't ask 'is any spot unusual', it asks 'does the *whole* thing match what I learned'. On our real captures, the old scoring passed the wrong case every time; the new design rejects it every time while still accepting the right part."**

**Then, quick contrast:**
> **B:** *(place the unrelated object)* → violet again
> **A:** "And something completely unrelated is rejected too."

*If a judge asks "how sure are you?"* → "That result is from our own real captures of three parts, measured offline — the next step is a bigger live test. We'll show you the numbers."

*Emphasise:* this is the differentiator. Slow down here.

---

## Scene 6 — Teach it more wrong objects (optional, 4:30–5:00) · B

**Screen:** Home → **Wrong objects**.

> **A:** "If a customer has a specific look-alike that causes trouble, the operator just shows it to the phone for half a second and it's remembered. Taught parts also teach each other automatically."

✂ *Skip in the short version.*

---

## Scene 7 — Proof, not promises: the certificate (5:00–5:40) · A

**Screen:** Home → **Certificate** (pre-calibrated part).

> **A:** **"Anyone can say 'it's accurate'. So the app measures it. After you show it about forty good parts, it computes a statistical upper limit on false alarms — 'at 95 % confidence, false alarms are no more than this' — using a standard, distribution-free bound. That number goes on a certificate an auditor can read."**
> **"A rejected good part costs money; a passed bad part costs customers. Now the customer can *see* the reliability."**

*If not calibrated on the day:* show the screen anyway and say "This is what an operator gets after calibrating with forty parts" — never fake numbers.

---

## Scene 8 — Under the hood in one breath (5:40–6:20) · A

**Screen:** Home → **Advanced ▾** → **Telemetry** (optional).

> **A:** **"Everything you saw runs on the phone. The app tests the phone's CPU, GPU and NPU at start-up and only uses — and only *claims* — an accelerator that is genuinely faster. On this phone the vision model runs on the GPU in about 57 milliseconds, twice as fast as the CPU. We measured the NPU too: for this transformer model it was slower, so the app refuses to claim it. That honesty is deliberate — every badge you see is measured."**
> **"The maths core is written twice — once in the app, once independently in Python — and the two agree to the fifteenth decimal place. Two hundred forty-four automated tests pass."**

*Emphasise:* measured, verified, honest. (Only show Telemetry if you're comfortable; the words work without it.)

---

## Scene 9 — Close (6:20–6:40) · A

> **"So: one video to teach, no dataset, no internet, live verdicts with the defect located, look-alikes caught, and a certificate to prove it — on a phone."**
> **"Where we go next: a live held-out test with published confidence intervals, an int8 model for the NPU, and a relay output to trigger a physical reject arm."**
> **"Quality inspection shouldn't need a server room. Thank you — happy to take questions."**

---

## Filler lines (use when something takes a few seconds)

| Moment | Say |
|---|---|
| Model still loading | "The app benchmarks the phone's chips when it starts, so the numbers you see are for *this* device." |
| Teach building (10–20 s) | "It's now choosing the most different views and building a compact memory — a few megabytes for the whole part." |
| Learning the sheet (2 s) | "It's locking exposure and focus so a new part entering can't change the background." |
| You're repositioning a part | "It refuses to judge if it isn't sure it's seeing exactly one whole part — it would rather ask than guess." |

---

## When a judge interrupts (short bridges)

| They ask | Answer, then return to the script |
|---|---|
| "What model is this?" | "DINOv2 from Meta for the image fingerprints, with a nearest-neighbour search on top — a patch-level method in the same family as PatchCore." |
| "Why not the NPU?" | "We tried: for this model it was slower than the GPU on this chip, so our own rule says don't claim it. The NPU path is proven on the ResNet18 model; an int8 export is the next step." |
| "Does it work with different lighting?" | "It locks exposure after learning the empty sheet, adapts slowly to drift, and warns you to re-learn if the background changes." |
| "How accurate?" | "We report what we measured: on our real look-alike captures it separated them perfectly, and the app produces its own false-alarm bound per part. A larger live test is our next step." |
| "Can it run several parts?" | "Yes — parts are stored separately, and each newly taught part also helps reject the others." |

---

## Recovery lines (stay calm)

| Problem | Say / do |
|---|---|
| Boxes flicker or "too many blobs" | "That's the safety check — the background changed. Two seconds to re-learn." → **Re-learn sheet** |
| Nothing is judged | "It waits for the part to be still." → hold still, hands out |
| Good part wrongly rejected | "Every site has its own tolerance." → move **Tolerance** slightly right, retry |
| Phone or app hiccup | "While that restarts, here is the recorded run" → play your backup screen video |
| Look-alike passes | "This one is genuinely close — this is exactly where we add it as a wrong object" → **Wrong objects**, show it ½ s, retry |

---

## What NOT to say

- ✗ "99 % accurate" / "100 % accurate" — say what was measured.
- ✗ "It uses the NPU for DINOv2" — it doesn't; say GPU, and the NPU story above.
- ✗ "Certified" / "guaranteed" — say "measured bound at 95 % confidence".
- ✗ "It never fails" — say "it refuses to judge when it isn't sure".
- ✗ Anything about the language model unless you saw it working on the phone today.

## What TO stress (memorise these five)

1. **One video, no dataset, no defect examples.**
2. **Fully offline — the internet permission is absent.**
3. **Finds *and locates* the defect, and alerts (beep + vibration).**
4. **Catches look-alikes — the part most systems miss.**
5. **Measured and honest — certificate, real benchmarks, independent verification.**
