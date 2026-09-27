# Kaizen Eye 2 — hands-on test on the phone (step by step)

Why the first try showed "a bunch of boxes and no verdict" (fixed in this build):
the old build learned the background silently the moment Inspect opened (often with a hand or the part in view),
did not lock exposure, and drew a box on every speck of background texture; a part is only judged when it stays still
for ½ s or crosses the dashed line, and nothing on screen said so. Now the background is learned only when you tap
**LEARN SHEET** (then exposure, colour and focus are locked), specks are ignored, and a banner says what to do next.

## 0. Install the latest build
```
adb install -r native\dist\apk\KaizenEye2-phonefree-debug.apk
```
(`-r` keeps your Twins. First launch: allow the camera; the app measures the accelerators once, ~10–30 s.)
Turn the **notification volume up** and Do Not Disturb off; the reject beep uses the notification sound.

## 1. Set up the scene (this decides whether it works)
- **Phone fixed**, not in your hand: on a stand / tripod / stack of books, camera looking **straight down**,
  about 20–30 cm above the table, closer for small parts: a part should be about a quarter of the picture width
  (the minimum is ~1/10). If the phone moves, nothing gets judged.
- **Plain matte sheet** that fills the **whole picture** (no table edge, no other objects, no pattern), in a colour
  that **contrasts with the part**: black card/cloth for shiny or light parts, white paper for dark parts.
- Even room light, no moving shadows (the app never uses the flash — a desk lamp helps in a dim room).
- Parts: 5+ identical good parts (same coins / nuts / screws), 2–3 different objects, 1–3 "defective" copies
  (marker dot, scratch, small chip or piece of tape).

## 2. Teach the good part
1. Home → **TEACH A NEW PART**.
2. **Step 1**: if anything besides the sheet is in view (table, stand, floor), drag **Zoom** until only the sheet is
   visible — only the zoomed view is used. Take everything off the sheet (no part, no hands) → **LEARN SHEET** → wait 2 s.
   The grey text must NOT say "background NOT plain". If it does, fix the sheet and tap **Re-learn sheet**.
3. **Step 2**: put ONE good part in the middle, then **draw a circle around it with your finger** (or tap it). The doodle
   turns green ("Got it") when the part is found inside, red when nothing is there — then fades. From now on only what is
   inside the circle is used (table, stand, shadows, hands outside are ignored; a part split by glare is joined back
   together); the circle follows the part while you turn it. **Whole view** goes back to automatic selection.
   A box appears on the part:
   **green "part ✓"** = good; **amber** = read the message (touches the edge / too small / more than one object…).
4. Type a **Part name** → **START RECORDING**. For 12 s, slowly turn and shift the part in place (nudge it with a pen,
   or turn it and pull your hand away — frames with your hand in them are skipped). Keep it inside the picture.
   Watch "frames x/y usable": 40+ is good; below ~15 the teach fails ("Not enough sharp frames").
5. Wait for **ARMED — <name>** (~5–15 s). If "Teach failed", read why and **TRY AGAIN**.

## 3. Inspect
1. **INSPECT NOW** (or Home → **INSPECT**).
2. **Step 1** panel: set **Zoom** so only the sheet is visible (same zoom as when teaching; it is remembered), take
   everything off the sheet → **LEARN SHEET** → wait 2 s (this also locks exposure, colour and
   focus for this screen). Do this again (**Re-learn sheet**) whenever the light or the phone position changes.
3. **Parts move** chips at the bottom:
   - **across** (default): judged when the part crosses the dashed line left↔right, *or* stays still for ½ s;
   - **down**: same, with a horizontal line (parts move top↔bottom);
   - **held still**: only the ½ s still-hold (best when placing parts by hand).
4. Put ONE part on the sheet, take your hand away, keep it still ~1 s — or **circle it with your finger** to judge it
   right away. After a circle, only that area is inspected ("Circled area only"; next parts go in the same spot);
   **Whole view** undoes it. The box on the part:
   - thin light grey = tracking, amber = being judged;
   - **green PASS** — same part, no defect;
   - **red REJECT #n** + red marks where it differs — defect (beep + vibration);
   - **violet NOT THE PART #n** — a different object (beep + vibration);
   - **dashed grey REFRAME** — could not judge (touching the edge, two objects…).
   Counters at the top (judged / pass / defect / not part / reframe); the card at the bottom explains the last reject.
5. Take the part completely out of the picture before the next one (each appearance is judged once).

## 4. What to try
- Several different good copies → PASS.
- A different object → NOT THE PART.
- A good copy with a marker dot / scratch → REJECT.
- Good parts rejected? Move **Sensitivity** right (1.2–1.5×). Defects passing? Move it left.
- Better "not the part": Home → **Negatives** → LEARN SHEET → hold 5–10 *different* wrong objects still, one by one
  → **DONE**.
- Certificate: Home → **Calibrate** → LEARN SHEET → present ≥ 40 *distinct* good parts (not the taught one) →
  **FINISH CALIBRATION** → the Certificate opens.

## 5. If it goes wrong
| You see | Why | Fix |
|---|---|---|
| Table / stand / floor visible around the sheet | the sheet is smaller than the view | **Zoom** in (Teach: "Zoom / sheet"; Inspect: "Zoom"), then LEARN SHEET again |
| "Too many blobs" / boxes on the background | background not plain, phone moved, light changed | plain sheet filling the view, phone on a stand, **Re-learn sheet** |
| No box on the part / "No part seen" | part colour too close to the sheet | contrasting sheet |
| "touches the edge" | part near the edge or your hand in view | move the part inward, hand out |
| "too small" | part tiny in the picture | move the phone closer (part ≥ ~1/10 of the picture width) |
| Nothing happens | phone moving, part not still, or hand still in view | stand; take the hand away; "held still" mode |
| Everything "NOT THE PART" | poor teach or different light | re-teach with a green box the whole time; re-learn sheet; add Negatives |
| No beep | notification volume muted / Do Not Disturb | turn it up (vibration still works) |
| Home chip says "model error — see Telemetry" | model/accelerator failed to load | Telemetry → **Accelerator** or **Re-benchmark all**; send the self-test |

## 6. Send results to the laptop
Home → **Self-test** → **RUN ALL**, then:
```
adb pull /sdcard/Android/data/com.kaizeneye.v2/files/selftest/selftest.json
adb pull /sdcard/Android/data/com.kaizeneye.v2/files/logs
```
