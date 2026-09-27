#!/usr/bin/env python3
r"""
Kaizen Eye 2 - SYNTHETIC "phone recording" for the on-device REPLAY regression self-test (mechanics only, not
evidence). Writes testdata/replay_synth/:

  frames/000000.jpg ...   landscape 1280x720 RGB frames, JPEG quality 90 (the phone decodes them like camera frames)
  timestamps.json         [tMs per frame] at 30 fps (round(i * 1000 / 30))
  script.json             what happens when (segments, parts crossing, parameters the self-test must use)
  expected.json           the verdicts the phone must reproduce, computed with tools/lab/twin_ref.py on the DECODED
                          JPEG frames: sheet fit on the empty frames -> teach on the teach window -> tracker + LINE
                          trigger on the line segment -> judge each fired track with the shipped ResNet18 TFLite

Scenario: 15 empty-sheet frames, a 4 s teach window (the bracket in the middle, slowly rotating and shifting), a
short empty gap, then a line segment: 10 parts crossing left -> right through the centre at 40 px/frame full-res
(10 px/frame at analysis resolution), 7 good, 2 seeded defects (chip, large dot) and 1 wrong object (hex nut), then
empty frames until every track has exited.

  .venv\Scripts\python tools\lab\make_replay_synth.py [--out testdata\replay_synth]

Borderline (expected.json): s within 10 % of 1.0 (|s - 1| <= 0.1) or sim within 0.02 of tau_id -> any verdict is
accepted there. The expected verdicts are computed WITHOUT borderline voting (spec 9); "voteVerdict" says what a vote
over the buffered crops would give when one would run.
"""
import argparse
import io
import json
import math
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import make_synthetic_export as mse  # noqa: E402  (renderer + R18 runner, same repo folder)
import twin_ref as tr  # noqa: E402

FPS = 30
W, H, FACTOR, N_IN, GRID = 1280, 720, 4, 320, 40
N_EMPTY, N_TEACH, N_GAP = 15, 120, 10
SPEED = 40.0                      # full-res px per frame (= 10 px/frame at analysis resolution)
SPACING = 17                      # frames between part entries
PARTS = ["good", "good", "defect:chip", "good", "wrong:hexnut", "good", "good", "defect:dot", "good", "good"]
BORDER_S, BORDER_SIM = 0.10, 0.02


def t_of(i):
    return int(math.floor(i * 1000.0 / FPS + 0.5))


def build(rng):
    """-> (per-frame object lists, script dict)."""
    frames = []
    for _ in range(N_EMPTY):
        frames.append([])
    teach0 = len(frames)
    for i in range(N_TEACH):
        u = i / (N_TEACH - 1)
        frames.append([dict(kind="bracket", surface="brushed", defect=None,
                            pose=dict(cx=W / 2 + 28.0 * math.sin(2 * math.pi * u), cy=H / 2 + 16.0 * math.sin(3 * math.pi * u),
                                      angle=-16.0 + 32.0 * u, scale=1.0, gain=1.0 + 0.015 * math.sin(4 * math.pi * u),
                                      blur=0.0))])
    teach1 = len(frames) - 1
    for _ in range(N_GAP):
        frames.append([])
    line0 = len(frames)
    parts = []
    x_start = -160.0
    for k, spec in enumerate(PARTS):
        kind, _, sub = spec.partition(":")
        entry = line0 + k * SPACING
        cy = H / 2 + float(rng.uniform(-30, 30))
        angle = float(rng.uniform(-5, 5))
        if kind == "good":
            obj = dict(kind="bracket", surface="brushed", defect=None)
        elif kind == "defect":
            obj = dict(kind="bracket", surface="brushed", defect=sub, seed=int(rng.integers(1 << 30)),
                       scale=1.4 if sub == "dot" else 1.0)
        else:
            obj = dict(kind=sub, seed=int(rng.integers(1 << 30)))
        parts.append(dict(part=k + 1, label=kind, detail=sub or None, entryFrame=entry, cy=cy, angle=angle, obj=obj))
    last_exit = 0
    for p in parts:
        # the part is fully off-screen again once its centre is > W + 200
        n_vis = int(math.ceil((W + 200 - x_start) / SPEED)) + 1
        p["exitFrame"] = p["entryFrame"] + n_vis
        last_exit = max(last_exit, p["exitFrame"])
    n_total = last_exit + 8                       # + frames so the tracker sees every track exit (maxMissed 5)
    while len(frames) < n_total:
        frames.append([])
    for p in parts:
        for fi in range(p["entryFrame"], p["exitFrame"] + 1):
            cx = x_start + SPEED * (fi - p["entryFrame"])
            o = dict(p["obj"])
            o["pose"] = dict(cx=cx, cy=p["cy"], angle=p["angle"], scale=1.0, gain=1.0, blur=0.0)
            frames[fi].append(o)
        p["crossFrame"] = p["entryFrame"] + int(math.ceil((W / 2 - x_start) / SPEED))
    script = dict(
        fps=FPS, frameW=W, frameH=H, analysisFactor=FACTOR, inputSize=N_IN, jpegQuality=90,
        segments=[
            dict(name="sheet", startFrame=0, endFrame=N_EMPTY - 1, startMs=t_of(0), endMs=t_of(N_EMPTY - 1),
                 action="empty-sheet fit (spec 2.1) over these frames, ROI = whole frame"),
            dict(name="teach", startFrame=teach0, endFrame=teach1, startMs=t_of(teach0), endMs=t_of(teach1),
                 action="teach (spec 7) from EVERY frame with startMs <= tMs <= endMs (no subsampling), no negatives"),
            dict(name="gap", startFrame=teach1 + 1, endFrame=line0 - 1, startMs=t_of(teach1 + 1),
                 endMs=t_of(line0 - 1), action="empty sheet, nothing to do"),
            dict(name="line", startFrame=line0, endFrame=n_total - 1, startMs=t_of(line0), endMs=t_of(n_total - 1),
                 action="inspect: a fresh tracker from startFrame on every frame; LINE trigger only is expected")],
        parameters=dict(
            note="spec defaults; the self-test must run with these",
            mask="twin_ref.MaskParams() defaults (kSigma 4, sigmaMin 3, minBlobPx 30, borderMargin 2, cropMargin "
                 "0.10, coreThreshold 0.5)",
            teach="twin_ref.TeachParams() defaults (keep P40, kMin 16, kMax 32, 5 segments, ratio 0.10, minK 256, "
                  "maxRows 2400, tauFactor 1.4)",
            tracker=dict(line="axis X at 0.5*W (analysis), direction ANY, crossFrames 2", steady="enabled (never "
                         "fires here: parts move at 0.3 px/ms)", gate="12 + 0.75 sqrt(area) + 0.5 |v| dt",
                         bestK=3),
            backbone="mobile/assets/models/backbone_r18_320.tflite (fp32), no L2 normalisation",
            sensitivity=1.0, voting=False),
        parts=[dict(part=p["part"], label=p["label"], detail=p["detail"], entryFrame=p["entryFrame"],
                    entryMs=t_of(p["entryFrame"]), crossesCentreFrame=p["crossFrame"],
                    crossesCentreMs=t_of(p["crossFrame"]), exitFrame=p["exitFrame"],
                    speedPxPerFrameAnalysis=SPEED / FACTOR, cyFull=round(p["cy"], 2), angleDeg=round(p["angle"], 3))
               for p in parts],
        frames=n_total,
        note="SYNTHETIC renders (PIL) - mechanics regression only, not evidence of accuracy")
    return frames, script


def render_iter(frames_objs, rng):
    """Yields one rendered uint8 frame at a time (never holds the whole clip in memory)."""
    sheet = mse.render_sheet(W, H, np.random.default_rng(99), noise=0.6)
    for objs in frames_objs:
        f = sheet.copy()
        for o in objs:
            if o["kind"] == "bracket":
                key = ("replay", o.get("defect"), o.get("seed"))
                sp = mse._SPRITES.get(key)
                if sp is None:
                    sp = mse.sprite_bracket("brushed", o.get("defect"), np.random.default_rng(o.get("seed", 0)),
                                            o.get("scale", 1.0))
                    mse._SPRITES[key] = sp
            else:
                sp = mse.sprite_for(o)
            p = o["pose"]
            mse.place(f, sp, p["cx"], p["cy"], p["angle"], p["scale"], p["gain"], p["blur"])
        yield mse.finish_frame(f, rng, noise=0.8)


class Frames:
    """Decoded full-resolution frames read back from the written JPEGs on demand (small LRU)."""

    def __init__(self, folder):
        self.folder, self.memo, self.order = folder, {}, []

    def __getitem__(self, i):
        if i not in self.memo:
            with open(os.path.join(self.folder, f"{i:06d}.jpg"), "rb") as fh:
                self.memo[i] = mse.decode_jpeg(fh.read())
            self.order.append(i)
            if len(self.order) > 8:
                self.memo.pop(self.order.pop(0), None)
        return self.memo[i]


def detections(small, seg):
    g = tr.grey(small)
    return [tr.Detection(c.area, c.minX, c.minY, c.maxX, c.maxY, c.cx, c.cy, c.touchesBorder,
                         tr.sharpness(g, c.minX - 1, c.minY - 1, c.maxX + 2, c.maxY + 2)[0]) for c in seg.components]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "testdata", "replay_synth"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    objs, script = build(rng)
    tmp = a.out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "frames"))
    small = []                                         # analysis frames of the DECODED JPEGs (what the phone sees)
    for i, im in enumerate(render_iter(objs, rng)):
        data = mse.jpeg_bytes(im, 90)
        with open(os.path.join(tmp, "frames", f"{i:06d}.jpg"), "wb") as fh:
            fh.write(data)
        small.append(mse.analysis_of(tr, mse.decode_jpeg(data), FACTOR))
    decoded = Frames(os.path.join(tmp, "frames"))
    ts = [t_of(i) for i in range(len(small))]
    print(f"[render] {len(small)} frames in {time.time() - t0:.0f} s")

    mp = tr.MaskParams(analysisFactor=FACTOR)
    seg_sheet, seg_teach, _, seg_line = script["segments"]
    sheet = tr.fit_sheet(small[seg_sheet["startFrame"]:seg_sheet["endFrame"] + 1], None, mp.sigmaMin)
    print(f"[sheet] mu {np.round(sheet.mu, 3).tolist()} sigma {np.round(sheet.sigma, 3).tolist()}")

    bb = mse.R18()
    teach_frames, teach_sanity = [], {}
    for i in range(seg_teach["startFrame"], seg_teach["endFrame"] + 1):
        info, comp, crop, c_in, _ = mse.cropinfo(tr, decoded[i], small[i], sheet, mp, GRID, GRID, N_IN)
        teach_sanity[info["sanity"]] = teach_sanity.get(info["sanity"], 0) + 1
        if info["sanity"] != tr.OK:
            teach_frames.append(tr.TeachFrame(ts[i], False))
            continue
        teach_frames.append(tr.TeachFrame(ts[i], True, info["sharpness"], np.array(info["cov"]), info["geometry"],
                                          bb(c_in)))
    twin = tr.teach(teach_frames, tr.TeachParams())
    ti = twin.teachInfo
    print(f"[teach] frames {len(teach_frames)} sanity {teach_sanity} kept {ti['framesKept']} keyframes "
          f"{len(twin.keyframes)} bank {twin.bank.shape[0]} tau {twin.tau:.4f} tauId {twin.tauId:.4f}")

    tracker = tr.Tracker(tr.TrackerParams(w=W // FACTOR, h=H // FACTOR))
    frame_of = []                                    # tracker frame index -> global frame index
    segs, assigns, fired, exited = {}, {}, [], []
    for i in range(seg_line["startFrame"], seg_line["endFrame"] + 1):
        seg = tr.segment(small[i], sheet, mp)
        dets = detections(small[i], seg)
        r = tracker.step(ts[i], dets)
        frame_of.append(i)
        segs[i], assigns[i] = seg, r["assign"]
        for e in r["events"]:
            if e["type"] == "FIRED":
                fired.append(dict(e, frame=i))
            else:
                exited.append(dict(e, frame=i))

    def judge_at(gi, track_id):
        seg = segs[gi]
        j = assigns[gi].index(track_id)
        comp = seg.components[j]
        hh, ww = seg.labels.shape
        crop = tr.crop_square(comp, FACTOR, W, H, mp.cropMargin)
        ps = tr.patch_sets(seg.labels, comp.label, crop, FACTOR, GRID, GRID, mp.coreThreshold)
        san = tr.sanity(comp, seg.components, ww, hh, mp, W, H, int(ps.core.sum()))
        geom = tr.component_geometry(seg.labels, comp)
        c_in = mse.crop_u8(tr, decoded[gi], crop, N_IN)
        if san != tr.OK:
            return tr.judge(twin, None, None, None, sanity_reason=san), None
        feat = bb(c_in)
        return tr.judge(twin, feat, ps.cov, geom.features()), (feat, ps.cov)

    verdicts = []
    for e in fired:
        rec = dict(tMs=ts[e["frame"]], frame=e["frame"], trackId=e["trackId"], trigger=e["trigger"])
        if e["judgeFrame"] is None:
            j = tr.Judgement(tr.REFRAME, tr.TOUCHES_BORDER, finalVerdict=tr.REFRAME)
            rec.update(judgeFrame=None, judgeTMs=None)
            extra = []
        else:
            gi = frame_of[e["judgeFrame"]]
            j, _ = judge_at(gi, e["trackId"])
            rec.update(judgeFrame=gi, judgeTMs=ts[gi])
            extra = []
            for b in e["buffer"][1:]:
                jb, fc = judge_at(frame_of[b], e["trackId"])
                if fc is not None:
                    extra.append(fc)
        border = False
        if j.s is not None and abs(j.s - 1.0) <= BORDER_S:
            border = True
        if j.sim is not None and abs(j.sim - twin.tauId) <= BORDER_SIM:
            border = True
        vote_verdict = None
        if j.verdict == tr.DEFECT and extra is not None:
            jv = tr.vote(twin, j, extra)
            if jv.vote is not None:
                vote_verdict = jv.vote["verdict"]
        rec.update(verdict=j.verdict, reason=j.reason, s=None if j.s is None else float(j.s),
                   sim=None if j.sim is None else float(j.sim), borderline=bool(border), voteVerdict=vote_verdict)
        verdicts.append(rec)

    # ground truth: which scripted part produced each track (the part whose centre is nearest at the judge frame)
    for v in verdicts:
        gi = v["judgeFrame"] if v["judgeFrame"] is not None else v["frame"]
        best, bd = None, 1e18
        for p in script["parts"]:
            if p["entryFrame"] <= gi <= p["exitFrame"]:
                cx = (-160.0 + SPEED * (gi - p["entryFrame"])) / FACTOR
                seg = segs[gi]
                j = assigns[gi].index(v["trackId"]) if v["trackId"] in assigns[gi] else None
                if j is None:
                    continue
                dcx = abs(seg.components[j].cx - cx)
                if dcx < bd:
                    best, bd = p, dcx
        v["part"] = None if best is None else best["part"]
        v["label"] = None if best is None else best["label"]
    expected = dict(
        analysisFactor=FACTOR, inputSize=N_IN,
        teach=dict(startMs=seg_teach["startMs"], endMs=seg_teach["endMs"]),
        line=dict(startMs=seg_line["startMs"]),
        verdicts=verdicts,
        exited=[dict(tMs=ts[e["frame"]], frame=e["frame"], trackId=e["trackId"], judged=e["judged"],
                     confirmed=e["confirmed"]) for e in exited],
        sheet=dict(mu=[float(v) for v in sheet.mu], sigma=[float(v) for v in sheet.sigma]),
        twin=dict(tau=twin.tau, tauId=twin.tauId, tauIdRule=twin.tauIdRule, keyframes=len(twin.keyframes),
                  bankRows=int(twin.bank.shape[0]), framesSeen=ti["framesSeen"], framesAccepted=ti["framesAccepted"],
                  framesKept=ti["framesKept"], segmentsUsed=twin.segmentsUsed, lso=[float(x) for x in twin.lso]),
        borderlineRule=f"|s - 1| <= {BORDER_S} or |sim - tauId| <= {BORDER_SIM}: any verdict accepted",
        generator="tools/lab/make_replay_synth.py (twin_ref + backbone_r18_320.tflite on the decoded JPEG frames)",
        twinRefSha256=mse.sha256_file(tr.__file__), backboneSha256=mse.sha256_file(mse.R18_TFLITE),
        note="SYNTHETIC - mechanics regression only, not evidence")
    mse.write_json(os.path.join(tmp, "timestamps.json"), ts)
    mse.write_json(os.path.join(tmp, "script.json"), script)
    mse.write_json(os.path.join(tmp, "expected.json"), expected)
    shutil.rmtree(a.out, ignore_errors=True)
    os.replace(tmp, a.out)
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(a.out) for fn in fns)
    print(f"[replay] wrote {a.out}: {len(small)} frames, {size / 1e6:.1f} MB in {time.time() - t0:.0f} s")
    for v in verdicts:
        print(f"   t={v['tMs']:6d} track {v['trackId']:2d} part {v['part']} ({v['label']}) {v['trigger']} -> "
              f"{v['verdict']} {v['reason'] or ''} s={v['s'] if v['s'] is None else round(v['s'], 3)} "
              f"sim={v['sim'] if v['sim'] is None else round(v['sim'], 4)}{' BORDERLINE' if v['borderline'] else ''}"
              f"{' vote->' + v['voteVerdict'] if v['voteVerdict'] else ''}")
    print(f"   exited: {len(expected['exited'])} tracks, unjudged {sum(1 for e in expected['exited'] if not e['judged'])}")


if __name__ == "__main__":
    main()
