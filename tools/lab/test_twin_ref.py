#!/usr/bin/env python3
"""Plain-assert checks of tools/lab/twin_ref.py (the numpy reference of docs/verification/twin-spec.md v1).

    .venv\\Scripts\\python.exe tools/lab/test_twin_ref.py            (all checks, incl. a realistic-size teach)
    .venv\\Scripts\\python.exe tools/lab/test_twin_ref.py --quick    (skip the realistic-size teach timing)

Prints PASS/FAIL per check and exits non-zero if any check fails. Independent cross-checks: scipy (incomplete beta,
Clopper-Pearson, labelling, morphology, convex hull), numpy.percentile, brute-force loops written straight from the
spec text (downscale, sharpness, bilinear sampling, patch coverage, masked smoothing, k-NN), the legacy
patchcore_ref.greedy_coreset, and the golden file (layout, determinism).
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import twin_ref as tr  # noqa: E402

RESULTS = []


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            fn()
            RESULTS.append((name, True, ""))
            print("PASS  %-52s (%.1f s)" % (name, time.time() - t0))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, repr(e)))
            print("FAIL  %-52s %r" % (name, e))
            traceback.print_exc()
        return fn
    return deco


def close(a, b, rel=1e-9, abs_=1e-12):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return bool(np.all(np.abs(a - b) <= rel * np.abs(b) + abs_))


def run_all(quick):
    rng = np.random.default_rng(7)

    # ------------------------------------------------------------------------------------------- s0 statistics
    @check("s0 percentile/median/MAD vs numpy")
    def _():
        for n in (1, 2, 3, 4, 7, 10, 37, 100):
            x = rng.normal(0, 3, n)
            for q in (0, 5, 25, 33.3, 40, 50, 95, 99, 100):
                assert close(tr.percentile(x, q), np.percentile(x, q), 1e-12, 1e-12), (n, q)
            m = np.median(x)
            assert close(tr.median(x), m, 1e-12, 1e-12)
            assert close(tr.mad(x), np.median(np.abs(x - m)), 1e-12, 1e-12)
            assert close(tr.std(x), np.std(x), 1e-12, 1e-12) and close(tr.mean(x), np.mean(x), 1e-12, 1e-12)

    @check("s14 AUROC vs pairwise count and scipy Mann-Whitney")
    def _():
        from scipy.stats import mannwhitneyu
        p = np.round(rng.normal(1, 1, 30), 1)
        q = np.round(rng.normal(0, 1, 40), 1)
        u = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in p for b in q)
        assert close(tr.auroc(p, q), u / (p.size * q.size), 1e-15, 0)
        assert close(tr.auroc(p, q), mannwhitneyu(p, q).statistic / (p.size * q.size), 1e-12, 0)

    # ------------------------------------------------------------------------------------------- s0 binary16
    @check("s0 binary16 RNE bits (ties, subnormals, overflow, NaN)")
    def _():
        cases = {1 + 2 ** -11: 0x3C00, 1 + 3 * 2 ** -11: 0x3C02, 2 ** -24: 0x0001, 2 ** -25: 0x0000,
                 3 * 2 ** -25: 0x0002, 65504.0: 0x7BFF, 65520.0: 0x7C00, -65520.0: 0xFC00, 2049.0: 0x6800,
                 2051.0: 0x6802, -0.0: 0x8000, 2 ** -14: 0x0400, 2 ** -14 - 2 ** -24: 0x03FF}
        for v, bits in cases.items():
            assert int(tr.f16_bits(np.float32(v))) == bits, (v, hex(int(tr.f16_bits(np.float32(v)))), hex(bits))
        nan = tr.f16_bits(np.array([0x7FC00000], np.uint32).view(np.float32))[0]
        assert (nan & 0x7C00) == 0x7C00 and (nan & 0x3FF) != 0
        # f64 -> f16 is a single rounding (differs from f64 -> f32 -> f16 here)
        x = 1 + 2 ** -11 + 2 ** -40
        assert int(tr.f16_bits(np.float64(x))) == 0x3C01 and int(tr.f16_bits(np.float32(x))) == 0x3C00

    @check("s8 .f16 file format round trip")
    def _():
        with tempfile.TemporaryDirectory() as d:
            for shape in [(3, 4), (2, 5, 7), (0, 8), (1, 1)]:
                a = rng.normal(0, 10, shape).astype(np.float32)
                p = os.path.join(d, "x.f16")
                tr.write_f16(p, a)
                raw = open(p, "rb").read()
                assert raw[:8] == b"KZF16v1\n"
                rows, cols = int.from_bytes(raw[8:12], "little"), int.from_bytes(raw[12:16], "little")
                assert rows == shape[0] and cols == int(np.prod(shape[1:])) and len(raw) == 16 + 2 * rows * cols
                b = tr.read_f16(p)
                assert b.shape == (rows, cols)
                assert np.array_equal(b.view(np.uint16).ravel(), tr.f16_bits(a).ravel())
                assert np.array_equal(tr.f16_round(b.astype(np.float32)), b.astype(np.float32))  # idempotent

    # ------------------------------------------------------------------------------------------- s10.2 binomial
    @check("s10.2 beta_inc / beta_inv vs scipy.special")
    def _():
        from scipy import special
        worst = 0.0
        for a in (0.5, 1.0, 2.0, 3.0, 7.5, 20.0, 60.0, 299.0):
            for b in (0.5, 1.0, 2.0, 5.0, 30.0, 150.0, 300.0):
                for x in (1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
                    worst = max(worst, abs(tr.beta_inc(x, a, b) - special.betainc(a, b, x)))
        assert worst < 1e-12, worst
        worst = 0.0
        for a in (1.0, 2.0, 5.0, 30.0, 297.0):
            for b in (1.0, 3.0, 20.0, 296.0):
                for p in (0.025, 0.05, 0.5, 0.95, 0.975):
                    worst = max(worst, abs(tr.beta_inv(p, a, b) - special.betaincinv(a, b, p)))
        assert worst < 1e-9, worst

    @check("s10.2 Clopper-Pearson / order-statistic vs scipy.stats")
    def _():
        from scipy.stats import beta
        for m, want in ((20, 0.139), (29, 0.098), (59, 0.0495)):
            assert abs(tr.alpha_order(m) - want) < 5e-4, (m, tr.alpha_order(m))
        for r, n in [(0, 1), (0, 20), (1, 20), (2, 29), (5, 100), (19, 20), (3, 299)]:
            assert abs(tr.cp_upper(r, n) - beta.ppf(0.95, r + 1, n - r)) < 1e-9, (r, n)
        assert tr.cp_upper(10, 10) == 1.0
        for k, n in [(0, 10), (10, 10), (5, 10), (1, 20), (50, 100), (297, 299)]:
            lo, hi = tr.two_sided(k, n)
            want_lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
            want_hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
            assert abs(lo - want_lo) < 1e-9 and abs(hi - want_hi) < 1e-9, (k, n)

    # ------------------------------------------------------------------------------------------- s1 image ops
    @check("s1.1 downscale vs spec loops (stride padding, odd f)")
    def _():
        for w, h, pad, f in [(16, 12, 8, 4), (16, 12, 0, 2), (12, 9, 4, 3)]:
            stride = 4 * w + pad
            buf = rng.integers(0, 256, h * stride, dtype=np.uint8)
            out = tr.downscale_rgba(buf, w, h, stride, f)
            for y in range(h // f):
                for x in range(w // f):
                    for c in range(3):
                        s = sum(int(buf[(f * y + i) * stride + (f * x + j) * 4 + c]) for i in range(f) for j in range(f))
                        assert out[y, x, c] == math.floor((s + f * f / 2) / (f * f))

    @check("s1.3 Laplacian-variance sharpness vs spec loops")
    def _():
        g = tr.grey(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8))
        for rect in [(0, 0, 16, 12), (3, 2, 11, 9), (-4, -3, 6, 5), (5, 5, 8, 7), (5, 5, 8, 8), (8, 8, 8, 10)]:
            x0, y0, x1, y1 = rect
            ls = [g[y - 1, x] + g[y + 1, x] + g[y, x - 1] + g[y, x + 1] - 4 * g[y, x]
                  for y in range(max(y0, 1), min(y1, 11)) for x in range(max(x0, 1), min(x1, 15))]
            want = float(np.var(ls)) if len(ls) >= 9 else 0.0
            got, cnt = tr.sharpness(g, *rect)
            assert cnt == len(ls) and close(got, want, 1e-9, 1e-9), (rect, got, want)

    @check("s1.4 bilinear sample / crop / rotated crop vs spec loops")
    def _():
        im = rng.integers(0, 256, (8, 10, 3), dtype=np.uint8).astype(np.float64)

        def samp(u, v):
            x, y = u - 0.5, v - 0.5
            x0, y0 = math.floor(x), math.floor(y)
            fx, fy = x - x0, y - y0
            cl = lambda a, n: min(max(a, 0), n - 1)  # noqa: E731
            xa, xb, ya, yb = cl(x0, 10), cl(x0 + 1, 10), cl(y0, 8), cl(y0 + 1, 8)
            return ((1 - fx) * (1 - fy) * im[ya, xa] + fx * (1 - fy) * im[ya, xb] + (1 - fx) * fy * im[yb, xa]
                    + fx * fy * im[yb, xb])

        for u, v in [(0.5, 0.5), (-3.0, 2.2), (9.99, 7.9), (12.0, -1.0), (3.3, 6.7)]:
            assert close(tr.sample(im, u, v), samp(u, v), 1e-14, 1e-12)
        x0, y0, side, n = 2.3, 1.7, 5.1, 5
        c = tr.crop_resize(im, x0, y0, side, n)
        for i in range(n):
            for j in range(n):
                assert close(c[i, j], samp(x0 + (j + 0.5) * side / n, y0 + (i + 0.5) * side / n), 1e-14, 1e-12)
        cx, cy, th = 5.0, 4.0, 0.3
        r = tr.crop_resize_rotated(im, cx, cy, side, th, n)
        for i in range(n):
            for j in range(n):
                a, b = (j + 0.5) * side / n - side / 2, (i + 0.5) * side / n - side / 2
                want = samp(cx + a * math.cos(th) - b * math.sin(th), cy + a * math.sin(th) + b * math.cos(th))
                assert close(r[i, j], want, 1e-13, 1e-10)
        bytes_ = tr.crop_bytes(np.array([0.49, 0.5, 1.5, 254.5, 255.4, -0.2]))
        assert list(bytes_) == [0, 1, 2, 255, 255, 0]

    # ------------------------------------------------------------------------------------------- s2 mask
    @check("s2.3 morphology vs scipy.ndimage (border rule)")
    def _():
        from scipy import ndimage
        st = np.ones((3, 3), bool)
        for _ in range(5):
            m = rng.random((30, 41)) > 0.45
            assert np.array_equal(tr.erode(m), ndimage.binary_erosion(m, st, border_value=1))
            assert np.array_equal(tr.dilate(m), ndimage.binary_dilation(m, st, border_value=0))
            assert np.array_equal(tr.dilate(m, 2), ndimage.binary_dilation(m, np.ones((5, 5), bool), border_value=0))

    @check("s2.4 labelling: partition, first-pixel order, renumbering")
    def _():
        from scipy import ndimage
        for _ in range(5):
            m = rng.random((40, 50)) > 0.55
            lab, areas = tr.label_components(m)
            sl, n = ndimage.label(m, structure=np.ones((3, 3)))
            assert len(areas) == n and len(set(zip(lab[m].tolist(), sl[m].tolist()))) == n
            firsts = [int(np.flatnonzero(lab.ravel() == k)[0]) for k in range(1, n + 1)]
            assert firsts == sorted(firsts)
            assert areas == [int(np.sum(lab == k)) for k in range(1, n + 1)]
        # renumbering after speck removal keeps the order
        fr = np.zeros((30, 40, 3), np.uint8)
        fr[2:4, 30:32] = 255                 # speck (removed by opening)
        fr[5:9, 5:9] = 255                   # 16 px -> removed by minBlobPx
        fr[12:20, 3:12] = 255                # 72 px
        fr[3:10, 20:27] = 255                # 49 px (first pixel earlier in the scan than the 72 px blob)
        seg = tr.segment(fr, tr.SheetModel(np.zeros(3), np.full(3, 3.0)), tr.MaskParams())
        assert seg.rawAreas == [49, 16, 72] and [c.area for c in seg.components] == [49, 72]
        assert seg.components[0].label == 1 and seg.labels[3, 20] == 1 and seg.labels[12, 3] == 2

    @check("s2.7 geometry: disc, rectangle, rotated rectangle, L-shape")
    def _():
        from scipy.spatial import ConvexHull
        yy, xx = np.mgrid[0:121, 0:121]
        disc = (xx - 60) ** 2 + (yy - 60) ** 2 <= 50 ** 2
        g = tr.geometry_features(*np.nonzero(disc)[::-1])
        assert abs(g.hu1 - 1 / (2 * math.pi)) < 1e-3 and abs(g.aspect - 1) < 1e-9 and g.solidity > 0.97
        assert abs(g.fill - math.pi / 4) < 0.02
        rect = np.zeros((60, 60), bool)
        rect[10:20, 5:45] = True
        g = tr.geometry_features(*np.nonzero(rect)[::-1])
        assert g.fill == 1.0 and g.solidity == 1.0 and abs(g.aspect - math.sqrt(99 / 1599)) < 1e-12
        assert abs(g.hu1 - (10 * 40 * (40 ** 2 - 1) / 12 + 40 * 10 * (10 ** 2 - 1) / 12) / 400 ** 2) < 1e-12
        u, v = np.mgrid[-30:30:0.25, -8:8:0.25]
        th = math.radians(30)
        rot = np.zeros((120, 120), bool)
        rot[np.round(60 + u * math.sin(th) + v * math.cos(th)).astype(int),
            np.round(60 + u * math.cos(th) - v * math.sin(th)).astype(int)] = True
        rot = tr.erode(tr.dilate(rot))                         # fill rasterisation pin-holes
        g = tr.geometry_features(*np.nonzero(rot)[::-1])
        assert abs(g.theta - th) < 0.02 and g.fill > 0.9 and g.solidity > 0.9 and abs(g.aspect - 16 / 60) < 0.05
        ell = np.zeros((40, 40), bool)
        ell[5:35, 5:15] = True
        ell[25:35, 15:35] = True
        ys, xs = np.nonzero(ell)
        g = tr.geometry_features(xs, ys)
        pts = np.concatenate([np.stack([xs + dx, ys + dy], 1) for dx in (0, 1) for dy in (0, 1)])
        assert g.hullArea == ConvexHull(pts).volume and abs(g.solidity - 500 / g.hullArea) < 1e-15
        assert 0.6 < g.solidity < 0.75

    # ------------------------------------------------------------------------------------------- s3/s4 patches
    @check("s3/s4 crop square + patch coverage vs spec loops")
    def _():
        lab = np.zeros((18, 24), np.int32)
        lab[4:13, 5:14] = 1
        lab[0:3, 0:4] = 2
        comps = tr.components_of(lab, 2, 2)
        for comp in comps:
            sq = tr.crop_square(comp, 4, 96, 72, 0.1)
            bx0, bx1 = 4 * comp.minX, 4 * (comp.maxX + 1)
            by0, by1 = 4 * comp.minY, 4 * (comp.maxY + 1)
            side = min(max(bx1 - bx0, by1 - by0) * 1.2, 96, 72)
            assert close(sq.side, side) and 0 <= sq.x0 <= 96 - sq.side and 0 <= sq.y0 <= 72 - sq.side
        for crop in [tr.crop_square(comps[0], 4, 96, 72, 0.1), tr.Crop(-3.1, -2.6, 40.3), tr.Crop(70.3, 50.1, 37.3)]:
            cov = tr.patch_cov(lab, 1, crop, 4, 6, 6)
            for r in range(6):
                for c in range(6):
                    hits = 0
                    for i in range(4):
                        for j in range(4):
                            u = crop.x0 + (c + (j + 0.5) / 4) * crop.side / 6
                            v = crop.y0 + (r + (i + 0.5) / 4) * crop.side / 6
                            ax, ay = math.floor(u / 4), math.floor(v / 4)
                            hits += 1 if (0 <= ax < 24 and 0 <= ay < 18 and lab[ay, ax] == 1) else 0
                    assert cov[r, c] == hits / 16

    # ------------------------------------------------------------------------------------------- s5 k-NN
    @check("s5 k-NN: exact and BLAS paths vs brute force, KNN_MASK")
    def _():
        q = rng.normal(0, 1, (37, 16)).astype(np.float32)
        b = rng.normal(0, 1, (53, 16)).astype(np.float32)
        ex = rng.random(53) < 0.3
        brute = np.array([min(float(np.sum((q[i].astype(np.float64) - b[j].astype(np.float64)) ** 2))
                              for j in range(53) if not ex[j]) for i in range(37)])
        assert close(tr.knn_min_sq(q, b, ex), brute, 1e-12, 1e-12)
        old = tr._DIRECT_LIMIT
        tr._DIRECT_LIMIT = 0                                # force the BLAS path
        try:
            assert close(tr.knn_min_sq(q, b, ex), brute, 1e-9, 1e-9)
        finally:
            tr._DIRECT_LIMIT = old
        assert np.all(tr.knn_min_sq(q, b, np.ones(53, bool)) >= 1e30)

    @check("s7.6 coreset == patchcore_ref.greedy_coreset (float64 copy)")
    def _():
        import patchcore_ref as pc
        x = rng.normal(0, 1, (1500, 24)).astype(np.float32)
        mine = tr.greedy_coreset(x, 150)
        legacy = pc.greedy_coreset(x.astype(np.float64), 150)
        assert np.array_equal(mine, legacy)
        x = rng.normal(0, 1, (33000, 128)).astype(np.float32)      # > _DIRECT_LIMIT -> BLAS path
        mine = tr.greedy_coreset(x, 60)
        legacy = pc.greedy_coreset(x.astype(np.float64), 60)
        assert np.array_equal(mine, legacy)

    # ------------------------------------------------------------------------------------------- s6 scoring
    @check("s6.2-6.4 masked smoothing / frame score vs spec loops")
    def _():
        d = rng.integers(0, 40, (9, 11)) / 8.0
        core = np.zeros((9, 11), bool)
        core[2:6, 3:8] = True
        S = tr.dilate(core)
        sm = tr.masked_smooth(d, S)
        for r in range(9):
            for c in range(11):
                if not S[r, c]:
                    continue
                vals = [d[rr, cc] for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                        if 0 <= rr < 9 and 0 <= cc < 11 and S[rr, cc]]
                assert sm[r, c] == sum(vals) / len(vals)
        fs = tr.frame_score(d, S, tr.SMOOTHED_MAX)
        assert fs.raw == max(sm[S]) and fs.peak[2] == int(np.argmax(np.where(S, sm, -1).ravel()))
        fs = tr.frame_score(d, S, tr.TOP1_MEAN)
        n = max(1, math.ceil(0.01 * S.sum()))
        assert fs.topN == n and close(fs.raw, np.mean(np.sort(d[S])[::-1][:n]), 1e-15, 0)

    # ------------------------------------------------------------------------------------------- s11 tracker
    @check("s11 tracker: 50 parts crossing -> 50 LINE events")
    def _():
        p = tr.TrackerParams(w=320, h=180, steadyEnabled=False)
        trk = tr.Tracker(p)
        fired, exited = [], []
        for i in range(50 * 20 + 40):
            t = int(round(33.3 * i))
            dets = []
            for k in range(50):                      # part k enters at frame 20k, moves right at 0.45 px/ms
                t0 = 33.3 * 20 * k
                if t < t0:
                    continue
                cx = -10 + 0.45 * (t - t0)
                if cx - 12 > 330:
                    continue
                x0, x1 = max(0, int(math.floor(cx - 12))), min(319, int(math.ceil(cx + 12)) - 1)
                if x1 < x0:
                    continue
                dets.append(tr.Detection((x1 - x0 + 1) * 18, x0, 81, x1, 98, (x0 + x1 + 1) / 2, 90.0,
                                         x0 < 2 or x1 >= 318, 400.0 + 30 * math.sin(i)))
            r = trk.step(t, dets)
            fired += [e for e in r["events"] if e["type"] == "FIRED"]
            exited += [e for e in r["events"] if e["type"] == "EXITED"]
        assert len(fired) == 50 and all(e["trigger"] == tr.LINE for e in fired), len(fired)
        assert len({e["trackId"] for e in fired}) == 50 and all(e["judgeFrame"] is not None for e in fired)
        assert len(exited) == 50 and all(e["judged"] for e in exited)

    @check("s11 tracker: wrong direction never fires, steady hold fires once")
    def _():
        trk = tr.Tracker(tr.TrackerParams(w=320, h=180, steadyEnabled=False, lineDirection="POSITIVE"))
        n_fired = 0
        for i in range(40):
            cx = 300 - 10.0 * i
            if cx < 5:
                break
            r = trk.step(33 * i, [tr.Detection(400, int(cx) - 10, 80, int(cx) + 9, 99, cx, 90.0, False, 400.0)])
            n_fired += sum(1 for e in r["events"] if e["type"] == "FIRED")
        assert n_fired == 0
        trk = tr.Tracker(tr.TrackerParams(w=320, h=180, lineEnabled=False))
        events = []
        for i in range(60):
            r = trk.step(33 * i, [tr.Detection(400, 90, 80, 109, 99, 100.0 + 0.25 * (i % 2), 90.0, False, 400.0)])
            events += [(i, e) for e in r["events"]]
        fires = [(i, e["trigger"]) for i, e in events if e["type"] == "FIRED"]
        assert len(fires) == 1 and fires[0][1] == tr.STEADY, fires

    # ------------------------------------------------------------------------------------------- s12 governor
    @check("s12 governor: rise immediately, fall after cooldown, blip resets")
    def _():
        g = tr.Governor()
        seq = [(0, 0, 0.3), (1000, 3, None), (2000, 0, 0.3), (31999, 0, 0.3), (32000, 0, 0.3), (33000, 2, None),
               (40000, 0, float("nan")), (60000, 2, None), (61000, 0, 0.5), (90999, 0, 0.5), (91000, 0, 0.5)]
        levels = [g.step(t, s, h, False)["level"] for t, s, h in seq]
        assert levels == ["L0", "L2", "L2", "L2", "L0", "L1", "L1", "L1", "L1", "L1", "L0"], levels
        st = g.step(92000, 0, 0.9, True)
        assert st["level"] == "L2" and st["fps"] == 2 and st["banner"] and not st["voting"] and st["vlm"] == "PAUSED"

    # ------------------------------------------------------------------------------------------- s8 storage
    @check("s8 canonical JSON + fingerprint")
    def _():
        pl = {"b": 1, "a": {"z": 4.0, "y": 0.1, "x": True}, "A": "s"}
        assert tr.canonical_json(pl) == '{"A":"s","a":{"x":true,"y":0.1,"z":4.0},"b":1}'
        import hashlib
        assert tr.fingerprint(pl) == hashlib.sha256(tr.canonical_json(pl).encode()).hexdigest()
        try:
            tr.canonical_json({"eps": 1e-5})
            raise AssertionError("expected a refusal for 1e-5")
        except ValueError:
            pass

    @check("s7/s8 Twin save -> load reproduces every judgement")
    def _():
        import make_golden_twin as mg
        _, twin, world, tp = mg.sec_teach()
        pl = dict(mg.PIPELINES["r18_default"], gh=twin.gh, gw=twin.gw, dim=twin.dim)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "twin_a")
            tr.save_twin(path, twin, pl)
            back = tr.load_twin(path)
        for name in ("bank", "globals", "kfCov"):
            assert np.array_equal(getattr(back, name), getattr(twin, name)), name
        assert np.array_equal(back.kfFeats, twin.kfFeats) and np.array_equal(back.bankSeg, twin.bankSeg)
        cov = world.cov(6100.0)
        geo = dict(area=1500, fill=0.62, aspect=0.8, hu1=0.17, solidity=0.93)
        for seed in range(5):
            f = world.features(6100.0, np.random.default_rng(seed), cov)
            for bank_rule in (tr.CORESET, tr.TOPK_VIEWS):
                a = tr.judge(twin, f, cov, geo, bank_rule=bank_rule)
                b = tr.judge(back, f, cov, geo, bank_rule=bank_rule)
                assert (a.verdict, a.reason, a.sim, a.s, a.peak) == (b.verdict, b.reason, b.sim, b.s, b.peak)

    # ------------------------------------------------------------------------------------------- golden file
    @check("golden: regenerates identically (determinism) and matches disk")
    def _():
        import make_golden_twin as mg
        a = mg.dumps(mg.build())
        b = mg.dumps(mg.build())
        assert a == b
        path = os.path.join(ROOT, "testdata", "golden_twin.json")
        if os.path.exists(path):
            disk = open(path, "rb").read()
            assert not disk.startswith(b"\xef\xbb\xbf"), "BOM"
            assert disk.decode("utf-8") == a, "testdata/golden_twin.json is stale - re-run make_golden_twin.py --out"
        assert len(a.encode("utf-8")) < 3_000_000

    @check("golden: layout matches the make_golden_twin.py docstring")
    def _():
        path = os.path.join(ROOT, "testdata", "golden_twin.json")
        doc = json.load(open(path, encoding="utf-8"))
        sections = ["stats", "half", "binomial", "auroc", "image", "mask", "crop_patch", "knn", "scoring", "teach",
                    "verdicts", "calibration", "tracker", "governor", "twin_json"]
        assert list(doc) == ["meta"] + sections
        need = {
            "stats": ({"x", "qs"}, {"percentiles", "median", "mad", "robustSigma", "mean", "std"}),
            "auroc": ({"pos", "neg"}, {"auroc"}),
            "mask": ({"frame", "params"}, {"sheet", "d2", "foreground", "afterOpen", "afterClose", "rawAreas",
                                           "labels", "components", "geometry", "mainObject", "perComponent",
                                           "sanity"}),
            "crop_patch": ({"labels", "label", "params"}, {"crop", "cov", "core", "S", "B"}),
            "knn": ({"feats", "bank", "excluded", "knnMask"}, {"minSq", "d", "allExcluded"}),
            "scoring": ({"gh", "gw", "dmap", "core", "S", "tau", "sensitivity", "scoreRule", "params"},
                        {"sm", "raw", "s", "topN", "peak", "coreCount", "anomalousCount", "anomalousFraction",
                         "areaPct"}),
            "calibration": ({"params", "tauTeach", "positives", "negSims", "segmentsUsed", "samples",
                             "sensitivities", "latencyMs"},
                            {"n", "nSanityOk", "m", "tauCal", "calMax", "alpha", "identity", "geometry",
                             "certificate", "valid"}),
            "tracker": ({"params", "frames"}, {"frames", "fired", "exited"}),
            "governor": ({"params", "samples"}, {"steps"}),
        }
        for sec in sections:
            cases = doc[sec]["cases"]
            assert cases, sec
            for c in cases:
                assert set(c) >= {"name", "inputs", "expected"}, (sec, c.keys())
                if sec in need:
                    assert set(c["inputs"]) >= need[sec][0], (sec, c["name"], set(c["inputs"]))
                    assert set(c["expected"]) >= need[sec][1], (sec, c["name"], set(c["expected"]))
        ok = [c for c in doc["teach"]["cases"] if c["kind"] == "ok"]
        for c in ok:
            assert set(c["expected"]) >= {"accepted", "sharpnessCut", "kept", "keyframes", "kcMinDist", "kcStop",
                                          "kcNext", "t0", "t1", "segmentsUsed", "pooledRows", "bankRows",
                                          "coresetIndices", "bank", "bankKf", "bankSeg", "globals", "lsoMode",
                                          "lso", "lsoKeyframes", "tauTeach", "tau", "positives", "identity",
                                          "geometry", "coverageCut"}
        res = doc["verdicts"]["cases"][0]["expected"]["results"]
        assert {r["finalVerdict"] for r in res} == {"PASS", "DEFECT", "NOT_ENROLLED", "REFRAME"}
        assert {r["reason"] for r in res} >= {"IDENTITY", "SHAPE", "COVERAGE", "NO_OBJECT", "TOUCHES_BORDER",
                                              "TOO_SMALL", "TOO_LARGE", "MULTIPLE", "NO_CORE"}
        assert any(r["vote"] is not None for r in res)
        # float32 inputs are exact float32 values
        feats = doc["knn"]["cases"][0]["inputs"]["feats"]["data"]
        assert all(float(np.float32(v)) == v for v in feats)
        s = open(path, encoding="utf-8").read()
        assert "NaN" not in s and "Infinity" not in s

    @check("golden: verdict twin == teach twin; LSO + a verdict re-derived by loops")
    def _():
        doc = json.load(open(os.path.join(ROOT, "testdata", "golden_twin.json"), encoding="utf-8"))
        te = doc["teach"]["cases"][0]["expected"]
        v = doc["verdicts"]["cases"][0]
        tw = v["inputs"]["twin"]
        for k in ("bank", "bankKf", "bankSeg", "globals", "tau", "coverageCut", "geometry"):
            assert tw[k] == te[k], k
        assert tw["tauId"] == te["identity"]["withNegatives"]["tauId"]
        gh, gw, dim = tw["gh"], tw["gw"], tw["dim"]
        npat = gh * gw
        bank = [tw["bank"]["data"][i * dim:(i + 1) * dim] for i in range(tw["bank"]["rows"])]
        kff = tw["kfFeats"]["data"]
        kfc = tw["kfCov"]["data"]

        def dmap(fm, allowed):
            out = []
            for p in range(npat):
                best = math.inf
                for m, row in enumerate(bank):
                    if allowed[m]:
                        best = min(best, sum((fm[p * dim + t] - row[t]) ** 2 for t in range(dim)))
                out.append(math.sqrt(best))
            return out

        def sets(cov):
            core = [c >= 0.5 for c in cov]
            S = [any(core[rr * gw + cc] for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                     if 0 <= rr < gh and 0 <= cc < gw) for r in range(gh) for c in range(gw)]
            return core, S

        def smoothed_max(d, S):
            best = -math.inf
            for r in range(gh):
                for c in range(gw):
                    if S[r * gw + c]:
                        vals = [d[rr * gw + cc] for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                                if 0 <= rr < gh and 0 <= cc < gw and S[rr * gw + cc]]
                        best = max(best, sum(vals) / len(vals))
            return best

        seg = [k["segment"] for k in te["keyframes"]]
        for k, want in zip(te["lsoKeyframes"], te["lso"]):
            fm = kff[k * npat * dim:(k + 1) * npat * dim]
            _, S = sets(kfc[k * npat:(k + 1) * npat])
            got = smoothed_max(dmap(fm, [tw["bankSeg"][m] != seg[k] for m in range(len(bank))]), S)
            assert close(got, want, 1e-12, 1e-12), (k, got, want)
        assert close(te["tau"], te["tauFactor"] if "tauFactor" in te else 1.4 * max(te["lso"]), 1e-15, 0)
        q, r = v["inputs"]["queries"][0], v["expected"]["results"][0]
        fm = q["features"]["data"]
        d = dmap(fm, [True] * len(bank))
        assert close(d, r["dmap"], 1e-12, 1e-12)
        core, S = sets(q["cov"])
        assert close(smoothed_max(d, S), r["raw"], 1e-12, 1e-12)
        sv = [sum(fm[p * dim + t] for p in range(npat) if core[p]) for t in range(dim)]
        nrm = math.sqrt(sum(x * x for x in sv))
        gl = tw["globals"]["data"]
        sim = max(sum(gl[k * dim + t] * sv[t] / nrm for t in range(dim)) for k in range(tw["globals"]["rows"]))
        assert close(sim, r["sim"], 1e-12, 1e-12) and r["verdict"] == "PASS"

    @check("API helpers: l2_normalize_patches, judge_best_of, teach(augment=)")
    def _():
        f = rng.normal(0, 1, (3, 4, 16)).astype(np.float32)
        f[0, 0] = 0
        n = tr.l2_normalize_patches(f)
        nr = np.linalg.norm(n.reshape(-1, 16).astype(np.float64), axis=1)
        assert np.all(n[0, 0] == 0) and np.allclose(nr[1:], 1.0, atol=1e-6)
        import make_golden_twin as mg
        teach_sec, twin, world, tp = mg.sec_teach()
        cov = world.cov(6100.0)
        a = world.features(6100.0, np.random.default_rng(1), cov)
        b = (a * np.float32(1.3)).astype(np.float32)
        geo = dict(area=1500, fill=0.62, aspect=0.8, hu1=0.17, solidity=0.93)
        ja, jb = tr.judge(twin, a, cov, geo), tr.judge(twin, b, cov, geo)
        best = tr.judge_best_of(twin, [(b, cov), (a, cov)], geo)
        assert best.raw == min(ja.raw, jb.raw)
        frames = [tr.TeachFrame(**{k: getattr(fr, k) for k in ("tMs", "sane", "sharpness", "cov", "geometry",
                                                                 "features")}) for fr in
                  [mg.tr.TeachFrame(int(t), True, float(300 + 7 * i), world.cov(t), dict(geo),
                                    world.features(t, np.random.default_rng(i), world.cov(t)))
                   for i, t in enumerate(np.linspace(0, 12000, 20))]]
        plain = tr.teach(frames, tp)
        aug = tr.teach(frames, tp, augment=lambda k, fi: [(np.rot90(frames[fi].features, r).copy(),
                                                          np.rot90(np.asarray(frames[fi].cov), r).copy())
                                                         for r in (1, 2, 3)])
        assert aug.teachInfo["pooledRows"] > plain.teachInfo["pooledRows"]

    # ------------------------------------------------------------------------------------------- performance
    if not quick:
        @check("realistic teach: 32 keyframes x 1600 patches x 128 dims < 120 s")
        def _():
            r = np.random.default_rng(3)
            gh = gw = 40
            d = 128
            base = np.abs(r.normal(1.0, 0.5, (gh, gw, d))).astype(np.float32)
            views = [r.normal(0, 0.25, (gh, gw, d)).astype(np.float32) for _ in range(8)]
            cov = np.zeros((gh, gw))
            cov[2:38, 2:38] = 1.0                    # core 36x36 -> B (5x5 dilation) = the whole 40x40 grid
            frames = []
            for i in range(60):
                t = i * 250
                w = r.dirichlet(np.ones(8))
                f = base + sum(w[k] * views[k] for k in range(8)) * 3.0 + r.normal(0, 0.05, (gh, gw, d))
                frames.append(tr.TeachFrame(t, True, float(r.uniform(100, 900)), cov,
                                            dict(area=1500, fill=0.6, aspect=0.8, hu1=0.17, solidity=0.93),
                                            f.astype(np.float32)))
            tp = tr.TeachParams(keepPercentile=10.0, kcEps=0.0)
            t0 = time.time()
            twin = tr.teach(frames, tp)
            dt = time.time() - t0
            print("      realistic teach: %.1f s, keyframes %d, pooled %d, bank %d, tau %.3f" % (
                dt, len(twin.keyframes), twin.teachInfo["pooledRows"], twin.bank.shape[0], twin.tau))
            assert len(twin.keyframes) == 32 and twin.teachInfo["pooledRows"] == 32 * 1600
            assert twin.bank.shape == (2400, 128) and dt < 120.0, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the realistic-size teach timing")
    a = ap.parse_args()
    run_all(a.quick)
    bad = [r for r in RESULTS if not r[1]]
    print("\n%d checks, %d passed, %d failed" % (len(RESULTS), len(RESULTS) - len(bad), len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
