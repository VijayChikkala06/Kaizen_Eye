/// <reference types="node" />
/**
 * Checks the TypeScript scoring maths against the Python reference's golden vectors.
 *   npm run verify-core
 * Pass criteria (README): coreset indices exactly, floats to ~1e-4 relative.
 */
import { readFileSync } from 'fs';
import { resolve } from 'path';

import { enrol, greedyCoreset, score } from '../src/core/patchcore';

type Golden = {
  params: { gh: number; gw: number; D: number; ratio: number; minK: number; folds: number; margin: number; start: number };
  frames: number[][][][];
  coreset_indices: number[];
  loo_scores: number[];
  tau: number;
  test_frame: number[][][];
  patch_dist: number[][];
  raw_score: number;
  normalised_score: number;
};

const path = resolve(__dirname, '../../testdata/golden_core.json');
const g: Golden = JSON.parse(readFileSync(path, 'utf8').replace(/^﻿/, ''));
const { gh, gw, D, ratio, minK, margin } = g.params;
const flat = (f: number[][][]) => Float32Array.from(f.flat(2));

let failures = 0;
let maxRel = 0;
function close(name: string, got: number, want: number, rel = 1e-4) {
  const r = Math.abs(got - want) / Math.max(Math.abs(want), 1e-12);
  maxRel = Math.max(maxRel, r);
  if (r > rel) {
    failures++;
    console.log(`FAIL ${name}: got ${got}, want ${want} (rel ${r.toExponential(2)})`);
  }
}

async function main() {
  const frames = g.frames.map(flat);
  const x = new Float32Array(frames.length * gh * gw * D);
  frames.forEach((f, i) => x.set(f, i * f.length));
  const k = Math.max(minK, Math.floor(ratio * frames.length * gh * gw));
  const sel = Array.from(await greedyCoreset(x, D, k, g.params.start));
  const idxOk = JSON.stringify(sel) === JSON.stringify(g.coreset_indices);
  if (!idxOk) {
    failures++;
    console.log('FAIL coreset_indices\n got ', sel.join(','), '\n want', g.coreset_indices.join(','));
  }

  const prof = await enrol(frames, gh, gw, D, { ratio, minK, margin });
  if (prof.looScores.length !== g.loo_scores.length) {
    failures++;
    console.log(`FAIL loo_scores length ${prof.looScores.length} vs ${g.loo_scores.length}`);
  }
  prof.looScores.forEach((v, i) => close(`loo_scores[${i}]`, v, g.loo_scores[i]));
  close('tau', prof.tau, g.tau);

  const res = await score(flat(g.test_frame), prof, 1.0);
  g.patch_dist.flat().forEach((v, i) => close(`patch_dist[${i}]`, res.dmap[i], v));
  close('raw_score', res.raw, g.raw_score);
  close('normalised_score', res.patchScore, g.normalised_score);

  console.log(
    `coreset indices: ${idxOk ? 'EXACT MATCH' : 'MISMATCH'} (${sel.length})  |  max relative float error: ${maxRel.toExponential(2)}  |  ` +
      `normalised_score ${res.patchScore.toFixed(6)} vs ${g.normalised_score.toFixed(6)}`,
  );
  if (failures) {
    console.log(`${failures} check(s) FAILED`);
    process.exit(1);
  }
  console.log('PASS - TypeScript core matches golden_core.json');
}

main();
