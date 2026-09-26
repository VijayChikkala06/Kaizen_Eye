import { useState } from 'react';
import { Alert, ScrollView, Text, View } from 'react-native';

import { borderFor, DETECTION } from '../config';
import { enrol, type Profile } from '../core/patchcore';
import { BACKBONE, embed } from '../ml/backbone';
import { loadSquare } from '../ml/image';
import { saveProfile, saveReferenceImage } from '../storage/profileStore';
import { Button, CameraCapture, ProgressBar, Thumb } from '../ui/components';
import { colors, ui } from '../ui/theme';

export const RECOMMENDED_FRAMES = 20;
export const MIN_FRAMES = 5;
export const MAX_FRAMES = 30;

const nextFrame = () => new Promise<void>((r) => setTimeout(r, 0));

interface Shot {
  uri: string;
  feats: Float32Array;
}

export function EnrolScreen({ onDone, onCancel }: { onDone: (p: Profile) => void; onCancel: () => void }) {
  const [shots, setShots] = useState<Shot[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ stage: string; fraction: number } | null>(null);
  const [embedMs, setEmbedMs] = useState<number | null>(null);

  const addImages = async (picked: string[]) => {
    const room = MAX_FRAMES - shots.length;
    const uris = picked.slice(0, Math.max(0, room));
    if (uris.length < picked.length) {
      Alert.alert('Enough photos', `Up to ${MAX_FRAMES} good photos are used; ${picked.length - uris.length} were skipped.`);
    }
    if (!uris.length) return;
    setBusy(true);
    try {
      for (let i = 0; i < uris.length; i++) {
        if (uris.length > 1) setProgress({ stage: `Embedding photo ${i + 1} of ${uris.length}`, fraction: i / uris.length });
        await nextFrame();
        const img = await loadSquare(uris[i], BACKBONE.size);
        const t0 = Date.now();
        const feats = await embed(img.rgb);
        setEmbedMs(Date.now() - t0);
        setShots((s) => [...s, { uri: img.uri, feats }]);
      }
    } catch (e) {
      Alert.alert('Could not process photo', String(e));
    } finally {
      setProgress(null);
      setBusy(false);
    }
  };

  const build = async () => {
    setBusy(true);
    try {
      setProgress({ stage: 'Preparing', fraction: 0 });
      await nextFrame();
      const t0 = Date.now();
      const prof = await enrol(
        shots.map((s) => s.feats),
        BACKBONE.gh,
        BACKBONE.gw,
        BACKBONE.dim,
        {
          margin: 1.0, // the app stores margin 1.0 and applies the sensitivity at runtime
          border: borderFor(BACKBONE.gh),
          smooth: DETECTION.smooth,
          trimOutliers: DETECTION.trimOutliers,
          onProgress: (stage, fraction) => setProgress({ stage, fraction }),
          yieldFn: nextFrame,
        },
      );
      try {
        prof.refImage = await saveReferenceImage(shots[0].uri);
      } catch (e) {
        console.warn('Could not save the alignment reference photo', e);
      }
      saveProfile(prof);
      const secs = ((Date.now() - t0) / 1000).toFixed(1);
      const sorted = [...prof.looScores].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      const worst = prof.looScores.indexOf(sorted[sorted.length - 1]);
      const outlier = sorted[sorted.length - 1] > 1.5 * median;
      const loose = prof.tau > DETECTION.tauWarning;
      const dropped = prof.droppedFrames ?? [];
      Alert.alert(
        outlier || loose ? 'Profile ready - but check your photos' : 'Profile ready - consistent photos',
        `Learned "normal" from ${prof.nFrames} photos in ${secs} s.\nThreshold tau = ${prof.tau.toFixed(3)} ` +
          `(${loose ? `above ${DETECTION.tauWarning}: lenient` : 'good'})` +
          (dropped.length
            ? `\n\nIgnored ${dropped.length} photo(s) that did not match the rest: #${dropped.map((i) => i + 1).join(', #')}.`
            : '') +
          (loose
            ? '\n\nYour good photos differ too much, so small defects may pass. Usual causes: other objects, hands or ' +
              'cables entering the frame, a changing background, blur, or moving the phone. Use a plain surface, keep ' +
              'only the part in view, hold still, then re-enrol.'
            : '') +
          (outlier && !loose
            ? `\n\nPhoto #${worst + 1} looks different from the others; re-enrol without it for a tighter check.`
            : ''),
      );
      onDone(prof);
    } catch (e) {
      Alert.alert('Enrolment failed', String(e));
    } finally {
      setProgress(null);
      setBusy(false);
    }
  };

  const n = shots.length;
  return (
    <ScrollView style={ui.screen} contentContainerStyle={ui.scroll}>
      <View style={ui.between}>
        <Text style={ui.h1}>Enrol</Text>
        <Button title="Cancel" kind="secondary" onPress={onCancel} disabled={busy} style={{ paddingVertical: 8 }} />
      </View>
      <Text style={[ui.muted, { marginTop: 6 }]}>
        Photograph {RECOMMENDED_FRAMES} GOOD parts with the same mount, distance and light you will inspect with.
        {'\n'}• Keep the whole part INSIDE the dashed square, with a little margin.
        {'\n'}• Plain surface, and nothing else in view - no hands, cables or other objects.
        {'\n'}• Hold still until the image is sharp; line every shot up with the faint alignment guide.
      </Text>

      <CameraCapture
        onImages={addImages}
        busy={busy}
        multiple
        captureLabel={n >= MAX_FRAMES ? 'Enough photos' : 'Capture good part'}
        ghostUri={shots[0]?.uri}
        disabled={n >= MAX_FRAMES}
      />

      <View style={ui.card}>
        <View style={ui.between}>
          <Text style={ui.h2}>
            {n} / {RECOMMENDED_FRAMES} good photos
          </Text>
          {embedMs != null ? <Text style={ui.muted}>embed {embedMs} ms</Text> : null}
        </View>
        <ProgressBar fraction={n / RECOMMENDED_FRAMES} />
        {n > 0 ? (
          <ScrollView horizontal style={{ marginTop: 10 }} showsHorizontalScrollIndicator={false}>
            {shots.map((s, i) => (
              <Thumb key={i} uri={s.uri} />
            ))}
          </ScrollView>
        ) : null}
        {progress ? <ProgressBar fraction={progress.fraction} label={progress.stage} /> : null}
        <View style={[ui.row, { marginTop: 12 }]}>
          <Button
            title="Undo last"
            kind="secondary"
            onPress={() => setShots((s) => s.slice(0, -1))}
            disabled={busy || n === 0}
            style={{ flex: 1, marginRight: 8 }}
          />
          <Button
            title={n < MIN_FRAMES ? `Need ${MIN_FRAMES - n} more` : 'Build profile'}
            onPress={build}
            disabled={busy || n < MIN_FRAMES}
            style={{ flex: 1.4 }}
          />
        </View>
        {n >= MIN_FRAMES && n < RECOMMENDED_FRAMES ? (
          <Text style={[ui.muted, { marginTop: 8, color: colors.warn }]}>
            Works with {MIN_FRAMES}+, but {RECOMMENDED_FRAMES} photos give a more reliable threshold.
          </Text>
        ) : null}
      </View>
    </ScrollView>
  );
}
