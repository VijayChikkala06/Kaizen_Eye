import { useState } from 'react';
import { Alert, ScrollView, Text, useWindowDimensions, View } from 'react-native';

import { score, type Profile } from '../core/patchcore';
import { BACKBONE, embed } from '../ml/backbone';
import { loadSquare } from '../ml/image';
import { Button, CameraCapture, HeatmapImage, Stat } from '../ui/components';
import { colors, ui } from '../ui/theme';

export interface Inspection {
  id: number;
  time: number;
  uri: string;
  dmap: Float32Array;
  raw: number;
  score: number;
  pass: boolean;
  threshold: number;
  sensitivity: number;
  peak: { row: number; col: number };
  embedMs: number;
  scoreMs: number;
}

const nextFrame = () => new Promise<void>((r) => setTimeout(r, 0));

export function ResultCard({ r, size }: { r: Inspection; size: number }) {
  const color = r.pass ? colors.pass : colors.reject;
  return (
    <View style={[ui.card, { borderColor: color, borderWidth: 2 }]}>
      <View style={[ui.between, { marginBottom: 12 }]}>
        <Text style={{ color, fontSize: 34, fontWeight: '900', letterSpacing: 1 }}>{r.pass ? 'PASS' : 'REJECT'}</Text>
        <View style={{ alignItems: 'flex-end' }}>
          <Text style={{ color, fontSize: 28, fontWeight: '800' }}>{r.score.toFixed(2)}</Text>
          <Text style={ui.muted}>score (reject &gt; 1.00)</Text>
        </View>
      </View>
      <HeatmapImage
        uri={r.uri}
        dmap={r.dmap}
        gh={BACKBONE.gh}
        gw={BACKBONE.gw}
        threshold={r.threshold}
        size={size}
        peak={r.peak}
        showPeak={!r.pass}
      />
      <View style={[ui.row, { flexWrap: 'wrap', marginTop: 8 }]}>
        <Stat label="max distance" value={r.raw.toFixed(3)} />
        <Stat label="threshold" value={r.threshold.toFixed(3)} />
        <Stat label="sensitivity" value={`×${r.sensitivity.toFixed(2)}`} />
        <Stat label="embed" value={`${r.embedMs} ms`} />
        <Stat label="k-NN" value={`${r.scoreMs} ms`} />
      </View>
      <Text style={[ui.muted, { marginTop: 8 }]}>
        Heat map: warm colours mark patches that look unlike anything in the good photos
        {r.pass ? '.' : '; the ring marks the most anomalous area.'}
      </Text>
    </View>
  );
}

export function InspectScreen({
  profile,
  sensitivity,
  onResult,
  onBack,
}: {
  profile: Profile;
  sensitivity: number;
  onResult: (r: Inspection) => void;
  onBack: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Inspection | null>(null);
  const { width } = useWindowDimensions();
  const size = Math.min(width - 60, 420);

  const inspect = async (uris: string[]) => {
    setBusy(true);
    setResult(null);
    try {
      await nextFrame();
      const img = await loadSquare(uris[0], BACKBONE.size);
      const t0 = Date.now();
      const feats = await embed(img.rgb);
      const t1 = Date.now();
      const res = await score(feats, profile, sensitivity, nextFrame);
      const t2 = Date.now();
      const r: Inspection = {
        id: t2,
        time: t2,
        uri: img.uri,
        dmap: res.dmap,
        raw: res.raw,
        score: res.score,
        pass: res.score <= 1.0,
        threshold: profile.tau * sensitivity,
        sensitivity,
        peak: res.peak,
        embedMs: t1 - t0,
        scoreMs: t2 - t1,
      };
      setResult(r);
      onResult(r);
    } catch (e) {
      Alert.alert('Inspection failed', String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <ScrollView style={ui.screen} contentContainerStyle={ui.scroll}>
      <View style={ui.between}>
        <Text style={ui.h1}>Inspect</Text>
        <Button title="Home" kind="secondary" onPress={onBack} disabled={busy} style={{ paddingVertical: 8 }} />
      </View>
      <Text style={[ui.muted, { marginTop: 6 }]}>
        Same mount and light as enrolment. Sensitivity ×{sensitivity.toFixed(2)} · threshold{' '}
        {(profile.tau * sensitivity).toFixed(3)}
      </Text>
      {result ? (
        <>
          <ResultCard r={result} size={size} />
          <Button title="Inspect next part" onPress={() => setResult(null)} style={{ marginTop: 12 }} />
        </>
      ) : (
        <CameraCapture onImages={inspect} busy={busy} multiple={false} captureLabel="Inspect part" />
      )}
    </ScrollView>
  );
}
