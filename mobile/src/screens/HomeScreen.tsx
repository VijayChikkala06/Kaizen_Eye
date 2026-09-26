import { Alert, Image, Pressable, ScrollView, Text, View } from 'react-native';

import type { Profile } from '../core/patchcore';
import { BACKBONE, type BackboneInfo } from '../ml/backbone';
import { Button, Stat } from '../ui/components';
import { colors, ui } from '../ui/theme';
import type { Inspection } from './InspectScreen';

export type ModelState = { status: 'loading' } | { status: 'ready'; info: BackboneInfo } | { status: 'error'; error: string };

export const SENSITIVITY_MIN = 0.8;
export const SENSITIVITY_MAX = 2.0;

export function HomeScreen({
  model,
  profile,
  sensitivity,
  setSensitivity,
  history,
  onEnrol,
  onInspect,
  onDeleteProfile,
  onOpenResult,
  onRetryModel,
}: {
  model: ModelState;
  profile: Profile | null;
  sensitivity: number;
  setSensitivity: (s: number) => void;
  history: Inspection[];
  onEnrol: () => void;
  onInspect: () => void;
  onDeleteProfile: () => void;
  onOpenResult: (r: Inspection) => void;
  onRetryModel: () => void;
}) {
  const ready = model.status === 'ready';
  const step = (d: number) =>
    setSensitivity(Math.round(Math.min(SENSITIVITY_MAX, Math.max(SENSITIVITY_MIN, sensitivity + d)) * 100) / 100);
  const passes = history.filter((h) => h.pass).length;

  return (
    <ScrollView style={ui.screen} contentContainerStyle={ui.scroll}>
      <Text style={ui.h1}>Kaizen Eye</Text>
      <Text style={ui.muted}>No-training visual inspection · learns “normal” from ~20 good photos</Text>

      <View style={ui.card}>
        <View style={ui.between}>
          <Text style={ui.h2}>Model</Text>
          <Text
            style={{
              color: model.status === 'ready' ? colors.pass : model.status === 'error' ? colors.reject : colors.warn,
              fontWeight: '700',
            }}
          >
            {model.status === 'ready' ? 'READY' : model.status === 'error' ? 'ERROR' : 'LOADING…'}
          </Text>
        </View>
        <Text style={[ui.muted, { marginTop: 4 }]}>
          {BACKBONE.name} · {BACKBONE.pretrained ? 'ImageNet-pretrained' : 'random weights'} · on-device LiteRT
        </Text>
        {model.status === 'ready' ? (
          <View style={[ui.row, { flexWrap: 'wrap' }]}>
            <Stat label="input" value={`[${model.info.inputShape.join(',')}]`} />
            <Stat label="patches" value={`${BACKBONE.gh}×${BACKBONE.gw}×${BACKBONE.dim}`} />
            <Stat label="load" value={`${model.info.loadMs} ms`} />
          </View>
        ) : null}
        {model.status === 'error' ? (
          <>
            <Text style={[ui.body, { color: colors.reject, marginTop: 6 }]}>{model.error}</Text>
            <Button title="Retry" kind="secondary" onPress={onRetryModel} style={{ marginTop: 8 }} />
          </>
        ) : null}
      </View>

      <View style={ui.card}>
        <View style={ui.between}>
          <Text style={ui.h2}>Good-part profile</Text>
          <Text style={{ color: profile ? colors.pass : colors.muted, fontWeight: '700' }}>
            {profile ? 'ENROLLED' : 'NOT SET'}
          </Text>
        </View>
        {profile ? (
          <View style={[ui.row, { flexWrap: 'wrap' }]}>
            <Stat label="photos" value={String(profile.nFrames)} />
            <Stat label="memory bank" value={`${profile.bankFrame.length} patches`} />
            <Stat label="tau" value={profile.tau.toFixed(3)} />
            <Stat label="LOO min / max" value={`${Math.min(...profile.looScores).toFixed(2)} / ${Math.max(...profile.looScores).toFixed(2)}`} />
          </View>
        ) : (
          <Text style={[ui.muted, { marginTop: 6 }]}>Enrol first: photograph ~20 good parts.</Text>
        )}
        <View style={[ui.row, { marginTop: 12 }]}>
          <Button
            title={profile ? 'Re-enrol' : 'Enrol good parts'}
            kind={profile ? 'secondary' : 'primary'}
            onPress={onEnrol}
            disabled={!ready}
            style={{ flex: 1, marginRight: profile ? 8 : 0 }}
          />
          {profile ? (
            <Button
              title="Delete"
              kind="danger"
              onPress={() =>
                Alert.alert('Delete profile?', 'The enrolled good-part profile will be removed.', [
                  { text: 'Cancel', style: 'cancel' },
                  { text: 'Delete', style: 'destructive', onPress: onDeleteProfile },
                ])
              }
              style={{ flex: 0.6 }}
            />
          ) : null}
        </View>
      </View>

      <View style={ui.card}>
        <Text style={ui.h2}>Sensitivity (margin)</Text>
        <Text style={[ui.muted, { marginTop: 4 }]}>
          Reject when score &gt; 1.0, where score = max patch distance ÷ (tau × margin). Higher margin = fewer false rejects;
          lower = catches smaller defects. Start near 1.10.
        </Text>
        <View style={[ui.between, { marginTop: 10 }]}>
          <Button title="−" kind="secondary" onPress={() => step(-0.05)} style={{ width: 64 }} />
          <Text style={{ color: colors.accent, fontSize: 30, fontWeight: '800' }}>×{sensitivity.toFixed(2)}</Text>
          <Button title="+" kind="secondary" onPress={() => step(0.05)} style={{ width: 64 }} />
        </View>
      </View>

      <Button title="Inspect a part" onPress={onInspect} disabled={!ready || !profile} style={{ marginTop: 16, paddingVertical: 18 }} />

      {history.length ? (
        <View style={ui.card}>
          <View style={ui.between}>
            <Text style={ui.h2}>This session</Text>
            <Text style={ui.muted}>
              {passes} pass · {history.length - passes} reject
            </Text>
          </View>
          {history.map((h) => (
            <Pressable key={h.id} onPress={() => onOpenResult(h)} style={[ui.between, { marginTop: 10 }]}>
              <View style={ui.row}>
                <Image source={{ uri: h.uri }} style={{ width: 44, height: 44, borderRadius: 6, marginRight: 10 }} />
                <View>
                  <Text style={{ color: h.pass ? colors.pass : colors.reject, fontWeight: '800' }}>
                    {h.pass ? 'PASS' : 'REJECT'}
                  </Text>
                  <Text style={ui.muted}>{new Date(h.time).toLocaleTimeString()}</Text>
                </View>
              </View>
              <Text style={[ui.body, { fontWeight: '700' }]}>{h.score.toFixed(2)}</Text>
            </Pressable>
          ))}
        </View>
      ) : null}
    </ScrollView>
  );
}
