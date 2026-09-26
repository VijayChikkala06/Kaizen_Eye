import { useRef, useState } from 'react';
import { ActivityIndicator, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';

import { colors, ui } from './theme';

type ButtonKind = 'primary' | 'secondary' | 'danger';

export function Button({
  title,
  onPress,
  disabled,
  kind = 'primary',
  style,
}: {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  kind?: ButtonKind;
  style?: object;
}) {
  const bg = disabled
    ? colors.disabled
    : kind === 'primary'
      ? colors.accent
      : kind === 'danger'
        ? colors.reject
        : colors.card;
  const fg = disabled ? colors.muted : kind === 'primary' ? colors.accentText : colors.text;
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: bg, opacity: pressed ? 0.8 : 1 },
        kind === 'secondary' && !disabled ? { borderColor: colors.cardBorder, borderWidth: 1 } : null,
        style,
      ]}
    >
      <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>
    </Pressable>
  );
}

export function ProgressBar({ fraction, label }: { fraction: number; label?: string }) {
  return (
    <View style={{ marginTop: 10 }}>
      {label ? <Text style={[ui.muted, { marginBottom: 6 }]}>{label}</Text> : null}
      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%` }]} />
      </View>
    </View>
  );
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={{ marginRight: 18, marginTop: 6 }}>
      <Text style={ui.muted}>{label}</Text>
      <Text style={[ui.body, { fontWeight: '700' }]}>{value}</Text>
    </View>
  );
}

/**
 * Square camera preview with a capture button and a gallery import button.
 * The square matches what the model sees: the centre square of the photo.
 */
export function CameraCapture({
  onImages,
  busy,
  multiple,
  captureLabel = 'Capture',
  ghostUri,
  disabled,
}: {
  onImages: (uris: string[]) => void;
  busy: boolean;
  multiple: boolean;
  captureLabel?: string;
  /** Reference photo drawn semi-transparently over the preview so every shot is framed the same way. */
  ghostUri?: string | null;
  disabled?: boolean;
}) {
  const [permission, requestPermission] = useCameraPermissions();
  const cam = useRef<CameraView>(null);
  const [ready, setReady] = useState(false);
  const [ghostOn, setGhostOn] = useState(true);
  const [pictureSize, setPictureSize] = useState<string | undefined>(undefined);

  const onCameraReady = async () => {
    setReady(true);
    try {
      const pick = choosePictureSize((await cam.current?.getAvailablePictureSizesAsync()) ?? []);
      if (pick) setPictureSize(pick);
    } catch {
      // keep the camera default
    }
  };

  const capture = async () => {
    if (!cam.current || busy) return;
    const photo = await cam.current.takePictureAsync({ quality: 1 });
    if (photo?.uri) onImages([photo.uri]);
  };

  const pick = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsMultipleSelection: multiple,
      selectionLimit: multiple ? 30 : 1,
      quality: 1,
    });
    if (!res.canceled && res.assets.length) onImages(res.assets.map((a) => a.uri));
  };

  let preview: React.ReactNode;
  if (!permission) {
    preview = <ActivityIndicator color={colors.accent} />;
  } else if (!permission.granted) {
    preview = (
      <View style={{ alignItems: 'center', padding: 16 }}>
        <Text style={[ui.body, { textAlign: 'center', marginBottom: 12 }]}>
          Camera access is needed to photograph parts. You can also import photos from the gallery.
        </Text>
        <Button title="Allow camera" onPress={requestPermission} />
      </View>
    );
  } else {
    preview = (
      <>
        <CameraView
          ref={cam}
          style={StyleSheet.absoluteFill}
          facing="back"
          autofocus="on"
          pictureSize={pictureSize}
          onCameraReady={onCameraReady}
        />
        {ghostUri && ghostOn ? (
          <View pointerEvents="none" style={StyleSheet.absoluteFill}>
            <Image source={{ uri: ghostUri }} style={[StyleSheet.absoluteFill, { opacity: 0.38 }]} />
          </View>
        ) : null}
        <View pointerEvents="none" style={styles.guide} />
        {ghostUri ? (
          <Pressable onPress={() => setGhostOn((g) => !g)} style={styles.ghostToggle}>
            <Text style={{ color: '#fff', fontWeight: '700', fontSize: 12 }}>
              {ghostOn ? 'Align guide ON' : 'Align guide OFF'}
            </Text>
          </Pressable>
        ) : null}
        {busy ? (
          <View style={styles.busyOverlay}>
            <ActivityIndicator size="large" color={colors.accent} />
          </View>
        ) : null}
      </>
    );
  }

  return (
    <View>
      <View style={styles.cameraBox}>{preview}</View>
      <View style={[ui.row, { marginTop: 10 }]}>
        <Button
          title={busy ? 'Working…' : captureLabel}
          onPress={capture}
          disabled={busy || disabled || !permission?.granted || !ready}
          style={{ flex: 1, marginRight: 8 }}
        />
        <Button title={multiple ? 'Import photos' : 'From gallery'} kind="secondary" onPress={pick} disabled={busy} style={{ flex: 1 }} />
      </View>
    </View>
  );
}

/**
 * Smallest 4:3 capture size whose short side is >= 1080 px. The model only needs 320 px and the result screen
 * 1024 px, so a ~1.6 MP photo is plenty and far faster to decode than 12 MP. 4:3 keeps the full sensor field of
 * view, so the centre square matches the preview and the alignment guide.
 */
function choosePictureSize(sizes: string[]): string | undefined {
  const parsed = sizes
    .map((s) => {
      const m = /^(\d+)\s*x\s*(\d+)$/i.exec(s.trim());
      return m ? { s, w: Number(m[1]), h: Number(m[2]) } : null;
    })
    .filter((p): p is { s: string; w: number; h: number } => p !== null)
    .filter((p) => Math.abs(Math.max(p.w, p.h) / Math.min(p.w, p.h) - 4 / 3) < 0.02 && Math.min(p.w, p.h) >= 1080)
    .sort((a, b) => a.w * a.h - b.w * b.h);
  return parsed[0]?.s;
}

/** Heat map like tools/lab/viz.py: 1.0 of the colour scale == 2 x threshold; only warm areas show. */
function jet(x: number): [number, number, number] {
  const c = (v: number) => Math.max(0, Math.min(1, v));
  return [c(1.5 - Math.abs(4 * x - 3)), c(1.5 - Math.abs(4 * x - 2)), c(1.5 - Math.abs(4 * x - 1))];
}

export function HeatmapImage({
  uri,
  dmap,
  gh,
  gw,
  threshold,
  size,
  peak,
  showPeak,
}: {
  uri: string;
  dmap: Float32Array;
  gh: number;
  gw: number;
  threshold: number;
  size: number;
  peak?: { row: number; col: number };
  showPeak?: boolean;
}) {
  const cw = size / gw;
  const chh = size / gh;
  const cells: React.ReactNode[] = [];
  for (let r = 0; r < gh; r++) {
    for (let c = 0; c < gw; c++) {
      // Colour only patches near / above the reject threshold (rel = distance / threshold).
      const rel = dmap[r * gw + c] / threshold;
      const a = Math.max(0, Math.min(1, rel / 2));
      const alpha = Math.max(0, Math.min(0.7, (rel - 0.75) * 1.4));
      if (alpha < 0.02) continue;
      const [R, G, B] = jet(a);
      cells.push(
        <View
          key={r * gw + c}
          style={{
            position: 'absolute',
            left: c * cw,
            top: r * chh,
            width: cw + 0.5,
            height: chh + 0.5,
            backgroundColor: `rgba(${Math.round(R * 255)},${Math.round(G * 255)},${Math.round(B * 255)},${alpha.toFixed(3)})`,
          }}
        />,
      );
    }
  }
  return (
    <View style={{ width: size, height: size, borderRadius: 10, overflow: 'hidden', alignSelf: 'center' }}>
      <Image source={{ uri }} style={{ width: size, height: size }} />
      {cells}
      {showPeak && peak ? (
        <View
          pointerEvents="none"
          style={{
            position: 'absolute',
            left: (peak.col + 0.5) * cw - 18,
            top: (peak.row + 0.5) * chh - 18,
            width: 36,
            height: 36,
            borderRadius: 18,
            borderWidth: 2,
            borderColor: '#fff',
          }}
        />
      ) : null}
    </View>
  );
}

export function Thumb({ uri, size = 56 }: { uri: string; size?: number }) {
  return <Image source={{ uri }} style={{ width: size, height: size, borderRadius: 8, marginRight: 6 }} />;
}

const styles = StyleSheet.create({
  button: { paddingVertical: 13, paddingHorizontal: 16, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  buttonText: { fontSize: 16, fontWeight: '700' },
  progressTrack: { height: 10, borderRadius: 5, backgroundColor: colors.disabled, overflow: 'hidden' },
  progressFill: { height: 10, backgroundColor: colors.accent },
  cameraBox: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 12,
  },
  guide: {
    position: 'absolute',
    // The scored area: the outer 10% of the photo is ignored by the frame score (config.DETECTION.borderFraction).
    left: '10%',
    top: '10%',
    right: '10%',
    bottom: '10%',
    borderWidth: 2,
    borderColor: 'rgba(34,211,238,0.8)',
    borderRadius: 10,
    borderStyle: 'dashed',
  },
  ghostToggle: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 14,
  },
  busyOverlay: {
    position: 'absolute',
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.45)',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
