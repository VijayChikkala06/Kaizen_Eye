import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useState } from 'react';
import { BackHandler, ScrollView, Text, useWindowDimensions, View } from 'react-native';

import { DETECTION } from './src/config';
import type { Profile } from './src/core/patchcore';
import { BACKBONE, loadBackbone } from './src/ml/backbone';
import { EnrolScreen } from './src/screens/EnrolScreen';
import { HomeScreen, type ModelState } from './src/screens/HomeScreen';
import { InspectScreen, ResultCard, type Inspection } from './src/screens/InspectScreen';
import { deleteProfile, loadProfile } from './src/storage/profileStore';
import { Button } from './src/ui/components';
import { ui } from './src/ui/theme';

type Screen = { name: 'home' } | { name: 'enrol' } | { name: 'inspect' } | { name: 'result'; r: Inspection };

const DEFAULT_SENSITIVITY = DETECTION.defaultSensitivity;
const MAX_HISTORY = 30;

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: 'home' });
  const [model, setModel] = useState<ModelState>({ status: 'loading' });
  const [profile, setProfile] = useState<Profile | null>(null);
  const [sensitivity, setSensitivity] = useState(DEFAULT_SENSITIVITY);
  const [history, setHistory] = useState<Inspection[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const { width } = useWindowDimensions();

  const initModel = useCallback(() => {
    setModel({ status: 'loading' });
    loadBackbone()
      .then(({ info }) => setModel({ status: 'ready', info }))
      .catch((e) => setModel({ status: 'error', error: String(e) }));
  }, []);

  useEffect(() => {
    initModel();
    loadProfile()
      .then((p) => {
        if (p && (p.gh !== BACKBONE.gh || p.gw !== BACKBONE.gw || p.dim !== BACKBONE.dim)) {
          // Enrolled with a different backbone: its memory bank is not comparable with the new features.
          setNotice('The detection model was upgraded, so the saved profile no longer applies. Please re-enrol.');
          setProfile(null);
        } else {
          setProfile(p);
        }
      })
      .catch((e) => console.warn('Could not load saved profile', e));
  }, [initModel]);

  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      if (screen.name === 'home') return false;
      setScreen({ name: 'home' });
      return true;
    });
    return () => sub.remove();
  }, [screen]);

  const home = () => setScreen({ name: 'home' });

  let body: React.ReactNode;
  switch (screen.name) {
    case 'enrol':
      body = (
        <EnrolScreen
          onCancel={home}
          onDone={(p) => {
            setNotice(null);
            setProfile(p);
            setHistory([]);
            home();
          }}
        />
      );
      break;
    case 'inspect':
      body = profile ? (
        <InspectScreen
          profile={profile}
          sensitivity={sensitivity}
          onBack={home}
          onResult={(r) => setHistory((h) => [r, ...h].slice(0, MAX_HISTORY))}
        />
      ) : null;
      break;
    case 'result':
      body = (
        <ScrollView style={ui.screen} contentContainerStyle={ui.scroll}>
          <View style={ui.between}>
            <Text style={ui.h1}>Result</Text>
            <Button title="Home" kind="secondary" onPress={home} style={{ paddingVertical: 8 }} />
          </View>
          <Text style={[ui.muted, { marginTop: 6 }]}>{new Date(screen.r.time).toLocaleString()}</Text>
          <ResultCard r={screen.r} size={Math.min(width - 60, 420)} />
        </ScrollView>
      );
      break;
    default:
      body = (
        <HomeScreen
          model={model}
          profile={profile}
          sensitivity={sensitivity}
          setSensitivity={setSensitivity}
          history={history}
          onEnrol={() => setScreen({ name: 'enrol' })}
          onInspect={() => setScreen({ name: 'inspect' })}
          onDeleteProfile={() => {
            deleteProfile();
            setProfile(null);
            setHistory([]);
          }}
          onOpenResult={(r) => setScreen({ name: 'result', r })}
          onRetryModel={initModel}
          notice={notice}
        />
      );
  }

  return (
    <View style={ui.screen}>
      {body}
      <StatusBar style="light" />
    </View>
  );
}
