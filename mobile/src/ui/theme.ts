import { Platform, StatusBar, StyleSheet } from 'react-native';

export const colors = {
  bg: '#0B1220',
  card: '#141C2E',
  cardBorder: '#223049',
  text: '#F1F5F9',
  muted: '#94A3B8',
  accent: '#22D3EE',
  accentText: '#04222A',
  pass: '#22C55E',
  reject: '#EF4444',
  warn: '#F59E0B',
  disabled: '#334155',
};

export const TOP_INSET = Platform.OS === 'android' ? (StatusBar.currentHeight ?? 24) + 8 : 52;
export const BOTTOM_INSET = 40;

export const ui = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  scroll: { paddingTop: TOP_INSET, paddingHorizontal: 16, paddingBottom: BOTTOM_INSET },
  h1: { color: colors.text, fontSize: 28, fontWeight: '800', letterSpacing: 0.5 },
  h2: { color: colors.text, fontSize: 18, fontWeight: '700' },
  body: { color: colors.text, fontSize: 15 },
  muted: { color: colors.muted, fontSize: 13 },
  card: {
    backgroundColor: colors.card,
    borderColor: colors.cardBorder,
    borderWidth: 1,
    borderRadius: 14,
    padding: 14,
    marginTop: 12,
  },
  row: { flexDirection: 'row', alignItems: 'center' },
  between: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
});
