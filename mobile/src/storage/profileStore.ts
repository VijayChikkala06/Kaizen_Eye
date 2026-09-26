/** Persist the enrolled profile (memory bank + threshold) in the app's document directory. */
import { File, Paths } from 'expo-file-system';

import type { Profile } from '../core/patchcore';

const metaFile = () => new File(Paths.document, 'kaizen_profile.json');
const bankFile = () => new File(Paths.document, 'kaizen_profile_bank.bin');

type Meta = Omit<Profile, 'bank' | 'bankFrame'> & { bankFrame: number[]; bankLength: number };

export function saveProfile(p: Profile): void {
  const { bank, bankFrame, ...rest } = p;
  const meta: Meta = { ...rest, bankFrame: Array.from(bankFrame), bankLength: bank.length };
  const bf = bankFile();
  if (bf.exists) bf.delete();
  bf.create();
  bf.write(new Uint8Array(bank.buffer, bank.byteOffset, bank.byteLength));
  const mf = metaFile();
  if (mf.exists) mf.delete();
  mf.create();
  mf.write(JSON.stringify(meta));
}

export async function loadProfile(): Promise<Profile | null> {
  const mf = metaFile();
  const bf = bankFile();
  if (!mf.exists || !bf.exists) return null;
  const meta = JSON.parse(await mf.text()) as Meta;
  const bytes = await bf.bytes();
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const bank = new Float32Array(copy.buffer);
  if (bank.length !== meta.bankLength) throw new Error('Saved profile is corrupt (bank size mismatch)');
  const { bankFrame, bankLength: _len, ...rest } = meta;
  return { ...rest, bank, bankFrame: Int32Array.from(bankFrame) };
}

export function deleteProfile(): void {
  for (const f of [metaFile(), bankFile()]) if (f.exists) f.delete();
}
