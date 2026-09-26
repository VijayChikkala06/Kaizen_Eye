/** Persist the enrolled profile (memory bank + threshold) in the app's document directory. */
import { File, Paths } from 'expo-file-system';

import type { Profile } from '../core/patchcore';

const metaFile = () => new File(Paths.document, 'kaizen_profile.json');
const bankFile = () => new File(Paths.document, 'kaizen_profile_bank.bin');

type Meta = Omit<Profile, 'bank' | 'bankFrame' | 'globals'> & { bankFrame: number[]; bankLength: number; globals?: number[] };

export function saveProfile(p: Profile): void {
  const { bank, bankFrame, globals, ...rest } = p;
  const meta: Meta = {
    ...rest,
    bankFrame: Array.from(bankFrame),
    bankLength: bank.length,
    globals: globals ? Array.from(globals) : undefined,
  };
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
  const { bankFrame, bankLength: _len, globals, ...rest } = meta;
  return {
    ...rest,
    bank,
    bankFrame: Int32Array.from(bankFrame),
    globals: globals ? Float32Array.from(globals) : undefined,
  };
}

const REF_PREFIX = 'kaizen_reference_';

/**
 * Keep a copy of one enrolment photo as the alignment "ghost" shown over the camera preview. A new file name per
 * enrolment avoids stale image caching; older reference files are removed.
 */
export async function saveReferenceImage(uri: string): Promise<string> {
  deleteReferenceImages();
  const dest = new File(Paths.document, `${REF_PREFIX}${Date.now()}.jpg`);
  await new File(uri).copy(dest);
  return dest.uri;
}

function deleteReferenceImages(): void {
  for (const item of Paths.document.list()) {
    if (item instanceof File && item.name.startsWith(REF_PREFIX)) item.delete();
  }
}

export function deleteProfile(): void {
  for (const f of [metaFile(), bankFile()]) if (f.exists) f.delete();
  deleteReferenceImages();
}
