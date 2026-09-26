// Metro bundles .tflite files as assets (see metro.config.js); require() returns an asset id.
declare module '*.tflite' {
  const asset: number;
  export default asset;
}

// jpeg-js ships no types for its decoder entry point (importing it alone keeps the encoder out of the bundle).
declare module 'jpeg-js/lib/decoder' {
  export interface DecodedJpeg {
    width: number;
    height: number;
    data: Uint8Array;
  }
  export default function decode(
    data: Uint8Array | ArrayBuffer,
    opts?: { useTArray?: boolean; formatAsRGBA?: boolean; maxResolutionInMP?: number; maxMemoryUsageInMB?: number },
  ): DecodedJpeg;
}
