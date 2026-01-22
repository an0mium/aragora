/**
 * Backwards-compatible re-export.
 * The SDK source is now in ./src/
 */
export * from './src/types';

// Legacy alias
export type { AragoraConfig as ApiConfig } from './src/types';
