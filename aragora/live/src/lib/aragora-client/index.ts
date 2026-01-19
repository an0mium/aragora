/**
 * Aragora SDK Client
 *
 * Modular structure for the Aragora client library.
 * Re-exports from the main client file for backward compatibility.
 *
 * Usage:
 * ```typescript
 * import { getClient, AragoraClient, AragoraError } from '@/lib/aragora-client';
 *
 * const client = getClient(token);
 * const debates = await client.debates.list();
 * ```
 *
 * Module Structure:
 * - types.ts - Type definitions
 * - http.ts - HTTP client and error handling
 * - index.ts - Main exports (this file)
 *
 * Future: API classes will be extracted to separate modules:
 * - debates.ts - Debates API
 * - agents.ts - Agents API
 * - analytics.ts - Analytics API
 * - etc.
 */

// Re-export everything from the main client file for backward compatibility
export * from '../aragora-client';

// The modular files (types.ts, http.ts) provide the foundation for future
// gradual migration. They are not re-exported here to avoid duplicate exports.
// New code can import directly from:
// - '@/lib/aragora-client/types' for type definitions
// - '@/lib/aragora-client/http' for HttpClient class
