/**
 * Gauntlet Namespace API
 *
 * Provides a namespaced interface for gauntlet (attack/defend cycle) operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  GauntletPersona,
  GauntletComparison,
  GauntletRunRequest,
  GauntletRunResponse,
  GauntletRun,
} from '../types';

/**
 * Interface for the internal client methods used by GauntletAPI.
 */
interface GauntletClientInterface {
  runGauntlet(request: GauntletRunRequest): Promise<GauntletRunResponse>;
  runGauntletAndWait(
    request: GauntletRunRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<GauntletRun>;
  listGauntletPersonas(params?: { category?: string; enabled?: boolean }): Promise<{ personas: GauntletPersona[] }>;
  compareGauntlets(gauntletId1: string, gauntletId2: string): Promise<GauntletComparison>;
}

/**
 * Gauntlet API namespace.
 *
 * Provides methods for running and managing gauntlets (attack/defend cycles):
 * - Running gauntlets against claims
 * - Comparing multiple gauntlet runs
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Run a gauntlet
 * const run = await client.gauntlet.runAndWait({
 *   input: 'Our API is secure against SQL injection',
 *   personas: ['adversarial', 'compliance'],
 * });
 *
 * // List available personas
 * const { personas } = await client.gauntlet.listPersonas();
 * ```
 */
export class GauntletAPI {
  constructor(private client: GauntletClientInterface) {}

  /**
   * Run a gauntlet (non-blocking).
   * Returns immediately with a gauntlet ID.
   */
  async run(request: GauntletRunRequest): Promise<GauntletRunResponse> {
    return this.client.runGauntlet(request);
  }

  /**
   * Run a gauntlet and wait for completion.
   * Polls until the gauntlet finishes and returns the full run.
   */
  async runAndWait(
    request: GauntletRunRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<GauntletRun> {
    return this.client.runGauntletAndWait(request, options);
  }

  /**
   * List available gauntlet personas.
   */
  async listPersonas(params?: { category?: string; enabled?: boolean }): Promise<{ personas: GauntletPersona[] }> {
    return this.client.listGauntletPersonas(params);
  }

  /**
   * Compare two gauntlet runs.
   */
  async compare(gauntletId1: string, gauntletId2: string): Promise<GauntletComparison> {
    return this.client.compareGauntlets(gauntletId1, gauntletId2);
  }

  /**
   * Get gauntlet details.
   */
  async getGauntlet(gauntletId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/gauntlet/${gauntletId}`) as Promise<Record<string, unknown>>;
  }

  /**
   * Compare gauntlets.
   */
  async compareGauntlets(gauntletId: string, otherGauntletId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/gauntlet/${gauntletId}1/compare/${otherGauntletId}`) as Promise<Record<string, unknown>>;
  }
}
