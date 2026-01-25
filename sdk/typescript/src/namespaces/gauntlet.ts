/**
 * Gauntlet Namespace API
 *
 * Provides a namespaced interface for gauntlet (attack/defend cycle) operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  GauntletReceipt,
  GauntletResult,
  GauntletPersona,
  GauntletHeatmap,
  GauntletComparison,
  PaginationParams,
} from '../types';

/**
 * Request for running a gauntlet.
 */
export interface GauntletRunRequest {
  /** The claim or assertion to test */
  claim: string;
  /** Optional context for the claim */
  context?: string;
  /** Personas to use for attack/defend cycles */
  personas?: string[];
  /** Number of rounds */
  rounds?: number;
  /** Minimum confidence threshold */
  min_confidence?: number;
}

/**
 * Interface for the internal client methods used by GauntletAPI.
 */
interface GauntletClientInterface {
  runGauntlet(request: GauntletRunRequest): Promise<{ gauntlet_id: string }>;
  runGauntletAndWait(
    request: GauntletRunRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<GauntletReceipt>;
  listGauntletReceipts(params?: PaginationParams): Promise<{ receipts: GauntletReceipt[] }>;
  getGauntletReceipt(gauntletId: string): Promise<GauntletReceipt>;
  verifyGauntletReceipt(gauntletId: string): Promise<{ valid: boolean; errors?: string[] }>;
  exportGauntletReceipt(
    gauntletId: string,
    format: 'json' | 'markdown' | 'pdf'
  ): Promise<{ content: string; format: string }>;
  getGauntletPersonas(gauntletId: string): Promise<{ personas: GauntletPersona[] }>;
  getGauntletResults(gauntletId: string): Promise<{ results: GauntletResult[] }>;
  getGauntletHeatmap(gauntletId: string): Promise<GauntletHeatmap>;
  compareGauntlets(gauntletIds: string[]): Promise<GauntletComparison>;
}

/**
 * Gauntlet API namespace.
 *
 * Provides methods for running and managing gauntlets (attack/defend cycles):
 * - Running gauntlets against claims
 * - Retrieving gauntlet results and receipts
 * - Exporting gauntlet reports
 * - Comparing multiple gauntlet runs
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Run a gauntlet
 * const receipt = await client.gauntlet.runAndWait({
 *   claim: 'Our API is secure against SQL injection',
 *   rounds: 3,
 * });
 *
 * // List gauntlet receipts
 * const { receipts } = await client.gauntlet.list({ limit: 10 });
 *
 * // Export as PDF
 * const report = await client.gauntlet.export(gauntletId, 'pdf');
 * ```
 */
export class GauntletAPI {
  constructor(private client: GauntletClientInterface) {}

  /**
   * Run a gauntlet (non-blocking).
   * Returns immediately with a gauntlet ID.
   */
  async run(request: GauntletRunRequest): Promise<{ gauntlet_id: string }> {
    return this.client.runGauntlet(request);
  }

  /**
   * Run a gauntlet and wait for completion.
   * Polls until the gauntlet finishes and returns the full receipt.
   */
  async runAndWait(
    request: GauntletRunRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<GauntletReceipt> {
    return this.client.runGauntletAndWait(request, options);
  }

  /**
   * List gauntlet receipts with optional pagination.
   */
  async list(params?: PaginationParams): Promise<{ receipts: GauntletReceipt[] }> {
    return this.client.listGauntletReceipts(params);
  }

  /**
   * Get a gauntlet receipt by ID.
   */
  async get(gauntletId: string): Promise<GauntletReceipt> {
    return this.client.getGauntletReceipt(gauntletId);
  }

  /**
   * Verify a gauntlet receipt's cryptographic integrity.
   */
  async verify(gauntletId: string): Promise<{ valid: boolean; errors?: string[] }> {
    return this.client.verifyGauntletReceipt(gauntletId);
  }

  /**
   * Export a gauntlet receipt in various formats.
   */
  async export(
    gauntletId: string,
    format: 'json' | 'markdown' | 'pdf'
  ): Promise<{ content: string; format: string }> {
    return this.client.exportGauntletReceipt(gauntletId, format);
  }

  /**
   * Get personas used in a gauntlet.
   */
  async getPersonas(gauntletId: string): Promise<{ personas: GauntletPersona[] }> {
    return this.client.getGauntletPersonas(gauntletId);
  }

  /**
   * Get detailed results from a gauntlet.
   */
  async getResults(gauntletId: string): Promise<{ results: GauntletResult[] }> {
    return this.client.getGauntletResults(gauntletId);
  }

  /**
   * Get a heatmap visualization of gauntlet results.
   */
  async getHeatmap(gauntletId: string): Promise<GauntletHeatmap> {
    return this.client.getGauntletHeatmap(gauntletId);
  }

  /**
   * Compare multiple gauntlet runs.
   */
  async compare(gauntletIds: string[]): Promise<GauntletComparison> {
    return this.client.compareGauntlets(gauntletIds);
  }
}
