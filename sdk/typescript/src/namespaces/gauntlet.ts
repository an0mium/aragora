/**
 * Gauntlet Namespace API
 *
 * Provides a namespaced interface for gauntlet (attack/defend cycle) operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  DecisionReceipt,
  GauntletResult,
  GauntletPersona,
  GauntletHeatmapExtended,
  GauntletComparison,
  GauntletRunRequest,
  GauntletRunResponse,
  GauntletRun,
  GauntletReceiptExport,
  PaginationParams,
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
  listGauntletReceipts(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }>;
  getGauntletReceipt(receiptId: string): Promise<DecisionReceipt>;
  verifyGauntletReceipt(receiptId: string): Promise<{ valid: boolean; hash: string }>;
  exportGauntletReceipt(
    receiptId: string,
    format: 'json' | 'html' | 'markdown' | 'sarif'
  ): Promise<GauntletReceiptExport>;
  listGauntletPersonas(params?: { category?: string; enabled?: boolean }): Promise<{ personas: GauntletPersona[] }>;
  listGauntletResults(params?: { gauntlet_id?: string; status?: string } & PaginationParams): Promise<{ results: GauntletResult[] }>;
  getGauntletHeatmap(gauntletId: string, format?: 'json' | 'svg'): Promise<GauntletHeatmapExtended>;
  compareGauntlets(gauntletId1: string, gauntletId2: string): Promise<GauntletComparison>;
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
 * const run = await client.gauntlet.runAndWait({
 *   input: 'Our API is secure against SQL injection',
 *   personas: ['adversarial', 'compliance'],
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
   * List gauntlet receipts with optional pagination.
   */
  async list(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }> {
    return this.client.listGauntletReceipts(params);
  }

  /**
   * Get a gauntlet receipt by ID.
   */
  async get(receiptId: string): Promise<DecisionReceipt> {
    return this.client.getGauntletReceipt(receiptId);
  }

  /**
   * Verify a gauntlet receipt's cryptographic integrity.
   */
  async verify(receiptId: string): Promise<{ valid: boolean; hash: string }> {
    return this.client.verifyGauntletReceipt(receiptId);
  }

  /**
   * Export a gauntlet receipt in various formats.
   */
  async export(
    receiptId: string,
    format: 'json' | 'html' | 'markdown' | 'sarif'
  ): Promise<GauntletReceiptExport> {
    return this.client.exportGauntletReceipt(receiptId, format);
  }

  /**
   * List available gauntlet personas.
   */
  async listPersonas(params?: { category?: string; enabled?: boolean }): Promise<{ personas: GauntletPersona[] }> {
    return this.client.listGauntletPersonas(params);
  }

  /**
   * List gauntlet results with optional filtering.
   */
  async listResults(params?: { gauntlet_id?: string; status?: string } & PaginationParams): Promise<{ results: GauntletResult[] }> {
    return this.client.listGauntletResults(params);
  }

  /**
   * Get a heatmap visualization of gauntlet results.
   */
  async getHeatmap(gauntletId: string, format?: 'json' | 'svg'): Promise<GauntletHeatmapExtended> {
    return this.client.getGauntletHeatmap(gauntletId, format);
  }

  /**
   * Compare two gauntlet runs.
   */
  async compare(gauntletId1: string, gauntletId2: string): Promise<GauntletComparison> {
    return this.client.compareGauntlets(gauntletId1, gauntletId2);
  }
}
