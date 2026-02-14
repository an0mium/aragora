/**
 * Receipts Namespace API
 *
 * Provides a namespaced interface for decision receipt management.
 * Critical for SME compliance, audit trails, and defensible decision-making.
 */

import type {
  DecisionReceipt,
  GauntletReceiptExport,
  PaginationParams,
} from '../types';

// Re-export types from ../types for convenience
export type { DecisionReceipt, GauntletReceiptExport } from '../types';

/**
 * Interface for the internal client methods used by ReceiptsAPI.
 */
interface ReceiptsClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
  listGauntletReceipts(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }>;
  getGauntletReceipt(receiptId: string): Promise<DecisionReceipt>;
  verifyGauntletReceipt(receiptId: string): Promise<{ valid: boolean; hash: string }>;
  exportGauntletReceipt(receiptId: string, format: 'json' | 'html' | 'markdown' | 'sarif'): Promise<GauntletReceiptExport>;
}

/**
 * Receipts API namespace.
 *
 * Provides methods for managing decision receipts:
 * - Gauntlet receipt listing and retrieval
 * - Verify receipt integrity (cryptographic hash)
 * - Export receipts in various formats
 * - Access findings and dissenting views
 *
 * Decision receipts provide audit-ready documentation of AI decisions,
 * essential for compliance, governance, and defensible decision-making.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List gauntlet receipts
 * const { receipts } = await client.receipts.listGauntlet();
 *
 * // Get a specific gauntlet receipt
 * const receipt = await client.receipts.getGauntlet('receipt-123');
 *
 * // Verify receipt integrity
 * const { valid, hash } = await client.receipts.verifyGauntlet('receipt-123');
 *
 * // Export as HTML for stakeholder review
 * const html = await client.receipts.exportGauntlet('receipt-123', 'html');
 * ```
 */
export class ReceiptsAPI {
  constructor(private client: ReceiptsClientInterface) {}

  // ===========================================================================
  // Generic Receipt Methods
  // ===========================================================================

  /**
   * List receipts with optional filtering.
   */
  async list(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }> {
    return this.client.listGauntletReceipts(params);
  }

  /**
   * Get a receipt by ID.
   */
  async get(receiptId: string): Promise<DecisionReceipt> {
    return this.client.getGauntletReceipt(receiptId);
  }

  /**
   * Verify a receipt's integrity.
   */
  async verify(receiptId: string): Promise<{ valid: boolean; hash: string }> {
    return this.client.verifyGauntletReceipt(receiptId);
  }

  /**
   * Verify a receipt with full signature validation.
   */
  async verifyFull(receiptId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/gauntlet/receipts/${encodeURIComponent(receiptId)}/verify-full`);
  }

  // ===========================================================================
  // Gauntlet Receipts
  // ===========================================================================

  /**
   * List gauntlet receipts with optional filtering.
   *
   * Gauntlet receipts are generated from attack/defend stress tests.
   */
  async listGauntlet(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }> {
    return this.client.listGauntletReceipts(params);
  }

  /**
   * Get a gauntlet receipt by ID.
   */
  async getGauntlet(receiptId: string): Promise<DecisionReceipt> {
    return this.client.getGauntletReceipt(receiptId);
  }

  /**
   * Verify a gauntlet receipt's integrity.
   */
  async verifyGauntlet(receiptId: string): Promise<{ valid: boolean; hash: string }> {
    return this.client.verifyGauntletReceipt(receiptId);
  }

  /**
   * Export a gauntlet receipt in various formats.
   *
   * @param receiptId - The receipt ID to export
   * @param format - Export format:
   *   - json: Machine-readable JSON
   *   - markdown: Human-readable Markdown
   *   - html: Styled HTML document
   *   - sarif: SARIF format for security tooling
   *
   * @example
   * ```typescript
   * // Export as HTML for stakeholder review
   * const html = await client.receipts.exportGauntlet('receipt-123', 'html');
   *
   * // Export as SARIF for security integration
   * const sarif = await client.receipts.exportGauntlet('receipt-123', 'sarif');
   * ```
   */
  async exportGauntlet(
    receiptId: string,
    format: 'json' | 'markdown' | 'html' | 'sarif'
  ): Promise<GauntletReceiptExport> {
    return this.client.exportGauntletReceipt(receiptId, format);
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  /**
   * Check if a receipt has any dissenting views.
   */
  hasDissent(receipt: DecisionReceipt): boolean {
    return (receipt.dissenting_agents?.length ?? 0) > 0;
  }

  /**
   * Get the consensus status from a receipt.
   */
  getConsensusStatus(receipt: DecisionReceipt): {
    reached: boolean;
    confidence: number;
    participatingAgents: number;
    dissentingAgents: number;
  } {
    return {
      reached: receipt.consensus_reached ?? false,
      confidence: receipt.confidence ?? 0,
      participatingAgents: receipt.participating_agents?.length ?? 0,
      dissentingAgents: receipt.dissenting_agents?.length ?? 0,
    };
  }
}
