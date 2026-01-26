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

/**
 * Interface for the internal client methods used by ReceiptsAPI.
 */
interface ReceiptsClientInterface {
  listDecisionReceipts(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }>;
  getDecisionReceipt(receiptId: string): Promise<DecisionReceipt>;
  verifyDecisionReceipt(receiptId: string): Promise<{ valid: boolean; hash: string; verified_at: string }>;
  listGauntletReceipts(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }>;
  getGauntletReceipt(receiptId: string): Promise<DecisionReceipt>;
  verifyGauntletReceipt(receiptId: string): Promise<{ valid: boolean; hash: string }>;
  exportGauntletReceipt(receiptId: string, format: 'json' | 'html' | 'markdown' | 'sarif'): Promise<GauntletReceiptExport>;
}

/**
 * Receipts API namespace.
 *
 * Provides methods for managing decision receipts:
 * - List and retrieve receipts
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
 * // List all receipts
 * const { receipts } = await client.receipts.list();
 *
 * // Get a specific receipt
 * const receipt = await client.receipts.get('receipt-123');
 *
 * // Verify receipt integrity
 * const { valid, hash } = await client.receipts.verify('receipt-123');
 * if (valid) {
 *   console.log('Receipt integrity verified:', hash);
 * }
 *
 * // Export as PDF for compliance documentation
 * const pdf = await client.receipts.exportGauntlet('receipt-123', 'html');
 *
 * // Filter by verdict
 * const rejected = await client.receipts.list({ verdict: 'REJECTED' });
 * ```
 */
export class ReceiptsAPI {
  constructor(private client: ReceiptsClientInterface) {}

  // ===========================================================================
  // Decision Receipts
  // ===========================================================================

  /**
   * List decision receipts with optional filtering.
   *
   * @param params.verdict - Filter by verdict (APPROVED, REJECTED, etc.)
   * @param params.limit - Maximum number of results
   * @param params.offset - Offset for pagination
   */
  async list(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }> {
    return this.client.listDecisionReceipts(params);
  }

  /**
   * Get a decision receipt by ID.
   */
  async get(receiptId: string): Promise<DecisionReceipt> {
    return this.client.getDecisionReceipt(receiptId);
  }

  /**
   * Verify a receipt's cryptographic integrity.
   *
   * Checks that the receipt has not been tampered with since creation.
   * The hash is computed over all decision-relevant fields.
   *
   * @example
   * ```typescript
   * const { valid, hash, verified_at } = await client.receipts.verify('receipt-123');
   * if (!valid) {
   *   console.error('Receipt integrity check failed!');
   * }
   * ```
   */
  async verify(receiptId: string): Promise<{ valid: boolean; hash: string; verified_at: string }> {
    return this.client.verifyDecisionReceipt(receiptId);
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

// Re-export types for convenience
export type { DecisionReceipt, GauntletReceiptExport };
