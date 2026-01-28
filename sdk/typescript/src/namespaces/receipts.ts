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
 * Receipt statistics response.
 */
export interface ReceiptStats {
  total_count: number;
  by_verdict: Record<string, number>;
  by_month: Record<string, number>;
  average_confidence: number;
  consensus_rate: number;
}

/**
 * Batch verification result for multiple receipts.
 */
export interface BatchVerificationResult {
  results: Array<{
    receipt_id: string;
    valid: boolean;
    hash: string;
    error?: string;
  }>;
  total_verified: number;
  total_valid: number;
  total_invalid: number;
}

/**
 * Receipt filters for querying and stats.
 */
export interface ReceiptFilters {
  verdict?: string;
  from_date?: string;
  to_date?: string;
  decision_type?: string;
}

/**
 * Interface for the internal client methods used by ReceiptsAPI.
 */
interface ReceiptsClientInterface {
  listDecisionReceipts(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }>;
  getDecisionReceipt(receiptId: string): Promise<DecisionReceipt>;
  verifyDecisionReceipt(receiptId: string): Promise<{ valid: boolean; hash: string; verified_at: string }>;
  verifyDecisionReceiptsBatch(receiptIds: string[]): Promise<BatchVerificationResult>;
  getReceiptStats(filters?: ReceiptFilters): Promise<ReceiptStats>;
  exportReceiptPdf(receiptId: string): Promise<Blob>;
  exportReceiptCsv(receiptIds: string[]): Promise<Blob>;
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

  /**
   * Verify multiple receipts in a single batch operation.
   *
   * More efficient than calling verify() for each receipt individually.
   * Returns detailed results for each receipt, including any errors.
   *
   * @example
   * ```typescript
   * const result = await client.receipts.verifyBatch(['receipt-1', 'receipt-2', 'receipt-3']);
   * console.log(`${result.total_valid}/${result.total_verified} receipts valid`);
   *
   * // Check individual results
   * for (const r of result.results) {
   *   if (!r.valid) {
   *     console.error(`Receipt ${r.receipt_id} invalid: ${r.error}`);
   *   }
   * }
   * ```
   */
  async verifyBatch(receiptIds: string[]): Promise<BatchVerificationResult> {
    return this.client.verifyDecisionReceiptsBatch(receiptIds);
  }

  /**
   * Get statistics about decision receipts.
   *
   * Provides aggregate metrics useful for compliance dashboards,
   * audits, and operational monitoring.
   *
   * @example
   * ```typescript
   * const stats = await client.receipts.getStats({ from_date: '2024-01-01' });
   * console.log(`Total decisions: ${stats.total_count}`);
   * console.log(`Consensus rate: ${(stats.consensus_rate * 100).toFixed(1)}%`);
   * console.log(`By verdict:`, stats.by_verdict);
   * ```
   */
  async getStats(filters?: ReceiptFilters): Promise<ReceiptStats> {
    return this.client.getReceiptStats(filters);
  }

  /**
   * Export a receipt as a PDF document.
   *
   * Returns a Blob containing the PDF, suitable for download
   * or storage. Ideal for compliance documentation and
   * stakeholder reports.
   *
   * @example
   * ```typescript
   * const pdfBlob = await client.receipts.exportPdf('receipt-123');
   *
   * // In a browser, trigger download
   * const url = URL.createObjectURL(pdfBlob);
   * const a = document.createElement('a');
   * a.href = url;
   * a.download = 'decision-receipt.pdf';
   * a.click();
   * ```
   */
  async exportPdf(receiptId: string): Promise<Blob> {
    return this.client.exportReceiptPdf(receiptId);
  }

  /**
   * Export multiple receipts as a CSV file.
   *
   * Returns a Blob containing CSV data with one row per receipt.
   * Useful for bulk analysis, spreadsheet import, and reporting.
   *
   * @example
   * ```typescript
   * const csvBlob = await client.receipts.exportCsv(['receipt-1', 'receipt-2']);
   *
   * // Parse the CSV
   * const text = await csvBlob.text();
   * console.log(text); // "id,verdict,confidence,created_at\n..."
   * ```
   */
  async exportCsv(receiptIds: string[]): Promise<Blob> {
    return this.client.exportReceiptCsv(receiptIds);
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
