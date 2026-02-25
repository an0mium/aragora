/**
 * Audit Trail Namespace API
 *
 * Provides endpoints for audit trail access and verification
 * including integrity checksums and multi-format export.
 */

import type { AragoraClient } from '../client';

/** Audit trail summary */
export interface AuditTrailSummary {
  trail_id: string;
  gauntlet_id?: string;
  created_at: string;
  verdict?: string;
  confidence?: number;
  total_findings?: number;
  duration_seconds?: number;
  checksum?: string;
}

/** Audit trail verification result */
export interface AuditTrailVerification {
  trail_id: string;
  valid: boolean;
  stored_checksum: string;
  computed_checksum: string;
  match: boolean;
  error?: string;
}

/** Audit trail export format */
export type AuditTrailExportFormat = 'json' | 'csv' | 'md';

/**
 * Audit Trail namespace for compliance and integrity verification.
 *
 * @example
 * ```typescript
 * const trails = await client.auditTrail.list();
 * const verification = await client.auditTrail.verify(trails.trails[0].trail_id);
 * console.log(`Trail valid: ${verification.valid}`);
 * ```
 */
export class AuditTrailAPI {
  constructor(private client: AragoraClient) {}

  /** List audit trails with pagination. */
  async list(options?: {
    verdict?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ trails: AuditTrailSummary[]; total: number }> {
    return this.client.request<{ trails: AuditTrailSummary[]; total: number }>(
      'GET',
      '/api/v1/audit-trails',
      { params: options }
    );
  }

  /** Get a specific audit trail by ID. */
  async get(trailId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/audit-trails/${encodeURIComponent(trailId)}`
    );
  }

  /** Export an audit trail in the specified format. */
  async export(
    trailId: string,
    format: AuditTrailExportFormat = 'json'
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/audit-trails/${encodeURIComponent(trailId)}/export`,
      { params: { format } }
    );
  }

  /** Verify audit trail integrity checksum. */
  async verify(trailId: string): Promise<AuditTrailVerification> {
    return this.client.request<AuditTrailVerification>(
      'POST',
      `/api/audit-trails/${encodeURIComponent(trailId)}/verify`
    );
  }

  /** List v1 decision receipts. */
  async listReceipts(options?: {
    verdict?: string;
    risk_level?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ receipts: Record<string, unknown>[]; total: number }> {
    return this.client.request<{ receipts: Record<string, unknown>[]; total: number }>(
      'GET',
      '/api/v1/receipts',
      { params: options }
    );
  }

  /** Get a v1 decision receipt by ID. */
  async getReceipt(receiptId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/receipts/${encodeURIComponent(receiptId)}`
    );
  }

  /** Verify a v1 receipt's integrity. */
  async verifyReceipt(receiptId: string): Promise<AuditTrailVerification> {
    return this.client.request<AuditTrailVerification>(
      'POST',
      `/api/v1/receipts/${encodeURIComponent(receiptId)}/verify`
    );
  }
}
