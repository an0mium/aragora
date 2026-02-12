/**
 * Approvals Namespace API
 *
 * Provides endpoints for the unified approvals inbox including
 * listing pending approval requests across subsystems.
 */

import type { AragoraClient } from '../client';

/** Approval request from a subsystem */
export interface ApprovalRequest {
  id: string;
  source: string;
  type: string;
  title: string;
  description?: string;
  requested_at: string;
  requested_by?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Approvals namespace for human-in-the-loop workflow.
 *
 * @example
 * ```typescript
 * const pending = await client.approvals.listPending();
 * console.log(`${pending.count} approvals pending`);
 * ```
 */
export class ApprovalsAPI {
  constructor(private client: AragoraClient) {}

  /** List pending approvals. */
  async listPending(options?: {
    sources?: string[];
    limit?: number;
  }): Promise<{ approvals: ApprovalRequest[]; count: number }> {
    const params: Record<string, unknown> = { status: 'pending' };
    if (options?.sources) {
      params.sources = options.sources.join(',');
    }
    if (options?.limit) {
      params.limit = options.limit;
    }
    return this.client.request<{ approvals: ApprovalRequest[]; count: number }>(
      'GET',
      '/api/v1/approvals/pending',
      { params }
    );
  }

  /** List all approvals. */
  async list(options?: {
    sources?: string[];
    limit?: number;
  }): Promise<{ approvals: ApprovalRequest[]; count: number }> {
    const params: Record<string, unknown> = {};
    if (options?.sources) {
      params.sources = options.sources.join(',');
    }
    if (options?.limit) {
      params.limit = options.limit;
    }
    return this.client.request<{ approvals: ApprovalRequest[]; count: number }>(
      'GET',
      '/api/v1/approvals',
      { params }
    );
  }
}
