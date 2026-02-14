/**
 * Reconciliation Namespace API
 *
 * Provides endpoints for financial reconciliation including
 * transaction matching, discrepancy detection, and reporting.
 */

import type { AragoraClient } from '../client';

/** Reconciliation job status */
export type ReconciliationStatus = 'pending' | 'running' | 'completed' | 'failed';

/** Reconciliation job */
export interface ReconciliationJob {
  id: string;
  name: string;
  status: ReconciliationStatus;
  source_account: string;
  target_account: string;
  matched_count: number;
  unmatched_count: number;
  discrepancy_amount: number;
  created_at: string;
  completed_at?: string;
}

/** Reconciliation discrepancy */
export interface Discrepancy {
  id: string;
  job_id: string;
  type: 'missing' | 'amount_mismatch' | 'duplicate' | 'timing';
  source_transaction_id?: string;
  target_transaction_id?: string;
  amount_difference: number;
  description: string;
  resolved: boolean;
}

/** Request to create a reconciliation job */
export interface CreateReconciliationRequest {
  name: string;
  source_account: string;
  target_account: string;
  date_range?: { start: string; end: string };
}

/**
 * Reconciliation namespace for financial transaction matching.
 *
 * @example
 * ```typescript
 * const job = await client.reconciliation.create({
 *   name: 'Monthly reconciliation',
 *   source_account: 'bank_main',
 *   target_account: 'ledger_main',
 * });
 * ```
 */
export class ReconciliationNamespace {
  constructor(private client: AragoraClient) {}

  /** List reconciliation jobs. */
  async list(options?: {
    status?: string;
    limit?: number;
  }): Promise<ReconciliationJob[]> {
    const response = await this.client.request<{ jobs: ReconciliationJob[] }>(
      'GET',
      '/api/v1/reconciliation',
      { params: options }
    );
    return response.jobs;
  }

  /** Create a reconciliation job. */
  async create(request: CreateReconciliationRequest): Promise<ReconciliationJob> {
    return this.client.request<ReconciliationJob>(
      'POST',
      '/api/v1/reconciliation',
      { body: request }
    );
  }

  /** Get a reconciliation job by ID. */
  async get(jobId: string): Promise<ReconciliationJob> {
    return this.client.request<ReconciliationJob>(
      'GET',
      `/api/v1/reconciliation/${encodeURIComponent(jobId)}`
    );
  }

  /** Get discrepancies for a reconciliation job. */
  async getDiscrepancies(jobId: string): Promise<Discrepancy[]> {
    const response = await this.client.request<{ discrepancies: Discrepancy[] }>(
      'GET',
      `/api/v1/reconciliation/${encodeURIComponent(jobId)}/discrepancies`
    );
    return response.discrepancies;
  }

  /** Resolve a discrepancy. */
  async resolveDiscrepancy(
    jobId: string,
    discrepancyId: string,
    resolution: { action: string; notes?: string }
  ): Promise<Discrepancy> {
    return this.client.request<Discrepancy>(
      'POST',
      `/api/v1/reconciliation/${encodeURIComponent(jobId)}/discrepancies/${encodeURIComponent(discrepancyId)}/resolve`,
      { body: resolution }
    );
  }

  /**
   * Get reconciliation status.
   */
  async getStatus(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/reconciliation/status', { params }) as Promise<Record<string, unknown>>;
  }
}
