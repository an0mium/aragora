/**
 * Quotas Namespace API
 *
 * Provides endpoints for managing resource quotas including
 * usage limits, quota policies, and usage tracking.
 */

import type { AragoraClient } from '../client';

/** Quota resource type */
export type QuotaResource = 'debates' | 'agents' | 'storage' | 'api_calls' | 'workflows';

/** Quota definition */
export interface Quota {
  id: string;
  resource: QuotaResource;
  limit: number;
  used: number;
  remaining: number;
  period: 'hourly' | 'daily' | 'monthly';
  resets_at: string;
}

/** Quota policy */
export interface QuotaPolicy {
  id: string;
  name: string;
  description?: string;
  limits: Record<string, number>;
  applies_to: string;
  created_at: string;
}

/** Quota usage history entry */
export interface QuotaUsageEntry {
  timestamp: string;
  resource: string;
  used: number;
  limit: number;
}

/**
 * Quotas namespace for resource limit management.
 *
 * @example
 * ```typescript
 * const quotas = await client.quotas.list();
 * const debateQuota = quotas.find(q => q.resource === 'debates');
 * console.log(`${debateQuota.remaining} debates remaining`);
 * ```
 */
export class QuotasNamespace {
  constructor(private client: AragoraClient) {}

  /** List all quotas for the current workspace. */
  async list(): Promise<Quota[]> {
    const response = await this.client.request<{ quotas: Quota[] }>(
      'GET',
      '/api/v1/quotas'
    );
    return response.quotas;
  }

  /** Get a specific quota by resource. */
  async get(resource: string): Promise<Quota> {
    return this.client.request<Quota>(
      'GET',
      `/api/v1/quotas/${encodeURIComponent(resource)}`
    );
  }

  /** Get quota usage history. */
  async getUsageHistory(options?: {
    resource?: string;
    period?: string;
    limit?: number;
  }): Promise<QuotaUsageEntry[]> {
    const response = await this.client.request<{ usage: QuotaUsageEntry[] }>(
      'GET',
      '/api/v1/quotas/usage',
      { params: options }
    );
    return response.usage;
  }

  /** Update a quota limit (admin only). */
  async updateLimit(resource: string, limit: number): Promise<Quota> {
    return this.client.request<Quota>(
      'PUT',
      `/api/v1/quotas/${encodeURIComponent(resource)}`,
      { body: { limit } }
    );
  }
}
