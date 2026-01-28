/**
 * Usage Metering Namespace API
 *
 * Provides usage tracking and billing information:
 * - Usage summaries and breakdowns
 * - Usage limits and quotas
 * - Export usage data
 */

/**
 * Usage period options.
 */
export type UsagePeriod = 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year';

/**
 * Billing tier.
 */
export type BillingTier = 'free' | 'starter' | 'professional' | 'enterprise' | 'enterprise_plus';

/**
 * Export format.
 */
export type UsageExportFormat = 'csv' | 'json';

/**
 * Token usage.
 */
export interface TokenUsage {
  input: number;
  output: number;
  total: number;
  cost: string;
}

/**
 * Usage counts.
 */
export interface UsageCounts {
  debates: number;
  api_calls: number;
}

/**
 * Usage summary.
 */
export interface UsageSummary {
  period_start: string;
  period_end: string;
  period_type: UsagePeriod;
  tokens: TokenUsage;
  counts: UsageCounts;
  by_model: Record<string, number>;
  by_provider: Record<string, number>;
  limits: {
    tokens: number;
    debates: number;
    api_calls: number;
  };
  usage_percent: {
    tokens: number;
    debates: number;
    api_calls: number;
  };
}

/**
 * Model usage breakdown.
 */
export interface ModelUsage {
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost: string;
  requests: number;
}

/**
 * Provider usage breakdown.
 */
export interface ProviderUsage {
  provider: string;
  total_tokens: number;
  cost: string;
  requests: number;
}

/**
 * Daily usage breakdown.
 */
export interface DailyUsage {
  day: string;
  total_tokens: number;
  cost: string;
  debates: number;
  api_calls: number;
}

/**
 * User usage breakdown.
 */
export interface UserUsage {
  user_id: string;
  total_tokens: number;
  cost: string;
  requests: number;
}

/**
 * Full usage breakdown.
 */
export interface UsageBreakdown {
  org_id: string;
  period_start: string;
  period_end: string;
  totals: {
    cost: string;
    tokens: number;
    debates: number;
    api_calls: number;
  };
  by_model: ModelUsage[];
  by_provider: ProviderUsage[];
  by_day: DailyUsage[];
  by_user: UserUsage[];
}

/**
 * Usage limits.
 */
export interface UsageLimits {
  org_id: string;
  tier: BillingTier;
  limits: {
    tokens: number;
    debates: number;
    api_calls: number;
  };
  used: {
    tokens: number;
    debates: number;
    api_calls: number;
  };
  percent: {
    tokens: number;
    debates: number;
    api_calls: number;
  };
  exceeded: {
    tokens: boolean;
    debates: boolean;
    api_calls: boolean;
  };
}

/**
 * Quota period.
 */
export type QuotaPeriod = 'hour' | 'day' | 'week' | 'month';

/**
 * Quota status.
 */
export interface QuotaStatus {
  limit: number;
  current: number;
  remaining: number;
  period: QuotaPeriod;
  percentage_used: number;
  is_exceeded: boolean;
  is_warning: boolean;
  resets_at?: string;
}

/**
 * All quotas response.
 */
export interface QuotasResponse {
  quotas: {
    debates: QuotaStatus;
    api_requests: QuotaStatus;
    tokens: QuotaStatus;
    storage_bytes: QuotaStatus;
    knowledge_bytes: QuotaStatus;
  };
}

/**
 * Usage breakdown options.
 */
export interface UsageBreakdownOptions {
  start?: string;
  end?: string;
}

/**
 * Usage export options.
 */
export interface UsageExportOptions {
  start?: string;
  end?: string;
  format?: UsageExportFormat;
}

/**
 * Client interface for usage metering operations.
 */
interface UsageMeteringClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Usage Metering API namespace.
 *
 * Provides methods for tracking and managing usage:
 * - Get usage summaries and breakdowns
 * - Check usage limits and quotas
 * - Export usage data
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get monthly usage summary
 * const { usage } = await client.usageMetering.getUsage('month');
 *
 * // Get detailed breakdown
 * const { breakdown } = await client.usageMetering.getBreakdown({
 *   start: '2025-01-01T00:00:00Z',
 *   end: '2025-01-31T23:59:59Z',
 * });
 *
 * // Check quotas
 * const { quotas } = await client.usageMetering.getQuotas();
 * ```
 */
export class UsageMeteringAPI {
  constructor(private client: UsageMeteringClientInterface) {}

  /**
   * Get usage summary for a billing period.
   */
  async getUsage(period: UsagePeriod = 'month'): Promise<{ usage: UsageSummary }> {
    return this.client.request('GET', '/api/v1/billing/usage', {
      params: { period },
    });
  }

  /**
   * Get detailed usage breakdown.
   */
  async getBreakdown(options?: UsageBreakdownOptions): Promise<{ breakdown: UsageBreakdown }> {
    return this.client.request('GET', '/api/v1/billing/usage/breakdown', {
      params: options as Record<string, unknown>,
    });
  }

  /**
   * Get current usage limits.
   */
  async getLimits(): Promise<{ limits: UsageLimits }> {
    return this.client.request('GET', '/api/v1/billing/limits');
  }

  /**
   * Get quota status for all resources.
   */
  async getQuotas(): Promise<QuotasResponse> {
    return this.client.request('GET', '/api/v1/quotas');
  }

  /**
   * Export usage data as CSV or JSON.
   */
  async exportUsage(options?: UsageExportOptions): Promise<UsageBreakdown | string> {
    return this.client.request('GET', '/api/v1/billing/usage/export', {
      params: options as Record<string, unknown>,
    });
  }
}
