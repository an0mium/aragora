/**
 * Usage Namespace API
 *
 * Provides methods for usage tracking and SME dashboard:
 * - Usage summary and breakdown
 * - ROI analysis
 * - Budget tracking
 * - Forecasting
 * - Industry benchmarks
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get usage summary
 * const summary = await client.usage.getSummary({ period: 'month' });
 *
 * // Get ROI analysis
 * const roi = await client.usage.getROI();
 *
 * // Get budget status
 * const budget = await client.usage.getBudgetStatus();
 * ```
 */

/**
 * Interface for usage client methods.
 */
interface UsageClientInterface {
  get<T>(path: string, params?: Record<string, unknown>): Promise<T>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown> }): Promise<T>;
}

/**
 * Time period options.
 */
export type UsagePeriod = 'day' | 'week' | 'month' | 'quarter' | 'year';

/**
 * Benchmark comparison types.
 */
export type BenchmarkType = 'sme' | 'enterprise' | 'tech_startup' | 'consulting';

/**
 * Export format options.
 */
export type ExportFormat = 'csv' | 'json';

/**
 * Group by dimension.
 */
export type GroupByDimension = 'agent' | 'model' | 'day' | 'debate_type';

/**
 * Usage summary response.
 */
export interface UsageSummary {
  period: string;
  total_debates: number;
  total_tokens: number;
  total_cost: number;
  active_users: number;
  avg_debate_duration_ms: number;
  consensus_rate: number;
  top_agents: Array<{ name: string; debates: number }>;
}

/**
 * Usage breakdown response.
 */
export interface UsageBreakdown {
  group_by: GroupByDimension;
  period: string;
  data: Array<{
    key: string;
    debates: number;
    tokens: number;
    cost: number;
  }>;
}

/**
 * ROI analysis response.
 */
export interface ROIAnalysis {
  time_saved_hours: number;
  cost_per_decision: number;
  benchmark_comparison: {
    industry_avg_cost: number;
    savings_percentage: number;
  };
  decision_quality_score: number;
  recommendations: string[];
}

/**
 * Budget status response.
 */
export interface BudgetStatus {
  daily_limit: number;
  daily_used: number;
  daily_remaining: number;
  monthly_limit: number;
  monthly_used: number;
  monthly_remaining: number;
  throttle_level: 'none' | 'light' | 'medium' | 'heavy' | 'blocked';
  reset_at: string;
}

/**
 * Usage forecast response.
 */
export interface UsageForecast {
  forecast_days: number;
  projected_debates: number;
  projected_cost: number;
  confidence: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  daily_projections: Array<{
    date: string;
    projected_debates: number;
    projected_cost: number;
  }>;
}

/**
 * Industry benchmark data.
 */
export interface IndustryBenchmarks {
  benchmark_type: BenchmarkType;
  metrics: {
    avg_decisions_per_month: number;
    avg_cost_per_decision: number;
    avg_time_per_decision_hours: number;
    consensus_rate: number;
  };
  your_position: {
    percentile: number;
    comparison: 'above' | 'at' | 'below';
  };
}

/**
 * Export result.
 */
export interface UsageExport {
  format: ExportFormat;
  period: string;
  download_url: string;
  expires_at: string;
}

/**
 * Usage API for SME dashboard.
 *
 * Provides methods for tracking and analyzing usage:
 * - Summary metrics
 * - Breakdown by dimension
 * - ROI analysis
 * - Budget monitoring
 * - Forecasting
 */
export class UsageAPI {
  constructor(private client: UsageClientInterface) {}

  /**
   * Get unified usage metrics summary.
   *
   * @param options - Query options
   * @param options.period - Time period (day, week, month, quarter, year)
   * @param options.organizationId - Optional organization filter
   */
  async getSummary(options?: {
    period?: UsagePeriod;
    organizationId?: string;
  }): Promise<UsageSummary> {
    const params: Record<string, unknown> = {};
    if (options?.period) params.period = options.period;
    if (options?.organizationId) params.organization_id = options.organizationId;
    return this.client.request('GET', '/api/v1/usage/summary', { params });
  }

  /**
   * Get usage breakdown by dimension.
   *
   * @param options - Query options
   * @param options.groupBy - Dimension to group by
   * @param options.period - Time period
   */
  async getBreakdown(options?: {
    groupBy?: GroupByDimension;
    period?: UsagePeriod;
  }): Promise<UsageBreakdown> {
    const params: Record<string, unknown> = {};
    if (options?.groupBy) params.group_by = options.groupBy;
    if (options?.period) params.period = options.period;
    return this.client.request('GET', '/api/v1/usage/breakdown', { params });
  }

  /**
   * Get ROI analysis with industry benchmarks.
   *
   * @param options - Query options
   * @param options.benchmarkType - Benchmark to compare against
   */
  async getROI(options?: { benchmarkType?: BenchmarkType }): Promise<ROIAnalysis> {
    const params: Record<string, unknown> = {};
    if (options?.benchmarkType) params.benchmark_type = options.benchmarkType;
    return this.client.request('GET', '/api/v1/usage/roi', { params });
  }

  /**
   * Get current budget utilization status.
   */
  async getBudgetStatus(): Promise<BudgetStatus> {
    return this.client.get('/api/v1/usage/budget-status');
  }

  /**
   * Get usage forecast.
   *
   * @param options - Query options
   * @param options.days - Number of days to forecast (default: 30)
   */
  async getForecast(options?: { days?: number }): Promise<UsageForecast> {
    const params: Record<string, unknown> = {};
    if (options?.days) params.days = options.days;
    return this.client.request('GET', '/api/v1/usage/forecast', { params });
  }

  /**
   * Get industry benchmarks for comparison.
   *
   * @param options - Query options
   * @param options.benchmarkType - Industry benchmark type
   */
  async getBenchmarks(options?: { benchmarkType?: BenchmarkType }): Promise<IndustryBenchmarks> {
    const params: Record<string, unknown> = {};
    if (options?.benchmarkType) params.type = options.benchmarkType;
    return this.client.request('GET', '/api/v1/usage/benchmarks', { params });
  }

  /**
   * Export usage data.
   *
   * @param options - Export options
   * @param options.format - Export format (csv or json)
   * @param options.period - Time period to export
   */
  async export(options?: {
    format?: ExportFormat;
    period?: UsagePeriod;
  }): Promise<UsageExport> {
    const params: Record<string, unknown> = {};
    if (options?.format) params.format = options.format;
    if (options?.period) params.period = options.period;
    return this.client.request('GET', '/api/v1/usage/export', { params });
  }
}
