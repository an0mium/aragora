/**
 * Costs Namespace API
 *
 * Provides endpoints for tracking and visualizing AI costs,
 * including budget tracking, alerts, optimization recommendations,
 * forecasting, and cost exports.
 */

import type { AragoraClient } from '../client';

/** Cost summary for the dashboard */
export interface CostSummary {
  total_cost: number;
  budget_limit: number;
  budget_used_pct: number;
  period_start: string;
  period_end: string;
  by_provider: Record<string, number>;
  by_feature: Record<string, number>;
}

/** Budget alert entry */
export interface BudgetAlert {
  id: string;
  threshold_pct: number;
  triggered: boolean;
  triggered_at?: string;
  message: string;
}

/** Cost optimization recommendation */
export interface CostRecommendation {
  id: string;
  category: string;
  description: string;
  estimated_savings: number;
  difficulty: string;
}

/** Cost timeline data point */
export interface CostTimelineEntry {
  date: string;
  cost: number;
  provider?: string;
}

/** Budget configuration */
export interface BudgetConfig {
  id: string;
  workspace_id: string;
  name: string;
  monthly_limit_usd: number;
  daily_limit_usd?: number;
  current_monthly_spend: number;
  current_daily_spend: number;
  active: boolean;
}

/** Cost forecast report */
export interface CostForecast {
  workspace_id: string;
  forecast_days: number;
  projected_cost: number;
  daily_forecasts?: Array<{
    date: string;
    projected_cost_usd: number;
    confidence_low?: number;
    confidence_high?: number;
  }>;
}

/** Cost estimate response */
export interface CostEstimate {
  estimated_cost_usd: number;
  breakdown: {
    input_tokens: number;
    output_tokens: number;
    input_cost_usd: number;
    output_cost_usd: number;
  };
  pricing: {
    model: string;
    provider: string;
    input_per_1m: number;
    output_per_1m: number;
  };
  operation: string;
}

/**
 * Costs namespace for AI cost tracking and optimization.
 *
 * @example
 * ```typescript
 * const summary = await client.costs.getSummary();
 * console.log(`Total spend: $${summary.total_cost}`);
 * ```
 */
export class CostsNamespace {
  constructor(private client: AragoraClient) {}

  // ===========================================================================
  // Core Cost Data
  // ===========================================================================

  /** Get cost summary dashboard data. */
  async getSummary(options?: { period?: string; workspace_id?: string }): Promise<CostSummary> {
    return this.client.request('GET', '/api/v1/costs', { params: options });
  }

  /** Get cost breakdown by provider, feature, or model. */
  async getBreakdown(options?: { group_by?: string; range?: string; workspace_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/costs/breakdown', { params: options });
  }

  /** Get cost timeline data. */
  async getTimeline(options?: { range?: string; workspace_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/costs/timeline', { params: options });
  }

  /** Get detailed usage tracking data. */
  async getUsage(options?: { range?: string; group_by?: string; workspace_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/costs/usage', { params: options });
  }

  /** Get cost efficiency metrics. */
  async getEfficiency(options?: { range?: string; workspace_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/costs/efficiency', { params: options });
  }

  /** Export cost data as CSV or JSON. */
  async export(options?: { format?: 'csv' | 'json'; range?: string; group_by?: string; workspace_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/costs/export', { params: options });
  }

  // ===========================================================================
  // Alerts
  // ===========================================================================

  /** Get budget alerts. */
  async getAlerts(options?: { workspace_id?: string }): Promise<{ alerts: BudgetAlert[] }> {
    return this.client.request('GET', '/api/v1/costs/alerts', { params: options });
  }

  /** Create a cost alert. */
  async createAlert(data: { name: string; type?: string; threshold?: number; notification_channels?: string[]; workspace_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/costs/alerts', { body: data });
  }

  /** Dismiss a budget alert. */
  async dismissAlert(alertId: string, options?: { workspace_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/costs/alerts/${alertId}/dismiss`, { params: options });
  }

  // ===========================================================================
  // Budgets
  // ===========================================================================

  /** Set budget limits (legacy endpoint). */
  async setBudget(data: { budget: number; workspace_id?: string; daily_limit?: number; name?: string }): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/costs/budget', { body: data });
  }

  /** List all budgets for the workspace. */
  async listBudgets(options?: { workspace_id?: string }): Promise<{ budgets: BudgetConfig[]; count: number }> {
    return this.client.request('GET', '/api/v1/costs/budgets', { params: options });
  }

  /** Create a new budget. */
  async createBudget(data: { monthly_limit_usd: number; workspace_id?: string; name?: string; daily_limit_usd?: number; alert_thresholds?: number[] }): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/costs/budgets', { body: data });
  }

  // ===========================================================================
  // Recommendations
  // ===========================================================================

  /** Get cost optimization recommendations. */
  async getRecommendations(options?: { workspace_id?: string; status?: string; type?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/costs/recommendations', { params: options });
  }

  /** Get detailed cost optimization recommendations with implementation steps. */
  async getRecommendationsDetailed(options?: { workspace_id?: string; include_implementation?: boolean; min_savings?: number }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/costs/recommendations/detailed', { params: options });
  }

  /** Get a specific cost optimization recommendation. */
  async getRecommendation(recommendationId: string): Promise<CostRecommendation> {
    return this.client.request('GET', `/api/v1/costs/recommendations/${recommendationId}`);
  }

  /** Apply a cost optimization recommendation. */
  async applyRecommendation(recommendationId: string, data?: { user_id?: string }): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/costs/recommendations/${recommendationId}/apply`, { body: data });
  }

  /** Dismiss a cost optimization recommendation. */
  async dismissRecommendation(recommendationId: string): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/costs/recommendations/${recommendationId}/dismiss`);
  }

  // ===========================================================================
  // Forecasting
  // ===========================================================================

  /** Get cost forecast. */
  async getForecast(options?: { workspace_id?: string; days?: number }): Promise<CostForecast> {
    return this.client.request('GET', '/api/v1/costs/forecast', { params: options });
  }

  /** Get detailed cost forecast with daily breakdowns and confidence intervals. */
  async getForecastDetailed(options?: { workspace_id?: string; days?: number; include_confidence?: boolean }): Promise<CostForecast> {
    return this.client.request('GET', '/api/v1/costs/forecast/detailed', { params: options });
  }

  /** Simulate a cost scenario. */
  async simulateForecast(data: { scenario: Record<string, unknown>; workspace_id?: string; days?: number }): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/costs/forecast/simulate', { body: data });
  }

  // ===========================================================================
  // Constraints and Estimates
  // ===========================================================================

  /** Pre-flight check if an operation would exceed budget constraints. */
  async checkConstraints(data: { estimated_cost_usd: number; workspace_id?: string; operation?: string }): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/costs/constraints/check', { body: data });
  }

  /** Estimate the cost of an operation. */
  async estimate(data: { operation: string; tokens_input?: number; tokens_output?: number; model?: string; provider?: string }): Promise<CostEstimate> {
    return this.client.request('POST', '/api/v1/costs/estimate', { body: data });
  }
}
