/**
 * Costs Namespace API
 *
 * Provides endpoints for tracking and visualizing AI costs,
 * including budget tracking, alerts, and optimization suggestions.
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

  /** Get cost summary dashboard data. */
  async getSummary(options?: { period?: string }): Promise<CostSummary> {
    return this.client.request<CostSummary>('GET', '/api/v1/costs', { params: options });
  }

  /** Get budget alerts. */
  async getAlerts(): Promise<BudgetAlert[]> {
    const response = await this.client.request<{ alerts: BudgetAlert[] }>(
      'GET',
      '/api/v1/costs/alerts'
    );
    return response.alerts;
  }

  /** Get cost optimization recommendations. */
  async getRecommendations(): Promise<CostRecommendation[]> {
    const response = await this.client.request<{ recommendations: CostRecommendation[] }>(
      'GET',
      '/api/v1/costs/recommendations'
    );
    return response.recommendations;
  }

  /** Get cost timeline data. */
  async getTimeline(options?: {
    start?: string;
    end?: string;
    granularity?: 'hour' | 'day' | 'week';
  }): Promise<CostTimelineEntry[]> {
    const response = await this.client.request<{ timeline: CostTimelineEntry[] }>(
      'GET',
      '/api/v1/costs/timeline',
      { params: options }
    );
    return response.timeline;
  }

  /** Export cost data. */
  async export(options?: { format?: 'csv' | 'json'; period?: string }): Promise<Blob> {
    return this.client.request<Blob>('GET', '/api/v1/costs/export', { params: options });
  }
}
