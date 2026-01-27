/**
 * Cost Management Namespace API
 *
 * Provides methods for tracking and managing AI costs:
 * - Cost dashboard and summaries
 * - Cost breakdown by provider, feature, model
 * - Budget management and alerts
 * - Usage timeline and forecasting
 * - Optimization recommendations
 * - Pre-execution cost estimation
 * - What-if scenario simulation
 */

// =============================================================================
// Enumerations
// =============================================================================

/**
 * Cost data granularity
 */
export type CostGranularity = 'hourly' | 'daily' | 'weekly' | 'monthly';

/**
 * Cost trend direction
 */
export type TrendDirection = 'increasing' | 'decreasing' | 'stable';

/**
 * Seasonal cost pattern
 */
export type SeasonalPattern = 'daily' | 'weekly' | 'monthly' | 'none';

/**
 * Alert severity levels
 */
export type AlertSeverity = 'info' | 'warning' | 'critical';

/**
 * Budget alert levels
 */
export type BudgetAlertLevel = 'info' | 'warning' | 'critical' | 'exceeded';

/**
 * Recommendation types
 */
export type RecommendationType =
  | 'model_downgrade'
  | 'caching'
  | 'batching'
  | 'rate_limiting'
  | 'prompt_optimization'
  | 'provider_switch'
  | 'time_shifting'
  | 'quota_adjustment';

/**
 * Recommendation priority levels
 */
export type RecommendationPriority = 'critical' | 'high' | 'medium' | 'low';

/**
 * Recommendation status
 */
export type RecommendationStatus = 'pending' | 'applied' | 'dismissed' | 'expired' | 'partial';

/**
 * Cost enforcement modes
 */
export type CostEnforcementMode = 'hard' | 'soft' | 'throttle' | 'estimate';

/**
 * Throttle levels based on budget usage
 */
export type ThrottleLevel = 'none' | 'light' | 'medium' | 'heavy' | 'blocked';

// =============================================================================
// Core Types
// =============================================================================

/**
 * Cost breakdown item
 */
export interface CostBreakdownItem {
  name: string;
  cost: number;
  percentage: number;
}

/**
 * Daily cost data point
 */
export interface DailyCost {
  date: string;
  cost: number;
  tokens?: number;
}

/**
 * Cost alert
 */
export interface CostAlert {
  id: string;
  type: 'budget_warning' | 'spike_detected' | 'limit_reached';
  message: string;
  severity: 'critical' | 'warning' | 'info';
  timestamp: string;
  acknowledged?: boolean;
}

/**
 * Cost summary data
 */
export interface CostSummary {
  total_cost: number;
  budget: number;
  tokens_used: number;
  api_calls: number;
  last_updated: string;
  cost_by_provider?: CostBreakdownItem[];
  cost_by_feature?: CostBreakdownItem[];
  daily_costs?: DailyCost[];
  alerts?: CostAlert[];
}

/**
 * Budget configuration
 */
export interface CostBudget {
  workspace_id: string;
  monthly_limit: number;
  daily_limit?: number;
  current_spend?: number;
  alert_thresholds?: number[];
}

/**
 * Cost optimization recommendation
 */
export interface CostRecommendation {
  id: string;
  type: 'model_downgrade' | 'caching' | 'batching' | string;
  title: string;
  description: string;
  estimated_savings: number;
  effort: 'low' | 'medium' | 'high';
  status: 'pending' | 'applied' | 'dismissed';
  created_at?: string;
}

/**
 * Efficiency metrics
 */
export interface EfficiencyMetrics {
  cost_per_1k_tokens: number;
  tokens_per_call: number;
  cost_per_call: number;
  total_tokens: number;
  total_calls: number;
  total_cost: number;
  model_utilization?: Array<{ model: string; calls: number; cost: number }>;
}

/**
 * Cost forecast data
 */
export interface CostForecast {
  workspace_id: string;
  forecast_days: number;
  projected_cost: number;
  confidence_interval?: [number, number];
  trend: 'increasing' | 'decreasing' | 'stable';
  daily_projections?: Array<{ date: string; cost: number }>;
}

/**
 * Time range options for cost queries
 */
export type TimeRange = '24h' | '7d' | '30d' | '90d';

/**
 * Group by options for cost breakdown
 */
export type GroupBy = 'provider' | 'feature' | 'model';

/**
 * Interface for the internal client used by CostManagementAPI.
 */
interface CostManagementClientInterface {
  request<T>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Cost Management API namespace.
 *
 * Provides methods for tracking and managing AI costs:
 * - View cost summaries and breakdowns
 * - Set and manage budgets
 * - Get cost optimization recommendations
 * - Forecast future costs
 *
 * Essential for SME cost management and preventing unexpected charges.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get cost summary
 * const summary = await client.costManagement.getSummary({ range: '30d' });
 * console.log(`Total cost: $${summary.total_cost}`);
 *
 * // Get breakdown by provider
 * const { breakdown, total } = await client.costManagement.getBreakdown({ group_by: 'provider' });
 *
 * // Set a budget
 * await client.costManagement.setBudget({ budget: 500, daily_limit: 20 });
 *
 * // Get optimization recommendations
 * const { recommendations } = await client.costManagement.getRecommendations();
 * ```
 */
export class CostManagementAPI {
  constructor(private client: CostManagementClientInterface) {}

  // ===========================================================================
  // Cost Dashboard
  // ===========================================================================

  /**
   * Get cost dashboard summary.
   */
  async getSummary(options?: {
    workspace_id?: string;
    range?: TimeRange;
  }): Promise<CostSummary> {
    return this.client.request('GET', '/api/costs', { params: options });
  }

  /**
   * Get detailed cost breakdown.
   */
  async getBreakdown(options?: {
    workspace_id?: string;
    range?: TimeRange;
    group_by?: GroupBy;
  }): Promise<{ breakdown: CostBreakdownItem[]; total: number }> {
    return this.client.request('GET', '/api/costs/breakdown', { params: options });
  }

  /**
   * Get usage timeline data.
   */
  async getTimeline(options?: {
    workspace_id?: string;
    range?: TimeRange;
  }): Promise<{ timeline: DailyCost[]; total: number; average: number }> {
    return this.client.request('GET', '/api/costs/timeline', { params: options });
  }

  // ===========================================================================
  // Alerts
  // ===========================================================================

  /**
   * Get budget alerts.
   */
  async getAlerts(options?: {
    workspace_id?: string;
  }): Promise<{ alerts: CostAlert[] }> {
    return this.client.request('GET', '/api/costs/alerts', { params: options });
  }

  /**
   * Dismiss a budget alert.
   */
  async dismissAlert(
    alertId: string,
    options?: { workspace_id?: string }
  ): Promise<{ success: boolean }> {
    return this.client.request('POST', `/api/costs/alerts/${alertId}/dismiss`, {
      json: options,
    });
  }

  // ===========================================================================
  // Budget Management
  // ===========================================================================

  /**
   * Set budget limits.
   */
  async setBudget(request: {
    budget: number;
    workspace_id?: string;
    daily_limit?: number;
    name?: string;
  }): Promise<CostBudget> {
    return this.client.request('POST', '/api/costs/budget', { json: request });
  }

  // ===========================================================================
  // Recommendations
  // ===========================================================================

  /**
   * Get cost optimization recommendations.
   */
  async getRecommendations(options?: {
    workspace_id?: string;
    status?: 'pending' | 'applied' | 'dismissed';
    type?: string;
  }): Promise<{ recommendations: CostRecommendation[] }> {
    return this.client.request('GET', '/api/costs/recommendations', { params: options });
  }

  /**
   * Apply a recommendation.
   */
  async applyRecommendation(
    recommendationId: string,
    options?: { user_id?: string }
  ): Promise<{ recommendation: CostRecommendation }> {
    return this.client.request('POST', `/api/costs/recommendations/${recommendationId}/apply`, {
      json: options,
    });
  }

  /**
   * Dismiss a recommendation.
   */
  async dismissRecommendation(recommendationId: string): Promise<{ success: boolean }> {
    return this.client.request('POST', `/api/costs/recommendations/${recommendationId}/dismiss`);
  }

  // ===========================================================================
  // Efficiency Metrics
  // ===========================================================================

  /**
   * Get efficiency metrics.
   */
  async getEfficiency(options?: {
    workspace_id?: string;
    range?: TimeRange;
  }): Promise<EfficiencyMetrics> {
    return this.client.request('GET', '/api/costs/efficiency', { params: options });
  }

  // ===========================================================================
  // Forecasting
  // ===========================================================================

  /**
   * Get cost forecast.
   */
  async getForecast(options?: {
    workspace_id?: string;
    days?: number;
  }): Promise<CostForecast> {
    return this.client.request('GET', '/api/costs/forecast', { params: options });
  }

  /**
   * Simulate a cost scenario.
   */
  async simulateScenario(request: {
    workspace_id: string;
    scenario: {
      name: string;
      description?: string;
      changes: Record<string, unknown>;
    };
    days?: number;
  }): Promise<CostForecast> {
    return this.client.request('POST', '/api/costs/forecast/simulate', { json: request });
  }
}
