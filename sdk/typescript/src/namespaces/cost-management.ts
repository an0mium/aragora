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
 * Basic cost forecast data
 */
export interface CostForecast {
  workspace_id: string;
  forecast_days: number;
  projected_cost: number;
  confidence_interval?: [number, number];
  trend: TrendDirection;
  daily_projections?: Array<{ date: string; cost: number }>;
}

// =============================================================================
// Advanced Forecasting Types
// =============================================================================

/** Detailed trend analysis for costs */
export interface TrendAnalysis {
  direction: TrendDirection;
  change_rate: number;
  change_rate_weekly: number;
  r_squared: number;
  trend_start?: string;
  confidence: number;
}

/** Daily forecast with confidence intervals */
export interface DailyForecast {
  date: string;
  predicted_cost: number;
  lower_bound: number;
  upper_bound: number;
  confidence: number;
}

/** Forecast alert for projected issues */
export interface ForecastAlert {
  type: 'budget_exceeded' | 'anomaly_detected' | 'trend_warning';
  message: string;
  severity: AlertSeverity;
  date: string;
  value?: number;
}

/** Comprehensive forecast report with trend analysis */
export interface ForecastReport {
  predicted_monthly_cost: number;
  predicted_daily_average: number;
  confidence_interval: number;
  trend?: TrendAnalysis;
  seasonal_pattern: SeasonalPattern;
  daily_forecasts: DailyForecast[];
  alerts: ForecastAlert[];
  projected_budget_usage?: number;
  days_until_budget_exceeded?: number | null;
  generated_at: string;
}

// =============================================================================
// Cost Estimation Types
// =============================================================================

/** Pre-execution cost estimate for a task */
export interface CostEstimate {
  estimated_cost_usd: number;
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_tokens: number;
  confidence: number;
  based_on_samples: number;
  model_suggestion?: string;
  estimated_savings_usd?: number;
  model_breakdown?: Array<{
    model: string;
    estimated_cost: number;
    estimated_tokens: number;
  }>;
}

/** Cost constraint validation result */
export interface CostConstraintResult {
  allowed: boolean;
  reason?: string;
  throttle_level: ThrottleLevel;
  enforcement_mode: CostEnforcementMode;
  budget_percentage_used: number;
  remaining_budget_usd?: number;
  estimated_cost?: CostEstimate;
  priority_adjustment: number;
}

// =============================================================================
// Simulation Types
// =============================================================================

/** Cost simulation scenario definition */
export interface SimulationScenario {
  name: string;
  description?: string;
  changes: {
    model_change?: string;
    volume_multiplier?: number;
    caching_enabled?: boolean;
    batch_size?: number;
    provider_change?: string;
    [key: string]: unknown;
  };
}

/** Cost simulation result */
export interface SimulationResult {
  baseline_cost: number;
  simulated_cost: number;
  cost_difference: number;
  percentage_change: number;
  daily_breakdown: Array<{
    date: string;
    baseline: number;
    simulated: number;
    difference: number;
  }>;
  quality_impact?: string;
  risk_level?: 'low' | 'medium' | 'high';
}

// =============================================================================
// Advanced Recommendation Types
// =============================================================================

/** Alternative model suggestion for cost optimization */
export interface ModelAlternative {
  provider: string;
  model: string;
  cost_per_1k_input: number;
  cost_per_1k_output: number;
  quality_score: number;
  latency_multiplier: number;
  suitable_for: string[];
}

/** Caching opportunity analysis */
export interface CachingOpportunity {
  pattern: 'system_prompt' | 'repeated_query' | 'prefix' | 'semantic';
  estimated_hit_rate: number;
  unique_queries: number;
  repeat_count: number;
  cache_strategy: 'exact' | 'semantic' | 'prefix';
  estimated_savings_usd: number;
}

/** Batching opportunity analysis */
export interface BatchingOpportunity {
  current_rpm: number;
  recommended_batch_size: number;
  latency_increase_ms: number;
  savings_percentage: number;
  batchable_tasks: string[];
}

/** Implementation step for a recommendation */
export interface ImplementationStep {
  order: number;
  description: string;
  code_snippet?: string;
  config_change?: Record<string, unknown>;
  effort_minutes?: number;
}

/** Enhanced cost optimization recommendation with full details */
export interface DetailedCostRecommendation {
  id: string;
  type: RecommendationType;
  priority: RecommendationPriority;
  title: string;
  description: string;
  estimated_savings: number;
  effort: 'low' | 'medium' | 'high';
  status: RecommendationStatus;
  model_alternative?: ModelAlternative;
  caching_opportunity?: CachingOpportunity;
  batching_opportunity?: BatchingOpportunity;
  implementation_steps?: ImplementationStep[];
  quality_impact?: string;
  quality_impact_score?: number;
  risk_level: 'low' | 'medium' | 'high';
  auto_apply_available: boolean;
  requires_approval: boolean;
  created_at: string;
  expires_at?: string;
}

/** Summary of all recommendations */
export interface RecommendationSummary {
  total_count: number;
  total_estimated_savings: number;
  by_type: Record<string, number>;
  by_priority: Record<string, number>;
  auto_apply_count: number;
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
 * - Estimate costs before execution
 * - Simulate what-if scenarios
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
 *
 * // Estimate cost before running a task
 * const estimate = await client.costManagement.estimateCost({ task: 'analyze document', model: 'claude-3-opus' });
 *
 * // Simulate switching to a cheaper model
 * const simulation = await client.costManagement.simulateScenario({
 *   workspace_id: 'ws-123',
 *   scenario: { name: 'Switch to Haiku', changes: { model_change: 'claude-3-haiku' } }
 * });
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
   * Get detailed recommendations with full analysis.
   */
  async getDetailedRecommendations(options?: {
    workspace_id?: string;
    status?: RecommendationStatus;
    type?: RecommendationType;
    priority?: RecommendationPriority;
  }): Promise<{ recommendations: DetailedCostRecommendation[]; summary: RecommendationSummary }> {
    return this.client.request('GET', '/api/costs/recommendations/detailed', { params: options });
  }

  /**
   * Get a specific recommendation by ID.
   */
  async getRecommendation(recommendationId: string): Promise<DetailedCostRecommendation> {
    return this.client.request('GET', `/api/costs/recommendations/${recommendationId}`);
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
  // Cost Estimation
  // ===========================================================================

  /**
   * Estimate cost for a task before execution.
   */
  async estimateCost(request: {
    task: string;
    model?: string;
    agents?: string[];
    rounds?: number;
    workspace_id?: string;
  }): Promise<CostEstimate> {
    return this.client.request('POST', '/api/costs/estimate', { json: request });
  }

  /**
   * Check if a task is allowed under current budget constraints.
   */
  async checkConstraints(request: {
    task: string;
    model?: string;
    workspace_id?: string;
  }): Promise<CostConstraintResult> {
    return this.client.request('POST', '/api/costs/constraints/check', { json: request });
  }

  // ===========================================================================
  // Forecasting
  // ===========================================================================

  /**
   * Get basic cost forecast.
   */
  async getForecast(options?: {
    workspace_id?: string;
    days?: number;
  }): Promise<CostForecast> {
    return this.client.request('GET', '/api/costs/forecast', { params: options });
  }

  /**
   * Get detailed forecast report with trend analysis.
   */
  async getDetailedForecast(options?: {
    workspace_id?: string;
    days?: number;
    include_alerts?: boolean;
  }): Promise<ForecastReport> {
    return this.client.request('GET', '/api/costs/forecast/detailed', { params: options });
  }

  /**
   * Simulate a cost scenario.
   */
  async simulateScenario(request: {
    workspace_id: string;
    scenario: SimulationScenario;
    days?: number;
  }): Promise<SimulationResult> {
    return this.client.request('POST', '/api/costs/forecast/simulate', { json: request });
  }
}
