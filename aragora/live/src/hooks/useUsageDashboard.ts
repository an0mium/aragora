'use client';

import { useMemo } from 'react';
import { useSWRFetch, type UseSWRFetchOptions } from './useSWRFetch';

// ============================================================================
// Types - Usage Summary
// ============================================================================

export interface UsageSummary {
  debates: {
    total: number;
    completed: number;
    today: number;
    this_week: number;
    this_month: number;
  };
  tokens: {
    total_in: number;
    total_out: number;
    today: number;
    this_week: number;
  };
  costs: {
    total_usd: number;
    today_usd: number;
    this_week_usd: number;
    this_month_usd: number;
  };
  consensus: {
    rate: number;
    avg_confidence: number;
    avg_time_seconds: number;
  };
  active_agents: number;
  period_start: string;
  period_end: string;
}

// ============================================================================
// Types - ROI Analysis
// ============================================================================

export interface ROIAnalysis {
  roi_percentage: number;
  time_saved_hours: number;
  cost_savings_usd: number;
  manual_equivalent_hours: number;
  cost_per_decision: number;
  value_generated_usd: number;
  benchmark: {
    industry: string;
    avg_roi: number;
    percentile: number;
  };
  trends: {
    roi_trend: 'increasing' | 'stable' | 'decreasing';
    efficiency_trend: 'improving' | 'stable' | 'declining';
  };
}

// ============================================================================
// Types - Budget Status
// ============================================================================

export interface BudgetStatus {
  monthly_limit_usd: number;
  spent_usd: number;
  remaining_usd: number;
  utilization_percent: number;
  projected_end_of_month_usd: number;
  will_exceed: boolean;
  alert_level: 'normal' | 'warning' | 'critical';
  daily_average_usd: number;
  days_remaining: number;
}

// ============================================================================
// Types - Usage Forecast
// ============================================================================

export interface UsageForecast {
  projected_monthly_tokens: number;
  projected_monthly_cost_usd: number;
  projected_monthly_debates: number;
  growth_rate_percent: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  confidence: number;
  recommendations: string[];
}

// ============================================================================
// Types - Industry Benchmarks
// ============================================================================

export interface IndustryBenchmark {
  industry: string;
  avg_consensus_rate: number;
  avg_decision_time_seconds: number;
  avg_cost_per_decision: number;
  avg_roi_percentage: number;
}

export type TimeRange = '24h' | '7d' | '30d' | '90d';

// ============================================================================
// Types - Cost Breakdown (used by CostBreakdown component)
// ============================================================================

export interface CostBreakdownItem {
  name: string;
  cost_usd: number;
  percentage: number;
  tokens: number;
  requests: number;
}

export interface CostBreakdown {
  total_cost_usd: number;
  by_agent: CostBreakdownItem[];
  by_model: CostBreakdownItem[];
}

// ============================================================================
// Types - Usage Trend
// ============================================================================

export interface UsageTrendPoint {
  date: string;
  debates: number;
  tokens: number;
  cost_usd: number;
  consensus_rate: number;
}

interface DashboardUsageSummaryResponse {
  period: {
    type: string;
    start: string;
    end: string;
    days: number;
  };
  debates: {
    total: number;
    completed: number;
    consensus_rate: number;
  };
  costs: {
    total_usd: string;
    avg_per_debate_usd: string;
    by_provider: Record<string, string>;
  };
  tokens: {
    total: number;
    input: number;
    output: number;
  };
  activity: {
    active_days: number;
    debates_per_day: number;
    api_calls: number;
  };
}

interface DashboardROIResponse {
  time_savings?: {
    estimated_hours_saved?: number;
    avg_debate_duration_minutes?: number;
    manual_equivalent_hours?: number;
  };
  cost?: {
    total_aragora_cost_usd?: string;
    total_manual_cost_usd?: string;
    cost_savings_usd?: string;
    cost_per_decision_usd?: string;
  };
  roi?: {
    roi_percentage?: number;
    payback_debates?: number;
  };
  quality?: {
    consensus_rate?: number;
    avg_confidence_score?: number;
  };
  benchmark?: {
    type?: string;
    cost_usd?: string;
    savings_vs_benchmark_pct?: number;
  };
}

interface DashboardBudgetStatusResponse {
  monthly?: {
    limit_usd?: string;
    spent_usd?: string;
    remaining_usd?: string;
    percent_used?: number;
    days_remaining?: number;
    projected_end_spend_usd?: string;
  };
  daily?: {
    limit_usd?: string;
    spent_usd?: string;
  };
  alert_level?: string | null;
}

interface DashboardForecastResponse {
  projections?: {
    debates_per_month?: number;
    monthly?: {
      manual_cost_usd?: string;
      aragora_cost_usd?: string;
      savings_usd?: string;
      hours_saved?: number;
    };
    annual?: {
      manual_cost_usd?: string;
      aragora_cost_usd?: string;
      savings_usd?: string;
      hours_saved?: number;
    };
  };
  assumptions?: {
    benchmark?: string;
    hourly_rate_usd?: string;
    hours_per_decision?: number;
    avg_participants?: number;
    avg_debate_duration_minutes?: number;
  };
}

interface DashboardAgentLeaderboardResponse {
  agents: Array<{
    rank: number;
    agent_id: string;
    agent_name: string;
    provider: string;
    model: string;
    elo: number;
    debates: number;
  }>;
  count: number;
  period: string;
}

function mapTimeRangeToUsagePeriod(timeRange: TimeRange): 'day' | 'week' | 'month' | 'quarter' {
  switch (timeRange) {
    case '24h':
      return 'day';
    case '7d':
      return 'week';
    case '90d':
      return 'quarter';
    case '30d':
    default:
      return 'month';
  }
}

function parseNumber(value: number | string | null | undefined): number {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : 0;
  }

  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  return 0;
}

// ============================================================================
// Individual Hooks
// ============================================================================

/**
 * Hook for fetching usage summary data
 */
export function useUsageSummary(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: UsageSummary }>
) {
  const result = useSWRFetch<{ data: UsageSummary }>(
    `/api/v1/usage/summary?range=${timeRange}`,
    {
      refreshInterval: 30000, // Refresh every 30 seconds
      ...options,
    }
  );

  return {
    ...result,
    summary: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching ROI analysis
 */
export function useROIAnalysis(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: ROIAnalysis }>
) {
  const result = useSWRFetch<{ data: ROIAnalysis }>(
    `/api/v1/usage/roi?range=${timeRange}`,
    {
      refreshInterval: 60000, // Refresh every minute
      ...options,
    }
  );

  return {
    ...result,
    roi: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching budget status
 */
export function useBudgetStatus(
  options?: UseSWRFetchOptions<{ data: BudgetStatus }>
) {
  const result = useSWRFetch<{ data: BudgetStatus }>(
    '/api/v1/usage/budget-status',
    {
      refreshInterval: 30000, // Check budget frequently
      ...options,
    }
  );

  return {
    ...result,
    budget: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching usage forecast
 */
export function useUsageForecast(
  options?: UseSWRFetchOptions<{ data: UsageForecast }>
) {
  const result = useSWRFetch<{ data: UsageForecast }>(
    '/api/v1/usage/forecast',
    {
      refreshInterval: 300000, // Refresh every 5 minutes
      ...options,
    }
  );

  return {
    ...result,
    forecast: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching industry benchmarks
 */
export function useIndustryBenchmarks(
  options?: UseSWRFetchOptions<{ data: { benchmarks: IndustryBenchmark[] } }>
) {
  const result = useSWRFetch<{ data: { benchmarks: IndustryBenchmark[] } }>(
    '/api/v1/usage/benchmarks',
    {
      refreshInterval: 3600000, // Refresh hourly (benchmarks don't change often)
      ...options,
    }
  );

  return {
    ...result,
    benchmarks: result.data?.data?.benchmarks ?? [],
  };
}

/**
 * Hook for fetching usage trend data over time
 */
export function useUsageTrend(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: { points: UsageTrendPoint[] } }>
) {
  const result = useSWRFetch<{ data: { points: UsageTrendPoint[] } }>(
    `/api/v1/usage/trend?range=${timeRange}`,
    {
      refreshInterval: 60000, // Refresh every minute
      ...options,
    }
  );

  return {
    ...result,
    trend: result.data?.data?.points ?? [],
  };
}

/**
 * Hook for fetching cost breakdown by agent/model
 */
export function useCostBreakdown(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: CostBreakdown }>
) {
  const result = useSWRFetch<{ data: CostBreakdown }>(
    `/api/v1/usage/cost-breakdown?range=${timeRange}`,
    {
      refreshInterval: 60000, // Refresh every minute
      ...options,
    }
  );

  return {
    ...result,
    breakdown: result.data?.data ?? null,
  };
}

// ============================================================================
// Unified Dashboard Hook
// ============================================================================

/**
 * Unified hook for all usage dashboard data.
 * Combines usage summary, ROI, budget status, and forecast.
 *
 * @param timeRange  Time window for the summary / ROI data.
 * @param options.refreshInterval  Override the default SWR polling intervals
 *   (e.g. use a longer interval when WebSocket push is active).
 */
export function useUsageDashboard(
  timeRange: TimeRange = '30d',
  options?: { refreshInterval?: number }
) {
  const ri = options?.refreshInterval;
  const usagePeriod = mapTimeRangeToUsagePeriod(timeRange);

  const {
    data: summaryResponse,
    isLoading: summaryLoading,
    error: summaryError,
  } = useSWRFetch<{ data: DashboardUsageSummaryResponse }>(
    `/api/v1/usage/summary?period=${usagePeriod}`,
    ri != null ? { refreshInterval: ri } : undefined
  );
  const {
    data: roiResponse,
    isLoading: roiLoading,
    error: roiError,
  } = useSWRFetch<{ data: DashboardROIResponse }>(
    `/api/v1/usage/roi?period=${usagePeriod}`,
    ri != null ? { refreshInterval: Math.max(ri, 60_000) } : undefined
  );
  const {
    data: budgetResponse,
    isLoading: budgetLoading,
    error: budgetError,
  } = useSWRFetch<{ data: DashboardBudgetStatusResponse }>(
    '/api/v1/usage/budget-status',
    ri != null ? { refreshInterval: ri } : undefined
  );
  const {
    data: forecastResponse,
    isLoading: forecastLoading,
    error: forecastError,
  } = useSWRFetch<{ data: DashboardForecastResponse }>(
    '/api/v1/usage/forecast',
    ri != null ? { refreshInterval: Math.max(ri, 120_000) } : undefined
  );
  const {
    data: agentsResponse,
    isLoading: agentsLoading,
    error: agentsError,
  } = useSWRFetch<{ data: DashboardAgentLeaderboardResponse }>(
    `/api/v1/outcome-dashboard/agents?period=${timeRange}`,
    ri != null ? { refreshInterval: Math.max(ri, 120_000) } : undefined
  );

  const summary = summaryResponse?.data ?? null;
  const roi = roiResponse?.data ?? null;
  const budget = budgetResponse?.data ?? null;
  const forecast = forecastResponse?.data ?? null;
  const agents = agentsResponse?.data ?? null;

  const isLoading =
    summaryLoading || roiLoading || budgetLoading || forecastLoading || agentsLoading;
  const error = summaryError || roiError || budgetError || forecastError || agentsError;

  // Transform to dashboard-friendly format
  const dashboardData = useMemo(() => {
    if (!summary) return null;

    const totalCost = parseNumber(summary.costs.total_usd);
    const avgCostPerDebate = parseNumber(summary.costs.avg_per_debate_usd);
    const consensusRatePct = parseNumber(
      roi?.quality?.consensus_rate ?? summary.debates.consensus_rate
    );
    const avgConfidence = parseNumber(roi?.quality?.avg_confidence_score);
    const avgDurationMinutes = parseNumber(roi?.time_savings?.avg_debate_duration_minutes);
    const budgetMonthly = budget?.monthly;
    const budgetLimit = parseNumber(budgetMonthly?.limit_usd);
    const projectedBudget = parseNumber(budgetMonthly?.projected_end_spend_usd);

    return {
      // Debate metrics
      debates: {
        total: summary.debates.total,
        completed: summary.debates.completed,
        periodLabel: summary.period.type,
      },
      // Consensus metrics
      consensus: {
        rate: consensusRatePct / 100,
        avgConfidence,
        avgTimeToDecision: Math.round(avgDurationMinutes * 60),
      },
      // Cost metrics
      costs: {
        totalTokens: summary.tokens.total,
        totalApiCalls: summary.activity.api_calls,
        estimatedCost: totalCost,
        totalCost,
        avgPerDebate: avgCostPerDebate,
      },
      // Agent metrics
      agents: {
        active: agents?.count ?? 0,
        total: agents?.count ?? 0,
        topPerformer: agents?.agents?.[0]?.agent_name ?? '-',
        topAgents: agents?.agents ?? [],
      },
      // ROI metrics
      roi: roi ? {
        percentage: parseNumber(roi.roi?.roi_percentage),
        timeSavedHours: parseNumber(roi.time_savings?.estimated_hours_saved),
        costSavingsUsd: parseNumber(roi.cost?.cost_savings_usd),
        costPerDecision: parseNumber(roi.cost?.cost_per_decision_usd),
        valueGenerated: parseNumber(roi.cost?.total_manual_cost_usd),
        industryBenchmark: parseNumber(roi.benchmark?.cost_usd),
        percentile: null,
        trend: 'stable' as const,
      } : null,
      // Budget status
      budget: budget && budgetMonthly ? {
        limit: budgetLimit,
        spent: parseNumber(budgetMonthly.spent_usd),
        remaining: parseNumber(budgetMonthly.remaining_usd),
        utilization: budgetMonthly.percent_used ?? 0,
        projectedTotal: projectedBudget,
        willExceed: budgetLimit > 0 ? projectedBudget > budgetLimit : false,
        alertLevel: budget.alert_level ?? 'normal',
        daysRemaining: budgetMonthly.days_remaining ?? 0,
      } : null,
      // Forecast
      forecast: forecast?.projections?.monthly ? {
        monthlyTokens: 0,
        monthlyCost: parseNumber(forecast.projections.monthly.aragora_cost_usd),
        monthlyDebates: forecast.projections.debates_per_month ?? 0,
        growthRate: 0,
        trend: 'stable' as const,
        confidence: 0,
      } : null,
      // Metadata
      lastUpdated: new Date().toISOString(),
    };
  }, [summary, roi, budget, forecast, agents]);

  return {
    // Raw data
    summary,
    roi,
    budget,
    forecast,

    // Transformed dashboard data
    dashboardData,

    // State
    isLoading,
    error,
  };
}

// ============================================================================
// Export
// ============================================================================

export default useUsageDashboard;
