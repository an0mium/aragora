'use client';

import { useCallback, useMemo } from 'react';
import { useSWRFetch, invalidateCache, type UseSWRFetchOptions } from './useSWRFetch';
import { useApi } from './useApi';

// ============================================================================
// Types
// ============================================================================

export interface CostSummary {
  total_cost_usd: number;
  budget_usd: number;
  tokens_in: number;
  tokens_out: number;
  api_calls: number;
  period_start: string;
  period_end: string;
}

export interface CostBreakdownItem {
  name: string;
  cost: number;
  percentage: number;
  tokens?: number;
  calls?: number;
}

export interface CostBreakdown {
  by_provider: CostBreakdownItem[];
  by_feature: CostBreakdownItem[];
  by_model?: CostBreakdownItem[];
}

export interface DailyCost {
  date: string;
  cost: number;
  tokens: number;
  calls?: number;
}

export interface CostTimeline {
  data_points: DailyCost[];
  total_cost: number;
  average_daily_cost: number;
}

export interface CostAlert {
  id: string;
  type: 'budget_warning' | 'budget_exceeded' | 'spike_detected' | 'forecast_warning';
  message: string;
  severity: 'info' | 'warning' | 'critical';
  timestamp: string;
  dismissed?: boolean;
}

export interface Budget {
  workspace_id?: string;
  monthly_limit_usd: number;
  daily_limit_usd?: number;
  alert_threshold_percent?: number;
}

export interface CostRecommendation {
  id: string;
  type: 'model_switch' | 'caching' | 'batching' | 'provider_switch';
  title: string;
  description: string;
  potential_savings_usd: number;
  effort: 'low' | 'medium' | 'high';
  applied?: boolean;
}

export interface EfficiencyMetrics {
  cost_per_1k_tokens: number;
  cost_per_call: number;
  cache_hit_rate: number;
  avg_tokens_per_call: number;
  efficiency_score: number;
}

export interface CostForecast {
  projected_monthly_cost: number;
  projected_end_of_month: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  confidence: number;
  recommended_tier?: string;
}

export type TimeRange = '24h' | '7d' | '30d' | '90d';

// ============================================================================
// Hooks
// ============================================================================

/**
 * Hook for fetching cost summary data
 */
export function useCostSummary(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: CostSummary }>
) {
  const result = useSWRFetch<{ data: CostSummary }>(
    `/api/v1/costs?range=${timeRange}`,
    {
      refreshInterval: 60000, // Refresh every minute
      ...options,
    }
  );

  return {
    ...result,
    summary: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching cost breakdown by provider/feature/model.
 * Note: This is distinct from the useCostBreakdown in useUsageDashboard
 * which fetches from /api/v1/usage/cost-breakdown (by agent/model).
 */
export function useCostsBreakdown(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: CostBreakdown }>
) {
  const result = useSWRFetch<{ data: CostBreakdown }>(
    `/api/v1/costs/breakdown?range=${timeRange}`,
    {
      refreshInterval: 60000,
      ...options,
    }
  );

  return {
    ...result,
    breakdown: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching cost timeline data
 */
export function useCostTimeline(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: CostTimeline }>
) {
  const result = useSWRFetch<{ data: CostTimeline }>(
    `/api/v1/costs/timeline?range=${timeRange}`,
    {
      refreshInterval: 60000,
      ...options,
    }
  );

  return {
    ...result,
    timeline: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching budget alerts
 */
export function useCostAlerts(options?: UseSWRFetchOptions<{ data: { alerts: CostAlert[] } }>) {
  const result = useSWRFetch<{ data: { alerts: CostAlert[] } }>(
    '/api/v1/costs/alerts',
    {
      refreshInterval: 30000, // Check alerts more frequently
      ...options,
    }
  );

  return {
    ...result,
    alerts: result.data?.data?.alerts ?? [],
  };
}

/**
 * Hook for fetching optimization recommendations
 */
export function useCostRecommendations(
  options?: UseSWRFetchOptions<{ data: { recommendations: CostRecommendation[] } }>
) {
  const result = useSWRFetch<{ data: { recommendations: CostRecommendation[] } }>(
    '/api/v1/costs/recommendations',
    {
      refreshInterval: 300000, // Refresh every 5 minutes
      ...options,
    }
  );

  return {
    ...result,
    recommendations: result.data?.data?.recommendations ?? [],
  };
}

/**
 * Hook for fetching efficiency metrics
 */
export function useCostEfficiency(
  timeRange: TimeRange = '30d',
  options?: UseSWRFetchOptions<{ data: EfficiencyMetrics }>
) {
  const result = useSWRFetch<{ data: EfficiencyMetrics }>(
    `/api/v1/costs/efficiency?range=${timeRange}`,
    {
      refreshInterval: 60000,
      ...options,
    }
  );

  return {
    ...result,
    efficiency: result.data?.data ?? null,
  };
}

/**
 * Hook for fetching cost forecast
 */
export function useCostForecast(options?: UseSWRFetchOptions<{ data: CostForecast }>) {
  const result = useSWRFetch<{ data: CostForecast }>(
    '/api/v1/costs/forecast',
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
 * Unified hook for all cost data with combined loading/error state
 */
export function useCosts(timeRange: TimeRange = '30d') {
  const { summary, isLoading: summaryLoading, error: summaryError, mutate: mutateSummary } = useCostSummary(timeRange);
  const { breakdown, isLoading: breakdownLoading, error: breakdownError } = useCostsBreakdown(timeRange);
  const { timeline, isLoading: timelineLoading, error: timelineError } = useCostTimeline(timeRange);
  const { alerts, isLoading: alertsLoading, error: alertsError, mutate: mutateAlerts } = useCostAlerts();

  const api = useApi();

  const isLoading = summaryLoading || breakdownLoading || timelineLoading || alertsLoading;
  const error = summaryError || breakdownError || timelineError || alertsError;

  // Transform data to match the CostDashboard expected format
  const costData = useMemo(() => {
    if (!summary) return null;

    return {
      totalCost: summary.total_cost_usd,
      budget: summary.budget_usd,
      tokensUsed: summary.tokens_in + summary.tokens_out,
      apiCalls: summary.api_calls,
      lastUpdated: new Date().toISOString(),
      costByProvider: breakdown?.by_provider?.map(item => ({
        name: item.name,
        cost: item.cost,
        percentage: item.percentage,
      })) ?? [],
      costByFeature: breakdown?.by_feature?.map(item => ({
        name: item.name,
        cost: item.cost,
        percentage: item.percentage,
      })) ?? [],
      dailyCosts: timeline?.data_points?.map(point => ({
        date: point.date,
        cost: point.cost,
        tokens: point.tokens,
      })) ?? [],
      alerts: alerts.map(alert => ({
        id: alert.id,
        type: alert.type,
        message: alert.message,
        severity: alert.severity,
        timestamp: alert.timestamp,
      })),
    };
  }, [summary, breakdown, timeline, alerts]);

  // Set budget
  const setBudget = useCallback(async (budget: Budget) => {
    await api.post('/api/v1/costs/budget', budget);
    // Invalidate and refetch
    invalidateCache('/api/v1/costs');
    mutateSummary();
  }, [api, mutateSummary]);

  // Dismiss alert
  const dismissAlert = useCallback(async (alertId: string) => {
    await api.post(`/api/v1/costs/alerts/${alertId}/dismiss`);
    mutateAlerts();
  }, [api, mutateAlerts]);

  // Apply recommendation
  const applyRecommendation = useCallback(async (recommendationId: string) => {
    await api.post(`/api/v1/costs/recommendations/${recommendationId}/apply`);
    invalidateCache('/api/v1/costs/recommendations');
  }, [api]);

  // Dismiss recommendation
  const dismissRecommendation = useCallback(async (recommendationId: string) => {
    await api.post(`/api/v1/costs/recommendations/${recommendationId}/dismiss`);
    invalidateCache('/api/v1/costs/recommendations');
  }, [api]);

  // Refresh all cost data
  const refresh = useCallback(() => {
    invalidateCache('/api/v1/costs');
    invalidateCache('/api/v1/costs/breakdown');
    invalidateCache('/api/v1/costs/timeline');
    invalidateCache('/api/v1/costs/alerts');
  }, []);

  return {
    // Data
    costData,
    summary,
    breakdown,
    timeline,
    alerts,

    // State
    isLoading,
    error,

    // Actions
    setBudget,
    dismissAlert,
    applyRecommendation,
    dismissRecommendation,
    refresh,
  };
}

// ============================================================================
// Default export
// ============================================================================

export default useCosts;
