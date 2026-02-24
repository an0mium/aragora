'use client';

import { useSWRFetch, type UseSWRFetchOptions } from './useSWRFetch';

// ============================================================================
// Types — match the backend SpendAnalytics dataclasses
// ============================================================================

export interface SpendDataPoint {
  date: string;
  cost_usd: number;
}

export interface SpendTrend {
  workspace_id: string;
  period: string;
  points: SpendDataPoint[];
  total_usd: number;
  avg_daily_usd: number;
}

export interface SpendForecast {
  workspace_id: string;
  forecast_days: number;
  projected_total_usd: number;
  projected_daily_avg_usd: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  confidence: number;
}

export interface SpendAnomaly {
  date: string;
  actual_usd: number;
  expected_usd: number;
  z_score: number;
  severity: 'warning' | 'critical';
  description: string;
}

/** Shape returned by GET /api/v1/spend/analytics */
export interface SpendAnalyticsData {
  trend: SpendTrend;
  by_provider: Record<string, number>;
  by_agent: Record<string, number>;
  forecast: SpendForecast;
  anomalies: SpendAnomaly[];
}

export type SpendPeriod = '7d' | '14d' | '30d' | '60d' | '90d';

// ============================================================================
// Hooks
// ============================================================================

/**
 * Fetch the full spend analytics summary in a single request.
 * Hits GET /api/v1/spend/analytics?period=...&workspace_id=...
 */
export function useSpendAnalytics(
  period: SpendPeriod = '30d',
  workspaceId: string = 'default',
  options?: UseSWRFetchOptions<{ data: SpendAnalyticsData }>,
) {
  const result = useSWRFetch<{ data: SpendAnalyticsData }>(
    `/api/v1/spend/analytics?period=${period}&workspace_id=${workspaceId}`,
    {
      refreshInterval: 60000,
      ...options,
    },
  );

  return {
    ...result,
    analytics: result.data?.data ?? null,
  };
}

/**
 * Fetch only the spend trend (daily timeline).
 */
export function useSpendTrend(
  period: SpendPeriod = '30d',
  workspaceId: string = 'default',
  options?: UseSWRFetchOptions<{ data: SpendTrend }>,
) {
  const result = useSWRFetch<{ data: SpendTrend }>(
    `/api/v1/spend/analytics/trend?period=${period}&workspace_id=${workspaceId}`,
    {
      refreshInterval: 60000,
      ...options,
    },
  );

  return {
    ...result,
    trend: result.data?.data ?? null,
  };
}

/**
 * Fetch the cost forecast.
 */
export function useSpendForecast(
  workspaceId: string = 'default',
  days: number = 30,
  options?: UseSWRFetchOptions<{ data: SpendForecast }>,
) {
  const result = useSWRFetch<{ data: SpendForecast }>(
    `/api/v1/spend/analytics/forecast?workspace_id=${workspaceId}&days=${days}`,
    {
      refreshInterval: 300000,
      ...options,
    },
  );

  return {
    ...result,
    forecast: result.data?.data ?? null,
  };
}

/**
 * Fetch spend anomalies.
 */
export function useSpendAnomalies(
  period: SpendPeriod = '30d',
  workspaceId: string = 'default',
  options?: UseSWRFetchOptions<{ data: { anomalies: SpendAnomaly[] } }>,
) {
  const result = useSWRFetch<{ data: { anomalies: SpendAnomaly[] } }>(
    `/api/v1/spend/analytics/anomalies?period=${period}&workspace_id=${workspaceId}`,
    {
      refreshInterval: 300000,
      ...options,
    },
  );

  return {
    ...result,
    anomalies: result.data?.data?.anomalies ?? [],
  };
}
