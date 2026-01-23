'use client';

import { useState, useCallback, useMemo } from 'react';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useAragoraClient } from '@/hooks/useAragoraClient';
import { useAsyncData } from '@/hooks/useAsyncData';

import {
  MetricCard,
  TrendChart,
  AgentLeaderboard,
  CostBreakdown,
  type DataPoint,
  type TimeRange,
  type AgentRankingEntry,
  type CostCategory,
} from '@/components/analytics';

/**
 * Analytics Dashboard Page (Phase 4.4)
 *
 * Comprehensive analytics visualization including:
 * - Debate Metrics: success rate, avg duration, consensus rate
 * - Agent Performance: ELO rankings, response times, usage by agent
 * - Usage Trends: debates/day, tokens/day, active users
 * - Cost Analysis: spending by model, projected costs
 */

type TabType = 'overview' | 'agents' | 'usage' | 'cost';

interface DebateOverviewData {
  time_range: string;
  total_debates: number;
  debates_this_period: number;
  debates_previous_period: number;
  growth_rate: number;
  consensus_reached: number;
  consensus_rate: number;
  avg_rounds: number;
  avg_agents_per_debate: number;
  avg_confidence: number;
}

interface DebateTrendData {
  period: string;
  total: number;
  consensus_reached: number;
  consensus_rate: number;
  avg_rounds: number;
}

interface LeaderboardResponse {
  leaderboard: AgentRankingEntry[];
  total_agents: number;
}

interface TokenUsageData {
  summary: {
    total_tokens_in: number;
    total_tokens_out: number;
    total_tokens: number;
    avg_tokens_per_day: number;
  };
  by_agent?: Record<string, string>;
  by_model?: Record<string, string>;
}

interface CostData {
  summary: {
    total_cost_usd: string;
    avg_cost_per_day: string;
    avg_cost_per_debate: string;
    total_api_calls: number;
  };
  by_provider: Record<string, { cost: string; percentage: number }>;
  by_model: Record<string, string>;
}

export default function AnalyticsPage() {
  const { config: backendConfig } = useBackend();
  // Client available for future SDK-based fetching
  const _client = useAragoraClient();
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');

  // Fetch debate overview
  const overviewFetcher = useCallback(async (): Promise<DebateOverviewData | null> => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/v1/analytics/debates/overview?time_range=${timeRange}`
      );
      if (!response.ok) return null;
      return response.json();
    } catch {
      return null;
    }
  }, [backendConfig.api, timeRange]);

  // Fetch debate trends
  const trendsFetcher = useCallback(async (): Promise<DebateTrendData[]> => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/v1/analytics/debates/trends?time_range=${timeRange}&granularity=daily`
      );
      if (!response.ok) return [];
      const data = await response.json();
      return data.data_points || [];
    } catch {
      return [];
    }
  }, [backendConfig.api, timeRange]);

  // Fetch agent leaderboard
  const leaderboardFetcher = useCallback(async (): Promise<LeaderboardResponse | null> => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/v1/analytics/agents/leaderboard?limit=20`
      );
      if (!response.ok) return null;
      return response.json();
    } catch {
      return null;
    }
  }, [backendConfig.api]);

  // Fetch token usage
  const tokenFetcher = useCallback(async (): Promise<TokenUsageData | null> => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/v1/analytics/usage/tokens?time_range=${timeRange}&org_id=default`
      );
      if (!response.ok) return null;
      return response.json();
    } catch {
      return null;
    }
  }, [backendConfig.api, timeRange]);

  // Fetch cost data
  const costFetcher = useCallback(async (): Promise<CostData | null> => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/v1/analytics/usage/costs?time_range=${timeRange}&org_id=default`
      );
      if (!response.ok) return null;
      return response.json();
    } catch {
      return null;
    }
  }, [backendConfig.api, timeRange]);

  const { data: overview, loading: overviewLoading } = useAsyncData(overviewFetcher, {
    immediate: true,
  });

  const { data: trends, loading: trendsLoading } = useAsyncData(trendsFetcher, {
    immediate: true,
  });

  const { data: leaderboard, loading: leaderboardLoading } = useAsyncData(leaderboardFetcher, {
    immediate: true,
  });

  const { data: tokenUsage, loading: tokenLoading } = useAsyncData(tokenFetcher, {
    immediate: true,
  });

  const { data: costData, loading: costLoading } = useAsyncData(costFetcher, {
    immediate: true,
  });

  // Transform trends for chart
  const debateTrendData: DataPoint[] = useMemo(() => {
    if (!trends) return [];
    return trends.map((t) => ({
      label: t.period.split('-').slice(1).join('/'), // Format: MM/DD
      value: t.total,
      date: t.period,
    }));
  }, [trends]);

  const consensusTrendData: DataPoint[] = useMemo(() => {
    if (!trends) return [];
    return trends.map((t) => ({
      label: t.period.split('-').slice(1).join('/'),
      value: t.consensus_rate,
      date: t.period,
    }));
  }, [trends]);

  // Transform cost data for breakdown
  const costCategories: CostCategory[] = useMemo(() => {
    if (!costData?.by_provider) return [];
    return Object.entries(costData.by_provider).map(([name, data]) => ({
      name,
      cost: parseFloat(data.cost),
      percentage: data.percentage,
    }));
  }, [costData]);

  const handleTimeRangeChange = (range: TimeRange) => {
    setTimeRange(range);
  };

  const tabs: { id: TabType; label: string }[] = [
    { id: 'overview', label: 'OVERVIEW' },
    { id: 'agents', label: 'AGENTS' },
    { id: 'usage', label: 'USAGE' },
    { id: 'cost', label: 'COST' },
  ];

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="container mx-auto px-4 py-6">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} ANALYTICS DASHBOARD
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Debate metrics, agent performance, usage trends, and cost analysis.
            </p>
          </div>

          {/* Tab Navigation */}
          <div className="flex flex-wrap gap-1 border-b border-acid-green/20 pb-2 mb-6">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-xs font-mono transition-colors ${
                  activeTab === tab.id
                    ? 'bg-acid-green text-bg'
                    : 'text-text-muted hover:text-acid-green'
                }`}
              >
                [{tab.label}]
              </button>
            ))}

            {/* Time Range Selector */}
            <div className="ml-auto flex gap-1">
              {(['7d', '30d', '90d'] as TimeRange[]).map((range) => (
                <button
                  key={range}
                  onClick={() => handleTimeRangeChange(range)}
                  className={`px-3 py-2 text-xs font-mono transition-colors ${
                    timeRange === range
                      ? 'bg-acid-cyan/20 text-acid-cyan border border-acid-cyan/40'
                      : 'text-text-muted hover:text-text'
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>

          {/* Tab Content */}
          <div className="space-y-6">
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <PanelErrorBoundary panelName="Overview">
                <div className="space-y-6">
                  {/* Key Metrics */}
                  <section>
                    <h2 className="text-lg font-mono text-acid-green mb-4">{'>'} KEY METRICS</h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricCard
                        title="Total Debates"
                        value={overview?.total_debates ?? '-'}
                        subtitle={`${overview?.debates_this_period ?? 0} this period`}
                        change={overview?.growth_rate}
                        changePeriod="vs prev"
                        color="green"
                        loading={overviewLoading}
                        icon="#"
                      />
                      <MetricCard
                        title="Consensus Rate"
                        value={overview ? `${overview.consensus_rate.toFixed(1)}%` : '-'}
                        subtitle={`${overview?.consensus_reached ?? 0} reached`}
                        color="cyan"
                        loading={overviewLoading}
                        icon="%"
                      />
                      <MetricCard
                        title="Avg Rounds"
                        value={overview?.avg_rounds.toFixed(1) ?? '-'}
                        subtitle="per debate"
                        color="yellow"
                        loading={overviewLoading}
                        icon="~"
                      />
                      <MetricCard
                        title="Avg Confidence"
                        value={overview ? `${(overview.avg_confidence * 100).toFixed(0)}%` : '-'}
                        subtitle="consensus confidence"
                        color="purple"
                        loading={overviewLoading}
                        icon="*"
                      />
                    </div>
                  </section>

                  {/* Debates Trend */}
                  <section>
                    <h2 className="text-lg font-mono text-acid-green mb-4">{'>'} DEBATE ACTIVITY</h2>
                    <TrendChart
                      title="Debates Over Time"
                      data={debateTrendData}
                      type="area"
                      color="green"
                      loading={trendsLoading}
                      showTimeRangeSelector={false}
                      height={250}
                    />
                  </section>

                  {/* Consensus Rate Trend */}
                  <section>
                    <TrendChart
                      title="Consensus Rate Trend"
                      data={consensusTrendData}
                      type="line"
                      color="cyan"
                      loading={trendsLoading}
                      showTimeRangeSelector={false}
                      height={200}
                      formatValue={(v) => `${v.toFixed(1)}%`}
                    />
                  </section>
                </div>
              </PanelErrorBoundary>
            )}

            {/* Agents Tab */}
            {activeTab === 'agents' && (
              <PanelErrorBoundary panelName="Agent Performance">
                <div className="space-y-6">
                  {/* Agent Stats Summary */}
                  <section>
                    <h2 className="text-lg font-mono text-acid-green mb-4">{'>'} AGENT STATISTICS</h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricCard
                        title="Total Agents"
                        value={leaderboard?.total_agents ?? '-'}
                        color="green"
                        loading={leaderboardLoading}
                        icon="#"
                      />
                      <MetricCard
                        title="Top ELO"
                        value={leaderboard?.leaderboard?.[0]?.elo.toFixed(0) ?? '-'}
                        subtitle={leaderboard?.leaderboard?.[0]?.agent_name ?? ''}
                        color="purple"
                        loading={leaderboardLoading}
                        icon="^"
                      />
                      <MetricCard
                        title="Avg Win Rate"
                        value={
                          leaderboard?.leaderboard
                            ? `${(
                                leaderboard.leaderboard.reduce((a, b) => a + b.win_rate, 0) /
                                leaderboard.leaderboard.length
                              ).toFixed(1)}%`
                            : '-'
                        }
                        color="cyan"
                        loading={leaderboardLoading}
                        icon="%"
                      />
                      <MetricCard
                        title="Total Games"
                        value={
                          leaderboard?.leaderboard
                            ? leaderboard.leaderboard.reduce((a, b) => a + b.games_played, 0)
                            : '-'
                        }
                        color="yellow"
                        loading={leaderboardLoading}
                        icon=">"
                      />
                    </div>
                  </section>

                  {/* Leaderboard */}
                  <section>
                    <AgentLeaderboard
                      agents={leaderboard?.leaderboard ?? []}
                      loading={leaderboardLoading}
                      title="AGENT RANKINGS"
                      limit={15}
                    />
                  </section>
                </div>
              </PanelErrorBoundary>
            )}

            {/* Usage Tab */}
            {activeTab === 'usage' && (
              <PanelErrorBoundary panelName="Usage Analytics">
                <div className="space-y-6">
                  {/* Token Usage Summary */}
                  <section>
                    <h2 className="text-lg font-mono text-acid-green mb-4">{'>'} TOKEN USAGE</h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricCard
                        title="Total Tokens"
                        value={
                          tokenUsage
                            ? (tokenUsage.summary.total_tokens / 1000000).toFixed(2) + 'M'
                            : '-'
                        }
                        color="green"
                        loading={tokenLoading}
                        icon="#"
                      />
                      <MetricCard
                        title="Input Tokens"
                        value={
                          tokenUsage
                            ? (tokenUsage.summary.total_tokens_in / 1000000).toFixed(2) + 'M'
                            : '-'
                        }
                        color="cyan"
                        loading={tokenLoading}
                        icon=">"
                      />
                      <MetricCard
                        title="Output Tokens"
                        value={
                          tokenUsage
                            ? (tokenUsage.summary.total_tokens_out / 1000000).toFixed(2) + 'M'
                            : '-'
                        }
                        color="yellow"
                        loading={tokenLoading}
                        icon="<"
                      />
                      <MetricCard
                        title="Avg/Day"
                        value={
                          tokenUsage
                            ? (tokenUsage.summary.avg_tokens_per_day / 1000).toFixed(1) + 'K'
                            : '-'
                        }
                        color="purple"
                        loading={tokenLoading}
                        icon="~"
                      />
                    </div>
                  </section>

                  {/* Usage by Model */}
                  {tokenUsage?.by_model && Object.keys(tokenUsage.by_model).length > 0 && (
                    <section>
                      <div className="card p-4">
                        <h3 className="font-mono text-sm text-acid-green mb-4">{'>'} USAGE BY MODEL</h3>
                        <div className="space-y-2">
                          {Object.entries(tokenUsage.by_model).map(([model, usage]) => (
                            <div
                              key={model}
                              className="flex items-center justify-between p-2 border border-acid-green/20 rounded"
                            >
                              <span className="font-mono text-sm text-acid-cyan">{model}</span>
                              <span className="font-mono text-sm text-text">{usage}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </section>
                  )}
                </div>
              </PanelErrorBoundary>
            )}

            {/* Cost Tab */}
            {activeTab === 'cost' && (
              <PanelErrorBoundary panelName="Cost Analysis">
                <div className="space-y-6">
                  {/* Cost Summary */}
                  <section>
                    <h2 className="text-lg font-mono text-acid-green mb-4">{'>'} COST SUMMARY</h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricCard
                        title="Total Cost"
                        value={costData ? `$${costData.summary.total_cost_usd}` : '-'}
                        subtitle={`${timeRange} period`}
                        color="green"
                        loading={costLoading}
                        icon="$"
                      />
                      <MetricCard
                        title="Avg/Day"
                        value={costData ? `$${costData.summary.avg_cost_per_day}` : '-'}
                        color="cyan"
                        loading={costLoading}
                        icon="~"
                      />
                      <MetricCard
                        title="Avg/Debate"
                        value={costData ? `$${costData.summary.avg_cost_per_debate}` : '-'}
                        color="yellow"
                        loading={costLoading}
                        icon="#"
                      />
                      <MetricCard
                        title="API Calls"
                        value={costData?.summary.total_api_calls ?? '-'}
                        color="purple"
                        loading={costLoading}
                        icon=">"
                      />
                    </div>
                  </section>

                  {/* Cost Breakdown */}
                  <section>
                    <CostBreakdown
                      data={costCategories}
                      totalCost={parseFloat(costData?.summary.total_cost_usd ?? '0')}
                      title="COST BY PROVIDER"
                      subtitle={`${timeRange} period`}
                      loading={costLoading}
                      showTokens={false}
                    />
                  </section>

                  {/* Cost by Model */}
                  {costData?.by_model && Object.keys(costData.by_model).length > 0 && (
                    <section>
                      <div className="card p-4">
                        <h3 className="font-mono text-sm text-acid-green mb-4">{'>'} COST BY MODEL</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {Object.entries(costData.by_model)
                            .sort(([, a], [, b]) => parseFloat(b) - parseFloat(a))
                            .map(([model, cost]) => (
                              <div
                                key={model}
                                className="flex items-center justify-between p-3 border border-acid-green/20 rounded hover:bg-acid-green/5 transition-colors"
                              >
                                <span className="font-mono text-sm text-acid-cyan">{model}</span>
                                <span className="font-mono text-sm text-acid-green">${cost}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </section>
                  )}
                </div>
              </PanelErrorBoundary>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">{'='.repeat(40)}</div>
          <p className="text-text-muted">{'>'} ARAGORA // ANALYTICS DASHBOARD</p>
        </footer>
      </main>
    </>
  );
}
