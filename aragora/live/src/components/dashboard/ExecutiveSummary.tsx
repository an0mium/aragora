'use client';

import { useMemo, useState } from 'react';
import { KPICard, KPIGrid, KPIMiniCard } from './KPICards';
import { useUsageDashboard, type TimeRange } from '@/hooks/useUsageDashboard';
import { useSWRFetch } from '@/hooks/useSWRFetch';

interface ExecutiveSummaryProps {
  refreshInterval?: number; // ms (now handled by hook)
}

interface LeaderboardAgent {
  name?: string;
  agent_name?: string;
  elo?: number;
  win_rate?: number;
}

interface LeaderboardResponse {
  agents?: LeaderboardAgent[];
  rankings?: LeaderboardAgent[];
}

interface TopAgent {
  name: string;
  elo: number | null;
  winRate: number | null;
}

export function ExecutiveSummary({
  refreshInterval = 30000,
}: ExecutiveSummaryProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const { dashboardData, isLoading, error } = useUsageDashboard(timeRange, { refreshInterval });
  const {
    data: leaderboardData,
    error: leaderboardError,
    isLoading: leaderboardLoading,
  } = useSWRFetch<LeaderboardResponse>('/api/leaderboard?limit=3', { refreshInterval });

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const formatCurrency = (amount: number): string => {
    if (amount >= 1000000) return `$${(amount / 1000000).toFixed(1)}M`;
    if (amount >= 1000) return `$${(amount / 1000).toFixed(1)}K`;
    return `$${amount.toFixed(2)}`;
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const _getTimeRangeLabel = (range: TimeRange): string => {
    switch (range) {
      case '24h':
        return '24 Hours';
      case '7d':
        return '7 Days';
      case '30d':
        return '30 Days';
      case '90d':
        return '90 Days';
      default:
        return range;
    }
  };

  const normalizePercent = (value: number | null | undefined): string => {
    if (value == null) return '-';
    const percent = value <= 1 ? value * 100 : value;
    return `${percent.toFixed(0)}%`;
  };

  const topAgents = useMemo<TopAgent[]>(() => {
    const entries = leaderboardData?.agents ?? leaderboardData?.rankings ?? [];
    return entries.slice(0, 3).map((agent) => ({
      name: agent.name ?? agent.agent_name ?? 'Unknown agent',
      elo: typeof agent.elo === 'number' ? agent.elo : null,
      winRate: typeof agent.win_rate === 'number' ? agent.win_rate : null,
    }));
  }, [leaderboardData]);

  const topAgent = topAgents[0] ?? null;

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 p-4 text-red-400 font-mono text-sm">
        Error loading dashboard: {error.message}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Time Range Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-mono text-[var(--acid-green)]">
          {'>'} EXECUTIVE SUMMARY
        </h2>
        <div className="flex items-center gap-4">
          {/* Time Range Selector */}
          <div className="flex items-center gap-2">
            {(['24h', '7d', '30d', '90d'] as TimeRange[]).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 text-xs font-mono border transition-colors ${
                  timeRange === range
                    ? 'bg-[var(--acid-green)]/20 text-[var(--acid-green)] border-[var(--acid-green)]/50'
                    : 'bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)] hover:border-[var(--acid-green)]/30'
                }`}
              >
                {range.toUpperCase()}
              </button>
            ))}
          </div>
          {dashboardData?.lastUpdated && (
            <span className="text-xs font-mono text-[var(--text-muted)]">
              Updated: {formatTime(new Date(dashboardData.lastUpdated))}
            </span>
          )}
        </div>
      </div>

      {/* Primary KPIs */}
      <KPIGrid columns={4}>
        <KPICard
          title="Total Debates"
          value={dashboardData?.debates.total ?? '-'}
          subtitle={`${dashboardData?.debates.completed ?? 0} completed`}
          color="green"
          loading={isLoading}
          icon=""
        />
        <KPICard
          title="Avg Confidence"
          value={dashboardData ? normalizePercent(dashboardData.consensus.avgConfidence) : '-'}
          subtitle={
            dashboardData
              ? `${normalizePercent(dashboardData.consensus.rate)} reached consensus`
              : undefined
          }
          color="cyan"
          loading={isLoading}
          icon=""
        />
        <KPICard
          title="Top Agent"
          value={topAgent?.name ?? '-'}
          subtitle={
            topAgent
              ? `${topAgent.elo !== null ? Math.round(topAgent.elo) : '-'} ELO | ${normalizePercent(topAgent.winRate)}`
              : 'Leaderboard unavailable'
          }
          color="yellow"
          loading={isLoading || leaderboardLoading}
          icon=""
        />
        <KPICard
          title="Total Spend"
          value={dashboardData ? formatCurrency(dashboardData.costs.totalCost) : '-'}
          subtitle={_getTimeRangeLabel(timeRange)}
          color="purple"
          loading={isLoading}
          icon=""
        />
      </KPIGrid>

      {/* Secondary Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Top agents */}
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-mono text-[var(--acid-cyan)] flex items-center gap-2">
              <span></span> TOP AGENTS
            </h3>
            <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
              {dashboardData?.agents.active ?? 0} active
            </span>
          </div>
          {leaderboardLoading ? (
            <div className="animate-pulse space-y-2">
              <div className="h-8 bg-[var(--border)] rounded" />
              <div className="h-8 bg-[var(--border)] rounded" />
              <div className="h-8 bg-[var(--border)] rounded" />
            </div>
          ) : topAgents.length > 0 ? (
            <div className="space-y-1">
              {topAgents.map((agent, index) => (
                <KPIMiniCard
                  key={agent.name}
                  label={`${index + 1}. ${agent.name}`}
                  value={`${agent.elo !== null ? Math.round(agent.elo) : '-'} ELO | ${normalizePercent(agent.winRate)}`}
                  color={index === 0 ? 'green' : index === 1 ? 'cyan' : 'yellow'}
                />
              ))}
            </div>
          ) : (
            <div className="text-xs font-mono text-[var(--text-muted)] py-2">
              {leaderboardError ? 'Leaderboard unavailable' : 'No agent rankings yet'}
            </div>
          )}
        </div>

        {/* ROI Summary */}
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <h3 className="text-sm font-mono text-[var(--acid-cyan)] mb-3 flex items-center gap-2">
            <span>$</span> ROI SUMMARY
          </h3>
          <div className="space-y-1">
            <KPIMiniCard
              label="ROI"
              value={dashboardData?.roi ? `${dashboardData.roi.percentage.toFixed(0)}%` : '-'}
              color="green"
            />
            <KPIMiniCard
              label="Time Saved"
              value={dashboardData?.roi ? `${dashboardData.roi.timeSavedHours.toFixed(0)} hrs` : '-'}
              color="cyan"
            />
            <KPIMiniCard
              label="Cost Savings"
              value={dashboardData?.roi ? `$${formatNumber(dashboardData.roi.costSavingsUsd)}` : '-'}
              color="yellow"
            />
          </div>
        </div>

        {/* Budget Status */}
        <div className={`bg-[var(--surface)] border p-4 ${
          dashboardData?.budget?.alertLevel === 'critical'
            ? 'border-red-500/50'
            : dashboardData?.budget?.alertLevel === 'warning'
            ? 'border-yellow-500/50'
            : 'border-[var(--border)]'
        }`}>
          <h3 className="text-sm font-mono text-[var(--acid-cyan)] mb-3 flex items-center gap-2">
            <span></span> BUDGET STATUS
            {dashboardData?.budget?.alertLevel && dashboardData.budget.alertLevel !== 'normal' && (
              <span className={`ml-auto px-2 py-0.5 text-xs uppercase ${
                dashboardData.budget.alertLevel === 'critical'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                {dashboardData.budget.alertLevel}
              </span>
            )}
          </h3>
          <div className="space-y-1">
            <KPIMiniCard
              label="Utilization"
              value={dashboardData?.budget ? `${dashboardData.budget.utilization.toFixed(0)}%` : '-'}
              color={
                dashboardData?.budget?.alertLevel === 'critical' ? 'red' :
                dashboardData?.budget?.alertLevel === 'warning' ? 'yellow' : 'green'
              }
            />
            <KPIMiniCard
              label="Remaining"
              value={dashboardData?.budget ? `$${formatNumber(dashboardData.budget.remaining)}` : '-'}
              color="cyan"
            />
            <KPIMiniCard
              label="Days Left"
              value={dashboardData?.budget?.daysRemaining ?? '-'}
              color="yellow"
            />
          </div>
        </div>
      </div>

      {/* Forecast Banner */}
      {dashboardData?.forecast && (
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h3 className="text-sm font-mono text-[var(--acid-cyan)]">
                MONTHLY FORECAST
              </h3>
              <span className={`text-xs font-mono px-2 py-0.5 ${
                dashboardData.forecast.trend === 'increasing'
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : dashboardData.forecast.trend === 'decreasing'
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-[var(--border)] text-[var(--text-muted)]'
              }`}>
                {dashboardData.forecast.trend.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center gap-6 text-sm font-mono">
              <div>
                <span className="text-[var(--text-muted)]">Debates: </span>
                <span className="text-[var(--acid-green)]">{formatNumber(dashboardData.forecast.monthlyDebates)}</span>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">Tokens: </span>
                <span className="text-[var(--acid-cyan)]">{formatNumber(dashboardData.forecast.monthlyTokens)}</span>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">Cost: </span>
                <span className="text-yellow-400">${formatNumber(dashboardData.forecast.monthlyCost)}</span>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">Growth: </span>
                <span className={dashboardData.forecast.growthRate >= 0 ? 'text-yellow-400' : 'text-green-400'}>
                  {dashboardData.forecast.growthRate >= 0 ? '+' : ''}{dashboardData.forecast.growthRate.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="flex items-center gap-2 pt-2">
        <span className="text-xs font-mono text-[var(--text-muted)]">Quick actions:</span>
        <a
          href="/arena"
          className="px-3 py-1 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
        >
          NEW DEBATE
        </a>
        <a
          href="/debates/provenance"
          className="px-3 py-1 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
        >
          AUDIT TRAIL
        </a>
        <a
          href="/control-plane"
          className="px-3 py-1 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
        >
          DASHBOARD
        </a>
        <a
          href="/usage"
          className="px-3 py-1 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
        >
          DETAILED USAGE
        </a>
      </div>
    </div>
  );
}

export default ExecutiveSummary;
