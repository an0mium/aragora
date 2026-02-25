'use client';

import { useState } from 'react';
import { KPICard, KPIGrid, KPIMiniCard } from './KPICards';
import { useUsageDashboard, type TimeRange } from '@/hooks/useUsageDashboard';

interface ExecutiveSummaryProps {
  refreshInterval?: number; // ms (now handled by hook)
}

export function ExecutiveSummary({
  refreshInterval = 30000,
}: ExecutiveSummaryProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const { dashboardData, isLoading, error } = useUsageDashboard(timeRange, { refreshInterval });

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
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
          title="Debates Today"
          value={dashboardData?.debates.today ?? '-'}
          subtitle={`${dashboardData?.debates.week ?? 0} this week`}
          change={dashboardData ? { value: 12, direction: 'up', period: 'yesterday' } : undefined}
          color="green"
          loading={isLoading}
          icon=""
        />
        <KPICard
          title="Consensus Rate"
          value={dashboardData ? `${(dashboardData.consensus.rate * 100).toFixed(0)}%` : '-'}
          subtitle={`${dashboardData ? (dashboardData.consensus.avgConfidence * 100).toFixed(0) : 0}% avg confidence`}
          change={dashboardData ? { value: 3, direction: 'up', period: 'last week' } : undefined}
          color="cyan"
          loading={isLoading}
          icon=""
        />
        <KPICard
          title="Avg Decision Time"
          value={dashboardData ? formatDuration(dashboardData.consensus.avgTimeToDecision) : '-'}
          subtitle="time to consensus"
          change={dashboardData ? { value: 8, direction: 'down', period: 'last week' } : undefined}
          color="yellow"
          loading={isLoading}
          icon=""
        />
        <KPICard
          title="Est. Cost Today"
          value={dashboardData ? `$${dashboardData.costs.estimatedCost.toFixed(2)}` : '-'}
          subtitle={`${formatNumber(dashboardData?.costs.todayTokens ?? 0)} tokens`}
          color="purple"
          loading={isLoading}
          icon=""
        />
      </KPIGrid>

      {/* Secondary Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Agent Health */}
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <h3 className="text-sm font-mono text-[var(--acid-cyan)] mb-3 flex items-center gap-2">
            <span></span> AGENT HEALTH
          </h3>
          <div className="space-y-1">
            <KPIMiniCard
              label="Active Agents"
              value={`${dashboardData?.agents.active ?? 0}/${dashboardData?.agents.total ?? 0}`}
              color="green"
            />
            <KPIMiniCard
              label="Avg Uptime"
              value={`${dashboardData?.agents.avgUptime ?? 0}%`}
              color="cyan"
            />
            <KPIMiniCard
              label="Top Performer"
              value={dashboardData?.agents.topPerformer ?? '-'}
              color="yellow"
            />
          </div>
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
