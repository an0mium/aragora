'use client';

import { useState } from 'react';
import { CostBreakdownChart } from './CostBreakdownChart';
import { BudgetAlerts } from './BudgetAlerts';
import { UsageTimeline } from './UsageTimeline';
import { OptimizationRecommendations } from './OptimizationRecommendations';
import { EfficiencyMetrics } from './EfficiencyMetrics';
import { BudgetForecast } from './BudgetForecast';
import { useCosts, type TimeRange } from '@/hooks/useCosts';

type TabView = 'overview' | 'recommendations' | 'efficiency' | 'forecast';

export function CostDashboard() {
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');
  const [activeTab, setActiveTab] = useState<TabView>('overview');

  const {
    costData,
    isLoading,
    error,
    dismissAlert,
    refresh,
  } = useCosts(timeRange);

  if (isLoading || !costData) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="h-8 bg-[var(--surface)] rounded w-1/3" />
        <div className="grid grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-24 bg-[var(--surface)] rounded" />
          ))}
        </div>
        <div className="h-64 bg-[var(--surface)] rounded" />
      </div>
    );
  }

  const budgetUsagePercent = (costData.totalCost / costData.budget) * 100;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-mono text-[var(--acid-green)]">
            {'>'} COST VISIBILITY
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            Monitor and optimize your AI spend
          </p>
        </div>
        <div className="flex items-center gap-2">
          <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
          <button
            onClick={refresh}
            disabled={isLoading}
            className="px-3 py-2 text-sm font-mono text-[var(--text-muted)] hover:text-[var(--text)] border border-[var(--border)] rounded hover:border-[var(--acid-green)]/30 transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-[var(--border)]">
        {(['overview', 'recommendations', 'efficiency', 'forecast'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-mono transition-colors ${
              activeTab === tab
                ? 'text-[var(--acid-green)] border-b-2 border-[var(--acid-green)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text)]'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <SummaryCard
              label="Total Cost"
              value={`$${costData.totalCost.toFixed(2)}`}
              subtext={`of $${costData.budget.toFixed(2)} budget`}
              color="text-[var(--acid-green)]"
              progress={budgetUsagePercent}
            />
            <SummaryCard
              label="Tokens Used"
              value={formatNumber(costData.tokensUsed)}
              subtext="input + output"
              color="text-[var(--acid-cyan)]"
            />
            <SummaryCard
              label="API Calls"
              value={formatNumber(costData.apiCalls)}
              subtext="total requests"
              color="text-purple-400"
            />
            <SummaryCard
              label="Avg Cost/Call"
              value={`$${(costData.totalCost / costData.apiCalls * 1000).toFixed(4)}`}
              subtext="per 1K calls"
              color="text-yellow-400"
            />
          </div>

          {/* Budget Progress */}
          <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-mono text-[var(--acid-green)]">
                {'>'} BUDGET PROGRESS
              </h3>
              <span className={`text-sm font-mono ${
                budgetUsagePercent >= 90 ? 'text-red-400' :
                budgetUsagePercent >= 75 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {budgetUsagePercent.toFixed(1)}% used
              </span>
            </div>
            <div className="h-4 bg-[var(--bg)] rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  budgetUsagePercent >= 90 ? 'bg-red-500' :
                  budgetUsagePercent >= 75 ? 'bg-yellow-500' : 'bg-[var(--acid-green)]'
                }`}
                style={{ width: `${Math.min(budgetUsagePercent, 100)}%` }}
              />
            </div>
            <div className="flex justify-between mt-2 text-xs text-[var(--text-muted)]">
              <span>$0</span>
              <span>${(costData.budget * 0.5).toFixed(0)}</span>
              <span>${costData.budget.toFixed(0)}</span>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded p-4 text-red-400 text-sm font-mono">
              Error loading cost data: {error.message}
            </div>
          )}

          {/* Alerts */}
          {costData.alerts.length > 0 && (
            <BudgetAlerts alerts={costData.alerts} onDismiss={dismissAlert} />
          )}

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Cost by Provider */}
            <CostBreakdownChart
              title="Cost by Provider"
              data={costData.costByProvider}
              colors={['#00ff9d', '#00d4ff', '#ff6b6b', '#ffd93d']}
            />

            {/* Cost by Feature */}
            <CostBreakdownChart
              title="Cost by Feature"
              data={costData.costByFeature}
              colors={['#a855f7', '#3b82f6', '#22c55e', '#f59e0b']}
            />
          </div>

          {/* Usage Timeline */}
          <UsageTimeline data={costData.dailyCosts} />

          {/* Last Updated */}
          <div className="text-xs text-[var(--text-muted)] text-center">
            Last updated: {new Date(costData.lastUpdated).toLocaleString()}
          </div>
        </>
      )}

      {activeTab === 'recommendations' && (
        <OptimizationRecommendations />
      )}

      {activeTab === 'efficiency' && (
        <EfficiencyMetrics timeRange={timeRange} />
      )}

      {activeTab === 'forecast' && (
        <BudgetForecast />
      )}
    </div>
  );
}

interface TimeRangeSelectorProps {
  value: TimeRange;
  onChange: (value: TimeRange) => void;
}

function TimeRangeSelector({ value, onChange }: TimeRangeSelectorProps) {
  const ranges: Array<{ id: TimeRange; label: string }> = [
    { id: '24h', label: '24h' },
    { id: '7d', label: '7d' },
    { id: '30d', label: '30d' },
    { id: '90d', label: '90d' },
  ];

  return (
    <div className="flex border border-[var(--border)] rounded overflow-hidden">
      {ranges.map(range => (
        <button
          key={range.id}
          onClick={() => onChange(range.id)}
          className={`px-3 py-1.5 text-xs font-mono transition-colors ${
            value === range.id
              ? 'bg-[var(--acid-green)] text-[var(--bg)]'
              : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg)]'
          }`}
        >
          {range.label}
        </button>
      ))}
    </div>
  );
}

interface SummaryCardProps {
  label: string;
  value: string;
  subtext: string;
  color: string;
  progress?: number;
}

function SummaryCard({ label, value, subtext, color, progress }: SummaryCardProps) {
  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
      <div className={`text-2xl font-mono font-bold ${color}`}>{value}</div>
      <div className="text-xs text-[var(--text-muted)] mt-1">{label}</div>
      <div className="text-xs text-[var(--text-muted)] opacity-70">{subtext}</div>
      {progress !== undefined && (
        <div className="mt-2 h-1 bg-[var(--bg)] rounded overflow-hidden">
          <div
            className={`h-full ${color.replace('text-', 'bg-').replace('[var(--acid-green)]', '[var(--acid-green)]')}`}
            style={{ width: `${Math.min(progress, 100)}%`, backgroundColor: 'var(--acid-green)' }}
          />
        </div>
      )}
    </div>
  );
}

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(1)}M`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}K`;
  }
  return num.toString();
}

export default CostDashboard;
