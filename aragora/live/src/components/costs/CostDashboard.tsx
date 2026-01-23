'use client';

import { useState, useEffect, useCallback } from 'react';
import { CostBreakdownChart } from './CostBreakdownChart';
import { BudgetAlerts } from './BudgetAlerts';
import { UsageTimeline } from './UsageTimeline';
import { useAuth } from '@/context/AuthContext';

type TimeRange = '24h' | '7d' | '30d' | '90d';

interface CostData {
  totalCost: number;
  budget: number;
  tokensUsed: number;
  apiCalls: number;
  lastUpdated: string;
  costByProvider: Array<{ name: string; cost: number; percentage: number }>;
  costByFeature: Array<{ name: string; cost: number; percentage: number }>;
  dailyCosts: Array<{ date: string; cost: number; tokens: number }>;
  alerts: Array<{ id: string; type: string; message: string; severity: string; timestamp: string }>;
}

const MOCK_COST_DATA: CostData = {
  totalCost: 127.45,
  budget: 500.00,
  tokensUsed: 4250000,
  apiCalls: 12847,
  lastUpdated: new Date().toISOString(),
  costByProvider: [
    { name: 'Anthropic', cost: 78.50, percentage: 61.6 },
    { name: 'OpenAI', cost: 35.20, percentage: 27.6 },
    { name: 'Mistral', cost: 8.75, percentage: 6.9 },
    { name: 'OpenRouter', cost: 5.00, percentage: 3.9 },
  ],
  costByFeature: [
    { name: 'Debates', cost: 55.00, percentage: 43.2 },
    { name: 'Email Triage', cost: 32.50, percentage: 25.5 },
    { name: 'Code Review', cost: 22.80, percentage: 17.9 },
    { name: 'Knowledge Work', cost: 17.15, percentage: 13.4 },
  ],
  dailyCosts: [
    { date: '2025-01-15', cost: 15.20, tokens: 380000 },
    { date: '2025-01-16', cost: 18.45, tokens: 461250 },
    { date: '2025-01-17', cost: 12.30, tokens: 307500 },
    { date: '2025-01-18', cost: 21.00, tokens: 525000 },
    { date: '2025-01-19', cost: 16.75, tokens: 418750 },
    { date: '2025-01-20', cost: 19.50, tokens: 487500 },
    { date: '2025-01-21', cost: 24.25, tokens: 606250 },
  ],
  alerts: [
    {
      id: '1',
      type: 'budget_warning',
      message: 'Projected to reach 80% of monthly budget by Jan 25',
      severity: 'warning',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      id: '2',
      type: 'spike_detected',
      message: 'Unusual spike in Debate costs detected (45% above average)',
      severity: 'info',
      timestamp: new Date(Date.now() - 7200000).toISOString(),
    },
  ],
};

export function CostDashboard() {
  const { isAuthenticated, isLoading: authLoading, tokens } = useAuth();
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');
  const [costData, setCostData] = useState<CostData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchCostData = useCallback(async () => {
    // Skip if not authenticated - use mock data instead
    if (!isAuthenticated || authLoading) {
      setCostData(MOCK_COST_DATA);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      if (tokens?.access_token) {
        headers['Authorization'] = `Bearer ${tokens.access_token}`;
      }
      const response = await fetch(`/api/costs?range=${timeRange}`, { headers });
      if (response.ok) {
        const data = await response.json();
        setCostData(data);
      } else {
        // Use mock data for demo
        setCostData(MOCK_COST_DATA);
      }
    } catch {
      // Use mock data on error
      setCostData(MOCK_COST_DATA);
    } finally {
      setLoading(false);
    }
  }, [timeRange, isAuthenticated, authLoading, tokens?.access_token]);

  useEffect(() => {
    fetchCostData();
    // Refresh every 5 minutes
    const interval = setInterval(fetchCostData, 300000);
    return () => clearInterval(interval);
  }, [fetchCostData]);

  if (loading || !costData) {
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
            onClick={fetchCostData}
            className="px-3 py-2 text-sm font-mono text-[var(--text-muted)] hover:text-[var(--text)] border border-[var(--border)] rounded hover:border-[var(--acid-green)]/30 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

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

      {/* Alerts */}
      {costData.alerts.length > 0 && (
        <BudgetAlerts alerts={costData.alerts} />
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

      {/* Cost Optimization Tips */}
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
        <h3 className="text-sm font-mono text-[var(--acid-green)] mb-4">
          {'>'} OPTIMIZATION SUGGESTIONS
        </h3>
        <div className="space-y-3">
          <OptimizationTip
            icon="💡"
            title="Switch to Haiku for triage"
            description="Email triage could use Claude Haiku instead of Sonnet, saving ~40% on this feature."
            savings="$13/mo estimated"
          />
          <OptimizationTip
            icon="🔄"
            title="Enable response caching"
            description="Cache similar debate responses to reduce redundant API calls."
            savings="$8/mo estimated"
          />
          <OptimizationTip
            icon="📊"
            title="Batch similar requests"
            description="Group knowledge queries together to reduce per-call overhead."
            savings="$5/mo estimated"
          />
        </div>
      </div>

      {/* Last Updated */}
      <div className="text-xs text-[var(--text-muted)] text-center">
        Last updated: {new Date(costData.lastUpdated).toLocaleString()}
      </div>
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

interface OptimizationTipProps {
  icon: string;
  title: string;
  description: string;
  savings: string;
}

function OptimizationTip({ icon, title, description, savings }: OptimizationTipProps) {
  return (
    <div className="flex items-start gap-3 p-3 bg-[var(--bg)] rounded">
      <span className="text-xl">{icon}</span>
      <div className="flex-1">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-mono text-[var(--text)]">{title}</h4>
          <span className="text-xs font-mono text-green-400">{savings}</span>
        </div>
        <p className="text-xs text-[var(--text-muted)] mt-1">{description}</p>
      </div>
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
