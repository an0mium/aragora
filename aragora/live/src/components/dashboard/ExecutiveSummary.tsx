'use client';

import { useState, useEffect } from 'react';
import { KPICard, KPIGrid, KPIMiniCard } from './KPICards';
import { API_BASE_URL } from '@/config';

interface DashboardStats {
  debates: {
    today: number;
    week: number;
    month: number;
    total: number;
  };
  consensus: {
    rate: number;
    avgConfidence: number;
    avgTimeToDecision: number; // seconds
  };
  agents: {
    active: number;
    total: number;
    topPerformer: string;
    avgUptime: number;
  };
  channels: {
    active: number;
    messagesProcessed: number;
    avgResponseTime: number; // ms
  };
  costs: {
    todayTokens: number;
    weekTokens: number;
    estimatedCost: number;
  };
  knowledge: {
    totalNodes: number;
    lastSync: string;
    coverage: number;
  };
}

interface ExecutiveSummaryProps {
  apiBase?: string;
  refreshInterval?: number; // ms
}

export function ExecutiveSummary({
  apiBase: _apiBase = API_BASE_URL,
  refreshInterval = 30000,
}: ExecutiveSummaryProps) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchStats = async () => {
    try {
      // In production, this would fetch from /api/dashboard/stats
      // For now, we generate demo data
      const demoStats: DashboardStats = {
        debates: {
          today: Math.floor(Math.random() * 20) + 5,
          week: Math.floor(Math.random() * 100) + 50,
          month: Math.floor(Math.random() * 400) + 200,
          total: Math.floor(Math.random() * 2000) + 1000,
        },
        consensus: {
          rate: 75 + Math.floor(Math.random() * 20),
          avgConfidence: 70 + Math.floor(Math.random() * 25),
          avgTimeToDecision: 45 + Math.floor(Math.random() * 60),
        },
        agents: {
          active: Math.floor(Math.random() * 8) + 4,
          total: 15,
          topPerformer: ['Claude', 'GPT-4', 'Gemini', 'Mistral'][Math.floor(Math.random() * 4)],
          avgUptime: 95 + Math.floor(Math.random() * 5),
        },
        channels: {
          active: Math.floor(Math.random() * 4) + 2,
          messagesProcessed: Math.floor(Math.random() * 500) + 100,
          avgResponseTime: 200 + Math.floor(Math.random() * 300),
        },
        costs: {
          todayTokens: Math.floor(Math.random() * 500000) + 100000,
          weekTokens: Math.floor(Math.random() * 3000000) + 500000,
          estimatedCost: Math.floor(Math.random() * 100) + 20,
        },
        knowledge: {
          totalNodes: Math.floor(Math.random() * 50000) + 10000,
          lastSync: new Date(Date.now() - Math.random() * 3600000).toISOString(),
          coverage: 80 + Math.floor(Math.random() * 15),
        },
      };

      setStats(demoStats);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load stats');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();

    // Set up refresh interval
    const interval = setInterval(fetchStats, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

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

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 p-4 text-red-400 font-mono text-sm">
        Error loading dashboard: {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Last Updated */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-mono text-[var(--acid-green)]">
          {'>'} EXECUTIVE SUMMARY
        </h2>
        {lastUpdated && (
          <span className="text-xs font-mono text-[var(--text-muted)]">
            Last updated: {formatTime(lastUpdated)} | Auto-refresh: {refreshInterval / 1000}s
          </span>
        )}
      </div>

      {/* Primary KPIs */}
      <KPIGrid columns={4}>
        <KPICard
          title="Debates Today"
          value={stats?.debates.today ?? '-'}
          subtitle={`${stats?.debates.week ?? 0} this week`}
          change={stats ? { value: 12, direction: 'up', period: 'yesterday' } : undefined}
          color="green"
          loading={loading}
          icon=""
        />
        <KPICard
          title="Consensus Rate"
          value={stats ? `${stats.consensus.rate}%` : '-'}
          subtitle={`${stats?.consensus.avgConfidence ?? 0}% avg confidence`}
          change={stats ? { value: 3, direction: 'up', period: 'last week' } : undefined}
          color="cyan"
          loading={loading}
          icon=""
        />
        <KPICard
          title="Avg Decision Time"
          value={stats ? formatDuration(stats.consensus.avgTimeToDecision) : '-'}
          subtitle="time to consensus"
          change={stats ? { value: 8, direction: 'down', period: 'last week' } : undefined}
          color="yellow"
          loading={loading}
          icon=""
        />
        <KPICard
          title="Est. Cost Today"
          value={stats ? `$${stats.costs.estimatedCost}` : '-'}
          subtitle={`${formatNumber(stats?.costs.todayTokens ?? 0)} tokens`}
          color="purple"
          loading={loading}
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
              value={`${stats?.agents.active ?? 0}/${stats?.agents.total ?? 0}`}
              color="green"
            />
            <KPIMiniCard
              label="Avg Uptime"
              value={`${stats?.agents.avgUptime ?? 0}%`}
              color="cyan"
            />
            <KPIMiniCard
              label="Top Performer"
              value={stats?.agents.topPerformer ?? '-'}
              color="yellow"
            />
          </div>
        </div>

        {/* Channel Activity */}
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <h3 className="text-sm font-mono text-[var(--acid-cyan)] mb-3 flex items-center gap-2">
            <span></span> CHANNEL ACTIVITY
          </h3>
          <div className="space-y-1">
            <KPIMiniCard
              label="Active Channels"
              value={stats?.channels.active ?? 0}
              color="green"
            />
            <KPIMiniCard
              label="Messages Today"
              value={formatNumber(stats?.channels.messagesProcessed ?? 0)}
              color="cyan"
            />
            <KPIMiniCard
              label="Avg Response"
              value={`${stats?.channels.avgResponseTime ?? 0}ms`}
              color="yellow"
            />
          </div>
        </div>

        {/* Knowledge Base */}
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <h3 className="text-sm font-mono text-[var(--acid-cyan)] mb-3 flex items-center gap-2">
            <span></span> KNOWLEDGE BASE
          </h3>
          <div className="space-y-1">
            <KPIMiniCard
              label="Total Nodes"
              value={formatNumber(stats?.knowledge.totalNodes ?? 0)}
              color="green"
            />
            <KPIMiniCard
              label="Coverage"
              value={`${stats?.knowledge.coverage ?? 0}%`}
              color="cyan"
            />
            <KPIMiniCard
              label="Last Sync"
              value={stats?.knowledge.lastSync ? new Date(stats.knowledge.lastSync).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '-'}
              color="yellow"
            />
          </div>
        </div>
      </div>

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
          CONTROL PLANE
        </a>
      </div>
    </div>
  );
}

export default ExecutiveSummary;
