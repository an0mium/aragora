'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { AdminLayout } from '@/components/admin/AdminLayout';
import { UsageChart, DataPoint } from '@/components/admin/UsageChart';
import { useBackend } from '@/components/BackendSelector';
import { useAuth } from '@/context/AuthContext';
import { useAragoraClient } from '@/hooks/useAragoraClient';

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime_seconds: number;
  version: string;
  components: {
    database: { status: string; latency_ms?: number };
    agents: { status: string; available: number; total: number };
    memory: { status: string; usage_mb?: number };
    websocket: { status: string; connections: number };
  };
  timestamp: string;
}

interface AdminStats {
  total_users: number;
  total_organizations: number;
  users_active_24h: number;
  new_users_7d: number;
  total_debates_this_month: number;
  total_api_calls_today?: number;
}

interface RecentActivity {
  id: string;
  type: 'user_signup' | 'debate_completed' | 'org_created' | 'payment_received' | 'api_error';
  description: string;
  timestamp: string;
  user_email?: string;
  org_name?: string;
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    healthy: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    degraded: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
    unhealthy: 'bg-acid-red/20 text-acid-red border-acid-red/40',
    ok: 'bg-acid-green/20 text-acid-green border-acid-green/40',
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${colors[status] || colors.degraded}`}>
      {status.toUpperCase()}
    </span>
  );
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (days > 0) return `${days}d ${hours}h ${minutes}m`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function getActivityIcon(type: string): string {
  const icons: Record<string, string> = {
    user_signup: '+',
    debate_completed: '*',
    org_created: '#',
    payment_received: '$',
    api_error: '!',
  };
  return icons[type] || '>';
}

function getActivityColor(type: string): string {
  const colors: Record<string, string> = {
    user_signup: 'text-acid-green',
    debate_completed: 'text-acid-cyan',
    org_created: 'text-acid-yellow',
    payment_received: 'text-acid-magenta',
    api_error: 'text-acid-red',
  };
  return colors[type] || 'text-text-muted';
}

export default function AdminOverviewPage() {
  const { config: backendConfig } = useBackend();
  const { isAuthenticated } = useAuth();
  const client = useAragoraClient();

  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [debateChartData, setDebateChartData] = useState<DataPoint[]>([]);
  const [apiCallsChartData, setApiCallsChartData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch health status
      const healthRes = await fetch(`${backendConfig.api}/api/health`);
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setHealth(healthData);
      }

      // Fetch admin stats
      if (isAuthenticated) {
        try {
          const statsData = await client.admin.stats();
          setStats(statsData.stats);
        } catch {
          // Stats may fail if not admin
        }
      }

      // Fetch recent activity (mock for now - would come from audit logs)
      try {
        const activityRes = await fetch(`${backendConfig.api}/api/admin/activity?limit=10`);
        if (activityRes.ok) {
          const activityData = await activityRes.json();
          setRecentActivity(activityData.activities || []);
        }
      } catch {
        // Activity endpoint may not exist, use mock data
        setRecentActivity([
          { id: '1', type: 'user_signup', description: 'New user registered', timestamp: new Date().toISOString(), user_email: 'user@example.com' },
          { id: '2', type: 'debate_completed', description: 'Debate completed with consensus', timestamp: new Date(Date.now() - 3600000).toISOString() },
          { id: '3', type: 'org_created', description: 'New organization created', timestamp: new Date(Date.now() - 7200000).toISOString(), org_name: 'Acme Corp' },
          { id: '4', type: 'payment_received', description: 'Payment received for Pro plan', timestamp: new Date(Date.now() - 10800000).toISOString() },
        ]);
      }

      // Generate chart data (would come from analytics API)
      try {
        const analyticsRes = await fetch(`${backendConfig.api}/api/v1/analytics/usage?period=30d`);
        if (analyticsRes.ok) {
          const analyticsData = await analyticsRes.json();
          if (analyticsData.daily_debates) {
            setDebateChartData(analyticsData.daily_debates.map((d: { date: string; count: number }) => ({
              label: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
              value: d.count,
              date: d.date,
            })));
          }
          if (analyticsData.daily_api_calls) {
            setApiCallsChartData(analyticsData.daily_api_calls.map((d: { date: string; count: number }) => ({
              label: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
              value: d.count,
              date: d.date,
            })));
          }
        }
      } catch {
        // Generate mock chart data
        const mockDates = Array.from({ length: 30 }, (_, i) => {
          const date = new Date();
          date.setDate(date.getDate() - (29 - i));
          return date;
        });
        setDebateChartData(mockDates.map(d => ({
          label: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          value: Math.floor(Math.random() * 100) + 20,
          date: d.toISOString(),
        })));
        setApiCallsChartData(mockDates.map(d => ({
          label: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          value: Math.floor(Math.random() * 5000) + 1000,
          date: d.toISOString(),
        })));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch admin data');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api, client, isAuthenticated]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [fetchData]);

  const quickActions = [
    { label: 'Invite User', href: '/admin/users?action=invite', icon: '+', color: 'acid-green' },
    { label: 'Create Organization', href: '/admin/organizations?action=create', icon: '#', color: 'acid-cyan' },
    { label: 'View Audit Logs', href: '/admin/audit', icon: '!', color: 'acid-yellow' },
    { label: 'Check Billing', href: '/admin/billing', icon: '$', color: 'acid-magenta' },
  ];

  return (
    <AdminLayout
      title="Admin Overview"
      description="System health, usage metrics, and recent activity at a glance."
      actions={
        <button
          onClick={fetchData}
          disabled={loading}
          className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      }
    >
      {error && (
        <div className="card p-4 mb-6 border-acid-red/40 bg-acid-red/10">
          <p className="text-acid-red font-mono text-sm">{error}</p>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {quickActions.map((action) => (
          <Link
            key={action.href}
            href={action.href}
            className={`card p-4 flex items-center gap-3 hover:border-${action.color}/60 transition-colors group`}
          >
            <span className={`text-2xl font-mono text-${action.color} group-hover:scale-110 transition-transform`}>
              {action.icon}
            </span>
            <span className="font-mono text-sm text-text group-hover:text-white transition-colors">
              {action.label}
            </span>
          </Link>
        ))}
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
        <div className="card p-4">
          <div className="font-mono text-xs text-text-muted mb-1">Total Users</div>
          <div className="font-mono text-2xl text-acid-green">{stats?.total_users || '-'}</div>
        </div>
        <div className="card p-4">
          <div className="font-mono text-xs text-text-muted mb-1">Organizations</div>
          <div className="font-mono text-2xl text-acid-cyan">{stats?.total_organizations || '-'}</div>
        </div>
        <div className="card p-4">
          <div className="font-mono text-xs text-text-muted mb-1">Active (24h)</div>
          <div className="font-mono text-2xl text-acid-yellow">{stats?.users_active_24h || '-'}</div>
        </div>
        <div className="card p-4">
          <div className="font-mono text-xs text-text-muted mb-1">New Users (7d)</div>
          <div className="font-mono text-2xl text-text">{stats?.new_users_7d || '-'}</div>
        </div>
        <div className="card p-4">
          <div className="font-mono text-xs text-text-muted mb-1">Debates (Month)</div>
          <div className="font-mono text-2xl text-acid-magenta">{stats?.total_debates_this_month || '-'}</div>
        </div>
        <div className="card p-4">
          <div className="font-mono text-xs text-text-muted mb-1">API Calls (Today)</div>
          <div className="font-mono text-2xl text-text">{stats?.total_api_calls_today?.toLocaleString() || '-'}</div>
        </div>
      </div>

      {/* System Health & Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* System Health Card */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-mono text-acid-green">System Health</h2>
            {health && <StatusBadge status={health.status} />}
          </div>
          {health ? (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="font-mono text-sm text-text-muted">Uptime</span>
                <span className="font-mono text-sm text-acid-cyan">{formatUptime(health.uptime_seconds)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-mono text-sm text-text-muted">Version</span>
                <span className="font-mono text-sm text-text">{health.version}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-mono text-sm text-text-muted">Agents</span>
                <span className="font-mono text-sm text-acid-green">
                  {health.components.agents.available}/{health.components.agents.total}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-mono text-sm text-text-muted">WebSocket</span>
                <span className="font-mono text-sm text-text">{health.components.websocket.connections} conn</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-mono text-sm text-text-muted">Database</span>
                <StatusBadge status={health.components.database.status} />
              </div>
              <Link
                href="/admin"
                className="block mt-4 text-center font-mono text-xs text-acid-cyan hover:text-acid-green transition-colors"
              >
                View Full System Status &gt;
              </Link>
            </div>
          ) : (
            <div className="font-mono text-sm text-text-muted animate-pulse">Loading...</div>
          )}
        </div>

        {/* Debates Chart */}
        <div className="lg:col-span-2">
          <UsageChart
            title="DEBATES PER DAY"
            data={debateChartData}
            type="bar"
            color="acid-green"
            loading={loading}
            height={240}
          />
        </div>
      </div>

      {/* API Calls Chart & Recent Activity Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* API Calls Chart */}
        <div className="lg:col-span-2">
          <UsageChart
            title="API CALLS PER DAY"
            data={apiCallsChartData}
            type="line"
            color="acid-cyan"
            loading={loading}
            height={240}
          />
        </div>

        {/* Recent Activity */}
        <div className="card p-6">
          <h2 className="font-mono text-acid-green mb-4">Recent Activity</h2>
          {recentActivity.length === 0 ? (
            <div className="font-mono text-sm text-text-muted">No recent activity</div>
          ) : (
            <div className="space-y-3">
              {recentActivity.slice(0, 6).map((activity) => (
                <div key={activity.id} className="flex items-start gap-3 pb-3 border-b border-acid-green/10 last:border-0">
                  <span className={`font-mono text-lg ${getActivityColor(activity.type)}`}>
                    {getActivityIcon(activity.type)}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-sm text-text truncate">
                      {activity.description}
                    </div>
                    <div className="font-mono text-xs text-text-muted">
                      {new Date(activity.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          <Link
            href="/admin/audit"
            className="block mt-4 text-center font-mono text-xs text-acid-cyan hover:text-acid-green transition-colors"
          >
            View All Activity &gt;
          </Link>
        </div>
      </div>
    </AdminLayout>
  );
}
