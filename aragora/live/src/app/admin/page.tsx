'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useAuth } from '@/context/AuthContext';

const MetricsPanel = dynamic(
  () => import('@/components/MetricsPanel').then(m => ({ default: m.MetricsPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-64 bg-surface rounded" />
      </div>
    ),
  }
);

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

interface RateLimitState {
  endpoint: string;
  limit: number;
  remaining: number;
  reset_at: string;
}

interface CircuitBreakerState {
  agent: string;
  state: 'closed' | 'open' | 'half_open';
  failures: number;
  last_failure?: string;
  last_success?: string;
}

interface RecentError {
  id: string;
  timestamp: string;
  level: string;
  message: string;
  endpoint?: string;
  user_id?: string;
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    healthy: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    degraded: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
    unhealthy: 'bg-acid-red/20 text-acid-red border-acid-red/40',
    closed: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    open: 'bg-acid-red/20 text-acid-red border-acid-red/40',
    half_open: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
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

export default function AdminPage() {
  const { config: backendConfig } = useBackend();
  const { user, isAuthenticated } = useAuth();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [rateLimits, setRateLimits] = useState<RateLimitState[]>([]);
  const [circuitBreakers, setCircuitBreakers] = useState<CircuitBreakerState[]>([]);
  const [recentErrors, setRecentErrors] = useState<RecentError[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'health' | 'agents' | 'errors' | 'metrics'>('health');

  const fetchAdminData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch health status
      const healthRes = await fetch(`${backendConfig.api}/api/health`);
      if (healthRes.ok) {
        const healthData = await healthRes.json();
        setHealth(healthData);
      }

      // Fetch circuit breaker states
      const cbRes = await fetch(`${backendConfig.api}/api/system/circuit-breakers`);
      if (cbRes.ok) {
        const cbData = await cbRes.json();
        setCircuitBreakers(cbData.breakers || []);
      }

      // Fetch recent errors (if available)
      try {
        const errRes = await fetch(`${backendConfig.api}/api/system/errors?limit=20`);
        if (errRes.ok) {
          const errData = await errRes.json();
          setRecentErrors(errData.errors || []);
        }
      } catch {
        // Errors endpoint may not exist
      }

      // Fetch rate limit states (if available)
      try {
        const rlRes = await fetch(`${backendConfig.api}/api/system/rate-limits`);
        if (rlRes.ok) {
          const rlData = await rlRes.json();
          setRateLimits(rlData.limits || []);
        }
      } catch {
        // Rate limits endpoint may not exist
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch admin data');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api]);

  useEffect(() => {
    fetchAdminData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchAdminData, 30000);
    return () => clearInterval(interval);
  }, [fetchAdminData]);

  // Check if user is admin
  const isAdmin = isAuthenticated && user?.role === 'admin';

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/settings"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [SETTINGS]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Sub Navigation */}
        <div className="border-b border-acid-green/20 bg-surface/40">
          <div className="container mx-auto px-4">
            <div className="flex gap-4 overflow-x-auto">
              <Link
                href="/admin"
                className="px-4 py-2 font-mono text-sm text-acid-green border-b-2 border-acid-green"
              >
                SYSTEM
              </Link>
              <Link
                href="/admin/organizations"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                ORGANIZATIONS
              </Link>
              <Link
                href="/admin/users"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                USERS
              </Link>
              <Link
                href="/admin/revenue"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                REVENUE
              </Link>
              <Link
                href="/admin/training"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                TRAINING
              </Link>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-mono text-acid-green mb-2">
                System Administration
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Server health, agent status, and system diagnostics.
              </p>
            </div>
            <button
              onClick={fetchAdminData}
              disabled={loading}
              className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
            >
              {loading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>

          {!isAdmin && (
            <div className="card p-6 mb-6 border-acid-yellow/40">
              <div className="flex items-center gap-2 text-acid-yellow font-mono text-sm">
                <span>!</span>
                <span>Viewing in read-only mode. Sign in as admin for full access.</span>
              </div>
            </div>
          )}

          {error && (
            <div className="card p-4 mb-6 border-acid-red/40 bg-acid-red/10">
              <p className="text-acid-red font-mono text-sm">{error}</p>
            </div>
          )}

          {/* Tab Navigation */}
          <div className="flex gap-2 border-b border-acid-green/20 pb-2 mb-6 overflow-x-auto">
            {(['health', 'agents', 'errors', 'metrics'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 font-mono text-sm whitespace-nowrap transition-colors ${
                  activeTab === tab
                    ? 'text-acid-green border-b-2 border-acid-green'
                    : 'text-text-muted hover:text-text'
                }`}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Health Tab */}
          {activeTab === 'health' && (
            <div className="space-y-6">
              {/* Overview Card */}
              {health && (
                <div className="card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-mono text-acid-green">System Health</h2>
                    <StatusBadge status={health.status} />
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <div className="font-mono text-xs text-text-muted">Uptime</div>
                      <div className="font-mono text-lg text-acid-cyan">{formatUptime(health.uptime_seconds)}</div>
                    </div>
                    <div>
                      <div className="font-mono text-xs text-text-muted">Version</div>
                      <div className="font-mono text-lg text-text">{health.version}</div>
                    </div>
                    <div>
                      <div className="font-mono text-xs text-text-muted">Agents Available</div>
                      <div className="font-mono text-lg text-acid-green">
                        {health.components.agents.available}/{health.components.agents.total}
                      </div>
                    </div>
                    <div>
                      <div className="font-mono text-xs text-text-muted">WebSocket Connections</div>
                      <div className="font-mono text-lg text-text">{health.components.websocket.connections}</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Component Status */}
              {health && (
                <div className="card p-6">
                  <h2 className="font-mono text-acid-green mb-4">Component Status</h2>
                  <div className="space-y-3">
                    {Object.entries(health.components).map(([name, component]) => (
                      <div key={name} className="flex items-center justify-between p-3 bg-surface rounded">
                        <div className="font-mono text-sm capitalize">{name.replace('_', ' ')}</div>
                        <div className="flex items-center gap-2">
                          {(component as { latency_ms?: number }).latency_ms && (
                            <span className="font-mono text-xs text-text-muted">
                              {(component as { latency_ms: number }).latency_ms}ms
                            </span>
                          )}
                          <StatusBadge status={(component as { status: string }).status} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Rate Limits */}
              {rateLimits.length > 0 && (
                <div className="card p-6">
                  <h2 className="font-mono text-acid-green mb-4">Rate Limit Status</h2>
                  <div className="space-y-2">
                    {rateLimits.map((rl, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2 bg-surface rounded">
                        <div className="font-mono text-xs">{rl.endpoint}</div>
                        <div className="flex items-center gap-2">
                          <div className="font-mono text-xs text-text-muted">
                            {rl.remaining}/{rl.limit}
                          </div>
                          <div className="w-24 h-2 bg-bg rounded overflow-hidden">
                            <div
                              className={`h-full ${rl.remaining / rl.limit > 0.5 ? 'bg-acid-green' : rl.remaining / rl.limit > 0.2 ? 'bg-acid-yellow' : 'bg-acid-red'}`}
                              style={{ width: `${(rl.remaining / rl.limit) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Agents Tab */}
          {activeTab === 'agents' && (
            <div className="space-y-6">
              {/* Circuit Breakers */}
              <div className="card p-6">
                <h2 className="font-mono text-acid-green mb-4">Circuit Breaker States</h2>
                {circuitBreakers.length === 0 ? (
                  <p className="font-mono text-sm text-text-muted">No circuit breaker data available.</p>
                ) : (
                  <div className="space-y-3">
                    {circuitBreakers.map((cb) => (
                      <div key={cb.agent} className="flex items-center justify-between p-3 bg-surface rounded">
                        <div>
                          <div className="font-mono text-sm">{cb.agent}</div>
                          <div className="font-mono text-xs text-text-muted">
                            Failures: {cb.failures}
                            {cb.last_failure && ` | Last: ${new Date(cb.last_failure).toLocaleTimeString()}`}
                          </div>
                        </div>
                        <StatusBadge status={cb.state} />
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Agent Configuration */}
              <div className="card p-6">
                <h2 className="font-mono text-acid-green mb-4">Agent Configuration</h2>
                <p className="font-mono text-sm text-text-muted mb-4">
                  Manage agent availability and fallback settings.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {['claude', 'gpt4', 'gemini', 'grok', 'deepseek', 'mistral'].map((agent) => (
                    <div key={agent} className="p-3 bg-surface rounded border border-acid-green/20">
                      <div className="font-mono text-sm capitalize">{agent}</div>
                      <div className="font-mono text-xs text-acid-green mt-1">Available</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Errors Tab */}
          {activeTab === 'errors' && (
            <div className="card p-6">
              <h2 className="font-mono text-acid-green mb-4">Recent Errors</h2>
              {recentErrors.length === 0 ? (
                <p className="font-mono text-sm text-text-muted">No recent errors recorded.</p>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {recentErrors.map((err) => (
                    <div key={err.id} className="p-3 bg-surface rounded border-l-2 border-acid-red">
                      <div className="flex items-center justify-between mb-1">
                        <span className={`font-mono text-xs px-1 rounded ${
                          err.level === 'error' ? 'bg-acid-red/20 text-acid-red' :
                          err.level === 'warning' ? 'bg-acid-yellow/20 text-acid-yellow' :
                          'bg-surface text-text-muted'
                        }`}>
                          {err.level.toUpperCase()}
                        </span>
                        <span className="font-mono text-xs text-text-muted">
                          {new Date(err.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <p className="font-mono text-sm text-text">{err.message}</p>
                      {err.endpoint && (
                        <p className="font-mono text-xs text-text-muted mt-1">Endpoint: {err.endpoint}</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Metrics Tab */}
          {activeTab === 'metrics' && (
            <PanelErrorBoundary panelName="Metrics">
              <MetricsPanel apiBase={backendConfig.api} />
            </PanelErrorBoundary>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // SYSTEM ADMINISTRATION
          </p>
        </footer>
      </main>
    </>
  );
}
