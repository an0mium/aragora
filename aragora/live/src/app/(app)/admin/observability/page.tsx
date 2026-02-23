'use client';

import { useState, useEffect, useCallback } from 'react';
import { AdminLayout } from '@/components/admin/AdminLayout';
import { useBackend } from '@/components/BackendSelector';

interface AgentRanking {
  name: string;
  rating: number;
  matches: number;
  win_rate: number;
}

interface CircuitBreaker {
  name: string;
  state: string;
  failure_count: number;
  success_count?: number;
}

interface RecentRun {
  id: string;
  goal: string;
  status: string;
  started_at: string;
}

interface DashboardData {
  timestamp: number;
  collection_time_ms: number;
  debate_metrics: {
    total_debates: number;
    avg_duration_seconds: number;
    consensus_rate: number;
    available: boolean;
  };
  agent_rankings: {
    top_agents: AgentRanking[];
    available: boolean;
  };
  circuit_breakers: {
    breakers: CircuitBreaker[];
    available: boolean;
  };
  self_improve: {
    total_cycles: number;
    successful: number;
    failed: number;
    recent_runs: RecentRun[];
    available: boolean;
  };
  system_health: {
    memory_percent: number | null;
    cpu_percent: number | null;
    pid: number;
    available: boolean;
  };
  error_rates: {
    total_requests: number;
    total_errors: number;
    error_rate: number;
    available: boolean;
  };
}

function StatusDot({ status }: { status: 'green' | 'yellow' | 'red' | 'gray' }) {
  const colors = {
    green: 'bg-acid-green shadow-[0_0_6px_var(--acid-green)]',
    yellow: 'bg-acid-yellow shadow-[0_0_6px_var(--acid-yellow)]',
    red: 'bg-acid-red shadow-[0_0_6px_var(--acid-red)]',
    gray: 'bg-text-muted',
  };
  return <span className={`inline-block w-2 h-2 rounded-full ${colors[status]}`} />;
}

function MetricCard({
  label,
  value,
  sub,
  color = 'acid-green',
}: {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="card p-4">
      <div className="font-mono text-xs text-text-muted mb-1">{label}</div>
      <div className={`font-mono text-2xl text-${color}`}>{value}</div>
      {sub && <div className="font-mono text-xs text-text-muted mt-1">{sub}</div>}
    </div>
  );
}

function BarChart({ value, max, color = 'acid-green' }: { value: number; max: number; color?: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="w-full h-3 bg-surface rounded overflow-hidden border border-border">
      <div
        className={`h-full bg-${color} transition-all duration-500`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function cbStateColor(state: string): 'green' | 'yellow' | 'red' | 'gray' {
  const s = state.toLowerCase();
  if (s === 'closed' || s === 'ok') return 'green';
  if (s === 'half_open' || s === 'half-open') return 'yellow';
  if (s === 'open') return 'red';
  return 'gray';
}

function runStatusColor(status: string): string {
  if (status === 'completed') return 'text-acid-green';
  if (status === 'failed') return 'text-acid-red';
  if (status === 'running' || status === 'in_progress') return 'text-acid-yellow';
  return 'text-text-muted';
}

export default function ObservabilityPage() {
  const { config: backendConfig } = useBackend();
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`${backendConfig.api}/api/v1/observability/dashboard`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const overallHealth = (): 'green' | 'yellow' | 'red' => {
    if (!data) return 'gray' as 'green';
    const openBreakers = data.circuit_breakers.breakers.filter(
      (b) => b.state.toLowerCase() === 'open'
    ).length;
    if (openBreakers > 0 || data.error_rates.error_rate > 0.05) return 'red';
    if (
      data.error_rates.error_rate > 0.01 ||
      (data.system_health.memory_percent && data.system_health.memory_percent > 85)
    )
      return 'yellow';
    return 'green';
  };

  const healthLabel = { green: 'HEALTHY', yellow: 'DEGRADED', red: 'UNHEALTHY' };

  return (
    <AdminLayout
      title="Observability"
      description="Real-time system metrics, agent rankings, and health indicators."
      actions={
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="font-mono text-xs text-text-muted">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={fetchData}
            disabled={loading}
            className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
          >
            {loading && !data ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      }
    >
      {error && (
        <div className="card p-4 mb-6 border-acid-red/40 bg-acid-red/10">
          <p className="text-acid-red font-mono text-sm">
            Failed to load observability data: {error}
          </p>
        </div>
      )}

      {/* System Health Banner */}
      <div className="card p-4 mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <StatusDot status={data ? overallHealth() : 'gray'} />
          <span className="font-mono text-sm text-text">
            System Status:{' '}
            <span
              className={`text-${
                overallHealth() === 'green'
                  ? 'acid-green'
                  : overallHealth() === 'yellow'
                  ? 'acid-yellow'
                  : 'acid-red'
              }`}
            >
              {data ? healthLabel[overallHealth()] : 'LOADING'}
            </span>
          </span>
        </div>
        {data && (
          <span className="font-mono text-xs text-text-muted">
            PID {data.system_health.pid} | Collected in {data.collection_time_ms}ms
          </span>
        )}
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
        <MetricCard
          label="Total Debates"
          value={data?.debate_metrics.total_debates ?? '-'}
          color="acid-green"
        />
        <MetricCard
          label="Avg Duration"
          value={data ? `${data.debate_metrics.avg_duration_seconds}s` : '-'}
          color="acid-cyan"
        />
        <MetricCard
          label="Consensus Rate"
          value={data ? `${(data.debate_metrics.consensus_rate * 100).toFixed(1)}%` : '-'}
          color="acid-yellow"
        />
        <MetricCard
          label="Error Rate"
          value={data ? `${(data.error_rates.error_rate * 100).toFixed(2)}%` : '-'}
          color={data && data.error_rates.error_rate > 0.01 ? 'acid-red' : 'acid-green'}
        />
        <MetricCard
          label="CPU"
          value={data?.system_health.cpu_percent != null ? `${data.system_health.cpu_percent}%` : '-'}
          color="acid-magenta"
        />
        <MetricCard
          label="Memory"
          value={
            data?.system_health.memory_percent != null
              ? `${data.system_health.memory_percent}%`
              : '-'
          }
          color={
            data?.system_health.memory_percent && data.system_health.memory_percent > 85
              ? 'acid-red'
              : 'acid-cyan'
          }
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Agent Leaderboard */}
        <div className="card p-6">
          <h2 className="font-mono text-acid-green mb-4">Agent Leaderboard</h2>
          {!data?.agent_rankings.available ? (
            <p className="font-mono text-xs text-text-muted">ELO system unavailable</p>
          ) : data.agent_rankings.top_agents.length === 0 ? (
            <p className="font-mono text-xs text-text-muted">No agent data yet</p>
          ) : (
            <div className="space-y-2">
              <div className="grid grid-cols-[2rem_1fr_4rem_4rem_4rem] gap-2 text-xs font-mono text-text-muted pb-2 border-b border-border">
                <span>#</span>
                <span>Agent</span>
                <span className="text-right">ELO</span>
                <span className="text-right">Matches</span>
                <span className="text-right">Win%</span>
              </div>
              {data.agent_rankings.top_agents.map((agent, idx) => {
                const maxRating = data.agent_rankings.top_agents[0]?.rating || 1500;
                return (
                  <div key={agent.name}>
                    <div className="grid grid-cols-[2rem_1fr_4rem_4rem_4rem] gap-2 text-sm font-mono items-center">
                      <span className="text-text-muted">{idx + 1}</span>
                      <span className="text-text truncate">{agent.name}</span>
                      <span className="text-right text-acid-green">{Math.round(agent.rating)}</span>
                      <span className="text-right text-text-muted">{agent.matches}</span>
                      <span className="text-right text-acid-cyan">
                        {(agent.win_rate * 100).toFixed(0)}%
                      </span>
                    </div>
                    <BarChart value={agent.rating} max={maxRating * 1.1} color="acid-green" />
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Circuit Breaker States */}
        <div className="card p-6">
          <h2 className="font-mono text-acid-green mb-4">Circuit Breakers</h2>
          {!data?.circuit_breakers.available ? (
            <p className="font-mono text-xs text-text-muted">Resilience registry unavailable</p>
          ) : data.circuit_breakers.breakers.length === 0 ? (
            <p className="font-mono text-xs text-text-muted">No circuit breakers registered</p>
          ) : (
            <div className="space-y-3">
              {data.circuit_breakers.breakers.map((cb) => (
                <div key={cb.name} className="flex items-center justify-between pb-2 border-b border-border last:border-0">
                  <div className="flex items-center gap-2">
                    <StatusDot status={cbStateColor(cb.state)} />
                    <span className="font-mono text-sm text-text truncate max-w-[200px]">
                      {cb.name}
                    </span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="font-mono text-xs text-text-muted">
                      {cb.state.toUpperCase()}
                    </span>
                    {cb.failure_count > 0 && (
                      <span className="font-mono text-xs text-acid-red">
                        {cb.failure_count} fail
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Self-Improvement Status */}
      <div className="card p-6 mb-6">
        <h2 className="font-mono text-acid-green mb-4">Self-Improvement Cycles</h2>
        {!data?.self_improve.available ? (
          <p className="font-mono text-xs text-text-muted">Self-improve store unavailable</p>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <div className="font-mono text-xs text-text-muted">Total Cycles</div>
                <div className="font-mono text-xl text-acid-cyan">{data.self_improve.total_cycles}</div>
              </div>
              <div>
                <div className="font-mono text-xs text-text-muted">Successful</div>
                <div className="font-mono text-xl text-acid-green">{data.self_improve.successful}</div>
              </div>
              <div>
                <div className="font-mono text-xs text-text-muted">Failed</div>
                <div className="font-mono text-xl text-acid-red">{data.self_improve.failed}</div>
              </div>
            </div>
            {data.self_improve.recent_runs.length > 0 && (
              <div className="space-y-2">
                <div className="font-mono text-xs text-text-muted border-b border-border pb-1">
                  Recent Runs
                </div>
                {data.self_improve.recent_runs.map((run) => (
                  <div key={run.id} className="flex items-center justify-between text-sm font-mono">
                    <span className="text-text truncate max-w-[60%]">{run.goal || run.id}</span>
                    <div className="flex items-center gap-3">
                      <span className={runStatusColor(run.status)}>{run.status.toUpperCase()}</span>
                      {run.started_at && (
                        <span className="text-xs text-text-muted">
                          {new Date(run.started_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {/* Error Rates */}
      {data?.error_rates.available && (
        <div className="card p-6">
          <h2 className="font-mono text-acid-green mb-4">Request Metrics</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="font-mono text-xs text-text-muted">Total Requests</div>
              <div className="font-mono text-xl text-text">
                {data.error_rates.total_requests.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="font-mono text-xs text-text-muted">Total Errors</div>
              <div className="font-mono text-xl text-acid-red">
                {data.error_rates.total_errors.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="font-mono text-xs text-text-muted">Error Rate</div>
              <div
                className={`font-mono text-xl ${
                  data.error_rates.error_rate > 0.01 ? 'text-acid-red' : 'text-acid-green'
                }`}
              >
                {(data.error_rates.error_rate * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </AdminLayout>
  );
}
