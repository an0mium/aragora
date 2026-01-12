'use client';

import { useState, useEffect, useCallback } from 'react';
import { TrendingTopicsPanel } from '@/components/TrendingTopicsPanel';
import { BackendSelector, useBackend } from '@/components/BackendSelector';

interface SchedulerStatus {
  state: 'stopped' | 'running' | 'paused';
  debates_run: number;
  debates_created: number;
  errors: number;
  last_poll_at: string | null;
  next_poll_at: string | null;
  last_debate_at: string | null;
  config: {
    poll_interval_seconds: number;
    max_debates_per_hour: number;
    min_volume_threshold: number;
    debate_rounds: number;
    allowed_categories: string[];
    blocked_categories: string[];
  };
  store_analytics?: {
    total_debates: number;
    debates_last_24h: number;
    avg_confidence: number;
    consensus_rate: number;
    by_platform: Record<string, number>;
  };
}

interface DebateHistory {
  id: string;
  topic: string;
  platform: string;
  category: string;
  volume: number;
  debate_id: string | null;
  created_at: string;
  hours_ago: number;
  consensus_reached: boolean | null;
  confidence: number | null;
}

export default function PulsePage() {
  const { config: backendConfig } = useBackend();
  const apiBase = backendConfig.api;
  const [status, setStatus] = useState<SchedulerStatus | null>(null);
  const [history, setHistory] = useState<DebateHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [configEditing, setConfigEditing] = useState(false);
  const [configValues, setConfigValues] = useState<Record<string, string | number>>({});

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/pulse/scheduler/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        setError(null);
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.error || 'Failed to fetch scheduler status');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    }
  }, [apiBase]);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/pulse/scheduler/history?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data.debates || []);
      }
    } catch {
      // Silently fail for history
    }
  }, [apiBase]);

  useEffect(() => {
    setLoading(true);
    Promise.all([fetchStatus(), fetchHistory()]).finally(() => setLoading(false));

    const interval = setInterval(() => {
      fetchStatus();
      fetchHistory();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchStatus, fetchHistory]);

  const handleSchedulerAction = async (action: 'start' | 'stop' | 'pause' | 'resume') => {
    setActionLoading(action);
    try {
      const res = await fetch(`${apiBase}/api/pulse/scheduler/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (res.ok) {
        await fetchStatus();
      } else {
        const data = await res.json().catch(() => ({}));
        setError(data.error || `Failed to ${action} scheduler`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setActionLoading(null);
    }
  };

  const handleConfigUpdate = async () => {
    setActionLoading('config');
    try {
      const res = await fetch(`${apiBase}/api/pulse/scheduler/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configValues),
      });
      if (res.ok) {
        await fetchStatus();
        setConfigEditing(false);
        setConfigValues({});
      } else {
        const data = await res.json().catch(() => ({}));
        setError(data.error || 'Failed to update config');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setActionLoading(null);
    }
  };

  const handleStartDebate = async (topic: string) => {
    setActionLoading('debate');
    try {
      const res = await fetch(`${apiBase}/api/pulse/debate-topic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, rounds: 3 }),
      });
      if (res.ok) {
        await fetchHistory();
      } else {
        const data = await res.json().catch(() => ({}));
        setError(data.error || 'Failed to start debate');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setActionLoading(null);
    }
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'running': return 'text-acid-green';
      case 'paused': return 'text-yellow-400';
      case 'stopped': return 'text-red-400';
      default: return 'text-text-muted';
    }
  };

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return '-';
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="min-h-screen bg-gray-1000 text-text font-mono">
      {/* Header */}
      <header className="border-b border-acid-green/30 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <span className="text-2xl">&#x1F525;</span>
            <h1 className="text-xl font-bold text-acid-green tracking-wide">
              PULSE DASHBOARD
            </h1>
          </div>
          <div className="flex items-center gap-4">
            {status && (
              <span className={`text-sm font-mono uppercase ${getStateColor(status.state)}`}>
                {status.state === 'running' && '\u25CF '}
                {status.state === 'paused' && '\u25D0 '}
                {status.state === 'stopped' && '\u25CB '}
                {status.state}
              </span>
            )}
            <BackendSelector />
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Error Banner */}
        {error && (
          <div className="bg-red-900/20 border border-red-500/50 px-4 py-3 text-sm text-red-400">
            {error}
            <button onClick={() => setError(null)} className="ml-4 text-red-300 hover:text-white">
              [dismiss]
            </button>
          </div>
        )}

        {/* Loading State */}
        {loading && !status && (
          <div className="text-center py-12 text-text-muted animate-pulse">
            Connecting to Pulse system...
          </div>
        )}

        {/* Main Grid */}
        {status && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Scheduler Controls */}
            <div className="panel p-6 space-y-6">
              <h2 className="text-acid-cyan text-sm font-mono mb-4">[SCHEDULER CONTROLS]</h2>

              {/* Action Buttons */}
              <div className="grid grid-cols-2 gap-3">
                {status.state === 'stopped' && (
                  <button
                    onClick={() => handleSchedulerAction('start')}
                    disabled={actionLoading === 'start'}
                    className="col-span-2 py-3 bg-acid-green/10 border border-acid-green text-acid-green hover:bg-acid-green/20 disabled:opacity-50"
                  >
                    {actionLoading === 'start' ? 'Starting...' : '\u25B6 START SCHEDULER'}
                  </button>
                )}

                {status.state === 'running' && (
                  <>
                    <button
                      onClick={() => handleSchedulerAction('pause')}
                      disabled={actionLoading === 'pause'}
                      className="py-3 bg-yellow-500/10 border border-yellow-500 text-yellow-500 hover:bg-yellow-500/20 disabled:opacity-50"
                    >
                      {actionLoading === 'pause' ? '...' : '\u23F8 PAUSE'}
                    </button>
                    <button
                      onClick={() => handleSchedulerAction('stop')}
                      disabled={actionLoading === 'stop'}
                      className="py-3 bg-red-500/10 border border-red-500 text-red-500 hover:bg-red-500/20 disabled:opacity-50"
                    >
                      {actionLoading === 'stop' ? '...' : '\u25A0 STOP'}
                    </button>
                  </>
                )}

                {status.state === 'paused' && (
                  <>
                    <button
                      onClick={() => handleSchedulerAction('resume')}
                      disabled={actionLoading === 'resume'}
                      className="py-3 bg-acid-green/10 border border-acid-green text-acid-green hover:bg-acid-green/20 disabled:opacity-50"
                    >
                      {actionLoading === 'resume' ? '...' : '\u25B6 RESUME'}
                    </button>
                    <button
                      onClick={() => handleSchedulerAction('stop')}
                      disabled={actionLoading === 'stop'}
                      className="py-3 bg-red-500/10 border border-red-500 text-red-500 hover:bg-red-500/20 disabled:opacity-50"
                    >
                      {actionLoading === 'stop' ? '...' : '\u25A0 STOP'}
                    </button>
                  </>
                )}
              </div>

              {/* Metrics */}
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Debates Created</span>
                  <span className="text-acid-green">{status.debates_created}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Poll Cycles</span>
                  <span>{status.debates_run}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Errors</span>
                  <span className={status.errors > 0 ? 'text-red-400' : ''}>{status.errors}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Last Poll</span>
                  <span className="text-xs">{formatTime(status.last_poll_at)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Next Poll</span>
                  <span className="text-xs">{formatTime(status.next_poll_at)}</span>
                </div>
              </div>

              {/* Configuration */}
              <div className="border-t border-acid-green/20 pt-4">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-acid-cyan text-xs">[CONFIG]</h3>
                  <button
                    onClick={() => setConfigEditing(!configEditing)}
                    className="text-xs text-text-muted hover:text-text"
                  >
                    {configEditing ? '[cancel]' : '[edit]'}
                  </button>
                </div>

                {configEditing ? (
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs text-text-muted">Poll Interval (seconds)</label>
                      <input
                        type="number"
                        defaultValue={status.config.poll_interval_seconds}
                        onChange={(e) => setConfigValues({...configValues, poll_interval_seconds: parseInt(e.target.value)})}
                        className="w-full bg-gray-900 border border-acid-green/30 px-2 py-1 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-text-muted">Max Debates/Hour</label>
                      <input
                        type="number"
                        defaultValue={status.config.max_debates_per_hour}
                        onChange={(e) => setConfigValues({...configValues, max_debates_per_hour: parseInt(e.target.value)})}
                        className="w-full bg-gray-900 border border-acid-green/30 px-2 py-1 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-text-muted">Min Volume Threshold</label>
                      <input
                        type="number"
                        defaultValue={status.config.min_volume_threshold}
                        onChange={(e) => setConfigValues({...configValues, min_volume_threshold: parseInt(e.target.value)})}
                        className="w-full bg-gray-900 border border-acid-green/30 px-2 py-1 text-sm"
                      />
                    </div>
                    <button
                      onClick={handleConfigUpdate}
                      disabled={actionLoading === 'config'}
                      className="w-full py-2 bg-acid-cyan/10 border border-acid-cyan text-acid-cyan hover:bg-acid-cyan/20 disabled:opacity-50"
                    >
                      {actionLoading === 'config' ? 'Saving...' : 'Save Config'}
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Poll Interval</span>
                      <span>{status.config.poll_interval_seconds}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Max/Hour</span>
                      <span>{status.config.max_debates_per_hour}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Min Volume</span>
                      <span>{status.config.min_volume_threshold}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Rounds</span>
                      <span>{status.config.debate_rounds}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Trending Topics */}
            <div className="lg:col-span-2">
              <TrendingTopicsPanel
                apiBase={apiBase}
                autoRefresh={true}
                refreshInterval={60000}
                onStartDebate={handleStartDebate}
              />

              {/* Analytics */}
              {status.store_analytics && (
                <div className="panel mt-6 p-6">
                  <h2 className="text-acid-cyan text-sm font-mono mb-4">[ANALYTICS]</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-acid-green">
                        {status.store_analytics.total_debates}
                      </div>
                      <div className="text-xs text-text-muted">Total Debates</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-acid-cyan">
                        {status.store_analytics.debates_last_24h}
                      </div>
                      <div className="text-xs text-text-muted">Last 24h</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-400">
                        {Math.round((status.store_analytics.consensus_rate || 0) * 100)}%
                      </div>
                      <div className="text-xs text-text-muted">Consensus Rate</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {(status.store_analytics.avg_confidence || 0).toFixed(2)}
                      </div>
                      <div className="text-xs text-text-muted">Avg Confidence</div>
                    </div>
                  </div>

                  {/* Platform Breakdown */}
                  {Object.keys(status.store_analytics.by_platform || {}).length > 0 && (
                    <div className="mt-4 pt-4 border-t border-acid-green/20">
                      <h3 className="text-xs text-text-muted mb-2">By Platform</h3>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(status.store_analytics.by_platform).map(([platform, count]) => (
                          <span key={platform} className="px-2 py-1 bg-gray-900 border border-acid-green/20 text-xs">
                            {platform}: {count}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Debate History */}
        {history.length > 0 && (
          <div className="panel p-6">
            <h2 className="text-acid-cyan text-sm font-mono mb-4">[RECENT SCHEDULED DEBATES]</h2>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {history.map((debate) => (
                <div
                  key={debate.id}
                  className="panel-item flex items-center justify-between"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-acid-green truncate">{debate.topic}</div>
                    <div className="text-xs text-text-muted flex gap-3 mt-1">
                      <span className="capitalize">{debate.platform}</span>
                      <span>{debate.category}</span>
                      <span>{debate.hours_ago}h ago</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    {debate.consensus_reached !== null && (
                      <span className={`text-xs ${debate.consensus_reached ? 'text-acid-green' : 'text-yellow-400'}`}>
                        {debate.consensus_reached ? '\u2713 Consensus' : '\u25CB No Consensus'}
                      </span>
                    )}
                    {debate.confidence !== null && (
                      <span className="text-xs text-text-muted">
                        {Math.round(debate.confidence * 100)}%
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
