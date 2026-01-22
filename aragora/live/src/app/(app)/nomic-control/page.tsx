'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useNomicLoopWebSocket } from '@/hooks/useNomicLoopWebSocket';

interface NomicState {
  running: boolean;
  paused: boolean;
  cycle: number;
  phase: string;
  started_at?: string;
  last_update?: string;
  target_cycles?: number;
  pid?: number;
}

interface NomicHealth {
  status: 'healthy' | 'stalled' | 'not_running' | 'error';
  cycle: number;
  phase: string;
  last_activity?: string;
  stall_duration_seconds?: number;
  warnings: string[];
}

interface Proposal {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  category: string;
  risk_level: 'low' | 'medium' | 'high';
}

interface LogEntry {
  lines: string[];
  total: number;
  showing: number;
}

const PHASES = ['context', 'debate', 'design', 'implement', 'verify'] as const;

export default function NomicControlPage() {
  const { config: backendConfig } = useBackend();
  const [httpState, setHttpState] = useState<NomicState | null>(null);
  const [health, setHealth] = useState<NomicHealth | null>(null);
  const [httpProposals, setHttpProposals] = useState<Proposal[]>([]);
  const [logs, setLogs] = useState<LogEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [cycleCount, setCycleCount] = useState(1);
  const [autoApprove, setAutoApprove] = useState(false);
  const [showLogs, setShowLogs] = useState(false);

  // WebSocket hook for real-time updates
  const {
    isConnected: wsConnected,
    loopState: wsLoopState,
    proposals: wsProposals,
  } = useNomicLoopWebSocket({
    enabled: true,
    autoReconnect: true,
  });

  // Merge WebSocket state with HTTP state (prefer WS when connected)
  const state = useMemo<NomicState | null>(() => {
    if (wsConnected && wsLoopState) {
      return {
        running: wsLoopState.running,
        paused: wsLoopState.paused,
        cycle: wsLoopState.currentCycle,
        phase: wsLoopState.currentPhase || 'not_running',
        started_at: wsLoopState.startedAt,
        target_cycles: wsLoopState.totalCycles,
      };
    }
    return httpState;
  }, [wsConnected, wsLoopState, httpState]);

  // Merge proposals from WebSocket and HTTP (prefer WS when available)
  const proposals = useMemo<Proposal[]>(() => {
    if (wsConnected && wsProposals.length > 0) {
      return wsProposals
        .filter(p => p.status === 'pending')
        .map(p => ({
          id: p.id,
          title: p.title,
          description: p.description,
          status: p.status,
          created_at: new Date(p.generatedAt).toISOString(),
          category: p.phase,
          risk_level: 'medium' as const,
        }));
    }
    return httpProposals;
  }, [wsConnected, wsProposals, httpProposals]);

  // Fetch nomic state (fallback for HTTP polling)
  const fetchState = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/nomic/state`);
      if (!response.ok) throw new Error('Failed to fetch state');
      const data = await response.json();
      setHttpState(data);
    } catch {
      setHttpState({
        running: false,
        paused: false,
        cycle: 0,
        phase: 'not_running',
      });
    }
  }, [backendConfig.api]);

  // Fetch health
  const fetchHealth = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/nomic/health`);
      if (!response.ok) throw new Error('Failed to fetch health');
      const data = await response.json();
      setHealth(data);
    } catch {
      setHealth({
        status: 'not_running',
        cycle: 0,
        phase: 'unknown',
        warnings: [],
      });
    }
  }, [backendConfig.api]);

  // Fetch proposals (fallback for HTTP polling)
  const fetchProposals = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/nomic/proposals`);
      if (!response.ok) throw new Error('Failed to fetch proposals');
      const data = await response.json();
      setHttpProposals(data.proposals || []);
    } catch {
      setHttpProposals([]);
    }
  }, [backendConfig.api]);

  // Fetch logs
  const fetchLogs = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/nomic/log?lines=50`);
      if (!response.ok) throw new Error('Failed to fetch logs');
      const data = await response.json();
      setLogs(data);
    } catch {
      setLogs({ lines: [], total: 0, showing: 0 });
    }
  }, [backendConfig.api]);

  // Fetch all data
  const fetchAll = useCallback(async () => {
    await Promise.all([fetchState(), fetchHealth(), fetchProposals()]);
    setLoading(false);
  }, [fetchState, fetchHealth, fetchProposals]);

  // Initial data fetch
  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // Fallback polling when WebSocket is disconnected
  useEffect(() => {
    if (wsConnected) return; // Don't poll when WebSocket is connected
    const interval = setInterval(fetchAll, 3000);
    return () => clearInterval(interval);
  }, [wsConnected, fetchAll]);

  // Control actions
  const startLoop = async () => {
    setActionLoading('start');
    try {
      const response = await fetch(`${backendConfig.api}/api/nomic/control/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cycles: cycleCount,
          auto_approve: autoApprove,
        }),
      });
      if (!response.ok) {
        const data = await response.json();
        alert(data.error || 'Failed to start');
      } else {
        await fetchAll();
      }
    } catch {
      alert('Failed to start nomic loop');
    }
    setActionLoading(null);
  };

  const stopLoop = async (graceful = true) => {
    setActionLoading('stop');
    try {
      await fetch(`${backendConfig.api}/api/nomic/control/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graceful }),
      });
      await fetchAll();
    } catch {
      alert('Failed to stop nomic loop');
    }
    setActionLoading(null);
  };

  const pauseLoop = async () => {
    setActionLoading('pause');
    try {
      await fetch(`${backendConfig.api}/api/nomic/control/pause`, {
        method: 'POST',
      });
      await fetchAll();
    } catch {
      alert('Failed to pause nomic loop');
    }
    setActionLoading(null);
  };

  const resumeLoop = async () => {
    setActionLoading('resume');
    try {
      await fetch(`${backendConfig.api}/api/nomic/control/resume`, {
        method: 'POST',
      });
      await fetchAll();
    } catch {
      alert('Failed to resume nomic loop');
    }
    setActionLoading(null);
  };

  const skipPhase = async () => {
    setActionLoading('skip');
    try {
      await fetch(`${backendConfig.api}/api/nomic/control/skip-phase`, {
        method: 'POST',
      });
      await fetchAll();
    } catch {
      alert('Failed to skip phase');
    }
    setActionLoading(null);
  };

  const approveProposal = async (proposalId: string) => {
    try {
      await fetch(`${backendConfig.api}/api/nomic/proposals/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: proposalId }),
      });
      await fetchProposals();
    } catch {
      alert('Failed to approve proposal');
    }
  };

  const rejectProposal = async (proposalId: string, reason?: string) => {
    try {
      await fetch(`${backendConfig.api}/api/nomic/proposals/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: proposalId, reason }),
      });
      await fetchProposals();
    } catch {
      alert('Failed to reject proposal');
    }
  };

  const getPhaseIndex = (phase: string) => {
    const idx = PHASES.indexOf(phase as typeof PHASES[number]);
    return idx >= 0 ? idx : 0;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-success';
      case 'running': return 'text-acid-cyan';
      case 'paused': return 'text-acid-yellow';
      case 'stalled': return 'text-crimson';
      case 'not_running': return 'text-text-muted';
      default: return 'text-text-muted';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-success border-success/30';
      case 'medium': return 'text-acid-yellow border-acid-yellow/30';
      case 'high': return 'text-crimson border-crimson/30';
      default: return 'text-text-muted border-border';
    }
  };

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
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-success animate-pulse' : 'bg-acid-yellow'}`} />
                <span className="text-xs font-mono text-text-muted">
                  {wsConnected ? 'WS LIVE' : 'POLLING'}
                </span>
              </div>
              <Link
                href="/control-plane"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [CONTROL PLANE]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <PanelErrorBoundary panelName="NomicControl">
            {/* Page Header */}
            <div className="mb-6">
              <h1 className="text-2xl font-mono text-acid-green mb-2">
                Nomic Loop Control
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Autonomous self-improvement cycle management and monitoring.
              </p>
            </div>

            {loading ? (
              <div className="card p-8 text-center">
                <div className="animate-pulse font-mono text-text-muted">Loading...</div>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Status and Controls */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Status Card */}
                  <div className="card p-4">
                    <h2 className="font-mono text-sm text-acid-green mb-4">Loop Status</h2>
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-4">
                        <div className={`w-4 h-4 rounded-full ${
                          state?.running && !state?.paused
                            ? 'bg-acid-cyan animate-pulse'
                            : state?.paused
                            ? 'bg-acid-yellow'
                            : 'bg-text-muted'
                        }`} />
                        <div>
                          <div className={`text-lg font-mono uppercase ${getStatusColor(health?.status || 'not_running')}`}>
                            {health?.status || 'Not Running'}
                          </div>
                          <div className="text-xs text-text-muted font-mono">
                            Cycle {state?.cycle || 0} / Phase: {state?.phase || 'N/A'}
                          </div>
                        </div>
                      </div>
                      {health?.warnings && health.warnings.length > 0 && (
                        <div className="text-crimson text-xs font-mono">
                          {health.warnings[0]}
                        </div>
                      )}
                    </div>

                    {/* Phase Progress */}
                    <div className="mb-6">
                      <div className="flex justify-between mb-2">
                        {PHASES.map((phase, idx) => (
                          <div
                            key={phase}
                            className={`text-xs font-mono ${
                              state?.phase === phase
                                ? 'text-acid-cyan'
                                : idx < getPhaseIndex(state?.phase || '')
                                ? 'text-success'
                                : 'text-text-muted'
                            }`}
                          >
                            {phase.toUpperCase()}
                          </div>
                        ))}
                      </div>
                      <div className="h-2 bg-surface rounded-full overflow-hidden flex">
                        {PHASES.map((phase, idx) => (
                          <div
                            key={phase}
                            className={`flex-1 ${
                              state?.phase === phase
                                ? 'bg-acid-cyan'
                                : idx < getPhaseIndex(state?.phase || '')
                                ? 'bg-success'
                                : 'bg-border'
                            } ${idx > 0 ? 'ml-0.5' : ''}`}
                          />
                        ))}
                      </div>
                    </div>

                    {/* Controls */}
                    <div className="flex flex-wrap gap-3">
                      {!state?.running ? (
                        <>
                          <div className="flex items-center gap-2">
                            <input
                              type="number"
                              min={1}
                              max={10}
                              value={cycleCount}
                              onChange={(e) => setCycleCount(parseInt(e.target.value) || 1)}
                              className="w-16 bg-surface border border-border rounded px-2 py-1 text-sm font-mono"
                            />
                            <span className="text-xs text-text-muted font-mono">cycles</span>
                          </div>
                          <label className="flex items-center gap-2 text-xs font-mono text-text-muted">
                            <input
                              type="checkbox"
                              checked={autoApprove}
                              onChange={(e) => setAutoApprove(e.target.checked)}
                              className="accent-acid-green"
                            />
                            Auto-approve
                          </label>
                          <button
                            onClick={startLoop}
                            disabled={actionLoading === 'start'}
                            className="px-4 py-1.5 bg-acid-green/20 border border-acid-green/50 rounded font-mono text-sm text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
                          >
                            {actionLoading === 'start' ? 'Starting...' : 'Start Loop'}
                          </button>
                        </>
                      ) : (
                        <>
                          {state.paused ? (
                            <button
                              onClick={resumeLoop}
                              disabled={actionLoading === 'resume'}
                              className="px-4 py-1.5 bg-acid-cyan/20 border border-acid-cyan/50 rounded font-mono text-sm text-acid-cyan hover:bg-acid-cyan/30 transition-colors disabled:opacity-50"
                            >
                              {actionLoading === 'resume' ? 'Resuming...' : 'Resume'}
                            </button>
                          ) : (
                            <button
                              onClick={pauseLoop}
                              disabled={actionLoading === 'pause'}
                              className="px-4 py-1.5 bg-acid-yellow/20 border border-acid-yellow/50 rounded font-mono text-sm text-acid-yellow hover:bg-acid-yellow/30 transition-colors disabled:opacity-50"
                            >
                              {actionLoading === 'pause' ? 'Pausing...' : 'Pause'}
                            </button>
                          )}
                          <button
                            onClick={skipPhase}
                            disabled={actionLoading === 'skip'}
                            className="px-4 py-1.5 bg-surface border border-border rounded font-mono text-sm hover:border-acid-cyan transition-colors disabled:opacity-50"
                          >
                            {actionLoading === 'skip' ? 'Skipping...' : 'Skip Phase'}
                          </button>
                          <button
                            onClick={() => stopLoop(true)}
                            disabled={actionLoading === 'stop'}
                            className="px-4 py-1.5 bg-crimson/20 border border-crimson/50 rounded font-mono text-sm text-crimson hover:bg-crimson/30 transition-colors disabled:opacity-50"
                          >
                            {actionLoading === 'stop' ? 'Stopping...' : 'Stop'}
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Proposals Card */}
                  <div className="card">
                    <div className="p-4 border-b border-border flex items-center justify-between">
                      <h2 className="font-mono text-sm text-acid-green">Pending Proposals</h2>
                      <span className="text-xs font-mono text-text-muted">{proposals.length} pending</span>
                    </div>
                    <div className="p-4">
                      {proposals.length === 0 ? (
                        <div className="text-center text-text-muted font-mono text-sm py-4">
                          No pending proposals
                        </div>
                      ) : (
                        <div className="space-y-3">
                          {proposals.map((proposal) => (
                            <div key={proposal.id} className="bg-surface p-3 rounded border border-border">
                              <div className="flex items-start justify-between mb-2">
                                <div>
                                  <div className="font-mono text-sm">{proposal.title}</div>
                                  <div className="text-xs text-text-muted font-mono mt-1">
                                    {proposal.category} | {new Date(proposal.created_at).toLocaleDateString()}
                                  </div>
                                </div>
                                <span className={`text-xs font-mono px-2 py-0.5 border rounded ${getRiskColor(proposal.risk_level)}`}>
                                  {proposal.risk_level}
                                </span>
                              </div>
                              <p className="text-xs text-text-muted font-mono mb-3">
                                {proposal.description}
                              </p>
                              <div className="flex gap-2">
                                <button
                                  onClick={() => approveProposal(proposal.id)}
                                  className="px-3 py-1 text-xs font-mono bg-success/20 border border-success/50 rounded text-success hover:bg-success/30 transition-colors"
                                >
                                  Approve
                                </button>
                                <button
                                  onClick={() => rejectProposal(proposal.id)}
                                  className="px-3 py-1 text-xs font-mono bg-crimson/20 border border-crimson/50 rounded text-crimson hover:bg-crimson/30 transition-colors"
                                >
                                  Reject
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Column - Logs and Metrics */}
                <div className="space-y-6">
                  {/* Metrics Card */}
                  <div className="card p-4">
                    <h2 className="font-mono text-sm text-acid-green mb-4">Cycle Metrics</h2>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-xs font-mono text-text-muted">Current Cycle</span>
                        <span className="font-mono">{state?.cycle || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs font-mono text-text-muted">Target Cycles</span>
                        <span className="font-mono">{state?.target_cycles || '-'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs font-mono text-text-muted">Started</span>
                        <span className="font-mono text-xs">
                          {state?.started_at
                            ? new Date(state.started_at).toLocaleTimeString()
                            : '-'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs font-mono text-text-muted">Last Update</span>
                        <span className="font-mono text-xs">
                          {health?.last_activity
                            ? new Date(health.last_activity).toLocaleTimeString()
                            : '-'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Logs Card */}
                  <div className="card">
                    <div
                      className="p-4 border-b border-border flex items-center justify-between cursor-pointer"
                      onClick={() => {
                        setShowLogs(!showLogs);
                        if (!showLogs) fetchLogs();
                      }}
                    >
                      <h2 className="font-mono text-sm text-acid-green">Loop Logs</h2>
                      <span className="text-xs font-mono text-text-muted">
                        {showLogs ? '[HIDE]' : '[SHOW]'}
                      </span>
                    </div>
                    {showLogs && (
                      <div className="p-4 max-h-80 overflow-y-auto">
                        {logs && logs.lines.length > 0 ? (
                          <pre className="text-xs font-mono text-text-muted whitespace-pre-wrap">
                            {logs.lines.join('\n')}
                          </pre>
                        ) : (
                          <div className="text-center text-text-muted font-mono text-xs py-4">
                            No logs available
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Quick Links */}
                  <div className="card p-4">
                    <h2 className="font-mono text-sm text-acid-green mb-4">Quick Links</h2>
                    <div className="space-y-2">
                      <Link
                        href="/admin"
                        className="block text-xs font-mono text-text-muted hover:text-acid-cyan transition-colors"
                      >
                        {'>'} Admin Dashboard
                      </Link>
                      <Link
                        href="/control-plane"
                        className="block text-xs font-mono text-text-muted hover:text-acid-cyan transition-colors"
                      >
                        {'>'} Control Plane
                      </Link>
                      <Link
                        href="/debates"
                        className="block text-xs font-mono text-text-muted hover:text-acid-cyan transition-colors"
                      >
                        {'>'} Debates History
                      </Link>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // NOMIC LOOP CONTROL
          </p>
        </footer>
      </main>
    </>
  );
}
