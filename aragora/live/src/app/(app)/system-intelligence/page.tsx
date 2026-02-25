'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import {
  useSystemIntelligence,
  useAgentPerformance,
  useInstitutionalMemory,
  useImprovementQueue,
} from '@/hooks/useSystemIntelligence';
import {
  useSystemHealth,
  useAgentPoolHealth,
} from '@/hooks/useSystemHealth';
import { useSWRFetch } from '@/hooks/useSWRFetch';

// ---------------------------------------------------------------------------
// Additional types for new sections
// ---------------------------------------------------------------------------

interface AnomalyAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  source: string;
  timestamp: string;
  resolved: boolean;
}

interface SystemEvent {
  id: string;
  type: string;
  message: string;
  timestamp: string;
  source: string;
}

interface KMSyncStatus {
  last_sync: string | null;
  pending_items: number;
  adapters_active: number;
  adapters_total: number;
  sync_healthy: boolean;
}

interface NomicCycleStatus {
  active: boolean;
  current_cycle: number;
  current_phase: string;
  last_completed_at: string | null;
  success_rate: number;
  total_cycles: number;
}

interface DebateQueueInfo {
  active_debates: number;
  queued_debates: number;
  completed_today: number;
  avg_duration_ms: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return 'never';
  try {
    const d = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    if (diffMs < 60_000) return `${Math.round(diffMs / 1000)}s ago`;
    if (diffMs < 3_600_000) return `${Math.round(diffMs / 60_000)}m ago`;
    if (diffMs < 86_400_000) return `${Math.round(diffMs / 3_600_000)}h ago`;
    return d.toLocaleDateString();
  } catch {
    return ts ?? 'unknown';
  }
}

function getSeverityStyle(severity: string) {
  switch (severity) {
    case 'critical': return { badge: 'bg-red-500/20 text-red-400', dot: 'bg-red-400' };
    case 'warning': return { badge: 'bg-yellow-500/20 text-yellow-400', dot: 'bg-yellow-400' };
    case 'info': return { badge: 'bg-blue-400/20 text-blue-400', dot: 'bg-blue-400' };
    default: return { badge: 'bg-surface text-text-muted', dot: 'bg-text-muted' };
  }
}

function getOverallStatusStyle(status: string) {
  switch (status) {
    case 'healthy': return 'text-acid-green border-acid-green/30 bg-acid-green/10';
    case 'degraded': return 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10';
    case 'critical': return 'text-red-400 border-red-500/30 bg-red-500/10';
    default: return 'text-text-muted border-border bg-surface';
  }
}

function getPhaseLabel(phase: string) {
  switch (phase) {
    case 'context': return 'Gathering Context';
    case 'debate': return 'Agent Debate';
    case 'design': return 'Architecture Design';
    case 'implement': return 'Implementation';
    case 'verify': return 'Verification';
    case 'complete': return 'Cycle Complete';
    default: return phase;
  }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

type Section = 'overview' | 'agents' | 'knowledge' | 'queue';

export default function SystemIntelligencePage() {
  const { config: _config } = useBackend();
  const [activeSection, setActiveSection] = useState<Section>('overview');

  // --- Core hooks ---
  const { overview, isLoading: overviewLoading } = useSystemIntelligence();
  const { agents: agentPerfAgents, isLoading: agentLoading } = useAgentPerformance();
  const { memory, isLoading: memoryLoading } = useInstitutionalMemory();
  const { items: queueItems, isLoading: queueLoading, addGoal } = useImprovementQueue();

  // --- System health ---
  const { health, isLoading: healthLoading } = useSystemHealth();
  const { agents: poolAgents, total: poolTotal, active: poolActive, available: poolAvailable } = useAgentPoolHealth();

  // --- Additional data from backend ---
  const { data: anomalyData } = useSWRFetch<{ data: { alerts: AnomalyAlert[] } }>(
    '/api/v1/system-intelligence/anomalies',
    { refreshInterval: 15000 }
  );
  const anomalies = anomalyData?.data?.alerts ?? [];
  const unresolvedAnomalies = anomalies.filter(a => !a.resolved);

  const { data: eventsData } = useSWRFetch<{ data: { events: SystemEvent[] } }>(
    '/api/v1/system-intelligence/events?limit=30',
    { refreshInterval: 10000 }
  );
  const systemEvents = eventsData?.data?.events ?? [];

  const { data: kmSyncData } = useSWRFetch<{ data: KMSyncStatus }>(
    '/api/v1/system-intelligence/km-sync',
    { refreshInterval: 30000 }
  );
  const kmSync = kmSyncData?.data ?? null;

  const { data: nomicData } = useSWRFetch<{ data: NomicCycleStatus }>(
    '/api/v1/system-intelligence/nomic-status',
    { refreshInterval: 15000 }
  );
  const nomicStatus = nomicData?.data ?? null;

  const { data: debateQueueData } = useSWRFetch<{ data: DebateQueueInfo }>(
    '/api/v1/system-intelligence/debate-queue',
    { refreshInterval: 10000 }
  );
  const debateQueue = debateQueueData?.data ?? null;

  // --- New goal form ---
  const [newGoal, setNewGoal] = useState('');
  const [submittingGoal, setSubmittingGoal] = useState(false);

  const handleAddGoal = async () => {
    if (!newGoal.trim() || submittingGoal) return;
    setSubmittingGoal(true);
    try {
      await addGoal(newGoal.trim());
      setNewGoal('');
    } catch {
      // Error handled by hook
    } finally {
      setSubmittingGoal(false);
    }
  };

  const sections: Array<{ key: Section; label: string }> = [
    { key: 'overview', label: 'Overview' },
    { key: 'agents', label: 'Agents' },
    { key: 'knowledge', label: 'Knowledge' },
    { key: 'queue', label: 'Improvement Queue' },
  ];

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-3">
              <Link href="/leaderboard" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [LEADERBOARD]
              </Link>
              <Link href="/self-improve" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [SELF-IMPROVE]
              </Link>
              <Link href="/memory-gateway" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [MEMORY]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} SYSTEM INTELLIGENCE
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Aggregated view of system health, agent performance, institutional memory,
              self-improvement cycles, and anomaly detection. The system&apos;s learning at a glance.
            </p>
          </div>

          {/* ================================================================ */}
          {/* SYSTEM HEALTH BANNER                                              */}
          {/* ================================================================ */}
          <PanelErrorBoundary panelName="System Health">
            {!healthLoading && health && (
              <div className={`mb-6 p-4 border rounded-lg ${getOverallStatusStyle(health.overall_status)}`}>
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div className="flex items-center gap-3">
                    <span className={`w-3 h-3 rounded-full ${
                      health.overall_status === 'healthy' ? 'bg-acid-green animate-pulse' :
                      health.overall_status === 'degraded' ? 'bg-yellow-400 animate-pulse' :
                      'bg-red-400 animate-pulse'
                    }`} />
                    <span className="font-mono text-sm font-bold uppercase">
                      System {health.overall_status}
                    </span>
                    <span className="text-xs font-mono opacity-70">
                      checked {formatTimestamp(health.last_check)} ({health.collection_time_ms}ms)
                    </span>
                  </div>
                  <div className="flex gap-4 text-xs font-mono">
                    {health.budget?.available && (
                      <span>
                        Budget: {(health.budget.utilization * 100).toFixed(0)}% used
                        {health.budget.forecast && (
                          <span className="ml-1 opacity-70">
                            ({health.budget.forecast.trend})
                          </span>
                        )}
                      </span>
                    )}
                    {health.circuit_breakers?.available && (
                      <span>
                        Breakers: {health.circuit_breakers.breakers.filter(b => b.state === 'open').length} open / {health.circuit_breakers.total}
                      </span>
                    )}
                    {health.slos?.available && (
                      <span className={health.slos.overall_healthy ? 'text-acid-green' : 'text-red-400'}>
                        SLOs: {health.slos.overall_healthy ? 'compliant' : 'breach'}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </PanelErrorBoundary>

          {/* ================================================================ */}
          {/* KEY METRICS ROW                                                    */}
          {/* ================================================================ */}
          <PanelErrorBoundary panelName="Key Metrics">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
              {/* Nomic cycles */}
              <div className="p-3 bg-surface border border-border rounded-lg text-center">
                <div className="text-2xl font-mono font-bold text-acid-green">
                  {overviewLoading ? '-' : overview?.totalCycles ?? 0}
                </div>
                <div className="text-xs text-text-muted uppercase">Nomic Cycles</div>
              </div>
              {/* Success rate */}
              <div className="p-3 bg-surface border border-border rounded-lg text-center">
                <div className="text-2xl font-mono font-bold text-blue-400">
                  {overviewLoading ? '-' : overview ? `${(overview.successRate * 100).toFixed(0)}%` : '0%'}
                </div>
                <div className="text-xs text-text-muted uppercase">Success Rate</div>
              </div>
              {/* Active agents */}
              <div className="p-3 bg-surface border border-border rounded-lg text-center">
                <div className="text-2xl font-mono font-bold text-purple-400">
                  {poolAvailable ? `${poolActive}/${poolTotal}` : overview?.activeAgents ?? '-'}
                </div>
                <div className="text-xs text-text-muted uppercase">Agents Active</div>
              </div>
              {/* Active debates */}
              <div className="p-3 bg-surface border border-border rounded-lg text-center">
                <div className="text-2xl font-mono font-bold text-yellow-400">
                  {debateQueue ? debateQueue.active_debates : '-'}
                </div>
                <div className="text-xs text-text-muted uppercase">Active Debates</div>
              </div>
              {/* Queue depth */}
              <div className="p-3 bg-surface border border-border rounded-lg text-center">
                <div className="text-2xl font-mono font-bold text-orange-400">
                  {debateQueue ? debateQueue.queued_debates : '-'}
                </div>
                <div className="text-xs text-text-muted uppercase">Queue Depth</div>
              </div>
              {/* Knowledge items */}
              <div className="p-3 bg-surface border border-border rounded-lg text-center">
                <div className="text-2xl font-mono font-bold text-gold">
                  {overviewLoading ? '-' : overview?.knowledgeItems ?? 0}
                </div>
                <div className="text-xs text-text-muted uppercase">Knowledge Items</div>
              </div>
            </div>
          </PanelErrorBoundary>

          {/* Section tabs */}
          <div className="flex gap-2 mb-6 flex-wrap">
            {sections.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveSection(key)}
                className={`px-4 py-2 text-sm font-mono rounded border transition-colors ${
                  activeSection === key
                    ? 'bg-acid-green/20 border-acid-green text-acid-green'
                    : 'border-border text-text-muted hover:border-acid-green/50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* ================================================================ */}
          {/* SECTION: OVERVIEW                                                 */}
          {/* ================================================================ */}
          {activeSection === 'overview' && (
            <div className="space-y-6">
              {/* Row: Nomic Loop + KM Sync + Debate Queue */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Self-Improvement Cycle Status */}
                <PanelErrorBoundary panelName="Nomic Cycle">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Self-Improvement Cycle
                    </h2>
                    {nomicStatus ? (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${nomicStatus.active ? 'bg-acid-green animate-pulse' : 'bg-text-muted'}`} />
                          <span className="font-mono text-sm text-text">
                            {nomicStatus.active ? 'Running' : 'Idle'}
                          </span>
                        </div>
                        {nomicStatus.active && (
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="text-text-muted">Cycle:</span>
                              <span className="font-mono text-text">{nomicStatus.current_cycle}</span>
                            </div>
                            <div className="flex justify-between text-xs">
                              <span className="text-text-muted">Phase:</span>
                              <span className="font-mono text-acid-green">{getPhaseLabel(nomicStatus.current_phase)}</span>
                            </div>
                          </div>
                        )}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-text-muted">Total Cycles:</span>
                            <span className="font-mono text-text">{nomicStatus.total_cycles}</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-text-muted">Success Rate:</span>
                            <span className="font-mono text-text">{(nomicStatus.success_rate * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-text-muted">Last Completed:</span>
                            <span className="font-mono text-text">{formatTimestamp(nomicStatus.last_completed_at)}</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">No cycle data available.</p>
                    )}
                  </div>
                </PanelErrorBoundary>

                {/* KM Sync Status */}
                <PanelErrorBoundary panelName="KM Sync">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Knowledge Mound Sync
                    </h2>
                    {kmSync ? (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${kmSync.sync_healthy ? 'bg-acid-green' : 'bg-red-400'}`} />
                          <span className="font-mono text-sm text-text">
                            {kmSync.sync_healthy ? 'Healthy' : 'Degraded'}
                          </span>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-text-muted">Adapters:</span>
                            <span className="font-mono text-text">{kmSync.adapters_active}/{kmSync.adapters_total}</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-text-muted">Pending:</span>
                            <span className={`font-mono ${kmSync.pending_items > 100 ? 'text-yellow-400' : 'text-text'}`}>
                              {kmSync.pending_items}
                            </span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-text-muted">Last Sync:</span>
                            <span className="font-mono text-text">{formatTimestamp(kmSync.last_sync)}</span>
                          </div>
                        </div>
                        {/* Adapter health bar */}
                        {kmSync.adapters_total > 0 && (
                          <div>
                            <div className="flex items-center justify-between text-xs text-text-muted mb-1">
                              <span>Adapter Coverage</span>
                              <span className="font-mono">
                                {((kmSync.adapters_active / kmSync.adapters_total) * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="w-full bg-bg rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full ${kmSync.sync_healthy ? 'bg-acid-green' : 'bg-yellow-400'}`}
                                style={{ width: `${(kmSync.adapters_active / kmSync.adapters_total) * 100}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">No sync data available.</p>
                    )}
                  </div>
                </PanelErrorBoundary>

                {/* Debate Queue */}
                <PanelErrorBoundary panelName="Debate Queue">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Debate Activity
                    </h2>
                    {debateQueue ? (
                      <div className="space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <div className="p-2 bg-bg rounded text-center">
                            <div className="text-lg font-mono font-bold text-acid-green">{debateQueue.active_debates}</div>
                            <div className="text-xs text-text-muted">Active</div>
                          </div>
                          <div className="p-2 bg-bg rounded text-center">
                            <div className="text-lg font-mono font-bold text-yellow-400">{debateQueue.queued_debates}</div>
                            <div className="text-xs text-text-muted">Queued</div>
                          </div>
                          <div className="p-2 bg-bg rounded text-center">
                            <div className="text-lg font-mono font-bold text-blue-400">{debateQueue.completed_today}</div>
                            <div className="text-xs text-text-muted">Today</div>
                          </div>
                          <div className="p-2 bg-bg rounded text-center">
                            <div className="text-lg font-mono font-bold text-purple-400">
                              {debateQueue.avg_duration_ms > 0
                                ? `${(debateQueue.avg_duration_ms / 1000).toFixed(1)}s`
                                : '-'}
                            </div>
                            <div className="text-xs text-text-muted">Avg Duration</div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">No debate queue data available.</p>
                    )}
                  </div>
                </PanelErrorBoundary>
              </div>

              {/* Row: Anomaly Alerts + Agent Pool */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Anomaly Detection */}
                <PanelErrorBoundary panelName="Anomaly Detection">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <h2 className="text-sm font-mono font-bold text-text-muted uppercase">
                        Anomaly Detection
                      </h2>
                      {unresolvedAnomalies.length > 0 && (
                        <span className="px-2 py-0.5 text-xs font-mono rounded bg-red-500/20 text-red-400">
                          {unresolvedAnomalies.length} active
                        </span>
                      )}
                    </div>
                    {anomalies.length === 0 ? (
                      <div className="text-center py-6">
                        <span className="text-acid-green font-mono text-sm">No anomalies detected</span>
                      </div>
                    ) : (
                      <div className="space-y-2 max-h-[300px] overflow-y-auto">
                        {anomalies.slice(0, 15).map((alert) => {
                          const style = getSeverityStyle(alert.severity);
                          return (
                            <div key={alert.id} className={`flex items-start gap-2 p-2 bg-bg rounded ${alert.resolved ? 'opacity-50' : ''}`}>
                              <span className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${style.dot}`} />
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className={`px-1.5 py-0.5 text-xs font-mono rounded ${style.badge}`}>
                                    {alert.severity}
                                  </span>
                                  <span className="text-xs text-text-muted font-mono">{alert.source}</span>
                                  <span className="text-xs text-text-muted">{formatTimestamp(alert.timestamp)}</span>
                                  {alert.resolved && (
                                    <span className="text-xs text-acid-green font-mono">RESOLVED</span>
                                  )}
                                </div>
                                <p className="text-sm text-text mt-1 line-clamp-2">{alert.message}</p>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </PanelErrorBoundary>

                {/* Agent Pool Utilization */}
                <PanelErrorBoundary panelName="Agent Pool">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Agent Pool Utilization
                    </h2>
                    {poolAvailable && poolAgents.length > 0 ? (
                      <div className="space-y-3">
                        {/* Utilization bar */}
                        <div>
                          <div className="flex items-center justify-between text-xs text-text-muted mb-1">
                            <span>Utilization</span>
                            <span className="font-mono">{poolTotal > 0 ? ((poolActive / poolTotal) * 100).toFixed(0) : 0}%</span>
                          </div>
                          <div className="w-full bg-bg rounded-full h-2">
                            <div
                              className={`h-2 rounded-full transition-all ${
                                poolTotal > 0 && (poolActive / poolTotal) > 0.9
                                  ? 'bg-red-400'
                                  : poolTotal > 0 && (poolActive / poolTotal) > 0.7
                                  ? 'bg-yellow-400'
                                  : 'bg-acid-green'
                              }`}
                              style={{ width: `${poolTotal > 0 ? (poolActive / poolTotal) * 100 : 0}%` }}
                            />
                          </div>
                        </div>
                        {/* Agent list */}
                        <div className="space-y-1 max-h-[250px] overflow-y-auto">
                          {poolAgents.slice(0, 20).map((agent) => (
                            <div key={agent.agent_id} className="flex items-center gap-2 p-2 bg-bg rounded text-xs">
                              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                                agent.status === 'active' ? 'bg-acid-green' :
                                agent.status === 'idle' ? 'bg-yellow-400' :
                                'bg-red-400'
                              }`} />
                              <span className="font-mono text-text flex-1 truncate">{agent.agent_id}</span>
                              <span className="text-text-muted">{agent.type}</span>
                              <span className={`px-1 py-0.5 rounded font-mono ${
                                agent.status === 'active' ? 'text-acid-green bg-acid-green/10' :
                                agent.status === 'idle' ? 'text-yellow-400 bg-yellow-500/10' :
                                'text-red-400 bg-red-500/10'
                              }`}>
                                {agent.status}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">
                        {poolAvailable ? 'No agents in pool.' : 'Agent pool data unavailable.'}
                      </p>
                    )}
                  </div>
                </PanelErrorBoundary>
              </div>

              {/* Recent System Events Timeline */}
              <PanelErrorBoundary panelName="System Events">
                <div className="p-4 bg-surface border border-border rounded-lg">
                  <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                    Recent System Events
                  </h2>
                  {systemEvents.length === 0 ? (
                    <p className="text-text-muted text-sm text-center py-4">No recent events.</p>
                  ) : (
                    <div className="relative">
                      {/* Timeline line */}
                      <div className="absolute left-3 top-0 bottom-0 w-px bg-border" />
                      <div className="space-y-0 max-h-[400px] overflow-y-auto">
                        {systemEvents.slice(0, 30).map((event, idx) => (
                          <div key={event.id || idx} className="relative pl-8 py-2">
                            {/* Timeline dot */}
                            <div className={`absolute left-2 top-3 w-2.5 h-2.5 rounded-full border-2 border-surface ${
                              event.type.includes('error') || event.type.includes('fail') ? 'bg-red-400' :
                              event.type.includes('warn') ? 'bg-yellow-400' :
                              event.type.includes('complete') || event.type.includes('success') ? 'bg-acid-green' :
                              'bg-blue-400'
                            }`} />
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="px-1.5 py-0.5 text-xs font-mono bg-surface border border-border rounded text-text-muted">
                                {event.type}
                              </span>
                              <span className="text-xs text-text-muted">{event.source}</span>
                              <span className="text-xs text-text-muted ml-auto">{formatTimestamp(event.timestamp)}</span>
                            </div>
                            <p className="text-sm text-text mt-1">{event.message}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </PanelErrorBoundary>
            </div>
          )}

          {/* ================================================================ */}
          {/* SECTION: AGENTS                                                   */}
          {/* ================================================================ */}
          {activeSection === 'agents' && (
            <div className="space-y-6">
              {/* Top Agents from Overview */}
              {overview?.topAgents && overview.topAgents.length > 0 && (
                <PanelErrorBoundary panelName="Top Agents">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Top Agents by ELO
                    </h2>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {overview.topAgents.slice(0, 8).map((agent, idx) => (
                        <div key={agent.id} className="p-3 bg-bg rounded-lg text-center">
                          <div className="text-xs text-text-muted mb-1">#{idx + 1}</div>
                          <div className="font-mono text-sm text-text truncate">{agent.id}</div>
                          <div className="text-lg font-mono font-bold text-acid-green">{agent.elo}</div>
                          <div className="text-xs text-text-muted">{agent.wins} wins</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </PanelErrorBoundary>
              )}

              {/* Full Agent Performance Table */}
              <PanelErrorBoundary panelName="Agent Performance">
                <div className="p-4 bg-surface border border-border rounded-lg">
                  <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                    Agent Performance (ELO + Calibration)
                  </h2>
                  {agentLoading ? (
                    <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                  ) : agentPerfAgents && agentPerfAgents.length > 0 ? (
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                      {agentPerfAgents.map((agent) => (
                        <div key={agent.id} className="p-3 bg-bg rounded flex items-center gap-3">
                          <div className="flex-1 min-w-0">
                            <div className="font-mono text-sm text-text truncate">{agent.name}</div>
                            <div className="flex gap-1 mt-1 flex-wrap">
                              {agent.domains.slice(0, 4).map((d) => (
                                <span key={d} className="px-1 py-0.5 text-xs bg-surface rounded text-text-muted">
                                  {d}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-mono text-sm text-acid-green">{agent.elo}</div>
                            <div className="text-xs text-text-muted">ELO</div>
                          </div>
                          <div className="text-right">
                            <div className="font-mono text-sm text-blue-400">
                              {(agent.calibration * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-text-muted">Cal.</div>
                          </div>
                          <div className="text-right">
                            <div className="font-mono text-sm text-purple-400">
                              {(agent.winRate * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-text-muted">Win</div>
                          </div>
                          {/* ELO sparkline placeholder */}
                          <div className="w-20 h-6 flex items-end gap-px">
                            {agent.eloHistory.slice(-10).map((point, i) => {
                              const min = Math.min(...agent.eloHistory.slice(-10).map(p => p.elo));
                              const max = Math.max(...agent.eloHistory.slice(-10).map(p => p.elo));
                              const range = max - min || 1;
                              const h = ((point.elo - min) / range) * 100;
                              return (
                                <div
                                  key={i}
                                  className="flex-1 bg-acid-green/40 rounded-sm"
                                  style={{ height: `${Math.max(10, h)}%` }}
                                />
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-text-muted text-sm">No agent performance data available.</p>
                  )}
                </div>
              </PanelErrorBoundary>
            </div>
          )}

          {/* ================================================================ */}
          {/* SECTION: KNOWLEDGE                                                */}
          {/* ================================================================ */}
          {activeSection === 'knowledge' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Institutional Memory */}
                <PanelErrorBoundary panelName="Institutional Memory">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Institutional Memory
                    </h2>
                    {memoryLoading ? (
                      <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                    ) : memory ? (
                      <div className="space-y-4">
                        {/* Stats */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-2 bg-bg rounded text-center">
                            <div className="text-lg font-mono font-bold text-acid-green">{memory.totalInjections}</div>
                            <div className="text-xs text-text-muted">Injections</div>
                          </div>
                          <div className="p-2 bg-bg rounded text-center">
                            <div className="text-lg font-mono font-bold text-blue-400">{memory.retrievalCount}</div>
                            <div className="text-xs text-text-muted">Retrievals</div>
                          </div>
                        </div>

                        {/* Learned Patterns */}
                        {memory.topPatterns && memory.topPatterns.length > 0 && (
                          <div>
                            <h3 className="text-xs text-text-muted uppercase mb-2">Learned Patterns</h3>
                            <div className="space-y-1">
                              {memory.topPatterns.slice(0, 10).map((p, i) => (
                                <div key={i} className="flex items-center gap-2 text-sm p-2 bg-bg rounded">
                                  <span className="text-text flex-1 line-clamp-1">{p.pattern}</span>
                                  <span className="text-xs font-mono text-acid-green whitespace-nowrap">
                                    {(p.confidence * 100).toFixed(0)}%
                                  </span>
                                  <span className="text-xs text-text-muted whitespace-nowrap">{p.frequency}x</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Confidence Changes */}
                        {memory.confidenceChanges && memory.confidenceChanges.length > 0 && (
                          <div>
                            <h3 className="text-xs text-text-muted uppercase mb-2">Confidence Shifts</h3>
                            <div className="space-y-1">
                              {memory.confidenceChanges.slice(0, 8).map((c, i) => {
                                const delta = c.after - c.before;
                                return (
                                  <div key={i} className="flex items-center gap-2 text-sm p-2 bg-bg rounded">
                                    <span className="text-text flex-1 line-clamp-1">{c.topic}</span>
                                    <span className="text-red-400 text-xs font-mono">{(c.before * 100).toFixed(0)}%</span>
                                    <span className="text-text-muted">&rarr;</span>
                                    <span className="text-acid-green text-xs font-mono">{(c.after * 100).toFixed(0)}%</span>
                                    <span className={`text-xs font-mono ${delta > 0 ? 'text-acid-green' : 'text-red-400'}`}>
                                      {delta > 0 ? '+' : ''}{(delta * 100).toFixed(0)}
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">No institutional memory data available.</p>
                    )}
                  </div>
                </PanelErrorBoundary>

                {/* Recent Improvements from Overview */}
                <PanelErrorBoundary panelName="Recent Improvements">
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Recent Improvements
                    </h2>
                    {overviewLoading ? (
                      <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                    ) : overview?.recentImprovements && overview.recentImprovements.length > 0 ? (
                      <div className="space-y-2 max-h-[400px] overflow-y-auto">
                        {overview.recentImprovements.map((imp) => (
                          <div key={imp.id} className="flex items-center gap-3 p-3 bg-bg rounded">
                            <span className={`px-2 py-0.5 text-xs font-mono rounded ${
                              imp.status === 'completed' ? 'bg-acid-green/20 text-acid-green' :
                              imp.status === 'in_progress' ? 'bg-blue-400/20 text-blue-400' :
                              imp.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                              'bg-surface text-text-muted'
                            }`}>
                              {imp.status}
                            </span>
                            <span className="text-sm text-text flex-1 line-clamp-1">{imp.goal}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-text-muted text-sm">No recent improvements recorded.</p>
                    )}
                  </div>
                </PanelErrorBoundary>
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* SECTION: IMPROVEMENT QUEUE                                         */}
          {/* ================================================================ */}
          {activeSection === 'queue' && (
            <PanelErrorBoundary panelName="Improvement Queue">
              <div className="space-y-4">
                {/* Add Goal Form */}
                <div className="p-4 bg-surface border border-border rounded-lg">
                  <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                    Submit Improvement Goal
                  </h2>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={newGoal}
                      onChange={(e) => setNewGoal(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleAddGoal()}
                      placeholder="Describe an improvement goal for the system..."
                      className="flex-1 px-4 py-2 bg-bg border border-border rounded font-mono text-sm text-text placeholder-text-muted focus:border-acid-green focus:outline-none"
                    />
                    <button
                      onClick={handleAddGoal}
                      disabled={submittingGoal || !newGoal.trim()}
                      className="px-4 py-2 bg-acid-green/20 border border-acid-green text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 disabled:opacity-50"
                    >
                      {submittingGoal ? 'Adding...' : 'Add Goal'}
                    </button>
                  </div>
                </div>

                {/* Queue List */}
                <div className="p-4 bg-surface border border-border rounded-lg">
                  <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                    Self-Improvement Queue
                  </h2>
                  {queueLoading ? (
                    <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                  ) : queueItems && queueItems.length > 0 ? (
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                      {queueItems.map((item) => (
                        <div key={item.id} className="flex items-center gap-3 p-3 bg-bg rounded">
                          <span className={`px-2 py-0.5 text-xs font-mono rounded whitespace-nowrap ${
                            item.priority >= 75 ? 'bg-red-500/20 text-red-400' :
                            item.priority >= 50 ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-blue-400/20 text-blue-400'
                          }`}>
                            P{item.priority}
                          </span>
                          <span className={`px-1.5 py-0.5 text-xs font-mono rounded ${
                            item.status === 'completed' ? 'bg-acid-green/20 text-acid-green' :
                            item.status === 'in_progress' ? 'bg-blue-400/20 text-blue-400' :
                            item.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                            'bg-surface text-text-muted'
                          }`}>
                            {item.status}
                          </span>
                          <span className="text-sm text-text flex-1 line-clamp-1">{item.goal}</span>
                          <span className="text-xs text-text-muted whitespace-nowrap">{item.source}</span>
                          <span className="text-xs text-text-muted whitespace-nowrap">{formatTimestamp(item.createdAt)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-text-muted text-sm">
                      No items in the improvement queue. Add a goal above to begin.
                    </p>
                  )}
                </div>
              </div>
            </PanelErrorBoundary>
          )}
        </div>

        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">{'='.repeat(40)}</div>
          <p className="text-text-muted">{'>'} ARAGORA // SYSTEM INTELLIGENCE</p>
        </footer>
      </main>
    </>
  );
}
