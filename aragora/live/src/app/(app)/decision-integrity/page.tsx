'use client';

/**
 * Decision Integrity Workbench
 *
 * Unified dashboard integrating all decision integrity subsystems:
 * debates, consensus, compliance, audit, memory, receipts, and agent performance.
 * Each section fetches data independently and degrades gracefully on 404.
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import {
  useDecisionIntegrity,
  type AuditEvent,
  type AgentRanking,
} from '@/hooks/useDecisionIntegrity';

// ============================================================================
// Tab types
// ============================================================================

type TabId = 'overview' | 'consensus' | 'compliance' | 'audit' | 'agents';

const TABS: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'OVERVIEW' },
  { id: 'consensus', label: 'CONSENSUS' },
  { id: 'compliance', label: 'COMPLIANCE' },
  { id: 'audit', label: 'AUDIT' },
  { id: 'agents', label: 'AGENTS' },
];

// ============================================================================
// Helpers
// ============================================================================

function StatusDot({ status }: { status: 'good' | 'warn' | 'critical' | 'unknown' }) {
  const color = {
    good: 'bg-acid-green',
    warn: 'bg-yellow-400',
    critical: 'bg-red-500',
    unknown: 'bg-text-muted/50',
  }[status];
  const pulse = status === 'critical' ? 'animate-pulse' : '';
  return <span className={`inline-block w-2 h-2 rounded-full ${color} ${pulse}`} />;
}

function scoreToStatus(score: number): 'good' | 'warn' | 'critical' | 'unknown' {
  if (score <= 0) return 'unknown';
  if (score >= 80) return 'good';
  if (score >= 50) return 'warn';
  return 'critical';
}

function MetricCard({
  label,
  value,
  unit,
  status,
  href,
}: {
  label: string;
  value: string | number;
  unit?: string;
  status: 'good' | 'warn' | 'critical' | 'unknown';
  href?: string;
}) {
  const content = (
    <div className="p-4 border border-acid-green/20 rounded bg-surface/30 hover:border-acid-green/40 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono text-text-muted uppercase tracking-wider">
          {label}
        </span>
        <StatusDot status={status} />
      </div>
      <div className="text-2xl font-mono text-acid-green">
        {value}
        {unit && <span className="text-sm text-text-muted ml-1">{unit}</span>}
      </div>
    </div>
  );

  if (href) {
    return (
      <Link href={href} className="block">
        {content}
      </Link>
    );
  }
  return content;
}

function EmptyState({ message, sub }: { message: string; sub?: string }) {
  return (
    <div className="p-8 border border-acid-green/20 rounded text-center">
      <p className="font-mono text-text-muted">{message}</p>
      {sub && (
        <p className="font-mono text-text-muted/60 text-xs mt-2">{sub}</p>
      )}
    </div>
  );
}

function LoadingPulse() {
  return (
    <div className="text-center py-8 text-acid-green font-mono animate-pulse">
      Loading...
    </div>
  );
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return '-';
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ============================================================================
// Tab content components
// ============================================================================

function OverviewTab({
  metrics,
  isLoading,
}: {
  metrics: ReturnType<typeof useDecisionIntegrity>['metrics'];
  isLoading: boolean;
}) {
  if (isLoading) return <LoadingPulse />;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <MetricCard
        label="Active Debates"
        value={metrics.activeDebates}
        status={metrics.activeDebates > 0 ? 'good' : 'unknown'}
        href="/debates"
      />
      <MetricCard
        label="Consensus Health"
        value={metrics.consensusHealth}
        unit="%"
        status={scoreToStatus(metrics.consensusHealth)}
        href="/consensus"
      />
      <MetricCard
        label="Compliance"
        value={metrics.complianceScore}
        unit="%"
        status={scoreToStatus(metrics.complianceScore)}
        href="/compliance"
      />
      <MetricCard
        label="Memory Pressure"
        value={metrics.memoryPressure}
        unit="%"
        status={
          metrics.memoryPressure <= 0
            ? 'unknown'
            : metrics.memoryPressure < 60
              ? 'good'
              : metrics.memoryPressure < 85
                ? 'warn'
                : 'critical'
        }
        href="/memory"
      />
      <MetricCard
        label="Receipt Delivery"
        value={metrics.receiptDeliveryRate}
        unit="%"
        status={scoreToStatus(metrics.receiptDeliveryRate)}
        href="/receipts"
      />
      <MetricCard
        label="System Integrity"
        value={metrics.systemIntegrity}
        unit="%"
        status={scoreToStatus(metrics.systemIntegrity)}
      />
    </div>
  );
}

function ConsensusTab({
  consensus,
  settled,
  isLoading,
}: {
  consensus: ReturnType<typeof useDecisionIntegrity>['consensus'];
  settled: ReturnType<typeof useDecisionIntegrity>['settled'];
  isLoading: boolean;
}) {
  if (isLoading) return <LoadingPulse />;

  const topics = settled?.topics ?? [];
  const strengthEntries = consensus?.by_strength
    ? Object.entries(consensus.by_strength).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="space-y-6">
      {/* Stats row */}
      {consensus && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-3 border border-acid-green/20 rounded bg-surface/30 text-center">
            <div className="text-2xl font-mono text-acid-green">
              {consensus.total_topics ?? 0}
            </div>
            <div className="text-xs font-mono text-text-muted">Total Topics</div>
          </div>
          <div className="p-3 border border-acid-green/20 rounded bg-surface/30 text-center">
            <div className="text-2xl font-mono text-acid-cyan">
              {consensus.high_confidence_count ?? 0}
            </div>
            <div className="text-xs font-mono text-text-muted">High Confidence</div>
          </div>
          <div className="p-3 border border-acid-green/20 rounded bg-surface/30 text-center">
            <div className="text-2xl font-mono text-text">
              {consensus.avg_confidence
                ? `${(consensus.avg_confidence * 100).toFixed(0)}%`
                : '-'}
            </div>
            <div className="text-xs font-mono text-text-muted">Avg Confidence</div>
          </div>
          <div className="p-3 border border-acid-green/20 rounded bg-surface/30 text-center">
            <div className="text-2xl font-mono text-yellow-400">
              {consensus.total_dissents ?? 0}
            </div>
            <div className="text-xs font-mono text-text-muted">Dissents</div>
          </div>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Settled Topics */}
        <div className="p-4 border border-acid-green/20 rounded bg-surface/30">
          <h3 className="font-mono text-acid-cyan text-sm mb-3">
            Recent Settled Topics
          </h3>
          {topics.length === 0 ? (
            <p className="text-xs font-mono text-text-muted">
              No settled topics yet.
            </p>
          ) : (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {topics.map((t, i) => (
                <div
                  key={i}
                  className="p-2 bg-bg/50 rounded border border-acid-green/10"
                >
                  <div className="font-mono text-sm text-text truncate">
                    {t.topic}
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs font-mono text-text-muted">
                    <span>
                      Confidence:{' '}
                      <span className="text-acid-cyan">
                        {(t.confidence * 100).toFixed(0)}%
                      </span>
                    </span>
                    {t.strength && (
                      <>
                        <span>|</span>
                        <span className="capitalize">{t.strength}</span>
                      </>
                    )}
                    {t.domain && (
                      <>
                        <span>|</span>
                        <span>{t.domain}</span>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Consensus Strength Distribution */}
        <div className="p-4 border border-acid-green/20 rounded bg-surface/30">
          <h3 className="font-mono text-acid-cyan text-sm mb-3">
            Consensus Strength Distribution
          </h3>
          {strengthEntries.length === 0 ? (
            <p className="text-xs font-mono text-text-muted">
              No strength data available.
            </p>
          ) : (
            <div className="space-y-2">
              {strengthEntries.map(([strength, count]) => {
                const total = consensus?.total_topics ?? 1;
                return (
                  <div key={strength} className="flex items-center gap-3">
                    <span className="text-xs font-mono text-text w-24 capitalize">
                      {strength}
                    </span>
                    <div className="flex-1 h-4 bg-bg rounded overflow-hidden">
                      <div
                        className="h-full bg-acid-green/40 rounded"
                        style={{
                          width: `${Math.min(100, (count / total) * 100)}%`,
                        }}
                      />
                    </div>
                    <span className="text-xs font-mono text-text-muted w-8 text-right">
                      {count}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div className="text-right">
        <Link
          href="/consensus"
          className="text-xs font-mono text-acid-green hover:text-acid-cyan transition-colors"
        >
          View full consensus dashboard {'\u2192'}
        </Link>
      </div>
    </div>
  );
}

function ComplianceTab({
  compliance,
  isLoading,
}: {
  compliance: ReturnType<typeof useDecisionIntegrity>['compliance'];
  isLoading: boolean;
}) {
  if (isLoading) return <LoadingPulse />;

  const frameworks = compliance?.frameworks ?? [];
  const findings = compliance?.findings ?? [];

  const statusColor: Record<string, string> = {
    compliant: 'text-acid-green border-acid-green/30 bg-acid-green/10',
    partial: 'text-yellow-400 border-yellow-400/30 bg-yellow-400/10',
    non_compliant: 'text-red-400 border-red-400/30 bg-red-400/10',
    not_assessed: 'text-text-muted border-text-muted/30 bg-text-muted/10',
  };

  return (
    <div className="space-y-6">
      {/* Overall score */}
      {compliance?.overall_score != null && (
        <div className="p-4 border border-acid-green/20 rounded bg-surface/30">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs font-mono text-text-muted uppercase tracking-wider mb-1">
                Overall Compliance Score
              </div>
              <div className="text-3xl font-mono text-acid-green">
                {Math.round(compliance.overall_score * 100)}%
              </div>
            </div>
            <StatusDot
              status={scoreToStatus(Math.round(compliance.overall_score * 100))}
            />
          </div>
        </div>
      )}

      {/* Framework status grid */}
      {frameworks.length > 0 ? (
        <div>
          <h3 className="font-mono text-acid-cyan text-sm mb-3">
            Framework Status
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {frameworks.map((fw, i) => (
              <div
                key={i}
                className={`p-3 border rounded font-mono text-sm ${
                  statusColor[fw.status] ?? statusColor.not_assessed
                }`}
              >
                <div className="font-bold truncate">{fw.name}</div>
                <div className="text-xs uppercase mt-1">{fw.status.replace('_', ' ')}</div>
                {fw.score != null && (
                  <div className="text-xs mt-1">
                    Score: {Math.round(fw.score * 100)}%
                  </div>
                )}
                {fw.last_assessed && (
                  <div className="text-xs text-text-muted/70 mt-1">
                    {formatDate(fw.last_assessed)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <EmptyState
          message="No compliance frameworks configured."
          sub="Configure frameworks in the Compliance section to see status here."
        />
      )}

      {/* Recent findings */}
      {findings.length > 0 && (
        <div>
          <h3 className="font-mono text-acid-cyan text-sm mb-3">
            Recent Findings
          </h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {findings.slice(0, 10).map((f, i) => {
              const sevColor: Record<string, string> = {
                critical: 'text-red-400',
                high: 'text-orange-400',
                medium: 'text-yellow-400',
                low: 'text-text-muted',
              };
              return (
                <div
                  key={f.id ?? i}
                  className="p-3 border border-acid-green/10 rounded bg-bg/50"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className={`text-xs font-mono px-2 py-0.5 rounded bg-surface uppercase ${
                        sevColor[f.severity] ?? 'text-text-muted'
                      }`}
                    >
                      {f.severity}
                    </span>
                    {f.framework && (
                      <span className="text-xs font-mono text-text-muted">
                        {f.framework}
                      </span>
                    )}
                  </div>
                  <p className="font-mono text-sm text-text">{f.description}</p>
                  {f.detected_at && (
                    <div className="text-xs font-mono text-text-muted/50 mt-1">
                      {formatDate(f.detected_at)}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="text-right">
        <Link
          href="/compliance"
          className="text-xs font-mono text-acid-green hover:text-acid-cyan transition-colors"
        >
          View full compliance dashboard {'\u2192'}
        </Link>
      </div>
    </div>
  );
}

function AuditTab({
  audit,
  receipts,
  isLoading,
}: {
  audit: ReturnType<typeof useDecisionIntegrity>['audit'];
  receipts: ReturnType<typeof useDecisionIntegrity>['receipts'];
  isLoading: boolean;
}) {
  if (isLoading) return <LoadingPulse />;

  const events: AuditEvent[] = audit?.events ?? [];
  const recentReceipts = receipts?.recent ?? [];

  const sevColor: Record<string, string> = {
    critical: 'border-l-red-500 bg-red-500/5',
    high: 'border-l-orange-400 bg-orange-400/5',
    medium: 'border-l-yellow-400 bg-yellow-400/5',
    low: 'border-l-text-muted bg-surface/30',
    info: 'border-l-acid-cyan bg-acid-cyan/5',
  };

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Audit Events */}
        <div>
          <h3 className="font-mono text-acid-cyan text-sm mb-3">
            Recent Audit Events
          </h3>
          {events.length === 0 ? (
            <EmptyState
              message="No audit events recorded yet."
              sub="Events appear as debates run and decisions are made."
            />
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {events.slice(0, 15).map((ev, i) => (
                <div
                  key={ev.id ?? i}
                  className={`p-3 border-l-2 rounded-r ${
                    sevColor[ev.severity ?? 'info'] ?? sevColor.info
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-sm text-text">
                      {ev.action}
                    </span>
                    <span className="text-xs font-mono text-text-muted">
                      {formatDate(ev.timestamp)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono text-text-muted">
                    <span>{ev.event_type}</span>
                    {ev.actor && (
                      <>
                        <span>|</span>
                        <span>{ev.actor}</span>
                      </>
                    )}
                    {ev.resource && (
                      <>
                        <span>|</span>
                        <span>{ev.resource}</span>
                      </>
                    )}
                  </div>
                  {ev.details && (
                    <p className="text-xs font-mono text-text-muted/70 mt-1 truncate">
                      {ev.details}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Receipt Delivery */}
        <div>
          <h3 className="font-mono text-acid-cyan text-sm mb-3">
            Decision Receipts
          </h3>

          {/* Receipt summary stats */}
          {receipts && (
            <div className="grid grid-cols-3 gap-2 mb-3">
              <div className="p-2 border border-acid-green/20 rounded bg-surface/30 text-center">
                <div className="text-lg font-mono text-acid-green">
                  {receipts.delivered ?? 0}
                </div>
                <div className="text-[10px] font-mono text-text-muted">
                  Delivered
                </div>
              </div>
              <div className="p-2 border border-yellow-400/20 rounded bg-surface/30 text-center">
                <div className="text-lg font-mono text-yellow-400">
                  {receipts.pending ?? 0}
                </div>
                <div className="text-[10px] font-mono text-text-muted">
                  Pending
                </div>
              </div>
              <div className="p-2 border border-red-500/20 rounded bg-surface/30 text-center">
                <div className="text-lg font-mono text-red-500">
                  {receipts.failed ?? 0}
                </div>
                <div className="text-[10px] font-mono text-text-muted">
                  Failed
                </div>
              </div>
            </div>
          )}

          {recentReceipts.length === 0 ? (
            <EmptyState
              message="No receipts generated yet."
              sub="Complete a debate to generate a decision receipt."
            />
          ) : (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {recentReceipts.slice(0, 10).map((r, i) => {
                const statusStyle: Record<string, string> = {
                  delivered:
                    'text-acid-green border-acid-green/30 bg-acid-green/10',
                  pending:
                    'text-yellow-400 border-yellow-400/30 bg-yellow-400/10',
                  failed: 'text-red-400 border-red-400/30 bg-red-400/10',
                };
                return (
                  <div
                    key={r.id ?? i}
                    className="p-2 border border-acid-green/10 rounded bg-bg/50 flex items-center justify-between"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="font-mono text-xs text-text truncate">
                        {r.debate_id ?? r.id}
                      </div>
                      <div className="text-[10px] font-mono text-text-muted">
                        {r.channel && <span>{r.channel} | </span>}
                        {formatDate(r.created_at)}
                      </div>
                    </div>
                    <span
                      className={`ml-2 px-2 py-0.5 text-[10px] font-mono rounded border ${
                        statusStyle[r.status] ?? statusStyle.pending
                      }`}
                    >
                      {r.status.toUpperCase()}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-between">
        <Link
          href="/audit"
          className="text-xs font-mono text-acid-green hover:text-acid-cyan transition-colors"
        >
          View full audit log {'\u2192'}
        </Link>
        <Link
          href="/receipts"
          className="text-xs font-mono text-acid-green hover:text-acid-cyan transition-colors"
        >
          View all receipts {'\u2192'}
        </Link>
      </div>
    </div>
  );
}

function AgentsTab({
  leaderboard,
  isLoading,
}: {
  leaderboard: ReturnType<typeof useDecisionIntegrity>['leaderboard'];
  isLoading: boolean;
}) {
  if (isLoading) return <LoadingPulse />;

  const agents: AgentRanking[] = useMemo(() => {
    const raw =
      leaderboard?.agents ?? leaderboard?.rankings ?? leaderboard?.leaderboard ?? [];
    return [...raw].sort((a, b) => b.elo - a.elo);
  }, [leaderboard]);

  if (agents.length === 0) {
    return (
      <EmptyState
        message="No agent rankings available yet."
        sub="Run debates to populate the agent leaderboard."
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Top agents table */}
      <div className="border border-acid-green/20 rounded overflow-hidden">
        <table className="w-full">
          <thead className="bg-surface/50 border-b border-acid-green/20">
            <tr>
              <th className="p-3 text-left font-mono text-xs text-text-muted">
                RANK
              </th>
              <th className="p-3 text-left font-mono text-xs text-text-muted">
                AGENT
              </th>
              <th className="p-3 text-right font-mono text-xs text-text-muted">
                ELO
              </th>
              <th className="p-3 text-right font-mono text-xs text-text-muted hidden md:table-cell">
                WINS
              </th>
              <th className="p-3 text-right font-mono text-xs text-text-muted hidden md:table-cell">
                LOSSES
              </th>
              <th className="p-3 text-right font-mono text-xs text-text-muted">
                WIN RATE
              </th>
              <th className="p-3 text-right font-mono text-xs text-text-muted hidden lg:table-cell">
                DEBATES
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-acid-green/10">
            {agents.slice(0, 20).map((agent, idx) => {
              const winRate =
                agent.win_rate != null
                  ? `${Math.round(agent.win_rate * 100)}%`
                  : agent.wins != null && agent.losses != null
                    ? `${Math.round(
                        (agent.wins / Math.max(agent.wins + agent.losses, 1)) *
                          100,
                      )}%`
                    : '-';
              return (
                <tr
                  key={agent.agent_id ?? agent.name}
                  className="hover:bg-surface/30 transition-colors"
                >
                  <td className="p-3 font-mono text-sm text-text-muted">
                    {idx + 1}
                  </td>
                  <td className="p-3">
                    <div className="font-mono text-sm text-acid-green">
                      {agent.name}
                    </div>
                    {agent.domains && agent.domains.length > 0 && (
                      <div className="flex gap-1 mt-1 flex-wrap">
                        {agent.domains.slice(0, 3).map((d) => (
                          <span
                            key={d}
                            className="text-[10px] font-mono px-1 py-0.5 bg-acid-green/10 text-text-muted rounded"
                          >
                            {d}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td className="p-3 text-right font-mono text-sm text-acid-cyan">
                    {Math.round(agent.elo)}
                  </td>
                  <td className="p-3 text-right font-mono text-sm text-text hidden md:table-cell">
                    {agent.wins ?? '-'}
                  </td>
                  <td className="p-3 text-right font-mono text-sm text-text hidden md:table-cell">
                    {agent.losses ?? '-'}
                  </td>
                  <td className="p-3 text-right font-mono text-sm text-text">
                    {winRate}
                  </td>
                  <td className="p-3 text-right font-mono text-sm text-text-muted hidden lg:table-cell">
                    {agent.debates_participated ?? '-'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="text-right">
        <Link
          href="/leaderboard"
          className="text-xs font-mono text-acid-green hover:text-acid-cyan transition-colors"
        >
          View full leaderboard {'\u2192'}
        </Link>
      </div>
    </div>
  );
}

// ============================================================================
// Page
// ============================================================================

export default function DecisionIntegrityPage() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const di = useDecisionIntegrity();

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="container mx-auto px-4 py-6">
          {/* Title */}
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} DECISION INTEGRITY WORKBENCH
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Unified view across debates, consensus, compliance, audit trails,
              receipts, and agent performance.
            </p>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 mb-6 flex-wrap">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 font-mono text-sm border transition-colors ${
                  activeTab === tab.id
                    ? 'border-acid-green bg-acid-green/10 text-acid-green'
                    : 'border-acid-green/30 text-text-muted hover:text-text hover:border-acid-green/50'
                }`}
              >
                [{tab.label}]
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <PanelErrorBoundary panelName="Decision Integrity Overview">
            {activeTab === 'overview' && (
              <OverviewTab metrics={di.metrics} isLoading={di.isLoading} />
            )}
          </PanelErrorBoundary>

          <PanelErrorBoundary panelName="Consensus">
            {activeTab === 'consensus' && (
              <ConsensusTab
                consensus={di.consensus}
                settled={di.settled}
                isLoading={di.isLoading}
              />
            )}
          </PanelErrorBoundary>

          <PanelErrorBoundary panelName="Compliance">
            {activeTab === 'compliance' && (
              <ComplianceTab
                compliance={di.compliance}
                isLoading={di.isLoading}
              />
            )}
          </PanelErrorBoundary>

          <PanelErrorBoundary panelName="Audit Trail">
            {activeTab === 'audit' && (
              <AuditTab
                audit={di.audit}
                receipts={di.receipts}
                isLoading={di.isLoading}
              />
            )}
          </PanelErrorBoundary>

          <PanelErrorBoundary panelName="Agent Performance">
            {activeTab === 'agents' && (
              <AgentsTab
                leaderboard={di.leaderboard}
                isLoading={di.isLoading}
              />
            )}
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // DECISION INTEGRITY WORKBENCH
          </p>
          <div className="flex justify-center gap-4 mt-2">
            <Link
              href="/debates"
              className="text-text-muted/60 hover:text-acid-green transition-colors"
            >
              Debates
            </Link>
            <Link
              href="/consensus"
              className="text-text-muted/60 hover:text-acid-green transition-colors"
            >
              Consensus
            </Link>
            <Link
              href="/compliance"
              className="text-text-muted/60 hover:text-acid-green transition-colors"
            >
              Compliance
            </Link>
            <Link
              href="/audit"
              className="text-text-muted/60 hover:text-acid-green transition-colors"
            >
              Audit
            </Link>
            <Link
              href="/receipts"
              className="text-text-muted/60 hover:text-acid-green transition-colors"
            >
              Receipts
            </Link>
            <Link
              href="/leaderboard"
              className="text-text-muted/60 hover:text-acid-green transition-colors"
            >
              Leaderboard
            </Link>
          </div>
        </footer>
      </main>
    </>
  );
}
