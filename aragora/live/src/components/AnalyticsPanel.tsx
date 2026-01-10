'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import type { StreamEvent } from '@/types/events';

interface Disagreement {
  debate_id: string;
  topic: string;
  agents: string[];
  dissent_count: number;
  consensus_reached: boolean;
  confidence: number;
  timestamp: string;
}

interface RoleRotation {
  agent: string;
  role_counts: Record<string, number>;
  total_debates: number;
}

interface EarlyStop {
  debate_id: string;
  topic: string;
  rounds_completed: number;
  rounds_planned: number;
  reason: string;
  consensus_early: boolean;
  timestamp: string;
}

interface GraphStats {
  node_count: number;
  edge_count: number;
  max_depth: number;
  avg_branching: number;
  complexity_score: number;
  claim_count: number;
  rebuttal_count: number;
}

interface AnalyticsPanelProps {
  apiBase: string;
  loopId?: string;
}

type TabType = 'disagreements' | 'roles' | 'early-stops' | 'graph';

export function AnalyticsPanel({ apiBase, loopId }: AnalyticsPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>('disagreements');
  const [disagreements, setDisagreements] = useState<Disagreement[]>([]);
  const [roleRotations, setRoleRotations] = useState<RoleRotation[]>([]);
  const [earlyStops, setEarlyStops] = useState<EarlyStop[]>([]);
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async (tab: TabType) => {
    setLoading(true);
    setError(null);
    try {
      if (tab === 'graph') {
        // Fetch debate graph stats if we have a loopId
        if (!loopId) {
          setGraphStats(null);
          setLoading(false);
          return;
        }
        const response = await fetch(`${apiBase}/api/debate/${encodeURIComponent(loopId)}/graph/stats`);
        if (response.ok) {
          const data = await response.json();
          setGraphStats(data);
        } else {
          setGraphStats(null);
        }
      } else {
        const endpoint = tab === 'roles' ? 'role-rotation' : tab;
        const response = await fetch(`${apiBase}/api/analytics/${endpoint}?limit=10`);
        if (!response.ok) throw new Error(`Failed to fetch ${tab}`);
        const data = await response.json();

        if (tab === 'disagreements') {
          setDisagreements(data.disagreements || []);
        } else if (tab === 'roles') {
          setRoleRotations(data.summary || []);
        } else if (tab === 'early-stops') {
          setEarlyStops(data.early_stops || []);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch analytics');
    } finally {
      setLoading(false);
    }
  }, [apiBase, loopId]);

  useEffect(() => {
    if (expanded) {
      fetchData(activeTab);
    }
  }, [expanded, activeTab, fetchData]);

  const formatTimestamp = (ts: string) => {
    if (!ts) return 'Unknown';
    const date = new Date(ts);
    return date.toLocaleDateString();
  };

  return (
    <div className="panel" style={{ padding: 0 }}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="panel-collapsible-header w-full"
      >
        <div className="flex items-center gap-2">
          <span className="text-acid-green font-mono text-sm">[ANALYTICS]</span>
          <span className="text-text-muted text-xs">Debate patterns & insights</span>
        </div>
        <span className="panel-toggle">{expanded ? '[-]' : '[+]'}</span>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          {/* Tabs */}
          <div className="flex gap-1 border-b border-acid-green/20 pb-2">
            {(['disagreements', 'roles', 'early-stops', 'graph'] as TabType[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-2 py-1 text-xs font-mono transition-colors ${
                  activeTab === tab
                    ? 'bg-acid-green text-bg'
                    : 'text-text-muted hover:text-acid-green'
                }`}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Content */}
          {loading ? (
            <div className="text-text-muted text-xs text-center py-4 animate-pulse">
              Loading analytics...
            </div>
          ) : error ? (
            <div className="text-warning text-xs text-center py-4">{error}</div>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {activeTab === 'disagreements' && (
                disagreements.length === 0 ? (
                  <div className="text-text-muted text-xs text-center py-4">
                    No disagreements recorded yet
                  </div>
                ) : (
                  disagreements.map((d) => (
                    <div
                      key={d.debate_id}
                      className="border border-warning/30 bg-warning/5 p-2 text-xs"
                    >
                      <div className="flex justify-between items-start">
                        <span className="text-warning font-mono truncate max-w-[60%]">
                          {d.topic || d.debate_id}
                        </span>
                        <span className="text-text-muted">
                          {Math.round(d.confidence * 100)}% conf
                        </span>
                      </div>
                      <div className="text-text-muted mt-1">
                        {d.agents.join(', ')} | {d.dissent_count} dissent(s)
                      </div>
                    </div>
                  ))
                )
              )}

              {activeTab === 'roles' && (
                roleRotations.length === 0 ? (
                  <div className="text-text-muted text-xs text-center py-4">
                    No role data available
                  </div>
                ) : (
                  roleRotations.map((r) => (
                    <div
                      key={r.agent}
                      className="border border-acid-cyan/30 bg-acid-cyan/5 p-2 text-xs"
                    >
                      <div className="font-mono text-acid-cyan">{r.agent}</div>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {Object.entries(r.role_counts).map(([role, count]) => (
                          <span key={role} className="text-text-muted">
                            {role}: {count}
                          </span>
                        ))}
                      </div>
                      <div className="text-text-muted/50 mt-1">
                        {r.total_debates} debates total
                      </div>
                    </div>
                  ))
                )
              )}

              {activeTab === 'early-stops' && (
                earlyStops.length === 0 ? (
                  <div className="text-text-muted text-xs text-center py-4">
                    No early terminations recorded
                  </div>
                ) : (
                  earlyStops.map((e) => (
                    <div
                      key={e.debate_id}
                      className="border border-acid-green/30 bg-surface p-2 text-xs"
                    >
                      <div className="flex justify-between items-start">
                        <span className="text-acid-green font-mono truncate max-w-[60%]">
                          {e.topic || e.debate_id}
                        </span>
                        <span className={e.consensus_early ? 'text-acid-green' : 'text-warning'}>
                          {e.consensus_early ? 'consensus' : e.reason}
                        </span>
                      </div>
                      <div className="text-text-muted mt-1">
                        Rounds: {e.rounds_completed}/{e.rounds_planned} | {formatTimestamp(e.timestamp)}
                      </div>
                    </div>
                  ))
                )
              )}

              {activeTab === 'graph' && (
                !loopId ? (
                  <div className="text-text-muted text-xs text-center py-4">
                    No debate selected for graph analysis
                  </div>
                ) : !graphStats ? (
                  <div className="text-text-muted text-xs text-center py-4">
                    No graph data available for this debate
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="border border-acid-cyan/30 bg-acid-cyan/5 p-2 text-xs">
                        <div className="text-text-muted">Nodes</div>
                        <div className="text-acid-cyan text-lg font-mono">{graphStats.node_count}</div>
                      </div>
                      <div className="border border-acid-cyan/30 bg-acid-cyan/5 p-2 text-xs">
                        <div className="text-text-muted">Edges</div>
                        <div className="text-acid-cyan text-lg font-mono">{graphStats.edge_count}</div>
                      </div>
                      <div className="border border-purple-500/30 bg-purple-500/5 p-2 text-xs">
                        <div className="text-text-muted">Max Depth</div>
                        <div className="text-purple-400 text-lg font-mono">{graphStats.max_depth}</div>
                      </div>
                      <div className="border border-purple-500/30 bg-purple-500/5 p-2 text-xs">
                        <div className="text-text-muted">Avg Branching</div>
                        <div className="text-purple-400 text-lg font-mono">{graphStats.avg_branching.toFixed(2)}</div>
                      </div>
                    </div>
                    <div className="border border-acid-green/30 bg-surface p-2 text-xs">
                      <div className="flex justify-between items-center">
                        <span className="text-text-muted">Complexity Score</span>
                        <span className="text-acid-green font-mono text-lg">
                          {(graphStats.complexity_score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="mt-2 h-2 bg-bg rounded-full overflow-hidden">
                        <div
                          className="h-full bg-acid-green"
                          style={{ width: `${graphStats.complexity_score * 100}%` }}
                        />
                      </div>
                    </div>
                    <div className="flex gap-2 text-xs text-text-muted">
                      <span>Claims: <span className="text-acid-green">{graphStats.claim_count}</span></span>
                      <span>Rebuttals: <span className="text-warning">{graphStats.rebuttal_count}</span></span>
                    </div>
                  </div>
                )
              )}
            </div>
          )}

          {/* Refresh */}
          <button
            onClick={() => fetchData(activeTab)}
            disabled={loading}
            className="w-full text-xs text-text-muted hover:text-acid-green transition-colors py-1"
          >
            [REFRESH]
          </button>
        </div>
      )}
    </div>
  );
}
