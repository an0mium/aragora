'use client';

import { useState, useEffect, useCallback } from 'react';
import { ErrorWithRetry } from './RetryButton';
import { fetchWithRetry } from '@/utils/retry';

interface DissentRecord {
  topic: string;
  majority_view: string;
  dissenting_view: string;
  dissenting_agent: string;
  confidence: number;
  reasoning?: string;
}

interface ContrarianView {
  agent: string;
  position: string;
  confidence: number;
  reasoning?: string;
  debate_id?: string;
}

interface RiskWarning {
  domain: string;
  risk_type: string;
  severity: string;
  description: string;
  mitigation?: string;
  detected_at: string;
}

interface ConsensusStats {
  total_topics: number;
  high_confidence_count: number;
  avg_confidence: number;
  total_dissents: number;
  domains: string[];
  by_strength: Record<string, number>;
}

interface EvidenceCitation {
  source: string;
  claim: string;
  agent: string;
  round: number;
  confidence?: number;
}

interface GraphNode {
  id: string;
  agent: string;
  content: string;
  type: string;
  parent_id?: string;
  children?: string[];
}

interface BackendConfig {
  apiUrl: string;
  wsUrl: string;
}

interface EvidenceVisualizerPanelProps {
  backendConfig?: BackendConfig;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

const SEVERITY_COLORS: Record<string, { text: string; bg: string }> = {
  critical: { text: 'text-acid-red', bg: 'bg-acid-red/20' },
  high: { text: 'text-acid-yellow', bg: 'bg-acid-yellow/20' },
  medium: { text: 'text-acid-cyan', bg: 'bg-acid-cyan/20' },
  low: { text: 'text-acid-green', bg: 'bg-acid-green/20' },
};

export function EvidenceVisualizerPanel({ backendConfig }: EvidenceVisualizerPanelProps) {
  const apiBase = backendConfig?.apiUrl || DEFAULT_API_BASE;

  const [dissents, setDissents] = useState<DissentRecord[]>([]);
  const [contrarianViews, setContrarianViews] = useState<ContrarianView[]>([]);
  const [riskWarnings, setRiskWarnings] = useState<RiskWarning[]>([]);
  const [consensusStats, setConsensusStats] = useState<ConsensusStats | null>(null);
  const [evidence, setEvidence] = useState<EvidenceCitation[]>([]);
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apiUnavailable, setApiUnavailable] = useState(false);
  const [activeTab, setActiveTab] = useState<'dissent' | 'evidence' | 'graph'>('dissent');
  const [searchDebateId, setSearchDebateId] = useState('');
  const [topicFilter, setTopicFilter] = useState('');

  const fetchDissentData = useCallback(async () => {
    try {
      setLoading(true);

      const topicParam = topicFilter ? `&topic=${encodeURIComponent(topicFilter)}` : '';
      const [dissentsRes, contrarianRes, warningsRes, statsRes] = await Promise.allSettled([
        fetchWithRetry(`${apiBase}/api/consensus/dissents?limit=20${topicParam}`, undefined, { maxRetries: 2 }),
        fetchWithRetry(`${apiBase}/api/consensus/contrarian-views?limit=15${topicParam}`, undefined, { maxRetries: 2 }),
        fetchWithRetry(`${apiBase}/api/consensus/risk-warnings?limit=10`, undefined, { maxRetries: 2 }),
        fetchWithRetry(`${apiBase}/api/consensus/stats`, undefined, { maxRetries: 2 }),
      ]);

      // Track if any API call succeeded
      let anySuccess = false;

      if (dissentsRes.status === 'fulfilled' && dissentsRes.value.ok) {
        const data = await dissentsRes.value.json();
        setDissents(data.dissents || []);
        anySuccess = true;
      } else {
        setDissents([]);
      }

      if (contrarianRes.status === 'fulfilled' && contrarianRes.value.ok) {
        const data = await contrarianRes.value.json();
        setContrarianViews(data.views || []);
        anySuccess = true;
      } else {
        setContrarianViews([]);
      }

      if (warningsRes.status === 'fulfilled' && warningsRes.value.ok) {
        const data = await warningsRes.value.json();
        setRiskWarnings(data.warnings || []);
        anySuccess = true;
      } else {
        setRiskWarnings([]);
      }

      if (statsRes.status === 'fulfilled' && statsRes.value.ok) {
        const data = await statsRes.value.json();
        setConsensusStats(data);
        anySuccess = true;
      }

      setApiUnavailable(!anySuccess);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch dissent data');
    } finally {
      setLoading(false);
    }
  }, [apiBase, topicFilter]);

  const fetchEvidence = useCallback(async () => {
    if (!searchDebateId.trim()) {
      setEvidence([]);
      return;
    }

    try {
      const response = await fetchWithRetry(
        `${apiBase}/api/debates/${searchDebateId}/evidence`,
        undefined,
        { maxRetries: 2 }
      );

      if (response.ok) {
        const data = await response.json();
        setEvidence(data.evidence || data.citations || []);
      } else {
        setEvidence([]);
      }
    } catch (err) {
      console.error('Failed to fetch evidence:', err);
      setEvidence([]);
    }
  }, [apiBase, searchDebateId]);

  const fetchGraphNodes = useCallback(async () => {
    if (!searchDebateId.trim()) {
      setGraphNodes([]);
      return;
    }

    try {
      const response = await fetchWithRetry(
        `${apiBase}/api/debates/graph/${searchDebateId}/nodes`,
        undefined,
        { maxRetries: 2 }
      );

      if (response.ok) {
        const data = await response.json();
        setGraphNodes(data.nodes || []);
      } else {
        setGraphNodes([]);
      }
    } catch (err) {
      console.error('Failed to fetch graph nodes:', err);
      setGraphNodes([]);
    }
  }, [apiBase, searchDebateId]);

  useEffect(() => {
    fetchDissentData();
  }, [fetchDissentData]);

  useEffect(() => {
    if (activeTab === 'evidence') {
      const debounce = setTimeout(fetchEvidence, 300);
      return () => clearTimeout(debounce);
    }
  }, [searchDebateId, activeTab, fetchEvidence]);

  useEffect(() => {
    if (activeTab === 'graph') {
      const debounce = setTimeout(fetchGraphNodes, 300);
      return () => clearTimeout(debounce);
    }
  }, [searchDebateId, activeTab, fetchGraphNodes]);

  if (loading && dissents.length === 0) {
    return (
      <div className="card p-6">
        <div className="flex items-center gap-3">
          <div className="animate-spin w-5 h-5 border-2 border-acid-green border-t-transparent rounded-full" />
          <span className="font-mono text-text-muted">Loading evidence data...</span>
        </div>
      </div>
    );
  }

  if (error && dissents.length === 0) {
    return (
      <ErrorWithRetry
        error={error || "Failed to load evidence and dissent data"}
        onRetry={fetchDissentData}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* API Unavailable Indicator */}
      {apiUnavailable && (
        <div className="bg-warning/10 border border-warning/30 rounded px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-warning">⚠</span>
            <span className="font-mono text-sm text-warning">
              Evidence API unavailable - No data to display
            </span>
          </div>
          <button
            onClick={fetchDissentData}
            className="font-mono text-xs text-warning hover:text-warning/80 transition-colors"
          >
            [RETRY]
          </button>
        </div>
      )}

      {/* Stats Overview */}
      {consensusStats && (
        <div className="card p-4">
          <h3 className="font-mono text-acid-green mb-4">Consensus Overview</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-mono text-acid-green">{consensusStats.total_topics}</div>
              <div className="text-xs font-mono text-text-muted">Total Topics</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-mono text-acid-cyan">{consensusStats.total_dissents}</div>
              <div className="text-xs font-mono text-text-muted">Total Dissents</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-mono text-acid-yellow">
                {(consensusStats.avg_confidence * 100).toFixed(1)}%
              </div>
              <div className="text-xs font-mono text-text-muted">Avg Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-mono text-acid-red">{riskWarnings.length}</div>
              <div className="text-xs font-mono text-text-muted">Active Warnings</div>
            </div>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-acid-green/20 pb-2">
        {(['dissent', 'evidence', 'graph'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 font-mono text-sm transition-colors ${
              activeTab === tab
                ? 'text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            {tab === 'dissent' ? 'DISSENT & CONTRARIAN' : tab === 'evidence' ? 'EVIDENCE TRAIL' : 'ARGUMENT GRAPH'}
          </button>
        ))}
      </div>

      {/* Dissent Tab */}
      {activeTab === 'dissent' && (
        <div className="space-y-6">
          {/* Topic Filter */}
          <div className="card p-4">
            <label className="block font-mono text-xs text-text-muted mb-2">
              Filter by Topic
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={topicFilter}
                onChange={(e) => setTopicFilter(e.target.value)}
                placeholder="Search topics..."
                className="flex-1 bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
              />
              <button
                onClick={fetchDissentData}
                className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors"
              >
                Search
              </button>
            </div>
          </div>

          {/* Risk Warnings */}
          {riskWarnings.length > 0 && (
            <div className="card p-4 border-l-4 border-acid-red">
              <h3 className="font-mono text-acid-red mb-4 flex items-center gap-2">
                Risk Warnings ({riskWarnings.length})
              </h3>
              <div className="space-y-3">
                {riskWarnings.map((warning, idx) => {
                  const severityStyle = SEVERITY_COLORS[warning.severity] || SEVERITY_COLORS.low;
                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded ${severityStyle.bg} border border-${warning.severity === 'critical' ? 'acid-red' : 'acid-green'}/30`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className={`font-mono text-xs uppercase ${severityStyle.text}`}>
                          {warning.severity} - {warning.risk_type}
                        </span>
                        <span className="font-mono text-xs text-text-muted">
                          {warning.domain}
                        </span>
                      </div>
                      <p className="font-mono text-sm text-text">{warning.description}</p>
                      {warning.mitigation && (
                        <p className="font-mono text-xs text-acid-green mt-2">
                          Mitigation: {warning.mitigation}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Dissenting Views */}
          <div className="card p-4">
            <h3 className="font-mono text-acid-yellow mb-4">
              Dissenting Views ({dissents.length})
            </h3>
            {dissents.length === 0 ? (
              <p className="text-text-muted font-mono text-sm">
                No dissenting views recorded yet. Dissents are captured when agents disagree during debates.
              </p>
            ) : (
              <div className="space-y-4">
                {dissents.map((dissent, idx) => (
                  <div
                    key={idx}
                    className="p-4 bg-surface rounded border border-acid-yellow/30"
                  >
                    <div className="mb-3">
                      <div className="font-mono text-xs text-acid-cyan mb-1">Topic</div>
                      <div className="font-mono text-sm text-text">{dissent.topic}</div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-3 bg-acid-green/10 rounded">
                        <div className="font-mono text-xs text-acid-green mb-1">Majority View</div>
                        <p className="font-mono text-sm text-text line-clamp-3">
                          {dissent.majority_view}
                        </p>
                      </div>

                      <div className="p-3 bg-acid-yellow/10 rounded">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-mono text-xs text-acid-yellow">Dissenting View</span>
                          <span className="font-mono text-xs text-text-muted">
                            by {dissent.dissenting_agent}
                          </span>
                        </div>
                        <p className="font-mono text-sm text-text line-clamp-3">
                          {dissent.dissenting_view}
                        </p>
                      </div>
                    </div>

                    {dissent.reasoning && (
                      <div className="mt-3 p-2 bg-surface/50 rounded">
                        <div className="font-mono text-xs text-text-muted mb-1">Reasoning</div>
                        <p className="font-mono text-xs text-text">{dissent.reasoning}</p>
                      </div>
                    )}

                    <div className="mt-2 flex items-center gap-4">
                      <span className="font-mono text-xs text-text-muted">
                        Confidence: {(dissent.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Contrarian Views */}
          {contrarianViews.length > 0 && (
            <div className="card p-4">
              <h3 className="font-mono text-acid-cyan mb-4">
                Contrarian Perspectives ({contrarianViews.length})
              </h3>
              <div className="space-y-3">
                {contrarianViews.map((view, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-surface rounded border border-acid-cyan/30"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-sm text-acid-cyan">{view.agent}</span>
                      <span className="font-mono text-xs text-text-muted">
                        {(view.confidence * 100).toFixed(0)}% confident
                      </span>
                    </div>
                    <p className="font-mono text-sm text-text">{view.position}</p>
                    {view.reasoning && (
                      <p className="font-mono text-xs text-text-muted mt-2">{view.reasoning}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Evidence Tab */}
      {activeTab === 'evidence' && (
        <div className="space-y-4">
          <div className="card p-4">
            <label className="block font-mono text-xs text-text-muted mb-2">
              Debate ID
            </label>
            <input
              type="text"
              value={searchDebateId}
              onChange={(e) => setSearchDebateId(e.target.value)}
              placeholder="Enter debate ID to view evidence trail..."
              className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
            />
          </div>

          <div className="card p-4">
            <h3 className="font-mono text-acid-green mb-4">
              Evidence Trail {evidence.length > 0 && `(${evidence.length} citations)`}
            </h3>
            {!searchDebateId ? (
              <p className="text-text-muted font-mono text-sm">
                Enter a debate ID to view its evidence citations and argument chain.
              </p>
            ) : evidence.length === 0 ? (
              <p className="text-text-muted font-mono text-sm">
                No evidence found for this debate ID.
              </p>
            ) : (
              <div className="space-y-3">
                {evidence.map((citation, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-surface rounded border border-acid-green/20"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-xs text-acid-cyan">
                        Round {citation.round} - {citation.agent}
                      </span>
                      {citation.confidence && (
                        <span className="font-mono text-xs text-text-muted">
                          {(citation.confidence * 100).toFixed(0)}% confidence
                        </span>
                      )}
                    </div>
                    <p className="font-mono text-sm text-text mb-2">{citation.claim}</p>
                    <div className="font-mono text-xs text-acid-green">
                      Source: {citation.source}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Graph Tab */}
      {activeTab === 'graph' && (
        <div className="space-y-4">
          <div className="card p-4">
            <label className="block font-mono text-xs text-text-muted mb-2">
              Graph Debate ID
            </label>
            <input
              type="text"
              value={searchDebateId}
              onChange={(e) => setSearchDebateId(e.target.value)}
              placeholder="Enter graph debate ID..."
              className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
            />
          </div>

          <div className="card p-4">
            <h3 className="font-mono text-acid-green mb-4">
              Argument Graph {graphNodes.length > 0 && `(${graphNodes.length} nodes)`}
            </h3>
            {!searchDebateId ? (
              <div className="text-center py-8">
                <p className="text-text-muted font-mono text-sm mb-4">
                  Enter a graph debate ID to visualize its argument structure.
                </p>
                <div className="text-xs font-mono text-acid-cyan">
                  Graph debates allow branching when agents fundamentally disagree.
                </div>
              </div>
            ) : graphNodes.length === 0 ? (
              <p className="text-text-muted font-mono text-sm">
                No graph nodes found for this debate ID. This may not be a graph debate.
              </p>
            ) : (
              <div className="space-y-4">
                {/* Simple tree visualization */}
                <div className="overflow-x-auto">
                  {graphNodes.map((node, idx) => {
                    const depth = node.parent_id ? 1 : 0;
                    return (
                      <div
                        key={node.id}
                        className={`p-3 bg-surface rounded border border-acid-green/20 ${
                          depth > 0 ? 'ml-8 border-l-2 border-acid-cyan/40' : ''
                        }`}
                        style={{ marginTop: idx > 0 ? '0.5rem' : 0 }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-0.5 rounded text-xs font-mono ${
                              node.type === 'argument' ? 'bg-acid-green/20 text-acid-green' :
                              node.type === 'rebuttal' ? 'bg-acid-red/20 text-acid-red' :
                              node.type === 'synthesis' ? 'bg-acid-cyan/20 text-acid-cyan' :
                              'bg-surface text-text-muted'
                            }`}>
                              {node.type}
                            </span>
                            <span className="font-mono text-sm text-acid-yellow">
                              {node.agent}
                            </span>
                          </div>
                          <span className="font-mono text-xs text-text-muted">
                            {node.id.slice(0, 8)}
                          </span>
                        </div>
                        <p className="font-mono text-sm text-text line-clamp-3">
                          {node.content}
                        </p>
                        {node.children && node.children.length > 0 && (
                          <div className="mt-2 text-xs font-mono text-text-muted">
                            {node.children.length} response{node.children.length > 1 ? 's' : ''}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Legend */}
                <div className="p-3 bg-surface/50 rounded">
                  <h4 className="font-mono text-xs text-acid-cyan mb-2">Node Types</h4>
                  <div className="flex flex-wrap gap-4 text-xs font-mono">
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-acid-green/40" />
                      <span className="text-text-muted">Argument</span>
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-acid-red/40" />
                      <span className="text-text-muted">Rebuttal</span>
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-acid-cyan/40" />
                      <span className="text-text-muted">Synthesis</span>
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={fetchDissentData}
          disabled={loading}
          className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh Data'}
        </button>
      </div>
    </div>
  );
}
