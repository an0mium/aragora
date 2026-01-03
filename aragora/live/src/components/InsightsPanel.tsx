'use client';

import { useState, useEffect, useCallback } from 'react';

interface Insight {
  id: string;
  type: string;
  title: string;
  description: string;
  confidence: number;
  agents_involved: string[];
  evidence: string[];
}

interface MemoryRecall {
  query: string;
  hits: Array<{ topic: string; similarity: number }>;
  count: number;
  timestamp: string;
}

interface InsightsPanelProps {
  wsMessages?: any[];
}

export function InsightsPanel({ wsMessages = [] }: InsightsPanelProps) {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [memoryRecalls, setMemoryRecalls] = useState<MemoryRecall[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'insights' | 'memory'>('insights');

  const fetchInsights = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/insights/recent?limit=10');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setInsights(data.insights || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch insights');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchInsights();
  }, [fetchInsights]);

  // Listen for memory_recall WebSocket events
  useEffect(() => {
    const recallMessages = wsMessages
      .filter((msg) => msg.type === 'memory_recall')
      .map((msg) => ({
        ...msg.data,
        timestamp: msg.timestamp || new Date().toISOString(),
      }));

    if (recallMessages.length > 0) {
      setMemoryRecalls((prev) => {
        const newRecalls = [...recallMessages, ...prev].slice(0, 20);
        return newRecalls;
      });
    }
  }, [wsMessages]);

  const getTypeColor = (type: string): string => {
    switch (type) {
      case 'consensus':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'pattern':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'agent_performance':
        return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
      case 'divergence':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text">Debate Insights</h3>
        <button
          onClick={fetchInsights}
          className="px-2 py-1 bg-surface border border-border rounded text-sm text-text hover:bg-surface-hover"
        >
          Refresh
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-bg border border-border rounded p-1 mb-4">
        <button
          onClick={() => setActiveTab('insights')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'insights'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Insights ({insights.length})
        </button>
        <button
          onClick={() => setActiveTab('memory')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'memory'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Memory Recalls ({memoryRecalls.length})
        </button>
      </div>

      {/* Insights Tab */}
      {activeTab === 'insights' && (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-4">Loading insights...</div>
          )}

          {error && (
            <div className="text-center text-red-400 py-4">
              <div className="text-sm">{error}</div>
            </div>
          )}

          {!loading && !error && insights.length === 0 && (
            <div className="text-center text-text-muted py-4">
              No insights extracted yet. Run a debate cycle to generate insights.
            </div>
          )}

          {insights.map((insight) => (
            <div
              key={insight.id}
              className="p-3 bg-bg border border-border rounded-lg hover:border-accent/50 transition-colors"
            >
              <div className="flex items-start justify-between gap-2">
                <span
                  className={`px-2 py-0.5 text-xs rounded border ${getTypeColor(insight.type)}`}
                >
                  {insight.type}
                </span>
                <span
                  className={`text-xs font-mono ${getConfidenceColor(insight.confidence)}`}
                >
                  {(insight.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <h4 className="text-sm font-medium text-text mt-2">{insight.title}</h4>

              <p className="text-xs text-text-muted mt-1 line-clamp-2">
                {insight.description}
              </p>

              {insight.agents_involved?.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {insight.agents_involved.map((agent, i) => (
                    <span
                      key={i}
                      className="px-1.5 py-0.5 text-xs bg-surface rounded text-text-muted"
                    >
                      {agent}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Memory Recalls Tab */}
      {activeTab === 'memory' && (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {memoryRecalls.length === 0 && (
            <div className="text-center text-text-muted py-4">
              No memory recalls yet. Historical context will appear here during debates.
            </div>
          )}

          {memoryRecalls.map((recall, index) => (
            <div
              key={`${recall.timestamp}-${index}`}
              className="p-3 bg-bg border border-border rounded-lg"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="px-2 py-0.5 text-xs bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 rounded">
                  Memory Recall
                </span>
                <span className="text-xs text-text-muted">
                  {new Date(recall.timestamp).toLocaleTimeString()}
                </span>
              </div>

              <p className="text-sm text-text-muted mb-2 line-clamp-1">
                Query: {recall.query}
              </p>

              <div className="space-y-1">
                {recall.hits?.map((hit, i) => (
                  <div key={i} className="flex justify-between text-xs">
                    <span className="text-text truncate flex-1 mr-2">{hit.topic}</span>
                    <span className="text-text-muted font-mono">
                      {(hit.similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>

              {recall.count > 3 && (
                <div className="text-xs text-text-muted mt-1">
                  +{recall.count - 3} more matches
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
