'use client';

import { useState, useEffect, useCallback } from 'react';

interface DebateSummary {
  id: string;
  slug?: string;
  task: string;
  created_at: string;
  agents: string[];
  winner?: string;
  consensus_reached: boolean;
  rounds_used: number;
  duration_seconds?: number;
}

interface DebateListPanelProps {
  onSelectDebate?: (debateId: string) => void;
  limit?: number;
}

export function DebateListPanel({ onSelectDebate, limit = 20 }: DebateListPanelProps) {
  const [debates, setDebates] = useState<DebateSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'consensus' | 'no-consensus'>('all');
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

  const fetchDebates = useCallback(async (reset = false) => {
    try {
      setLoading(true);
      const currentOffset = reset ? 0 : offset;
      const response = await fetch(
        `${apiBase}/api/debates?limit=${limit}&offset=${currentOffset}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch debates: ${response.status}`);
      }

      const data = await response.json();
      const newDebates = data.debates || [];

      if (reset) {
        setDebates(newDebates);
        setOffset(limit);
      } else {
        setDebates(prev => [...prev, ...newDebates]);
        setOffset(prev => prev + limit);
      }

      setHasMore(newDebates.length === limit);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load debates');
    } finally {
      setLoading(false);
    }
  }, [apiBase, limit, offset]);

  useEffect(() => {
    fetchDebates(true);
  }, []);

  const filteredDebates = debates.filter(debate => {
    if (filter === 'consensus') return debate.consensus_reached;
    if (filter === 'no-consensus') return !debate.consensus_reached;
    return true;
  });

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateStr;
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '-';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    return `${Math.round(seconds / 60)}m`;
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Debate History</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1 text-sm rounded ${
              filter === 'all'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilter('consensus')}
            className={`px-3 py-1 text-sm rounded ${
              filter === 'consensus'
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Consensus
          </button>
          <button
            onClick={() => setFilter('no-consensus')}
            className={`px-3 py-1 text-sm rounded ${
              filter === 'no-consensus'
                ? 'bg-yellow-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            No Consensus
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">
          {error}
        </div>
      )}

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {filteredDebates.length === 0 && !loading ? (
          <div className="text-gray-400 text-center py-8">
            No debates found
          </div>
        ) : (
          filteredDebates.map((debate) => (
            <div
              key={debate.id}
              onClick={() => onSelectDebate?.(debate.id)}
              className="p-3 bg-gray-800 rounded border border-gray-700 hover:border-gray-500 cursor-pointer transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm font-medium truncate">
                    {debate.task}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-gray-400 text-xs">
                      {formatDate(debate.created_at)}
                    </span>
                    <span className="text-gray-500">|</span>
                    <span className="text-gray-400 text-xs">
                      {debate.rounds_used} rounds
                    </span>
                    <span className="text-gray-500">|</span>
                    <span className="text-gray-400 text-xs">
                      {formatDuration(debate.duration_seconds)}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 mt-2">
                    {debate.agents.slice(0, 4).map((agent, i) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300"
                      >
                        {agent}
                      </span>
                    ))}
                    {debate.agents.length > 4 && (
                      <span className="text-gray-500 text-xs">
                        +{debate.agents.length - 4}
                      </span>
                    )}
                  </div>
                </div>
                <div className="ml-3 flex-shrink-0">
                  {debate.consensus_reached ? (
                    <span className="px-2 py-1 bg-green-900/50 text-green-400 text-xs rounded">
                      Consensus
                    </span>
                  ) : (
                    <span className="px-2 py-1 bg-yellow-900/50 text-yellow-400 text-xs rounded">
                      No Consensus
                    </span>
                  )}
                </div>
              </div>
              {debate.winner && (
                <div className="mt-2 text-xs text-gray-400">
                  Winner: <span className="text-blue-400">{debate.winner}</span>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {hasMore && !loading && (
        <button
          onClick={() => fetchDebates(false)}
          className="mt-4 w-full py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 text-sm rounded transition-colors"
        >
          Load More
        </button>
      )}

      {loading && (
        <div className="flex justify-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
        </div>
      )}
    </div>
  );
}
