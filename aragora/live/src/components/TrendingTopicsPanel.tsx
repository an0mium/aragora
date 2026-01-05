'use client';

import { useState, useEffect, useCallback } from 'react';

interface TrendingTopic {
  topic: string;
  source: string;
  score: number;
  debate_count?: number;
  last_active?: string;
  category?: string;
}

interface TrendingTopicsPanelProps {
  apiBase: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function TrendingTopicsPanel({
  apiBase,
  autoRefresh = true,
  refreshInterval = 60000,
}: TrendingTopicsPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [topics, setTopics] = useState<TrendingTopic[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchTrending = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/pulse/trending?limit=10`);
      if (res.ok) {
        const data = await res.json();
        setTopics(data.topics || []);
        setLastUpdated(new Date());
      } else {
        setError('Failed to fetch trending topics');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    if (expanded) {
      fetchTrending();
    }
  }, [expanded, fetchTrending]);

  useEffect(() => {
    if (!expanded || !autoRefresh) return;
    const interval = setInterval(fetchTrending, refreshInterval);
    return () => clearInterval(interval);
  }, [expanded, autoRefresh, refreshInterval, fetchTrending]);

  const getSourceIcon = (source: string): string => {
    switch (source.toLowerCase()) {
      case 'hackernews': return '🔶';
      case 'arxiv': return '📄';
      case 'reddit': return '🤖';
      case 'twitter': return '🐦';
      case 'github': return '🐙';
      case 'debate': return '💬';
      default: return '📡';
    }
  };

  const getScoreColor = (score: number): string => {
    if (score >= 0.8) return 'text-yellow-400';
    if (score >= 0.6) return 'text-green-400';
    if (score >= 0.4) return 'text-blue-400';
    return 'text-text-muted';
  };

  const formatTimeAgo = (timestamp: string): string => {
    const now = new Date();
    const then = new Date(timestamp);
    const diffMs = now.getTime() - then.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return then.toLocaleDateString();
  };

  return (
    <div className="border border-acid-green/30 bg-surface/50">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-surface/80 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">🔥</span>
          <span className="text-acid-cyan font-mono text-sm">[TRENDING]</span>
          {topics.length > 0 && !expanded && (
            <span className="text-xs text-text-muted">
              {topics[0]?.topic?.slice(0, 30)}...
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {lastUpdated && expanded && (
            <span className="text-xs text-text-muted">
              {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
          <span className="text-acid-green">{expanded ? '[-]' : '[+]'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          {loading && topics.length === 0 && (
            <div className="text-text-muted text-xs text-center py-4 animate-pulse">
              Scanning for trending topics...
            </div>
          )}

          {error && (
            <div className="text-warning text-xs text-center py-4">{error}</div>
          )}

          {!loading && !error && topics.length === 0 && (
            <div className="text-text-muted text-xs text-center py-4">
              No trending topics detected
            </div>
          )}

          {topics.length > 0 && (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {topics.map((topic, idx) => (
                <div
                  key={idx}
                  className="p-2 bg-bg border border-acid-green/20 hover:border-acid-green/40 transition-colors"
                >
                  <div className="flex items-start gap-2">
                    <span className="text-sm" title={topic.source}>
                      {getSourceIcon(topic.source)}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-mono text-acid-green truncate">
                          {topic.topic}
                        </span>
                        <span className={`text-xs ${getScoreColor(topic.score)}`}>
                          {Math.round(topic.score * 100)}%
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-text-muted">
                        <span className="capitalize">{topic.source}</span>
                        {topic.debate_count !== undefined && topic.debate_count > 0 && (
                          <span>{topic.debate_count} debates</span>
                        )}
                        {topic.last_active && (
                          <span>{formatTimeAgo(topic.last_active)}</span>
                        )}
                        {topic.category && (
                          <span className="px-1 border border-text-muted/30 rounded">
                            {topic.category}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Refresh Button */}
          <button
            onClick={fetchTrending}
            disabled={loading}
            className="w-full py-2 text-xs text-text-muted hover:text-text border border-acid-green/20 hover:border-acid-green/40 transition-colors disabled:opacity-50"
          >
            {loading ? 'Refreshing...' : 'Refresh Trending'}
          </button>

          {/* Legend */}
          <div className="text-xs text-text-muted/50 flex flex-wrap gap-3">
            <span>🔶 HN</span>
            <span>📄 arXiv</span>
            <span>🐙 GitHub</span>
            <span>💬 Debates</span>
          </div>
        </div>
      )}
    </div>
  );
}
