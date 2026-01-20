'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { TrendingTopicCard, type TrendingTopic } from './TrendingTopicCard';
import { PulseFilters, type PulseFiltersState, defaultFilters } from './PulseFilters';

export interface TrendingTopicsGridProps {
  apiBase: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onStartDebate?: (topic: TrendingTopic) => void;
  onTopicSelect?: (topic: TrendingTopic | null) => void;
  selectedTopic?: TrendingTopic | null;
  showFilters?: boolean;
  maxTopics?: number;
}

export function TrendingTopicsGrid({
  apiBase,
  autoRefresh = true,
  refreshInterval = 60000,
  onStartDebate,
  onTopicSelect,
  selectedTopic,
  showFilters = true,
  maxTopics = 50,
}: TrendingTopicsGridProps) {
  const [topics, setTopics] = useState<TrendingTopic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [filters, setFilters] = useState<PulseFiltersState>(defaultFilters);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  const fetchTrending = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/pulse/trending?limit=${maxTopics}`);
      if (res.ok) {
        const data = await res.json();
        setTopics(data.topics || []);
        setLastUpdated(new Date());
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.error || 'Failed to fetch trending topics');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setLoading(false);
    }
  }, [apiBase, maxTopics]);

  // Initial fetch
  useEffect(() => {
    fetchTrending();
  }, [fetchTrending]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchTrending, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchTrending]);

  // Filter topics based on current filters
  const filteredTopics = useMemo(() => {
    return topics.filter((topic) => {
      // Search filter
      if (filters.search) {
        const query = filters.search.toLowerCase();
        if (!topic.topic.toLowerCase().includes(query)) {
          return false;
        }
      }

      // Source filter
      if (filters.sources.length > 0) {
        if (!filters.sources.includes(topic.source.toLowerCase())) {
          return false;
        }
      }

      // Category filter
      if (filters.categories.length > 0) {
        if (!topic.category || !filters.categories.includes(topic.category.toLowerCase())) {
          return false;
        }
      }

      // Score filter
      if (filters.minScore > 0 && topic.score < filters.minScore) {
        return false;
      }

      // Time filter
      if (filters.timeRange !== 'all' && topic.last_active) {
        const now = Date.now();
        const topicTime = new Date(topic.last_active).getTime();
        const diff = now - topicTime;
        const hour = 3600000;
        const day = hour * 24;
        const week = day * 7;

        if (filters.timeRange === 'hour' && diff > hour) return false;
        if (filters.timeRange === 'day' && diff > day) return false;
        if (filters.timeRange === 'week' && diff > week) return false;
      }

      return true;
    });
  }, [topics, filters]);

  // Get unique sources and categories from topics for filter options
  const availableSources = useMemo(() => {
    return [...new Set(topics.map((t) => t.source.toLowerCase()))];
  }, [topics]);

  const availableCategories = useMemo(() => {
    return [...new Set(topics.filter((t) => t.category).map((t) => t.category!.toLowerCase()))];
  }, [topics]);

  const handleTopicClick = useCallback(
    (topic: TrendingTopic) => {
      if (selectedTopic?.topic === topic.topic) {
        onTopicSelect?.(null);
      } else {
        onTopicSelect?.(topic);
      }
    },
    [selectedTopic, onTopicSelect]
  );

  const handleStartDebate = useCallback(
    (topic: TrendingTopic) => {
      onStartDebate?.(topic);
    },
    [onStartDebate]
  );

  if (error && topics.length === 0) {
    return (
      <div className="bg-surface border border-border rounded-lg p-8 text-center">
        <div className="text-4xl mb-4">⚠️</div>
        <p className="text-text-muted mb-4">{error}</p>
        <button
          onClick={fetchTrending}
          className="px-4 py-2 bg-acid-green/20 border border-acid-green text-acid-green text-sm font-mono rounded hover:bg-acid-green/30"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with stats and controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-sm font-mono text-acid-green uppercase">
            Trending Topics ({filteredTopics.length})
          </h2>
          {lastUpdated && (
            <span className="text-xs text-text-muted">
              Updated {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
          {loading && (
            <span className="text-xs text-acid-cyan animate-pulse">Refreshing...</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* View mode toggle */}
          <div className="flex gap-1 border border-border rounded overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-2 py-1 text-xs font-mono ${
                viewMode === 'grid'
                  ? 'bg-acid-green text-bg'
                  : 'bg-surface text-text-muted hover:text-text'
              }`}
              title="Grid view"
            >
              ▦
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`px-2 py-1 text-xs font-mono ${
                viewMode === 'list'
                  ? 'bg-acid-green text-bg'
                  : 'bg-surface text-text-muted hover:text-text'
              }`}
              title="List view"
            >
              ≡
            </button>
          </div>

          {/* Refresh button */}
          <button
            onClick={fetchTrending}
            disabled={loading}
            className="px-3 py-1 text-xs font-mono text-acid-green border border-acid-green/30 rounded hover:bg-acid-green/10 disabled:opacity-50"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Filters sidebar */}
        {showFilters && (
          <div className="lg:col-span-1">
            <PulseFilters
              filters={filters}
              onChange={setFilters}
              availableSources={availableSources.length > 0 ? availableSources : undefined}
              availableCategories={availableCategories.length > 0 ? availableCategories : undefined}
            />
          </div>
        )}

        {/* Topics grid/list */}
        <div className={showFilters ? 'lg:col-span-3' : 'lg:col-span-4'}>
          {filteredTopics.length === 0 ? (
            <div className="bg-surface border border-border rounded-lg p-8 text-center">
              <div className="text-4xl mb-4">🔍</div>
              <p className="text-text-muted mb-2">No topics match your filters</p>
              <button
                onClick={() => setFilters(defaultFilters)}
                className="text-xs font-mono text-acid-green hover:underline"
              >
                Clear filters
              </button>
            </div>
          ) : viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filteredTopics.map((topic, idx) => (
                <TrendingTopicCard
                  key={`${topic.source}-${topic.topic}-${idx}`}
                  topic={topic}
                  isSelected={selectedTopic?.topic === topic.topic}
                  onClick={() => handleTopicClick(topic)}
                  onStartDebate={() => handleStartDebate(topic)}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredTopics.map((topic, idx) => (
                <TrendingTopicCard
                  key={`${topic.source}-${topic.topic}-${idx}`}
                  topic={topic}
                  isSelected={selectedTopic?.topic === topic.topic}
                  onClick={() => handleTopicClick(topic)}
                  onStartDebate={() => handleStartDebate(topic)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TrendingTopicsGrid;
