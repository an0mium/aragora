'use client';

import { useState, useCallback } from 'react';
import useSWR from 'swr';
import { API_BASE_URL } from '@/config';
import { useUnifiedMemoryQuery, useMemorySources } from '@/hooks/useUnifiedMemory';

interface LearningInsight {
  content: string;
  source: 'continuum' | 'km' | 'supermemory' | 'claude_mem';
  relevance: number;
  metadata: Record<string, unknown>;
}

const sourceLabels: Record<string, string> = {
  continuum: 'Continuum Memory',
  km: 'Knowledge Mound',
  supermemory: 'Supermemory',
  claude_mem: 'Claude-Mem',
};

const sourceColors: Record<string, string> = {
  continuum: 'text-acid-green border-acid-green/30 bg-acid-green/10',
  km: 'text-acid-cyan border-acid-cyan/30 bg-acid-cyan/10',
  supermemory: 'text-acid-purple border-purple-400/30 bg-purple-900/10',
  claude_mem: 'text-yellow-400 border-yellow-400/30 bg-yellow-900/10',
};

interface DebateOutcome {
  debate_id: string;
  task: string;
  consensus_score: number;
  outcome: string;
  timestamp: string;
  insights: string[];
}

interface CrossDebateResponse {
  outcomes: DebateOutcome[];
  total: number;
}

const fetcher = (url: string) => fetch(url).then((r) => r.json());

/**
 * Cross-debate learning panel: surfaces institutional knowledge from past debates,
 * supermemory context, and cross-system learning insights.
 */
export function CrossDebateLearning() {
  const [searchQuery, setSearchQuery] = useState('');
  const { search, results, perSystem, loading: searchLoading } = useUnifiedMemoryQuery();
  const { sources, loading: sourcesLoading } = useMemorySources();

  // Fetch recent debate outcomes from KM
  const { data: debateOutcomes, isLoading: outcomesLoading } = useSWR<CrossDebateResponse>(
    `${API_BASE_URL}/api/v1/knowledge/mound/query?query=debate+outcome&limit=10`,
    fetcher,
    { refreshInterval: 60000 }
  );

  const handleSearch = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!searchQuery.trim()) return;
      // Search across all systems with debate-focused query
      await search(`debate insights: ${searchQuery}`, ['continuum', 'km', 'supermemory', 'claude_mem']);
    },
    [search, searchQuery]
  );

  const handleQuickQuery = useCallback(
    async (query: string) => {
      setSearchQuery(query);
      await search(`debate insights: ${query}`, ['continuum', 'km', 'supermemory', 'claude_mem']);
    },
    [search]
  );

  const activeSources = sources.filter((s) => s.status === 'active');
  const outcomes = debateOutcomes?.outcomes ?? [];

  return (
    <div className="space-y-6">
      {/* Memory Sources Status */}
      <div className="grid grid-cols-4 gap-3">
        {(['continuum', 'km', 'supermemory', 'claude_mem'] as const).map((system) => {
          const src = sources.find((s) => s.name === system);
          const isActive = src?.status === 'active';
          return (
            <div
              key={system}
              className={`p-3 rounded border ${
                isActive
                  ? sourceColors[system]
                  : 'border-border text-text-muted bg-surface/50'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-mono uppercase tracking-wider">
                  {sourceLabels[system]}
                </span>
                <span
                  className={`w-2 h-2 rounded-full ${
                    isActive ? 'bg-green-400' : 'bg-red-400'
                  }`}
                />
              </div>
              <div className="text-lg font-mono">
                {sourcesLoading ? '...' : src?.entry_count?.toLocaleString() ?? '0'}
              </div>
              <div className="text-xs text-text-muted">
                {src?.last_activity
                  ? `Active ${new Date(src.last_activity).toLocaleDateString()}`
                  : 'No activity'}
              </div>
            </div>
          );
        })}
      </div>

      {/* Search Across All Memory Systems */}
      <div className="p-4 border border-acid-green/30 rounded bg-surface/50">
        <h3 className="text-sm font-mono text-acid-green mb-3">
          Cross-System Knowledge Search
        </h3>
        <form onSubmit={handleSearch} className="flex gap-2 mb-3">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search institutional knowledge across all memory systems..."
            className="flex-1 px-3 py-2 bg-bg border border-border rounded text-sm text-text placeholder-text-muted font-mono focus:border-acid-green focus:outline-none"
          />
          <button
            type="submit"
            disabled={searchLoading || !searchQuery.trim()}
            className="px-4 py-2 text-sm font-mono bg-acid-green/10 text-acid-green border border-acid-green/30 rounded hover:bg-acid-green/20 disabled:opacity-50 transition-colors"
          >
            {searchLoading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {/* Quick queries */}
        <div className="flex gap-2 flex-wrap">
          {[
            'recent decisions',
            'consensus patterns',
            'failed debates',
            'architecture decisions',
            'security findings',
          ].map((q) => (
            <button
              key={q}
              onClick={() => handleQuickQuery(q)}
              className="px-2 py-1 text-xs font-mono text-text-muted border border-border rounded hover:border-acid-green/30 hover:text-acid-green transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-mono text-text-muted">
              {results.length} results across {activeSources.length} systems
            </h3>
            <div className="flex gap-3 text-xs font-mono">
              {Object.entries(perSystem).map(([sys, count]) =>
                count > 0 ? (
                  <span key={sys} className={sourceColors[sys]?.split(' ')[0] || 'text-text-muted'}>
                    {sourceLabels[sys] || sys}: {count}
                  </span>
                ) : null
              )}
            </div>
          </div>

          {results.map((result: LearningInsight, idx: number) => (
            <div
              key={idx}
              className="p-3 border border-border rounded hover:border-acid-green/30 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className={`text-xs font-mono px-2 py-0.5 rounded border ${
                    sourceColors[result.source] || 'text-text-muted border-border'
                  }`}
                >
                  {sourceLabels[result.source] || result.source}
                </span>
                <span className="text-xs font-mono text-text-muted">
                  relevance: {(result.relevance * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-text line-clamp-3">{result.content}</p>
              {result.metadata?.debate_id != null ? (
                <div className="mt-2 text-xs font-mono text-text-muted">
                  debate: {String(result.metadata.debate_id).slice(0, 12)}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      )}

      {/* Recent Debate Outcomes */}
      <div>
        <h3 className="text-sm font-mono text-acid-cyan mb-3">
          Recent Debate Outcomes (Knowledge Mound)
        </h3>
        {outcomesLoading ? (
          <div className="text-xs text-text-muted font-mono">Loading outcomes...</div>
        ) : outcomes.length > 0 ? (
          <div className="space-y-2">
            {outcomes.map((outcome) => (
              <div
                key={outcome.debate_id}
                className="p-3 border border-acid-cyan/20 rounded bg-acid-cyan/5"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-mono text-text truncate mr-4">
                    {outcome.task}
                  </span>
                  <span
                    className={`text-xs font-mono px-2 py-0.5 rounded ${
                      outcome.consensus_score > 0.7
                        ? 'bg-green-900/20 text-green-400'
                        : outcome.consensus_score > 0.4
                          ? 'bg-yellow-900/20 text-yellow-400'
                          : 'bg-red-900/20 text-red-400'
                    }`}
                  >
                    consensus: {(outcome.consensus_score * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-xs text-text-muted">{outcome.outcome}</p>
                {outcome.insights && outcome.insights.length > 0 && (
                  <div className="mt-2 flex gap-1 flex-wrap">
                    {outcome.insights.slice(0, 3).map((insight, i) => (
                      <span
                        key={i}
                        className="text-xs font-mono px-1.5 py-0.5 bg-acid-cyan/10 text-acid-cyan rounded"
                      >
                        {insight}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-text-muted text-sm font-mono">
            No debate outcomes recorded yet. Run debates to build institutional knowledge.
          </div>
        )}
      </div>
    </div>
  );
}
