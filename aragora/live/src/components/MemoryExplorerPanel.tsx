'use client';

import { useState, useEffect, useCallback } from 'react';
import { ErrorWithRetry } from './RetryButton';
import { fetchWithRetry } from '@/utils/retry';

interface TierStats {
  count: number;
  avg_importance: number;
  ttl_hours?: number;
  max_entries?: number;
}

interface MemoryEntry {
  id: string;
  content: string;
  tier: string;
  importance: number;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

interface TierTransition {
  from_tier: string;
  to_tier: string;
  count: number;
  timestamp?: string;
}

interface MemoryStats {
  tiers: Record<string, TierStats>;
  total_memories: number;
  transitions: TierTransition[];
}

interface MemoryPressure {
  fast: { usage: number; limit: number };
  medium: { usage: number; limit: number };
  slow: { usage: number; limit: number };
  glacial: { usage: number; limit: number };
  overall_pressure: number;
}

interface BackendConfig {
  apiUrl: string;
  wsUrl: string;
}

interface MemoryExplorerPanelProps {
  backendConfig?: BackendConfig;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

const TIER_COLORS: Record<string, string> = {
  fast: 'text-acid-red',
  medium: 'text-acid-yellow',
  slow: 'text-acid-cyan',
  glacial: 'text-acid-blue',
};

const TIER_BG_COLORS: Record<string, string> = {
  fast: 'bg-acid-red/20 border-acid-red/40',
  medium: 'bg-acid-yellow/20 border-acid-yellow/40',
  slow: 'bg-acid-cyan/20 border-acid-cyan/40',
  glacial: 'bg-acid-blue/20 border-acid-blue/40',
};

const TIER_DESCRIPTIONS: Record<string, string> = {
  fast: 'Immediate context (TTL: ~1 min)',
  medium: 'Session memory (TTL: ~1 hour)',
  slow: 'Cross-session (TTL: ~1 day)',
  glacial: 'Long-term patterns (TTL: ~1 week)',
};

export function MemoryExplorerPanel({ backendConfig }: MemoryExplorerPanelProps) {
  const apiBase = backendConfig?.apiUrl || DEFAULT_API_BASE;

  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [pressure, setPressure] = useState<MemoryPressure | null>(null);
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [usingDemoData, setUsingDemoData] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'search' | 'transitions'>('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTiers, setSelectedTiers] = useState<string[]>(['fast', 'medium', 'slow', 'glacial']);
  const [minImportance, setMinImportance] = useState(0);

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true);
      const [statsRes, pressureRes] = await Promise.allSettled([
        fetchWithRetry(`${apiBase}/api/memory/tier-stats`, undefined, { maxRetries: 2 }),
        fetchWithRetry(`${apiBase}/api/memory/pressure`, undefined, { maxRetries: 2 }),
      ]);

      if (statsRes.status === 'fulfilled' && statsRes.value.ok) {
        const data = await statsRes.value.json();
        setStats(data);
        setUsingDemoData(false);
      } else {
        // Demo data when API unavailable
        setUsingDemoData(true);
        setStats({
          tiers: {
            fast: { count: 0, avg_importance: 0, ttl_hours: 0.017, max_entries: 100 },
            medium: { count: 0, avg_importance: 0, ttl_hours: 1, max_entries: 500 },
            slow: { count: 0, avg_importance: 0, ttl_hours: 24, max_entries: 1000 },
            glacial: { count: 0, avg_importance: 0, ttl_hours: 168, max_entries: 5000 },
          },
          total_memories: 0,
          transitions: [],
        });
      }

      if (pressureRes.status === 'fulfilled' && pressureRes.value.ok) {
        const data = await pressureRes.value.json();
        setPressure(data);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch memory stats');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const searchMemories = useCallback(async () => {
    if (!searchQuery.trim()) {
      setMemories([]);
      return;
    }

    try {
      const tiersParam = selectedTiers.join(',');
      const response = await fetchWithRetry(
        `${apiBase}/api/memory/continuum/retrieve?query=${encodeURIComponent(searchQuery)}&tiers=${tiersParam}&limit=20&min_importance=${minImportance}`,
        undefined,
        { maxRetries: 2 }
      );

      if (response.ok) {
        const data = await response.json();
        setMemories(data.memories || []);
      }
    } catch (err) {
      console.error('Search failed:', err);
    }
  }, [apiBase, searchQuery, selectedTiers, minImportance]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  useEffect(() => {
    const debounce = setTimeout(() => {
      if (activeTab === 'search') {
        searchMemories();
      }
    }, 300);
    return () => clearTimeout(debounce);
  }, [searchQuery, selectedTiers, minImportance, activeTab, searchMemories]);

  if (loading && !stats) {
    return (
      <div className="card p-6">
        <div className="flex items-center gap-3">
          <div className="animate-spin w-5 h-5 border-2 border-acid-green border-t-transparent rounded-full" />
          <span className="font-mono text-text-muted">Loading memory stats...</span>
        </div>
      </div>
    );
  }

  if (error && !stats) {
    return (
      <ErrorWithRetry
        error={error || "Failed to load memory statistics"}
        onRetry={fetchStats}
      />
    );
  }

  const tiers = ['fast', 'medium', 'slow', 'glacial'];

  return (
    <div className="space-y-6">
      {/* Demo Mode Indicator */}
      {usingDemoData && (
        <div className="bg-warning/10 border border-warning/30 rounded px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-warning">⚠</span>
            <span className="font-mono text-sm text-warning">
              Demo Mode - Memory API unavailable
            </span>
          </div>
          <button
            onClick={fetchStats}
            className="font-mono text-xs text-warning hover:text-warning/80 transition-colors"
          >
            [RETRY]
          </button>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-acid-green/20 pb-2">
        {(['overview', 'search', 'transitions'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 font-mono text-sm transition-colors ${
              activeTab === tab
                ? 'text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && stats && (
        <div className="space-y-6">
          {/* Summary Stats */}
          <div className="card p-4">
            <h3 className="font-mono text-acid-green mb-4">Memory System Status</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-3xl font-mono text-acid-green">{stats.total_memories}</div>
                <div className="text-xs font-mono text-text-muted">Total Memories</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono text-acid-cyan">{tiers.length}</div>
                <div className="text-xs font-mono text-text-muted">Active Tiers</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono text-acid-yellow">
                  {stats.transitions?.length || 0}
                </div>
                <div className="text-xs font-mono text-text-muted">Transitions</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-mono text-acid-red">
                  {pressure ? `${Math.round(pressure.overall_pressure * 100)}%` : 'N/A'}
                </div>
                <div className="text-xs font-mono text-text-muted">Pressure</div>
              </div>
            </div>
          </div>

          {/* Tier Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {tiers.map((tier) => {
              const tierStats = stats.tiers?.[tier] || { count: 0, avg_importance: 0 };
              const tierPressure = pressure?.[tier as keyof MemoryPressure];
              const usagePercent = tierPressure && typeof tierPressure === 'object'
                ? Math.round((tierPressure.usage / tierPressure.limit) * 100)
                : 0;

              return (
                <div
                  key={tier}
                  className={`card p-4 border ${TIER_BG_COLORS[tier]}`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className={`font-mono font-bold uppercase ${TIER_COLORS[tier]}`}>
                      {tier}
                    </h4>
                    <span className="text-xs font-mono text-text-muted">
                      {tierStats.count} entries
                    </span>
                  </div>

                  <p className="text-xs font-mono text-text-muted mb-3">
                    {TIER_DESCRIPTIONS[tier]}
                  </p>

                  {/* Progress Bar */}
                  <div className="mb-2">
                    <div className="h-2 bg-surface rounded-full overflow-hidden">
                      <div
                        className={`h-full ${TIER_COLORS[tier].replace('text-', 'bg-')} transition-all`}
                        style={{ width: `${Math.min(usagePercent, 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs font-mono text-text-muted mt-1">
                      <span>{usagePercent}% used</span>
                      <span>
                        {tierPressure && typeof tierPressure === 'object'
                          ? `${tierPressure.usage}/${tierPressure.limit}`
                          : 'N/A'}
                      </span>
                    </div>
                  </div>

                  <div className="text-xs font-mono text-text-muted">
                    Avg importance: {(tierStats.avg_importance || 0).toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Search Tab */}
      {activeTab === 'search' && (
        <div className="space-y-4">
          {/* Search Controls */}
          <div className="card p-4 space-y-4">
            <div>
              <label className="block font-mono text-xs text-text-muted mb-2">
                Search Query
              </label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search memories..."
                className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
              />
            </div>

            <div className="flex flex-wrap gap-4">
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">
                  Tiers
                </label>
                <div className="flex gap-2">
                  {tiers.map((tier) => (
                    <button
                      key={tier}
                      onClick={() => {
                        setSelectedTiers((prev) =>
                          prev.includes(tier)
                            ? prev.filter((t) => t !== tier)
                            : [...prev, tier]
                        );
                      }}
                      className={`px-3 py-1 font-mono text-xs rounded border transition-colors ${
                        selectedTiers.includes(tier)
                          ? `${TIER_BG_COLORS[tier]} ${TIER_COLORS[tier]}`
                          : 'border-text-muted/30 text-text-muted'
                      }`}
                    >
                      {tier}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">
                  Min Importance: {minImportance.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={minImportance}
                  onChange={(e) => setMinImportance(parseFloat(e.target.value))}
                  className="w-32"
                />
              </div>
            </div>
          </div>

          {/* Search Results */}
          <div className="card p-4">
            <h3 className="font-mono text-acid-green mb-4">
              Results ({memories.length})
            </h3>
            {memories.length === 0 ? (
              <p className="text-text-muted font-mono text-sm">
                {searchQuery ? 'No memories found.' : 'Enter a search query to find memories.'}
              </p>
            ) : (
              <div className="space-y-3">
                {memories.map((memory) => (
                  <div
                    key={memory.id}
                    className={`p-3 rounded border ${TIER_BG_COLORS[memory.tier] || 'border-text-muted/30'}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={`font-mono text-xs uppercase ${TIER_COLORS[memory.tier]}`}>
                        {memory.tier}
                      </span>
                      <span className="font-mono text-xs text-text-muted">
                        importance: {memory.importance.toFixed(2)}
                      </span>
                    </div>
                    <p className="font-mono text-sm text-text line-clamp-3">
                      {memory.content}
                    </p>
                    {memory.created_at && (
                      <p className="font-mono text-xs text-text-muted mt-2">
                        Created: {new Date(memory.created_at).toLocaleString()}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Transitions Tab */}
      {activeTab === 'transitions' && stats && (
        <div className="card p-4">
          <h3 className="font-mono text-acid-green mb-4">Tier Transitions</h3>
          {(!stats.transitions || stats.transitions.length === 0) ? (
            <p className="text-text-muted font-mono text-sm">
              No tier transitions recorded yet. Transitions occur when memories
              are promoted or demoted based on importance and access patterns.
            </p>
          ) : (
            <div className="space-y-2">
              {stats.transitions.map((transition, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 bg-surface rounded font-mono text-sm"
                >
                  <span className={TIER_COLORS[transition.from_tier]}>
                    {transition.from_tier}
                  </span>
                  <span className="text-text-muted">→</span>
                  <span className={TIER_COLORS[transition.to_tier]}>
                    {transition.to_tier}
                  </span>
                  <span className="text-text-muted ml-auto">
                    {transition.count} entries
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Transition Legend */}
          <div className="mt-6 p-3 bg-surface/50 rounded">
            <h4 className="font-mono text-xs text-acid-cyan mb-2">
              How Transitions Work
            </h4>
            <ul className="font-mono text-xs text-text-muted space-y-1">
              <li>• <span className="text-acid-red">Fast</span> → <span className="text-acid-yellow">Medium</span>: Frequently accessed memories promote up</li>
              <li>• <span className="text-acid-yellow">Medium</span> → <span className="text-acid-cyan">Slow</span>: Important patterns persist longer</li>
              <li>• <span className="text-acid-cyan">Slow</span> → <span className="text-acid-blue">Glacial</span>: Critical insights archive long-term</li>
              <li>• Reverse transitions occur when TTL expires or limits exceed</li>
            </ul>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={fetchStats}
          disabled={loading}
          className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh Stats'}
        </button>
      </div>
    </div>
  );
}
