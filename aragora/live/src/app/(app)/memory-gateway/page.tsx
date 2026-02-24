'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useSWRFetch } from '@/hooks/useSWRFetch';

interface MemorySource {
  name: string;
  status: 'connected' | 'disconnected' | 'degraded';
  item_count: number;
  last_query_ms: number;
}

interface SourcesResponse {
  sources: MemorySource[];
  gateway_enabled: boolean;
}

interface RetentionDecision {
  memory_id: string;
  action: 'retain' | 'demote' | 'forget' | 'consolidate';
  surprise_score: number;
  confidence: number;
  source: string;
  timestamp: string;
}

interface RetentionResponse {
  decisions: RetentionDecision[];
  count: number;
}

interface DedupCluster {
  cluster_id: string;
  items: Array<{ id: string; source: string; preview: string; similarity: number }>;
  duplicate_type: 'exact' | 'near';
}

interface DedupResponse {
  clusters: DedupCluster[];
  total_duplicates: number;
}

interface SearchResult {
  id: string;
  content: string;
  source: string;
  confidence: number;
  surprise: number;
  metadata: Record<string, unknown>;
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
  sources_queried: string[];
  query_time_ms: number;
}

export default function MemoryGatewayPage() {
  const { config } = useBackend();
  const [activeTab, setActiveTab] = useState<'search' | 'sources' | 'retention' | 'dedup'>('sources');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [searching, setSearching] = useState(false);

  const { data: sourcesData, isLoading: sourcesLoading } = useSWRFetch<{ data: SourcesResponse }>(
    '/api/v1/memory/unified/sources',
    { refreshInterval: 15000, baseUrl: config.api }
  );

  const { data: retentionData, isLoading: retentionLoading } = useSWRFetch<{ data: RetentionResponse }>(
    activeTab === 'retention' ? '/api/v1/memory/unified/retention?limit=50' : null,
    { refreshInterval: 30000, baseUrl: config.api }
  );

  const { data: dedupData, isLoading: dedupLoading } = useSWRFetch<{ data: DedupResponse }>(
    activeTab === 'dedup' ? '/api/v1/memory/unified/dedup?limit=20' : null,
    { refreshInterval: 60000, baseUrl: config.api }
  );

  const sources = sourcesData?.data?.sources || [];
  const retention = retentionData?.data?.decisions || [];
  const dedupClusters = dedupData?.data?.clusters || [];

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const response = await fetch(`${config.api}/api/v1/memory/unified/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, limit: 20 }),
        signal: AbortSignal.timeout(15000),
      });
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.data || data);
      }
    } catch {
      // Search failed silently
    } finally {
      setSearching(false);
    }
  }, [searchQuery, config.api]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-acid-green bg-acid-green/20';
      case 'degraded': return 'text-yellow-400 bg-yellow-500/20';
      case 'disconnected': return 'text-red-400 bg-red-500/20';
      default: return 'text-text-muted bg-surface';
    }
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'retain': return 'text-acid-green bg-acid-green/20';
      case 'demote': return 'text-yellow-400 bg-yellow-500/20';
      case 'forget': return 'text-red-400 bg-red-500/20';
      case 'consolidate': return 'text-blue-400 bg-blue-400/20';
      default: return 'text-text-muted bg-surface';
    }
  };

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
              <Link href="/memory" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [MEMORY]
              </Link>
              <Link href="/intelligence" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [INTELLIGENCE]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} UNIFIED MEMORY GATEWAY
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Fan-out search across ContinuumMemory, Knowledge Mound, Supermemory, and claude-mem.
              View retention decisions, near-duplicate clusters, and memory source status.
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mb-6">
            {([
              { key: 'sources', label: 'Sources' },
              { key: 'search', label: 'Search' },
              { key: 'retention', label: 'Retention Gate' },
              { key: 'dedup', label: 'Dedup Clusters' },
            ] as const).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-4 py-2 text-sm font-mono rounded border transition-colors ${
                  activeTab === key
                    ? 'bg-acid-green/20 border-acid-green text-acid-green'
                    : 'border-border text-text-muted hover:border-acid-green/50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <PanelErrorBoundary panelName="Memory Gateway">
            {activeTab === 'sources' && (
              <div>
                {sourcesLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-center py-12">Loading sources...</div>
                ) : sources.length === 0 ? (
                  <div className="p-8 bg-surface border border-border rounded-lg text-center">
                    <p className="text-text-muted font-mono">
                      No memory sources configured. Enable unified memory in ArenaConfig.
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {sources.map((src) => (
                      <div key={src.name} className="p-4 bg-surface border border-border rounded-lg">
                        <div className="flex items-center justify-between mb-3">
                          <span className="font-mono text-sm text-text font-bold">{src.name}</span>
                          <span className={`px-2 py-0.5 text-xs font-mono rounded ${getStatusColor(src.status)}`}>
                            {src.status}
                          </span>
                        </div>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-text-muted">Items:</span>
                            <span className="text-text font-mono">{src.item_count.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-muted">Latency:</span>
                            <span className="text-text font-mono">{src.last_query_ms}ms</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'search' && (
              <div className="space-y-4">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Search across all memory systems..."
                    className="flex-1 px-4 py-2 bg-surface border border-border rounded font-mono text-sm text-text placeholder-text-muted focus:border-acid-green focus:outline-none"
                  />
                  <button
                    onClick={handleSearch}
                    disabled={searching || !searchQuery.trim()}
                    className="px-4 py-2 bg-acid-green/20 border border-acid-green text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 disabled:opacity-50"
                  >
                    {searching ? 'Searching...' : 'Search'}
                  </button>
                </div>

                {searchResults && (
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-mono font-bold text-text-muted">
                        {searchResults.total} results in {searchResults.query_time_ms}ms
                      </h3>
                      <span className="text-xs text-text-muted">
                        Sources: {searchResults.sources_queried.join(', ')}
                      </span>
                    </div>
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                      {searchResults.results.map((result) => (
                        <div key={result.id} className="p-3 bg-bg rounded">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="px-1.5 py-0.5 text-xs font-mono bg-blue-400/20 text-blue-400 rounded">
                              {result.source}
                            </span>
                            <span className="text-xs text-text-muted font-mono">
                              conf: {(result.confidence * 100).toFixed(0)}%
                            </span>
                            {result.surprise > 0.5 && (
                              <span className="text-xs text-yellow-400 font-mono">
                                surprise: {result.surprise.toFixed(2)}
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-text line-clamp-3">{result.content}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'retention' && (
              <div>
                {retentionLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-center py-12">Loading...</div>
                ) : retention.length === 0 ? (
                  <div className="p-8 bg-surface border border-border rounded-lg text-center">
                    <p className="text-text-muted font-mono">No retention decisions recorded yet.</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <p className="text-xs text-text-muted font-mono mb-3">
                      Titans/MIRAS-inspired retention decisions based on surprise scores and confidence decay.
                    </p>
                    {retention.map((decision, i) => (
                      <div key={i} className="flex items-center gap-3 p-3 bg-surface border border-border rounded">
                        <span className={`px-2 py-0.5 text-xs font-mono rounded ${getActionColor(decision.action)}`}>
                          {decision.action.toUpperCase()}
                        </span>
                        <span className="text-xs text-text-muted font-mono">{decision.source}</span>
                        <span className="text-xs text-text flex-1 font-mono">{decision.memory_id.substring(0, 16)}...</span>
                        <span className="text-xs text-text-muted">
                          surprise: {decision.surprise_score.toFixed(2)}
                        </span>
                        <span className="text-xs text-text-muted">
                          conf: {(decision.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'dedup' && (
              <div>
                {dedupLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-center py-12">Loading...</div>
                ) : dedupClusters.length === 0 ? (
                  <div className="p-8 bg-surface border border-border rounded-lg text-center">
                    <p className="text-text-muted font-mono">No duplicate clusters detected.</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <p className="text-xs text-text-muted font-mono mb-2">
                      SHA-256 exact + Jaccard near-duplicate detection across memory systems. Total: {dedupData?.data?.total_duplicates || 0}
                    </p>
                    {dedupClusters.map((cluster) => (
                      <div key={cluster.cluster_id} className="p-4 bg-surface border border-border rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-mono text-text-muted">
                            Cluster: {cluster.cluster_id.substring(0, 8)}
                          </span>
                          <span className={`px-1.5 py-0.5 text-xs font-mono rounded ${
                            cluster.duplicate_type === 'exact' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
                          }`}>
                            {cluster.duplicate_type}
                          </span>
                          <span className="text-xs text-text-muted">{cluster.items.length} items</span>
                        </div>
                        <div className="space-y-1">
                          {cluster.items.map((item) => (
                            <div key={item.id} className="text-xs p-2 bg-bg rounded flex items-center gap-2">
                              <span className="px-1 py-0.5 bg-blue-400/20 text-blue-400 rounded font-mono">{item.source}</span>
                              <span className="text-text flex-1 line-clamp-1">{item.preview}</span>
                              <span className="text-text-muted font-mono">{(item.similarity * 100).toFixed(0)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </PanelErrorBoundary>
        </div>

        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">{'='.repeat(40)}</div>
          <p className="text-text-muted">{'>'} ARAGORA // UNIFIED MEMORY GATEWAY</p>
        </footer>
      </main>
    </>
  );
}
