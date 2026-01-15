'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { fetchRecentDebates, type DebateArtifact } from '@/utils/supabase';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';
import { getAgentColors } from '@/utils/agentColors';
import { logger } from '@/utils/logger';

const PAGE_SIZE = 20;

export default function DebatesPage() {
  const [debates, setDebates] = useState<DebateArtifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [filter, setFilter] = useState<'all' | 'consensus' | 'no-consensus'>('all');

  useEffect(() => {
    async function loadDebates() {
      try {
        setLoading(true);
        const data = await fetchRecentDebates(PAGE_SIZE);
        setDebates(data);
        setHasMore(data.length === PAGE_SIZE);
      } catch (e) {
        logger.error('Failed to load debates:', e);
      } finally {
        setLoading(false);
      }
    }

    loadDebates();
  }, []);

  const loadMore = async () => {
    if (loadingMore || !hasMore) return;

    try {
      setLoadingMore(true);
      const nextPage = page + 1;
      const offset = page * PAGE_SIZE;
      // Fetch next batch (simplified - using limit + offset simulation)
      const data = await fetchRecentDebates(PAGE_SIZE * (nextPage + 1));
      const newDebates = data.slice(offset, offset + PAGE_SIZE);

      if (newDebates.length < PAGE_SIZE) {
        setHasMore(false);
      }

      setDebates(prev => [...prev, ...newDebates]);
      setPage(nextPage);
    } catch (e) {
      logger.error('Failed to load more debates:', e);
    } finally {
      setLoadingMore(false);
    }
  };

  // Filter debates
  const filteredDebates = debates.filter(debate => {
    if (filter === 'consensus') return debate.consensus_reached;
    if (filter === 'no-consensus') return !debate.consensus_reached;
    return true;
  });

  const handleCopyLink = async (debateId: string) => {
    const url = `${window.location.origin}/debate/${debateId}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopiedId(debateId);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      logger.error('Failed to copy link:', err);
    }
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  // Group filtered debates by date
  const groupedDebates = filteredDebates.reduce((acc, debate) => {
    const date = new Date(debate.created_at).toLocaleDateString();
    if (!acc[date]) acc[date] = [];
    acc[date].push(debate);
    return acc;
  }, {} as Record<string, DebateArtifact[]>);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>

            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [BACK TO LIVE]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          {/* Page Title & Filters */}
          <div className="mb-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h1 className="text-xl font-mono text-acid-green mb-2">
                  {'>'} DEBATE ARCHIVE
                </h1>
                <p className="text-xs text-text-muted font-mono">
                  Browse and share past debates with permalinks
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <Link
                    href="/debates/graph"
                    className="px-2 py-1 text-xs font-mono bg-acid-cyan/10 text-teal-700 dark:text-acid-cyan border border-acid-cyan/30 hover:bg-acid-cyan/20 transition-colors"
                  >
                    GRAPH DEBATES
                  </Link>
                  <Link
                    href="/debates/matrix"
                    className="px-2 py-1 text-xs font-mono bg-gold/10 text-amber-700 dark:text-gold border border-gold/30 hover:bg-gold/20 transition-colors"
                  >
                    SCENARIO MATRIX
                  </Link>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-text-muted font-mono">Filter:</span>
                {(['all', 'consensus', 'no-consensus'] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`px-2 py-1 text-xs font-mono border transition-colors ${
                      filter === f
                        ? 'bg-acid-green/20 text-acid-green border-acid-green/40'
                        : 'bg-surface text-text-muted border-border hover:border-acid-green/40'
                    }`}
                  >
                    {f === 'all' ? 'ALL' : f === 'consensus' ? 'CONSENSUS' : 'NO CONSENSUS'}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-2 text-xs text-text-muted font-mono">
              Showing {filteredDebates.length} of {debates.length} debates
            </div>
          </div>

          {loading && (
            <div className="flex items-center justify-center py-20">
              <div className="text-acid-green font-mono animate-pulse">
                {'>'} LOADING DEBATES...
              </div>
            </div>
          )}

          {!loading && debates.length === 0 && (
            <div className="bg-surface border border-acid-green/30 p-6 text-center">
              <div className="text-text-muted text-sm font-mono">
                No debates archived yet. Run the nomic loop to generate debates.
              </div>
            </div>
          )}

          {/* Debates by Date */}
          <div className="space-y-6">
            {Object.entries(groupedDebates).map(([date, dateDebates]) => (
              <div key={date}>
                <div className="text-xs font-mono text-text-muted mb-2 flex items-center gap-2">
                  <span className="text-acid-green">{'>'}</span>
                  {date}
                  <span className="text-text-muted/50">({dateDebates.length} debates)</span>
                </div>

                <div className="space-y-2">
                  {dateDebates.map((debate) => (
                    <div
                      key={debate.id}
                      className="bg-surface border border-acid-green/30 p-4 hover:border-acid-green/50 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <Link
                            href={`/debate/${debate.id}`}
                            className="text-sm font-mono text-acid-green hover:text-acid-cyan transition-colors block mb-2"
                          >
                            {debate.task}
                          </Link>

                          <div className="flex flex-wrap items-center gap-3 text-xs">
                            {/* Agents */}
                            <div className="flex items-center gap-1">
                              {debate.agents.map((agent, i) => {
                                const colors = getAgentColors(agent);
                                return (
                                  <span
                                    key={i}
                                    className={`px-1.5 py-0.5 ${colors.bg} ${colors.text} font-mono`}
                                    title={agent}
                                  >
                                    {agent.split('-')[0].toUpperCase()}
                                  </span>
                                );
                              })}
                            </div>

                            {/* Status */}
                            <div className="flex items-center gap-2 text-text-muted">
                              <span
                                className={`w-1.5 h-1.5 rounded-full ${
                                  debate.consensus_reached ? 'bg-green-400' : 'bg-yellow-400'
                                }`}
                              />
                              <span>
                                {debate.consensus_reached ? 'Consensus' : 'No consensus'}
                              </span>
                              <span className="text-text-muted/50">|</span>
                              <span>{Math.round(debate.confidence * 100)}% conf</span>
                            </div>

                            {/* Phase and Cycle */}
                            <div className="text-text-muted">
                              C{debate.cycle_number} / {debate.phase}
                            </div>
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex flex-col items-end gap-2 flex-shrink-0">
                          <button
                            onClick={() => handleCopyLink(debate.id)}
                            className="px-2 py-1 text-xs font-mono bg-acid-green/10 text-acid-green border border-acid-green/30 hover:bg-acid-green hover:text-bg transition-colors"
                          >
                            {copiedId === debate.id ? 'COPIED!' : 'SHARE'}
                          </button>
                          <span className="text-[10px] text-text-muted font-mono">
                            {formatDate(debate.created_at).split(',')[1]}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* Load More Button */}
            {hasMore && !loading && (
              <div className="text-center py-6">
                <button
                  onClick={loadMore}
                  disabled={loadingMore}
                  className="px-6 py-3 font-mono text-sm bg-surface border border-acid-green/30 text-acid-green hover:bg-acid-green/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loadingMore ? (
                    <span className="animate-pulse">LOADING MORE...</span>
                  ) : (
                    <span>LOAD MORE DEBATES</span>
                  )}
                </button>
                <p className="text-xs text-text-muted mt-2">
                  Page {page} • {PAGE_SIZE} debates per page
                </p>
              </div>
            )}

            {!hasMore && debates.length > 0 && (
              <div className="text-center py-6 text-xs text-text-muted font-mono">
                {'>'} END OF ARCHIVE • {debates.length} total debates
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'═'.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} AGORA DEBATE ARCHIVE // {debates.length} DEBATES
          </p>
          <p className="text-acid-cyan mt-2">
            <Link
              href="/"
              className="hover:text-acid-green transition-colors"
            >
              [ RETURN TO LIVE ]
            </Link>
          </p>
          <div className="text-acid-green/50 mt-4">
            {'═'.repeat(40)}
          </div>
        </footer>
      </main>
    </>
  );
}
