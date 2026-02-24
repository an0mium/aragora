'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { fetchRecentDebates, type DebateArtifact } from '@/utils/supabase';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { getAgentColors } from '@/utils/agentColors';
import { logger } from '@/utils/logger';
import { DebatesEmptyState } from '@/components/ui/EmptyState';
import { useRightSidebar } from '@/context/RightSidebarContext';
import { API_BASE_URL } from '@/config';
import { UseCaseWizard } from '@/components/wizards/UseCaseWizard';

const PAGE_SIZE = 20;

// Backend API response shape for debates list
interface BackendDebatesResponse {
  debates?: Array<{
    id: string;
    debate_id?: string;
    task?: string;
    question?: string;
    agents: string[];
    consensus_reached: boolean;
    confidence: number;
    winning_proposal?: string | null;
    vote_tally?: Record<string, number> | null;
    created_at: string;
    loop_id?: string;
    cycle_number?: number;
    phase?: string;
  }>;
  results?: Array<{
    id: string;
    debate_id?: string;
    task?: string;
    question?: string;
    agents: string[];
    consensus_reached: boolean;
    confidence: number;
    winning_proposal?: string | null;
    vote_tally?: Record<string, number> | null;
    created_at: string;
    loop_id?: string;
    cycle_number?: number;
    phase?: string;
  }>;
  total?: number;
  has_more?: boolean;
}

// Normalize backend debate data to DebateArtifact shape
function normalizeBackendDebate(d: NonNullable<BackendDebatesResponse['debates']>[number]): DebateArtifact {
  return {
    id: d.debate_id || d.id,
    loop_id: d.loop_id || '',
    cycle_number: d.cycle_number || 0,
    phase: d.phase || 'completed',
    task: d.task || d.question || 'Untitled debate',
    agents: d.agents || [],
    transcript: [],
    consensus_reached: d.consensus_reached ?? false,
    confidence: d.confidence ?? 0,
    winning_proposal: d.winning_proposal ?? null,
    vote_tally: d.vote_tally ?? null,
    created_at: d.created_at || new Date().toISOString(),
  };
}

export default function DebatesPage() {
  const router = useRouter();
  const [debates, setDebates] = useState<DebateArtifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [filter, setFilter] = useState<'all' | 'consensus' | 'no-consensus'>('all');
  const [dataSource, setDataSource] = useState<'backend' | 'supabase' | 'none'>('none');
  const [showWizard, setShowWizard] = useState(false);

  const { setContext, clearContext } = useRightSidebar();

  // Set up right sidebar content
  useEffect(() => {
    const consensusCount = debates.filter(d => d.consensus_reached).length;
    const consensusRate = debates.length > 0 ? Math.round((consensusCount / debates.length) * 100) : 0;
    const avgConfidence = debates.length > 0
      ? Math.round(debates.reduce((sum, d) => sum + (d.confidence || 0), 0) / debates.length * 100)
      : 0;

    setContext({
      title: 'Debate Archive',
      subtitle: `${debates.length} debates`,
      statsContent: (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Total</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">{debates.length}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Consensus Rate</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">{consensusRate}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Avg Confidence</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">{avgConfidence}%</span>
          </div>
        </div>
      ),
      actionsContent: (
        <div className="space-y-2">
          <Link
            href="/arena"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
          >
            + NEW DEBATE
          </Link>
          <button
            onClick={() => setShowWizard(true)}
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-cyan)]/10 text-[var(--acid-cyan)] border border-[var(--acid-cyan)]/30 hover:bg-[var(--acid-cyan)]/20 transition-colors"
          >
            USE CASE WIZARD
          </button>
          <Link
            href="/debates/graph"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            GRAPH VIEW
          </Link>
          <Link
            href="/debates/matrix"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            MATRIX VIEW
          </Link>
          <Link
            href="/debates/provenance"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-cyan)]/10 text-[var(--acid-cyan)] border border-[var(--acid-cyan)]/30 hover:bg-[var(--acid-cyan)]/20 transition-colors"
          >
            PROVENANCE
          </Link>
        </div>
      ),
    });

    return () => clearContext();
  }, [debates, setContext, clearContext]);

  // Fetch debates from backend API, falling back to Supabase
  const fetchDebatesFromBackend = useCallback(async (limit: number, offset: number): Promise<{ debates: DebateArtifact[]; hasMore: boolean; source: 'backend' | 'supabase' } | null> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/debates?limit=${limit}&offset=${offset}&sort=created_at:desc`,
        {
          headers: { 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(10000),
        }
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: BackendDebatesResponse = await response.json();
      const debateList = data.debates || data.results || [];
      const normalized = debateList.map(normalizeBackendDebate);
      const moreAvailable = data.has_more ?? normalized.length === limit;
      return { debates: normalized, hasMore: moreAvailable, source: 'backend' };
    } catch (err) {
      logger.warn('Backend debates fetch failed, trying Supabase:', err);
      return null;
    }
  }, []);

  const fetchDebatesFromSupabase = useCallback(async (limit: number): Promise<{ debates: DebateArtifact[]; hasMore: boolean; source: 'supabase' }> => {
    const data = await fetchRecentDebates(limit);
    return { debates: data, hasMore: data.length === limit, source: 'supabase' };
  }, []);

  useEffect(() => {
    async function loadDebates() {
      try {
        setLoading(true);

        // Try backend first
        const backendResult = await fetchDebatesFromBackend(PAGE_SIZE, 0);
        if (backendResult && backendResult.debates.length > 0) {
          setDebates(backendResult.debates);
          setHasMore(backendResult.hasMore);
          setDataSource('backend');
          return;
        }

        // Fall back to Supabase
        const supabaseResult = await fetchDebatesFromSupabase(PAGE_SIZE);
        setDebates(supabaseResult.debates);
        setHasMore(supabaseResult.hasMore);
        setDataSource(supabaseResult.debates.length > 0 ? 'supabase' : 'none');
      } catch (e) {
        logger.error('Failed to load debates:', e);
      } finally {
        setLoading(false);
      }
    }

    loadDebates();
  }, [fetchDebatesFromBackend, fetchDebatesFromSupabase]);

  const loadMore = async () => {
    if (loadingMore || !hasMore) return;

    try {
      setLoadingMore(true);
      const nextPage = page + 1;
      const offset = page * PAGE_SIZE;

      if (dataSource === 'backend') {
        // Fetch next page from backend API with offset
        const result = await fetchDebatesFromBackend(PAGE_SIZE, offset);
        if (result) {
          if (result.debates.length < PAGE_SIZE) {
            setHasMore(false);
          }
          setDebates(prev => [...prev, ...result.debates]);
          setPage(nextPage);
          return;
        }
      }

      // Supabase fallback for pagination
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
    const url = `${window.location.origin}/debates/${debateId}`;
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

      <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="container mx-auto px-4 py-6">
          {/* Page Title & Filters */}
          <div className="mb-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h1 className="text-xl font-mono text-[var(--acid-green)] mb-2">
                  {'>'} DEBATE ARCHIVE
                </h1>
                <p className="text-xs text-[var(--text-muted)] font-mono">
                  Browse and share past debates with permalinks
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--text-muted)] font-mono">Filter:</span>
                {(['all', 'consensus', 'no-consensus'] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`px-2 py-1 text-xs font-mono border transition-colors ${
                      filter === f
                        ? 'bg-[var(--acid-green)]/20 text-[var(--acid-green)] border-[var(--acid-green)]/40'
                        : 'bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)] hover:border-[var(--acid-green)]/40'
                    }`}
                  >
                    {f === 'all' ? 'ALL' : f === 'consensus' ? 'CONSENSUS' : 'NO CONSENSUS'}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-2 text-xs text-[var(--text-muted)] font-mono flex items-center gap-2">
              <span>Showing {filteredDebates.length} of {debates.length} debates</span>
              {dataSource !== 'none' && (
                <span className="text-[10px] text-[var(--text-muted)]" title={dataSource === 'backend' ? 'Fetched from API' : 'Fetched from Supabase'}>
                  [{dataSource === 'backend' ? 'API' : 'DB'}]
                </span>
              )}
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
            <div className="space-y-4">
              <div className="bg-surface border border-acid-green/30">
                <DebatesEmptyState onStart={() => window.location.href = '/'} />
              </div>
              <button
                onClick={() => setShowWizard(true)}
                className="w-full px-4 py-3 font-mono text-sm bg-[var(--acid-cyan)]/10 text-[var(--acid-cyan)] border border-[var(--acid-cyan)]/30 hover:bg-[var(--acid-cyan)]/20 transition-colors"
              >
                OR USE THE GUIDED WIZARD TO GET STARTED
              </button>
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
                            href={`/debates/${debate.id}`}
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

                            {/* Status Badge */}
                            <span
                              className={`px-1.5 py-0.5 text-[10px] font-mono border ${
                                debate.winning_proposal
                                  ? 'bg-acid-green/10 text-acid-green border-acid-green/30'
                                  : debate.consensus_reached
                                    ? 'bg-acid-cyan/10 text-acid-cyan border-acid-cyan/30'
                                    : 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30'
                              }`}
                            >
                              {debate.winning_proposal ? 'COMPLETED' : debate.consensus_reached ? 'CONSENSUS' : 'NO CONSENSUS'}
                            </span>

                            {/* Confidence */}
                            <span className="text-text-muted">
                              {Math.round(debate.confidence * 100)}% conf
                            </span>

                            {/* Receipt indicator */}
                            {debate.vote_tally && (
                              <span className="text-[10px] font-mono text-acid-green" title="Has receipt">
                                [RCV]
                              </span>
                            )}

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

        {/* Use Case Wizard Modal */}
        {showWizard && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="w-full max-w-2xl max-h-[80vh] overflow-y-auto mx-4">
              <UseCaseWizard
                onComplete={(debateId) => {
                  setShowWizard(false);
                  router.push(`/debates/${debateId}`);
                }}
                onCancel={() => setShowWizard(false)}
                apiBase={API_BASE_URL}
              />
            </div>
          </div>
        )}

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
