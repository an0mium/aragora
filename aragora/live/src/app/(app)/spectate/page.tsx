'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { API_BASE_URL } from '@/config';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { TrustBadge } from '@/components/TrustBadge';
import { useRightSidebar } from '@/context/RightSidebarContext';
import { fetchWithRetry } from '@/utils/retry';

interface LiveDebate {
  id: string;
  task: string;
  agents: string[];
  round: number;
  started_at: string;
  spectator_count: number;
}

export default function SpectatePage() {
  const [liveDebates, setLiveDebates] = useState<LiveDebate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { setContext, clearContext } = useRightSidebar();

  // Fetch live debates
  useEffect(() => {
    const fetchLiveDebates = async () => {
      try {
        setLoading(true);
        setError(null);

        // Try to fetch from API
        const apiUrl = API_BASE_URL;
        const response = await fetchWithRetry(`${apiUrl}/api/debates/live`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (response.ok) {
          const data = await response.json();
          setLiveDebates(data.debates || []);
        } else {
          // No live debates or API not available - show empty state
          setLiveDebates([]);
        }
      } catch {
        // API might not be running - show empty state
        setLiveDebates([]);
        setError(null); // Don't show error for expected case
      } finally {
        setLoading(false);
      }
    };

    fetchLiveDebates();

    // Poll for updates every 10 seconds
    const interval = setInterval(fetchLiveDebates, 10000);
    return () => clearInterval(interval);
  }, []);

  // Set up right sidebar
  useEffect(() => {
    setContext({
      title: 'Spectate Mode',
      subtitle: 'Watch debates live',
      statsContent: (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Live Debates</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">{liveDebates.length}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Total Spectators</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">
              {liveDebates.reduce((sum, d) => sum + d.spectator_count, 0)}
            </span>
          </div>
        </div>
      ),
      actionsContent: (
        <div className="space-y-2">
          <Link
            href="/arena"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
          >
            + START DEBATE
          </Link>
          <Link
            href="/debates"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            VIEW ARCHIVE
          </Link>
        </div>
      ),
    });

    return () => clearContext();
  }, [liveDebates, setContext, clearContext]);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
              <h1 className="text-2xl font-mono text-acid-green">SPECTATE MODE</h1>
            </div>
            <p className="text-text-muted text-sm font-mono">
              Watch debates in real-time without participating. Read-only observation mode.
            </p>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-16">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-acid-green/30 border-t-acid-green rounded-full animate-spin mx-auto mb-4" />
                <p className="text-text-muted text-sm font-mono">Scanning for live debates...</p>
              </div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="border border-warning/30 bg-warning/10 p-4 mb-6">
              <p className="text-warning text-sm font-mono">{error}</p>
            </div>
          )}

          {/* Live Debates List */}
          {!loading && liveDebates.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-sm font-mono text-acid-cyan uppercase tracking-wider">
                Live Debates ({liveDebates.length})
              </h2>

              <div className="grid gap-4">
                {liveDebates.map((debate) => (
                  <Link
                    key={debate.id}
                    href={`/spectate/${debate.id}`}
                    className="block border border-acid-green/30 bg-surface/50 p-4 hover:border-acid-green/60 hover:bg-surface/80 transition-all group"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        {/* Task */}
                        <h3 className="text-sm font-mono text-text truncate group-hover:text-acid-green transition-colors">
                          {debate.task}
                        </h3>

                        {/* Agents */}
                        <div className="flex flex-wrap gap-2 mt-2">
                          {debate.agents.map((agent) => (
                            <span
                              key={agent}
                              className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono bg-acid-cyan/10 text-acid-cyan border border-acid-cyan/30"
                            >
                              {agent}
                              {/* TODO: Wire agent.calibration when API provides it */}
                              <TrustBadge calibration={null} size="sm" />
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Stats */}
                      <div className="flex flex-col items-end gap-1 text-xs font-mono">
                        <div className="flex items-center gap-2">
                          <span className="text-text-muted">Round</span>
                          <span className="text-acid-green">{debate.round}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-text-muted">Watching</span>
                          <span className="text-acid-cyan">{debate.spectator_count}</span>
                        </div>
                        <div className="flex items-center gap-1 text-red-400">
                          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                          LIVE
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!loading && liveDebates.length === 0 && (
            <div className="border border-acid-green/20 bg-surface/30 p-8 text-center">
              <div className="text-4xl mb-4">👁️</div>
              <h2 className="text-lg font-mono text-acid-green mb-2">No Live Debates</h2>
              <p className="text-text-muted text-sm font-mono mb-6 max-w-md mx-auto">
                There are no debates currently in progress. Start a new debate to begin spectating,
                or check back later.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                <Link
                  href="/arena"
                  className="px-6 py-2 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors"
                >
                  START DEBATE
                </Link>
                <Link
                  href="/debates"
                  className="px-6 py-2 border border-acid-green/30 text-acid-green font-mono hover:border-acid-green transition-colors"
                >
                  VIEW ARCHIVE
                </Link>
              </div>
            </div>
          )}

          {/* Info Box */}
          <div className="mt-8 border border-acid-cyan/20 bg-acid-cyan/5 p-4">
            <h3 className="text-sm font-mono text-acid-cyan mb-2">About Spectate Mode</h3>
            <ul className="text-xs font-mono text-text-muted space-y-1">
              <li>• Watch debates unfold in real-time</li>
              <li>• See all agent proposals, critiques, and votes</li>
              <li>• Monitor convergence and consensus progress</li>
              <li>• Read-only: spectators cannot influence debates</li>
              <li>• Perfect for stakeholders who want to observe decisions</li>
            </ul>
          </div>
        </div>
      </main>
    </>
  );
}
