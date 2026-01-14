'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const EvidenceVisualizerPanel = dynamic(
  () => import('@/components/EvidenceVisualizerPanel').then(m => ({ default: m.EvidenceVisualizerPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-96 bg-surface rounded" />
      </div>
    ),
  }
);

interface EvidenceStats {
  total_evidence: number;
  by_type: Record<string, number>;
  by_source: Record<string, number>;
  recent_count: number;
  verified_count: number;
  avg_relevance_score: number;
}

export default function EvidencePage() {
  const { config: backendConfig } = useBackend();
  const [stats, setStats] = useState<EvidenceStats | null>(null);

  // Fetch evidence statistics
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${backendConfig.api}/api/evidence/statistics`);
        if (res.ok) {
          const data = await res.json();
          setStats(data.statistics || data);
        }
      } catch {
        // Statistics endpoint may not exist
      }
    };

    fetchStats();
  }, [backendConfig.api]);

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
                [DASHBOARD]
              </Link>
              <Link
                href="/memory"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [MEMORY]
              </Link>
              <Link
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
              </Link>
              <Link
                href="/verify"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [PROOFS]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} EVIDENCE & DISSENT
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Explore dissenting views, contrarian perspectives, risk warnings, and evidence trails from debates.
            </p>
          </div>

          {/* Evidence Statistics */}
          {stats && (
            <div className="mb-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
              <h3 className="text-sm font-mono text-acid-cyan mb-3">Evidence Statistics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-2xl font-mono text-acid-green">{stats.total_evidence}</div>
                  <div className="text-xs font-mono text-text-muted">Total Evidence</div>
                </div>
                <div>
                  <div className="text-2xl font-mono text-acid-cyan">{stats.verified_count}</div>
                  <div className="text-xs font-mono text-text-muted">Verified</div>
                </div>
                <div>
                  <div className="text-2xl font-mono text-gold">{stats.recent_count}</div>
                  <div className="text-xs font-mono text-text-muted">Recent (24h)</div>
                </div>
                <div>
                  <div className="text-2xl font-mono text-text">{(stats.avg_relevance_score * 100).toFixed(0)}%</div>
                  <div className="text-xs font-mono text-text-muted">Avg Relevance</div>
                </div>
              </div>
              {stats.by_type && Object.keys(stats.by_type).length > 0 && (
                <div className="mt-4 pt-3 border-t border-acid-cyan/20">
                  <div className="text-xs font-mono text-text-muted mb-2">By Type</div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(stats.by_type).map(([type, count]) => (
                      <span key={type} className="px-2 py-1 text-xs font-mono bg-surface rounded">
                        {type}: <span className="text-acid-green">{count}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <PanelErrorBoundary panelName="Evidence Visualizer">
            <EvidenceVisualizerPanel backendConfig={{ apiUrl: backendConfig.api, wsUrl: backendConfig.ws }} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // EVIDENCE & DISSENT EXPLORER
          </p>
        </footer>
      </main>
    </>
  );
}
