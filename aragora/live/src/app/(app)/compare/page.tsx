'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const AgentComparePanel = dynamic(
  () => import('@/components/AgentComparePanel').then(m => ({ default: m.AgentComparePanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[600px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function ComparePage() {
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
                href="/agents"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [AGENTS]
              </Link>
              <Link
                href="/leaderboard"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [RANKS]
              </Link>
              <Link
                href="/calibration"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [CALIBRATE]
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
              {'>'} AGENT COMPARISON
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Head-to-head agent comparison with ELO ratings, win rates,
              consistency scores, and calibration metrics.
            </p>
          </div>

          <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 border border-acid-green/30 bg-acid-green/5 rounded">
              <h3 className="text-sm font-mono text-acid-green mb-2">ELO Rating</h3>
              <p className="text-xs font-mono text-text-muted">
                Skill-based ranking derived from debate outcomes.
                Higher ELO indicates stronger performance.
              </p>
            </div>
            <div className="p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
              <h3 className="text-sm font-mono text-acid-cyan mb-2">Consistency</h3>
              <p className="text-xs font-mono text-text-muted">
                How reliably an agent maintains positions and
                produces coherent arguments across debates.
              </p>
            </div>
            <div className="p-4 border border-gold/30 bg-gold/5 rounded">
              <h3 className="text-sm font-mono text-gold mb-2">Calibration</h3>
              <p className="text-xs font-mono text-text-muted">
                How well confidence matches actual accuracy.
                Well-calibrated agents know what they know.
              </p>
            </div>
          </div>

          <PanelErrorBoundary panelName="Agent Compare">
            <AgentComparePanel />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // AGENT COMPARISON
          </p>
        </footer>
      </main>
    </>
  );
}
