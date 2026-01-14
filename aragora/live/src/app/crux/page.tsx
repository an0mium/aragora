'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const CruxPanel = dynamic(
  () => import('@/components/CruxPanel').then(m => ({ default: m.CruxPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[500px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function CruxPage() {
  const { config: backendConfig } = useBackend();

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
                href="/uncertainty"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [UNCERTAINTY]
              </Link>
              <Link
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
              </Link>
              <Link
                href="/debates"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DEBATES]
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
              {'>'} CRUX ANALYSIS
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Identify load-bearing claims that debates hinge upon.
              Analyze centrality and entropy scores for key arguments.
            </p>
          </div>

          <div className="mb-6 p-4 border border-acid-purple/30 bg-acid-purple/5 rounded">
            <h3 className="text-sm font-mono text-acid-purple mb-2">What is a Crux?</h3>
            <p className="text-xs font-mono text-text-muted mb-3">
              A <span className="text-acid-green">crux</span> is a claim that, if resolved, would significantly
              change the outcome of a debate. Finding cruxes helps focus discussion on what matters most.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-xs font-mono text-text-muted">
              <div>
                <span className="text-acid-green">Centrality Score</span>
                <p>How many arguments depend on this claim</p>
              </div>
              <div>
                <span className="text-acid-green">Entropy Score</span>
                <p>Uncertainty/disagreement level</p>
              </div>
              <div>
                <span className="text-acid-green">Impact Score</span>
                <p>Effect on final conclusion</p>
              </div>
            </div>
          </div>

          <PanelErrorBoundary panelName="Crux Analysis">
            <CruxPanel apiBase={backendConfig.api} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // CRUX ANALYSIS
          </p>
        </footer>
      </main>
    </>
  );
}
