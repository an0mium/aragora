'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const InsightsPanel = dynamic(
  () => import('@/components/InsightsPanel').then(m => ({ default: m.InsightsPanel })),
  {
    ssr: false,
    loading: () => <div className="card p-4 animate-pulse"><div className="h-96 bg-surface rounded" /></div>,
  }
);

export default function InsightsPage() {
  useBackend();

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
                href="/evidence"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [EVIDENCE]
              </Link>
              <Link
                href="/memory"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [MEMORY]
              </Link>
              <Link
                href="/network"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [NETWORK]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">Insights</h1>
            <p className="text-text-muted font-mono text-sm">
              Pattern analysis, position flips, and learning metrics from agent debates.
            </p>
          </div>

          <PanelErrorBoundary panelName="Insights">
            <InsightsPanel wsMessages={[]} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // INSIGHTS VIEW
          </p>
        </footer>
      </main>
    </>
  );
}
