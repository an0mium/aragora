'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const ConsensusQualityDashboard = dynamic(
  () => import('@/components/ConsensusQualityDashboard').then(m => ({ default: m.ConsensusQualityDashboard })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[500px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function QualityPage() {
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
                href="/analytics"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ANALYTICS]
              </Link>
              <Link
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
              </Link>
              <Link
                href="/leaderboard"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [RANKS]
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
              {'>'} CONSENSUS QUALITY
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Monitor consensus quality metrics, hollow consensus detection,
              and debate outcome reliability across all discussions.
            </p>
          </div>

          <div className="mb-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
            <h3 className="text-sm font-mono text-acid-cyan mb-2">Quality Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono text-text-muted">
              <div>
                <span className="text-acid-green">Agreement Depth</span>
                <p>Genuine vs shallow consensus</p>
              </div>
              <div>
                <span className="text-acid-green">Dissent Inclusion</span>
                <p>Minority view representation</p>
              </div>
              <div>
                <span className="text-acid-green">Reasoning Quality</span>
                <p>Argument validity scores</p>
              </div>
              <div>
                <span className="text-acid-green">Evidence Support</span>
                <p>Citation coverage metrics</p>
              </div>
            </div>
          </div>

          <PanelErrorBoundary panelName="Consensus Quality">
            <ConsensusQualityDashboard apiBase={backendConfig.api} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // CONSENSUS QUALITY
          </p>
        </footer>
      </main>
    </>
  );
}
