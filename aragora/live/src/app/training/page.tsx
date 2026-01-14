'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const TrainingExportPanel = dynamic(
  () => import('@/components/TrainingExportPanel').then(m => ({ default: m.TrainingExportPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-96 bg-surface rounded" />
      </div>
    ),
  }
);

export default function TrainingPage() {
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
                href="/gauntlet"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [GAUNTLET]
              </Link>
              <Link
                href="/evolution"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [EVOLUTION]
              </Link>
              <Link
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
              </Link>
              <Link
                href="/tournaments"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [RANKINGS]
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
              Training Data Export
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Export debate outcomes as training data for ML fine-tuning. Supports SFT, DPO, and Gauntlet formats.
            </p>
          </div>

          <PanelErrorBoundary panelName="Training Export">
            <TrainingExportPanel />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // TRAINING DATA EXPORT
          </p>
        </footer>
      </main>
    </>
  );
}
