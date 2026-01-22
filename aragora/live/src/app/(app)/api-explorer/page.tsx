'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const ApiExplorerPanel = dynamic(
  () => import('@/components/ApiExplorerPanel').then(m => ({ default: m.ApiExplorerPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-96 bg-surface rounded" />
      </div>
    ),
  }
);

export default function ApiExplorerPage() {
  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/" aria-label="Go to dashboard">
              <AsciiBannerCompact connected={true} />
            </Link>
            <nav className="flex items-center gap-4" aria-label="Main navigation">
              <Link
                href="/"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/debates"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [DEBATES]
              </Link>
              <Link
                href="/settings"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [SETTINGS]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </nav>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              API Explorer
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Browse and test all available API endpoints interactively.
            </p>
          </div>

          <PanelErrorBoundary panelName="API Explorer">
            <ApiExplorerPanel />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2" aria-hidden="true">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // API EXPLORER
          </p>
        </footer>
      </main>
    </>
  );
}
