'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const GauntletPanel = dynamic(
  () => import('@/components/GauntletPanel').then(m => ({ default: m.GauntletPanel })),
  {
    ssr: false,
    loading: () => <div className="card p-4 animate-pulse"><div className="h-96 bg-surface rounded" /></div>,
  }
);

export default function GauntletPage() {
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
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2 flex items-center gap-3">
              <span>⚔️</span> Gauntlet
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Adversarial stress-testing results. Find vulnerabilities before they find you.
            </p>
          </div>

          <div className="grid gap-6">
            <PanelErrorBoundary panelName="Gauntlet Results">
              <GauntletPanel apiBase={backendConfig.api} />
            </PanelErrorBoundary>

            {/* Usage hints */}
            <div className="bg-surface border border-border rounded-lg p-4">
              <h3 className="text-sm font-mono text-acid-green mb-3">Quick Start</h3>
              <ul className="text-xs text-text-muted space-y-2 font-mono">
                <li className="flex items-start gap-2">
                  <span className="text-acid-cyan">&gt;</span>
                  <code className="bg-bg px-2 py-0.5">aragora gauntlet spec.md --profile quick</code>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-acid-cyan">&gt;</span>
                  <code className="bg-bg px-2 py-0.5">aragora gauntlet policy.yaml --persona gdpr</code>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-acid-cyan">&gt;</span>
                  <code className="bg-bg px-2 py-0.5">aragora gauntlet arch.md --profile thorough --output report.html</code>
                </li>
              </ul>
              <div className="mt-4 pt-3 border-t border-border">
                <p className="text-xs text-text-muted">
                  See{' '}
                  <a
                    href="https://github.com/an0mium/aragora/blob/main/docs/GAUNTLET.md"
                    className="text-acid-cyan hover:text-acid-green"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    GAUNTLET.md
                  </a>{' '}
                  for full documentation.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // GAUNTLET STRESS TESTING
          </p>
        </footer>
      </main>
    </>
  );
}
