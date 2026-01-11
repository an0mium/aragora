'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { GauntletLive } from '@/components/gauntlet/GauntletLive';
import type { GauntletVerdict } from '@/hooks/useGauntletWebSocket';

export default function GauntletLivePage() {
  const params = useParams();
  const router = useRouter();
  const { config: backendConfig } = useBackend();
  const [copied, setCopied] = useState(false);
  const [verdict, setVerdict] = useState<GauntletVerdict | null>(null);

  const gauntletId = params.id as string;

  // Derive WebSocket URL from API base
  const wsUrl = backendConfig.api.replace(/^http/, 'ws');

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, []);

  const handleComplete = useCallback((v: GauntletVerdict) => {
    setVerdict(v);
  }, []);

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
                href="/gauntlet"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [ALL GAUNTLETS]
              </Link>
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
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-mono text-acid-green mb-2 flex items-center gap-3">
                <span className="animate-pulse">&#x2694;</span> Live Gauntlet
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Real-time adversarial stress-testing in progress.
              </p>
            </div>
            <button
              onClick={handleCopy}
              className="px-3 py-1 text-xs font-mono bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
            >
              {copied ? '[COPIED!]' : '[SHARE LINK]'}
            </button>
          </div>

          <PanelErrorBoundary panelName="Gauntlet Live View">
            <GauntletLive
              gauntletId={gauntletId}
              wsUrl={wsUrl}
              onComplete={handleComplete}
            />
          </PanelErrorBoundary>

          {/* Post-completion actions */}
          {verdict && (
            <div className="mt-6 bg-surface border border-acid-green/30 p-4 rounded">
              <h3 className="text-sm font-mono text-acid-green mb-3">Next Steps</h3>
              <div className="flex flex-wrap gap-3">
                <Link
                  href={`/api/gauntlet/${gauntletId}/receipt?format=html`}
                  target="_blank"
                  className="px-4 py-2 text-xs font-mono bg-acid-green/20 text-acid-green border border-acid-green/40 hover:bg-acid-green/30 transition-colors rounded"
                >
                  [DOWNLOAD RECEIPT]
                </Link>
                <Link
                  href="/gauntlet"
                  className="px-4 py-2 text-xs font-mono bg-surface text-acid-cyan border border-acid-cyan/40 hover:bg-acid-cyan/10 transition-colors rounded"
                >
                  [VIEW ALL RESULTS]
                </Link>
                <button
                  onClick={() => router.push('/gauntlet')}
                  className="px-4 py-2 text-xs font-mono bg-surface text-text-muted border border-border hover:border-acid-cyan/40 transition-colors rounded"
                >
                  [RUN NEW GAUNTLET]
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // LIVE GAUNTLET STRESS TEST
          </p>
        </footer>
      </main>
    </>
  );
}
