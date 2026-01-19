'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';
import { ScenarioMatrixView } from '@/components/scenario-matrix';

function MatrixDebatesContent() {
  const searchParams = useSearchParams();
  const initialMatrixId = searchParams.get('id');

  return (
    <div className="container mx-auto px-4 py-6">
      {/* Breadcrumb */}
      <div className="mb-4 text-xs font-mono text-text-muted">
        <Link href="/debates" className="hover:text-acid-green">
          Debates
        </Link>
        <span className="mx-2">/</span>
        <span className="text-acid-green">Matrix</span>
      </div>

      <ScenarioMatrixView initialMatrixId={initialMatrixId} />
    </div>
  );
}

export default function MatrixDebatesPage() {
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
                href="/debates/graph"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [GRAPH]
              </Link>
              <Link
                href="/debates"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ARCHIVE]
              </Link>
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [LIVE]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <Suspense fallback={
          <div className="container mx-auto px-4 py-6">
            <div className="animate-pulse text-acid-green font-mono">Loading matrix debate...</div>
          </div>
        }>
          <MatrixDebatesContent />
        </Suspense>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'═'.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} SCENARIO MATRIX // PARALLEL COMPARISON
          </p>
          <p className="text-acid-cyan mt-2">
            <Link href="/" className="hover:text-acid-green transition-colors">
              [ RETURN TO LIVE ]
            </Link>
          </p>
          <div className="text-acid-green/50 mt-4">
            {'═'.repeat(40)}
          </div>
        </footer>
      </main>
    </>
  );
}
