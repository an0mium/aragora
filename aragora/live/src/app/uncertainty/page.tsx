'use client';

import { useState } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const UncertaintyPanel = dynamic(
  () => import('@/components/UncertaintyPanel').then(m => ({ default: m.UncertaintyPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[500px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function UncertaintyPage() {
  const [debateId, setDebateId] = useState<string>('');
  const [activeDebateId, setActiveDebateId] = useState<string | null>(null);

  const handleLoadDebate = () => {
    if (debateId.trim()) {
      setActiveDebateId(debateId.trim());
    }
  };

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
                href="/crux"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [CRUX]
              </Link>
              <Link
                href="/quality"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [QUALITY]
              </Link>
              <Link
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
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
              {'>'} UNCERTAINTY ANALYSIS
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Analyze collective confidence, identify sources of disagreement,
              and assess calibration quality across debate participants.
            </p>
          </div>

          <div className="mb-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
            <h3 className="text-sm font-mono text-acid-cyan mb-2">Uncertainty Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono text-text-muted">
              <div>
                <span className="text-acid-green">Collective Confidence</span>
                <p>Aggregate certainty level</p>
              </div>
              <div>
                <span className="text-acid-green">Disagreement Sources</span>
                <p>Where agents diverge</p>
              </div>
              <div>
                <span className="text-acid-green">Calibration Quality</span>
                <p>Confidence vs accuracy</p>
              </div>
              <div>
                <span className="text-acid-green">Crux Detection</span>
                <p>Key disputed claims</p>
              </div>
            </div>
          </div>

          {/* Debate ID Input */}
          <div className="mb-6 p-4 border border-acid-green/30 rounded">
            <label className="block text-sm font-mono text-text-muted mb-2">
              Enter Debate ID to Analyze
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={debateId}
                onChange={(e) => setDebateId(e.target.value)}
                placeholder="debate-uuid-here"
                className="flex-1 bg-bg border border-acid-green/30 px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
              />
              <button
                onClick={handleLoadDebate}
                className="px-4 py-2 bg-acid-green/10 border border-acid-green/30 text-acid-green text-sm font-mono hover:bg-acid-green/20 transition-colors"
              >
                [ANALYZE]
              </button>
            </div>
          </div>

          {activeDebateId ? (
            <PanelErrorBoundary panelName="Uncertainty Analysis">
              <UncertaintyPanel events={[]} debateId={activeDebateId} />
            </PanelErrorBoundary>
          ) : (
            <div className="p-8 border border-acid-green/20 rounded text-center">
              <p className="text-text-muted font-mono text-sm">
                Enter a debate ID above to analyze uncertainty and confidence patterns.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // UNCERTAINTY ANALYSIS
          </p>
        </footer>
      </main>
    </>
  );
}
