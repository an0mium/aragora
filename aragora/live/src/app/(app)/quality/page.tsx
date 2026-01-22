'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const ConsensusQualityDashboard = dynamic(
  () => import('@/components/ConsensusQualityDashboard').then(m => ({ default: m.ConsensusQualityDashboard })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[400px] bg-surface rounded" />
      </div>
    ),
  }
);

const EvaluationPanel = dynamic(
  () => import('@/components/EvaluationPanel').then(m => ({ default: m.EvaluationPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[500px] bg-surface rounded" />
      </div>
    ),
  }
);

type ActiveSection = 'consensus' | 'evaluation';

export default function QualityPage() {
  const { config: backendConfig } = useBackend();
  const [activeSection, setActiveSection] = useState<ActiveSection>('consensus');

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
              <Link
                href="/scheduler"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [SCHEDULER]
              </Link>
              <Link
                href="/selection"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [SELECTION]
              </Link>
              <Link
                href="/ml"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ML]
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
              {'>'} QUALITY & EVALUATION
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Monitor consensus quality metrics and evaluate responses using LLM-as-Judge.
            </p>
          </div>

          {/* Section Toggle */}
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setActiveSection('consensus')}
              className={`px-4 py-2 text-sm font-mono rounded transition-colors ${
                activeSection === 'consensus'
                  ? 'bg-acid-green/20 text-acid-green border border-acid-green/50'
                  : 'bg-surface text-text-muted hover:text-acid-green border border-transparent'
              }`}
            >
              [CONSENSUS QUALITY]
            </button>
            <button
              onClick={() => setActiveSection('evaluation')}
              className={`px-4 py-2 text-sm font-mono rounded transition-colors ${
                activeSection === 'evaluation'
                  ? 'bg-acid-green/20 text-acid-green border border-acid-green/50'
                  : 'bg-surface text-text-muted hover:text-acid-green border border-transparent'
              }`}
            >
              [LLM-AS-JUDGE]
            </button>
          </div>

          {/* Consensus Section */}
          {activeSection === 'consensus' && (
            <>
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
            </>
          )}

          {/* Evaluation Section */}
          {activeSection === 'evaluation' && (
            <>
              <div className="mb-6 p-4 border border-purple-500/30 bg-purple-500/5 rounded">
                <h3 className="text-sm font-mono text-purple-400 mb-2">LLM-as-Judge Evaluation</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono text-text-muted">
                  <div>
                    <span className="text-acid-green">Response Evaluation</span>
                    <p>Score responses on dimensions</p>
                  </div>
                  <div>
                    <span className="text-acid-green">Pairwise Comparison</span>
                    <p>Compare two responses</p>
                  </div>
                  <div>
                    <span className="text-acid-green">Dimension Explorer</span>
                    <p>View scoring rubrics</p>
                  </div>
                  <div>
                    <span className="text-acid-green">Evaluation Profiles</span>
                    <p>Use-case weight presets</p>
                  </div>
                </div>
              </div>

              <PanelErrorBoundary panelName="Evaluation Panel">
                <EvaluationPanel apiBase={backendConfig.api} />
              </PanelErrorBoundary>
            </>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // QUALITY & EVALUATION
          </p>
        </footer>
      </main>
    </>
  );
}
