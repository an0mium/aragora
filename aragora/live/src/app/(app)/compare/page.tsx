'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { useSearchParams } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const AgentComparePanel = dynamic(
  () => import('@/components/AgentComparePanel').then(m => ({ default: m.AgentComparePanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[600px] bg-surface rounded" />
      </div>
    ),
  }
);

const DebateResultComparePanel = dynamic(
  () =>
    import('@/components/DebateResultComparePanel').then((m) => ({
      default: m.DebateResultComparePanel,
    })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[420px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function ComparePage() {
  const searchParams = useSearchParams();
  const initialLeftDebateId = searchParams.get('left') ?? '';
  const initialRightDebateId = searchParams.get('right') ?? '';

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
                href="/agents"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [AGENTS]
              </Link>
              <Link
                href="/leaderboard"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [RANKS]
              </Link>
              <Link
                href="/calibration"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [CALIBRATE]
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
              {'>'} COMPARISON WORKBENCH
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Compare completed debate outcomes side by side, then inspect head-to-head agent
              performance and calibration below.
            </p>
          </div>

          <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 border border-acid-green/30 bg-acid-green/5 rounded">
              <h3 className="text-sm font-mono text-acid-green mb-2">Debate Outcomes</h3>
              <p className="text-xs font-mono text-text-muted">
                Load two decision packages and inspect verdict, confidence, cost, and agent-model
                drift side by side.
              </p>
            </div>
            <div className="p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
              <h3 className="text-sm font-mono text-acid-cyan mb-2">Configuration Drift</h3>
              <p className="text-xs font-mono text-text-muted">
                See which agents were added, removed, or reconfigured between two runs before you
                interpret an outcome shift.
              </p>
            </div>
            <div className="p-4 border border-gold/30 bg-gold/5 rounded">
              <h3 className="text-sm font-mono text-gold mb-2">Agent Benchmarks</h3>
              <p className="text-xs font-mono text-text-muted">
                Keep the existing agent leaderboard view for baseline skill, consistency, and
                calibration checks.
              </p>
            </div>
          </div>

          <div className="mb-8">
            <PanelErrorBoundary panelName="Debate Compare">
              <DebateResultComparePanel
                initialLeftDebateId={initialLeftDebateId}
                initialRightDebateId={initialRightDebateId}
              />
            </PanelErrorBoundary>
          </div>

          <div className="mb-3">
            <h2 className="text-lg font-mono text-acid-cyan mb-2">{'>'} AGENT BENCHMARKS</h2>
            <p className="text-xs font-mono text-text-muted">
              Use this section when you want to compare agents themselves rather than specific debate
              results.
            </p>
          </div>
          <PanelErrorBoundary panelName="Agent Compare">
            <AgentComparePanel />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // COMPARISON WORKBENCH
          </p>
        </footer>
      </main>
    </>
  );
}
