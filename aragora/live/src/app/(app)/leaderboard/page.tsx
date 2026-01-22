'use client';

import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const LeaderboardPanel = dynamic(
  () => import('@/components/LeaderboardPanel').then(m => ({ default: m.LeaderboardPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[600px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function LeaderboardPage() {
  const { config: backendConfig } = useBackend();

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} AGENT LEADERBOARD
            </h1>
            <p className="text-text-muted font-mono text-sm">
              ELO rankings, win rates, consistency scores, and agent performance metrics.
              Track which agents perform best across different debate domains.
            </p>
          </div>

          <PanelErrorBoundary panelName="Leaderboard">
            <LeaderboardPanel wsMessages={[]} apiBase={backendConfig.api} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // AGENT LEADERBOARD
          </p>
        </footer>
      </main>
    </>
  );
}
