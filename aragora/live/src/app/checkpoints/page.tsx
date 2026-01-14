'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { useRouter } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const CheckpointPanel = dynamic(
  () => import('@/components/CheckpointPanel').then(m => ({ default: m.CheckpointPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[500px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function CheckpointsPage() {
  const { config: backendConfig } = useBackend();
  const router = useRouter();

  const handleResume = (debateId: string, checkpointId: string) => {
    // Navigate to debate page with checkpoint context
    router.push(`/debate/${debateId}?checkpoint=${checkpointId}`);
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
                href="/debates"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DEBATES]
              </Link>
              <Link
                href="/replays"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [REPLAYS]
              </Link>
              <Link
                href="/analytics"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ANALYTICS]
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
              {'>'} DEBATE CHECKPOINTS
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Browse saved debate states and resume from any checkpoint.
              Automatic checkpoints are created at each round boundary.
            </p>
          </div>

          <div className="mb-6 p-4 border border-acid-yellow/30 bg-acid-yellow/5 rounded">
            <h3 className="text-sm font-mono text-acid-yellow mb-2">Checkpoint Features</h3>
            <ul className="text-xs font-mono text-text-muted space-y-1">
              <li>• <span className="text-acid-green">Auto-save</span>: Checkpoints created after each round</li>
              <li>• <span className="text-acid-green">State recovery</span>: Resume debates from any saved point</li>
              <li>• <span className="text-acid-green">Branching</span>: Create alternate debate paths from checkpoints</li>
              <li>• <span className="text-acid-green">Comparison</span>: Compare outcomes from different branches</li>
            </ul>
          </div>

          <PanelErrorBoundary panelName="Checkpoints">
            <CheckpointPanel
              backendConfig={{ apiUrl: backendConfig.api, wsUrl: backendConfig.ws }}
              onResume={handleResume}
            />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // DEBATE CHECKPOINTS
          </p>
        </footer>
      </main>
    </>
  );
}
