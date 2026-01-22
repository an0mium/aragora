'use client';

import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const MLDashboard = dynamic(
  () => import('@/components/MLDashboard').then(m => ({ default: m.MLDashboard })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[500px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function MLPage() {
  const { config: backendConfig } = useBackend();

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} ML INTELLIGENCE
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Machine learning capabilities for agent routing, quality scoring,
              consensus prediction, and training data management.
            </p>
          </div>

          <div className="mb-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
            <h3 className="text-sm font-mono text-acid-cyan mb-2">ML Capabilities</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono text-text-muted">
              <div>
                <span className="text-acid-green">Agent Routing</span>
                <p>Task-based team selection</p>
              </div>
              <div>
                <span className="text-acid-green">Quality Scoring</span>
                <p>Response quality analysis</p>
              </div>
              <div>
                <span className="text-acid-green">Consensus Prediction</span>
                <p>Convergence likelihood</p>
              </div>
              <div>
                <span className="text-acid-green">Training Export</span>
                <p>SFT/DPO data generation</p>
              </div>
            </div>
          </div>

          <PanelErrorBoundary panelName="ML Dashboard">
            <MLDashboard apiBase={backendConfig.api} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // ML INTELLIGENCE
          </p>
        </footer>
      </main>
    </>
  );
}
