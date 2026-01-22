'use client';

import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const SettingsPanel = dynamic(
  () => import('@/components/settings-panel').then(m => ({ default: m.SettingsPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-96 bg-surface rounded" />
      </div>
    ),
  }
);

export default function SettingsPage() {
  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              Settings
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Configure your preferences, API keys, and integrations.
            </p>
          </div>

          <PanelErrorBoundary panelName="Settings">
            <SettingsPanel />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2" aria-hidden="true">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // SETTINGS
          </p>
        </footer>
      </main>
    </>
  );
}
