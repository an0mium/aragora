'use client';

import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const SchedulerDashboard = dynamic(
  () => import('@/components/SchedulerDashboard').then(m => ({ default: m.SchedulerDashboard })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-[500px] bg-surface rounded" />
      </div>
    ),
  }
);

export default function SchedulerPage() {
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
            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/workflows"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [WORKFLOWS]
              </Link>
              <Link
                href="/analytics"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ANALYTICS]
              </Link>
              <Link
                href="/integrations"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INTEGRATIONS]
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
              {'>'} AUDIT SCHEDULER
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Automate your audits with cron schedules, webhooks, and CI/CD integration.
              Schedule recurring security scans, compliance checks, and quality audits.
            </p>
          </div>

          <div className="mb-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
            <h3 className="text-sm font-mono text-acid-cyan mb-2">Trigger Types</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs font-mono text-text-muted">
              <div>
                <span className="text-acid-green">Cron Schedule</span>
                <p>Time-based triggers</p>
              </div>
              <div>
                <span className="text-acid-green">Interval</span>
                <p>Run every N minutes</p>
              </div>
              <div>
                <span className="text-acid-green">Webhook</span>
                <p>External HTTP triggers</p>
              </div>
              <div>
                <span className="text-acid-green">Git Push</span>
                <p>CI/CD integration</p>
              </div>
              <div>
                <span className="text-acid-green">File Upload</span>
                <p>New document events</p>
              </div>
            </div>
          </div>

          <PanelErrorBoundary panelName="Scheduler Dashboard">
            <SchedulerDashboard apiBase={backendConfig.api} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // AUDIT SCHEDULER
          </p>
        </footer>
      </main>
    </>
  );
}
