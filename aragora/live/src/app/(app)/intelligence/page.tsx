'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { OrgSummary } from '@/components/intelligence/OrgSummary';
import { TeamMetrics } from '@/components/intelligence/TeamMetrics';
import { ROIAnalysis } from '@/components/intelligence/ROIAnalysis';
import { QualityMetrics } from '@/components/intelligence/QualityMetrics';

export default function IntelligencePage() {
  const { config } = useBackend();
  const backendUrl = config.api;

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
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
              </Link>
              <Link
                href="/billing"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [BILLING]
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
              Organizational Intelligence
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Executive dashboard with KPIs, team performance, ROI analysis, and quality metrics.
            </p>
          </div>

          {/* Grid layout for dashboard panels */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Organization Summary - Full width on top */}
            <div className="lg:col-span-2">
              <PanelErrorBoundary panelName="Organization Summary">
                <OrgSummary backendUrl={backendUrl} />
              </PanelErrorBoundary>
            </div>

            {/* Team Metrics */}
            <PanelErrorBoundary panelName="Team Metrics">
              <TeamMetrics backendUrl={backendUrl} />
            </PanelErrorBoundary>

            {/* ROI Analysis */}
            <PanelErrorBoundary panelName="ROI Analysis">
              <ROIAnalysis backendUrl={backendUrl} />
            </PanelErrorBoundary>

            {/* Quality Metrics - Full width */}
            <div className="lg:col-span-2">
              <PanelErrorBoundary panelName="Quality Metrics">
                <QualityMetrics backendUrl={backendUrl} />
              </PanelErrorBoundary>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // ORGANIZATIONAL INTELLIGENCE
          </p>
        </footer>
      </main>
    </>
  );
}
