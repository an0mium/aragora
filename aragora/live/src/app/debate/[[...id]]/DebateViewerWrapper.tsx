'use client';

import { useState, useEffect } from 'react';
import { DebateViewer } from '@/components/DebateViewer';
import { CruxPanel } from '@/components/CruxPanel';
import { AnalyticsPanel } from '@/components/AnalyticsPanel';
import { RedTeamAnalysisPanel } from '@/components/RedTeamAnalysisPanel';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { ImpasseDetectionPanel } from '@/components/ImpasseDetectionPanel';
import { CalibrationPanel } from '@/components/CalibrationPanel';
import { ConsensusKnowledgeBase } from '@/components/ConsensusKnowledgeBase';
import { TrendingTopicsPanel } from '@/components/TrendingTopicsPanel';
import { MemoryInspector } from '@/components/MemoryInspector';
import { MetricsPanel } from '@/components/MetricsPanel';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useBackend } from '@/components/BackendSelector';

export function DebateViewerWrapper() {
  const [debateId, setDebateId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [showDeepAnalysis, setShowDeepAnalysis] = useState(false);
  const { config } = useBackend();

  useEffect(() => {
    // Extract debate ID from actual browser URL: /debate/abc123 -> abc123
    const pathSegments = window.location.pathname.split('/').filter(Boolean);
    const id = pathSegments[1] || null; // ['debate', 'abc123'] -> 'abc123'
    setDebateId(id);
    setIsLoading(false);
  }, []);

  // Show loading while determining debate ID
  if (isLoading) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="text-acid-green font-mono animate-pulse">LOADING...</div>
      </div>
    );
  }

  // No ID provided - show message
  if (!debateId) {
    return (
      <>
        <Scanlines opacity={0.02} />
        <CRTVignette />
        <main className="min-h-screen bg-bg text-text relative z-10">
          <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
            <div className="container mx-auto px-4 py-3 flex items-center justify-between">
              <Link href="/">
                <AsciiBannerCompact connected={true} />
              </Link>
              <ThemeToggle />
            </div>
          </header>
          <div className="container mx-auto px-4 py-20 text-center">
            <div className="text-acid-green font-mono text-xl mb-4">{'>'} NO DEBATE ID PROVIDED</div>
            <Link href="/" className="text-acid-cyan hover:text-acid-green transition-colors font-mono">
              [RETURN TO DASHBOARD]
            </Link>
          </div>
        </main>
      </>
    );
  }

  // Live debates start with 'adhoc_' - hide analysis during streaming for better UX
  const isLiveDebate = debateId.startsWith('adhoc_');

  return (
    <div className="min-h-screen bg-bg">
      {/* Main Debate Viewer */}
      <PanelErrorBoundary panelName="Debate Viewer">
        <DebateViewer debateId={debateId} wsUrl={config.ws} />
      </PanelErrorBoundary>

      {/* Analysis Panels Toggle - hidden during live debates for maximum viewport space */}
      {!isLiveDebate && (
        <div className="container mx-auto px-4 py-4">
          <button
            onClick={() => setShowAnalysis(!showAnalysis)}
            className="w-full py-3 border border-acid-green/30 bg-surface hover:bg-surface/80 transition-colors font-mono text-sm text-acid-green"
          >
            {showAnalysis ? '[-] HIDE ANALYSIS PANELS' : '[+] SHOW ANALYSIS PANELS'}
          </button>
        </div>
      )}

      {/* Collapsible Analysis Section - only for archived debates */}
      {!isLiveDebate && showAnalysis && (
        <div className="container mx-auto px-4 pb-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Crux Analysis Panel */}
            <div className="lg:col-span-1">
              <PanelErrorBoundary panelName="Crux Analysis">
                <CruxPanel debateId={debateId} apiBase={config.api} />
              </PanelErrorBoundary>
            </div>

            {/* Analytics Panel (with Graph Stats) */}
            <div className="lg:col-span-1">
              <PanelErrorBoundary panelName="Analytics">
                <AnalyticsPanel apiBase={config.api} loopId={debateId} />
              </PanelErrorBoundary>
            </div>

            {/* Red Team Analysis Panel */}
            <div className="lg:col-span-1">
              <PanelErrorBoundary panelName="Red Team Analysis">
                <RedTeamAnalysisPanel debateId={debateId} apiBase={config.api} />
              </PanelErrorBoundary>
            </div>
          </div>

          {/* Deep Analysis Toggle */}
          <button
            onClick={() => setShowDeepAnalysis(!showDeepAnalysis)}
            className="w-full py-2 mt-4 border border-acid-cyan/30 bg-surface hover:bg-surface/80 transition-colors font-mono text-xs text-acid-cyan"
          >
            {showDeepAnalysis ? '[-] HIDE DEEP ANALYSIS' : '[+] SHOW DEEP ANALYSIS'}
          </button>

          {/* Deep Analysis Panels */}
          {showDeepAnalysis && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
              <PanelErrorBoundary panelName="Impasse Detection">
                <ImpasseDetectionPanel debateId={debateId} apiBase={config.api} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Calibration">
                <CalibrationPanel apiBase={config.api} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Consensus Knowledge">
                <ConsensusKnowledgeBase apiBase={config.api} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Trending Topics">
                <TrendingTopicsPanel apiBase={config.api} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Memory Inspector">
                <MemoryInspector apiBase={config.api} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Metrics">
                <MetricsPanel apiBase={config.api} />
              </PanelErrorBoundary>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
