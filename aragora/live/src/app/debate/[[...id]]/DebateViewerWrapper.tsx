'use client';

import { useState, useEffect } from 'react';
import { DebateViewer } from '@/components/DebateViewer';
import { CruxPanel } from '@/components/CruxPanel';
import { AnalyticsPanel } from '@/components/AnalyticsPanel';
import { RedTeamAnalysisPanel } from '@/components/RedTeamAnalysisPanel';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

export function DebateViewerWrapper() {
  const [debateId, setDebateId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showAnalysis, setShowAnalysis] = useState(false);

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

  return (
    <div className="min-h-screen bg-bg">
      {/* Main Debate Viewer */}
      <DebateViewer debateId={debateId} />

      {/* Analysis Panels Toggle */}
      <div className="container mx-auto px-4 py-4">
        <button
          onClick={() => setShowAnalysis(!showAnalysis)}
          className="w-full py-3 border border-acid-green/30 bg-surface hover:bg-surface/80 transition-colors font-mono text-sm text-acid-green"
        >
          {showAnalysis ? '[-] HIDE ANALYSIS PANELS' : '[+] SHOW ANALYSIS PANELS'}
        </button>
      </div>

      {/* Collapsible Analysis Section */}
      {showAnalysis && (
        <div className="container mx-auto px-4 pb-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Crux Analysis Panel */}
            <div className="lg:col-span-1">
              <CruxPanel debateId={debateId} apiBase={API_BASE} />
            </div>

            {/* Analytics Panel (with Graph Stats) */}
            <div className="lg:col-span-1">
              <AnalyticsPanel apiBase={API_BASE} loopId={debateId} />
            </div>

            {/* Red Team Analysis Panel */}
            <div className="lg:col-span-1">
              <RedTeamAnalysisPanel debateId={debateId} apiBase={API_BASE} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
