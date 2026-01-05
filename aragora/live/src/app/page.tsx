'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { useNomicStream } from '@/hooks/useNomicStream';
import { ConnectionStatus } from '@/components/ConnectionStatus';
import { MetricsCards } from '@/components/MetricsCards';
import { PhaseProgress } from '@/components/PhaseProgress';
import { AgentPanel } from '@/components/AgentPanel';
import { AgentTabs } from '@/components/AgentTabs';
import { RoundProgress } from '@/components/RoundProgress';
import { HistoryPanel } from '@/components/HistoryPanel';
import { UserParticipation } from '@/components/UserParticipation';
import { ReplayBrowser } from '@/components/ReplayBrowser';
import { DebateBrowser } from '@/components/DebateBrowser';
import { TournamentPanel } from '@/components/TournamentPanel';
import { CruxPanel } from '@/components/CruxPanel';
import { MemoryInspector } from '@/components/MemoryInspector';
import { CitationsPanel } from '@/components/CitationsPanel';
import { CapabilityProbePanel } from '@/components/CapabilityProbePanel';
import { OperationalModesPanel } from '@/components/OperationalModesPanel';
import { RedTeamAnalysisPanel } from '@/components/RedTeamAnalysisPanel';
import { ContraryViewsPanel } from '@/components/ContraryViewsPanel';
import { RiskWarningsPanel } from '@/components/RiskWarningsPanel';
import { AnalyticsPanel } from '@/components/AnalyticsPanel';
import { CalibrationPanel } from '@/components/CalibrationPanel';
import { ConsensusKnowledgeBase } from '@/components/ConsensusKnowledgeBase';
import { DebateListPanel } from '@/components/DebateListPanel';
import { AgentComparePanel } from '@/components/AgentComparePanel';
import { TrendingTopicsPanel } from '@/components/TrendingTopicsPanel';
import { VerdictCard } from '@/components/VerdictCard';
import { CompareButton } from '@/components/CompareView';
import { DeepAuditToggle } from '@/components/DeepAuditView';
import { ThemeToggle } from '@/components/ThemeToggle';
import { DocumentUpload } from '@/components/DocumentUpload';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { StatusBar, StatusPill } from '@/components/StatusBar';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { BackendSelector, useBackend, BACKENDS } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import type { NomicState } from '@/types/events';

// Dynamic imports for heavy/conditionally-shown components
// These are code-split to reduce initial bundle size

// Shown once per session
const BootSequence = dynamic(() => import('@/components/BootSequence').then(m => ({ default: m.BootSequence })), {
  ssr: false,
});

// Shown only on aragora.ai domain
const LandingPage = dynamic(() => import('@/components/LandingPage').then(m => ({ default: m.LandingPage })), {
  ssr: false,
  loading: () => <div className="min-h-screen bg-bg flex items-center justify-center"><div className="text-acid-green font-mono animate-pulse">LOADING...</div></div>,
});

// Modal - only shown on button click
const CompareView = dynamic(() => import('@/components/CompareView').then(m => ({ default: m.CompareView })), {
  ssr: false,
});

// Only shown in deep-audit view mode
const DeepAuditView = dynamic(() => import('@/components/DeepAuditView').then(m => ({ default: m.DeepAuditView })), {
  ssr: false,
});

// Heavy panels (788, 566, 498, 485, 451 lines)
const LeaderboardPanel = dynamic(() => import('@/components/LeaderboardPanel').then(m => ({ default: m.LeaderboardPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-40 bg-surface rounded" /></div>,
});

const AgentNetworkPanel = dynamic(() => import('@/components/AgentNetworkPanel').then(m => ({ default: m.AgentNetworkPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-40 bg-surface rounded" /></div>,
});

const InsightsPanel = dynamic(() => import('@/components/InsightsPanel').then(m => ({ default: m.InsightsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-40 bg-surface rounded" /></div>,
});

const LaboratoryPanel = dynamic(() => import('@/components/LaboratoryPanel').then(m => ({ default: m.LaboratoryPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-40 bg-surface rounded" /></div>,
});

const MetricsPanel = dynamic(() => import('@/components/MetricsPanel').then(m => ({ default: m.MetricsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-40 bg-surface rounded" /></div>,
});

type ViewMode = 'tabs' | 'stream' | 'deep-audit';
type SiteMode = 'landing' | 'dashboard' | 'loading';

export default function Home() {
  const router = useRouter();

  // Backend selection (production vs development)
  const { config: backendConfig } = useBackend();
  const [apiBase, setApiBase] = useState(BACKENDS.production.api);
  const [wsUrl, setWsUrl] = useState(BACKENDS.production.ws);

  // Update URLs when backend changes
  useEffect(() => {
    setApiBase(backendConfig.api);
    setWsUrl(backendConfig.ws);
  }, [backendConfig]);

  const { events, connected, nomicState: wsNomicState, activeLoops, selectedLoopId, selectLoop, sendMessage, onAck, onError } = useNomicStream(wsUrl);

  // Domain detection - show landing page on aragora.ai, dashboard on live.aragora.ai
  const [siteMode, setSiteMode] = useState<SiteMode>('loading');

  useEffect(() => {
    const hostname = window.location.hostname;
    // Show landing page on aragora.ai (but not live.aragora.ai or localhost)
    if (hostname === 'aragora.ai' || hostname === 'www.aragora.ai') {
      setSiteMode('landing');
    } else {
      setSiteMode('dashboard');
    }
  }, []);

  // Handle debate started from landing page - navigate to debate viewer
  const handleDebateStarted = useCallback((debateId: string) => {
    // Navigate to the dedicated debate viewer page
    router.push(`/debate/${debateId}`);
  }, [router]);
  // Local state for nomicState, initialized from wsNomicState and updated by events
  const [localNomicState, setLocalNomicState] = useState<NomicState | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Merge wsNomicState (from WebSocket) with local state - prefer WS state for cycle/phase
  const nomicState: NomicState | null = wsNomicState || localNomicState ? {
    ...localNomicState,
    ...wsNomicState,
    // Local state can override for things updated by events
    completed_tasks: localNomicState?.completed_tasks ?? wsNomicState?.completed_tasks,
    last_success: localNomicState?.last_success ?? wsNomicState?.last_success,
  } : null;
  const [viewMode, setViewMode] = useState<ViewMode>('tabs');
  const [showCompare, setShowCompare] = useState(false);
  const [showBoot, setShowBoot] = useState(true);
  const [skipBoot, setSkipBoot] = useState(false);

  // Check if boot was shown before (session storage)
  useEffect(() => {
    const bootShown = sessionStorage.getItem('aragora-boot-shown');
    if (bootShown === 'true') {
      setSkipBoot(true);
      setShowBoot(false);
    }
  }, []);

  const handleBootComplete = useCallback(() => {
    setShowBoot(false);
    sessionStorage.setItem('aragora-boot-shown', 'true');
  }, []);

  // Compute effective loop ID - auto-select if only one loop active (fixes race condition)
  const effectiveLoopId = selectedLoopId || (activeLoops.length === 1 ? activeLoops[0].loop_id : null);

  // Note: Initial state now comes from wsNomicState (via loop_list WebSocket event)
  // HTTP API fetch removed - api.aragora.ai only serves WebSocket, not HTTP API

  // Update local nomic state from events
  useEffect(() => {
    if (events.length === 0) return;

    const lastEvent = events[events.length - 1];

    // Update state based on event type
    switch (lastEvent.type) {
      case 'cycle_start':
        setLocalNomicState((prev) => ({
          ...prev,
          cycle: lastEvent.data.cycle as number,
          phase: 'debate',
          last_success: undefined,
        }));
        break;
      case 'phase_start':
        setLocalNomicState((prev) => ({
          ...prev,
          phase: lastEvent.data.phase as string,
        }));
        break;
      case 'task_complete':
        setLocalNomicState((prev) => ({
          ...prev,
          completed_tasks: (prev?.completed_tasks || 0) + 1,
        }));
        break;
      case 'cycle_end':
        setLocalNomicState((prev) => ({
          ...prev,
          last_success: lastEvent.data.success as boolean,
        }));
        break;
      case 'error':
        setError(lastEvent.data.error as string);
        break;
    }
  }, [events]);

  // Derive current phase from state or latest phase event
  const currentPhase = nomicState?.phase || 'idle';

  // Check if we have a verdict
  const hasVerdict = events.some(
    (e) => e.type === 'grounded_verdict' || e.type === 'verdict' || e.type === 'consensus'
  );

  // Format timestamp as relative time
  const formatRelativeTime = (timestamp: number) => {
    const seconds = Math.floor((Date.now() / 1000) - timestamp);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  // User participation handlers (use effectiveLoopId to auto-select single active loop)
  const handleUserVote = (choice: string, intensity?: number) => {
    if (!effectiveLoopId) {
      setError('No active debate loop selected. Please wait for a debate to start.');
      return;
    }
    sendMessage({
      type: 'user_vote',
      loop_id: effectiveLoopId,
      payload: { choice, intensity: intensity ?? 5 }  // Default to neutral intensity
    });
  };

  const handleUserSuggestion = (suggestion: string) => {
    if (!effectiveLoopId) {
      setError('No active debate loop selected. Please wait for a debate to start.');
      return;
    }
    sendMessage({
      type: 'user_suggestion',
      loop_id: effectiveLoopId,
      payload: { suggestion }
    });
  };

  // Loading state - show minimal loading indicator
  if (siteMode === 'loading') {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="text-acid-green font-mono animate-pulse">INITIALIZING...</div>
      </div>
    );
  }

  // Landing page for aragora.ai
  if (siteMode === 'landing') {
    return (
      <LandingPage
        apiBase={apiBase}
        wsUrl={wsUrl}
        onDebateStarted={handleDebateStarted}
      />
    );
  }

  // Dashboard for live.aragora.ai and localhost
  return (
    <>
      {/* Boot Sequence */}
      {showBoot && <BootSequence onComplete={handleBootComplete} skip={skipBoot} />}

      {/* CRT Effects */}
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Compare Modal */}
        {showCompare && (
          <CompareView events={events} onClose={() => setShowCompare(false)} />
        )}

        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            {/* ASCII Logo */}
            <AsciiBannerCompact connected={connected} />

            <div className="flex items-center gap-3">
              {/* View Mode Toggle */}
              <div className="flex items-center gap-0.5 bg-bg border border-acid-green/30 p-0.5 font-mono text-xs">
                <button
                  onClick={() => setViewMode('tabs')}
                  className={`px-2 py-1 transition-colors ${
                    viewMode === 'tabs'
                      ? 'bg-acid-green text-bg'
                      : 'text-text-muted hover:text-acid-green'
                  }`}
                >
                  [TABS]
                </button>
                <button
                  onClick={() => setViewMode('stream')}
                  className={`px-2 py-1 transition-colors ${
                    viewMode === 'stream'
                      ? 'bg-acid-green text-bg'
                      : 'text-text-muted hover:text-acid-green'
                  }`}
                >
                  [STREAM]
                </button>
              </div>

              {/* Deep Audit Toggle */}
              <DeepAuditToggle
                isActive={viewMode === 'deep-audit'}
                onToggle={() => setViewMode(viewMode === 'deep-audit' ? 'tabs' : 'deep-audit')}
              />

              {/* Compare Button */}
              <CompareButton onClick={() => setShowCompare(true)} />

              {/* Status Pill */}
              <StatusPill connected={connected} phase={currentPhase} />

              {/* Backend Selector */}
              <BackendSelector compact />

              {/* Theme Toggle */}
              <ThemeToggle />

              {/* Loop Selector - Only show if multiple loops */}
              {activeLoops.length > 1 && (
                <div className="flex items-center gap-2">
                  <span className="text-text-muted text-xs font-mono">{activeLoops.length} LOOPS</span>
                  <select
                    value={selectedLoopId || ''}
                    onChange={(e) => selectLoop(e.target.value)}
                    className="bg-bg border border-acid-green/30 px-2 py-1 text-xs font-mono text-acid-green"
                  >
                    {activeLoops.map((loop) => (
                      <option key={loop.loop_id} value={loop.loop_id}>
                        {loop.name} (C{loop.cycle}, {formatRelativeTime(loop.started_at)})
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {/* Single loop indicator */}
              {activeLoops.length === 1 && (
                <span className="text-acid-cyan text-xs font-mono">
                  {activeLoops[0].name}
                </span>
              )}
            </div>
          </div>
        </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6 space-y-6">
        {/* Error Banner */}
        {error && (
          <div className="bg-warning/10 border border-warning/30 rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-warning text-xl">⚠️</span>
              <span className="text-warning">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-warning hover:text-warning/80"
            >
              ✕
            </button>
          </div>
        )}

        {/* Phase Progress */}
        <PhaseProgress events={events} currentPhase={currentPhase} />

        {/* Round Progress (new - Heavy3-inspired) */}
        {currentPhase === 'debate' && (
          <RoundProgress events={events} />
        )}

        {/* Metrics */}
        <MetricsCards nomicState={nomicState} events={events} />

        {/* Verdict Card (when available) */}
        {hasVerdict && <VerdictCard events={events} />}

        {/* Main Panels */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Agent Activity - Switch based on view mode */}
          <div className="lg:col-span-2 min-h-[500px]">
            {viewMode === 'deep-audit' ? (
              <DeepAuditView
                events={events}
                isActive={true}
                onToggle={() => setViewMode('tabs')}
              />
            ) : viewMode === 'tabs' ? (
              <PanelErrorBoundary panelName="Debate Dashboard">
                <AgentTabs events={events} />
              </PanelErrorBoundary>
            ) : (
              <AgentPanel events={events} />
            )}
          </div>

          {/* Side Panel */}
          <div className="lg:col-span-1 space-y-4">
            <PanelErrorBoundary panelName="Document Upload">
              <DocumentUpload apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="User Participation">
              <UserParticipation
                events={events}
                onVote={handleUserVote}
                onSuggest={handleUserSuggestion}
                onAck={onAck}
                onError={onError}
              />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Citations">
              <CitationsPanel events={events} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="History">
              <HistoryPanel />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Trending Topics">
              <TrendingTopicsPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Debate List">
              <DebateListPanel />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Agent Compare">
              <AgentComparePanel />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Leaderboard">
              <LeaderboardPanel wsMessages={events} loopId={effectiveLoopId} apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Calibration">
              <CalibrationPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Tournament">
              <TournamentPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Analytics">
              <AnalyticsPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Server Metrics">
              <MetricsPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Consensus KB">
              <ConsensusKnowledgeBase apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Debate Browser">
              <DebateBrowser />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Insights">
              <InsightsPanel wsMessages={events} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Crux Analysis">
              <CruxPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Memory Inspector">
              <MemoryInspector apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Laboratory">
              <LaboratoryPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Agent Network">
              <AgentNetworkPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Capability Probes">
              <CapabilityProbePanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Operational Modes">
              <OperationalModesPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Red Team">
              <RedTeamAnalysisPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Contrary Views">
              <ContraryViewsPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Risk Warnings">
              <RiskWarningsPanel apiBase={apiBase} />
            </PanelErrorBoundary>
            <PanelErrorBoundary panelName="Replay Browser">
              <ReplayBrowser />
            </PanelErrorBoundary>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'═'.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA NOMIC LOOP // REAL-TIME MONITORING
          </p>
          <p className="text-acid-cyan mt-2">
            <a
              href="https://aragora.ai"
              className="hover:text-acid-green transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              [ ARAGORA.AI ]
            </a>
          </p>
          <div className="mt-4 text-text-muted/50 text-[10px] max-w-2xl mx-auto">
            Built on{' '}
            <a href="https://github.com/AI-Counsel/ai-counsel" className="text-acid-green/50 hover:text-acid-green" target="_blank" rel="noopener noreferrer">ai-counsel</a>{' | '}
            <a href="https://github.com/Tsinghua-MARS-Lab/DebateLLM" className="text-acid-green/50 hover:text-acid-green" target="_blank" rel="noopener noreferrer">DebateLLM</a>{' | '}
            <a href="https://github.com/camel-ai/camel" className="text-acid-green/50 hover:text-acid-green" target="_blank" rel="noopener noreferrer">CAMEL-AI</a>{' | '}
            <a href="https://github.com/joonspk-research/generative_agents" className="text-acid-green/50 hover:text-acid-green" target="_blank" rel="noopener noreferrer">Generative Agents</a>{' | '}
            <a href="https://heavy3.ai" className="text-acid-green/50 hover:text-acid-green" target="_blank" rel="noopener noreferrer">Heavy3.ai</a>
          </div>
          <div className="text-acid-green/50 mt-4">
            {'═'.repeat(40)}
          </div>
        </footer>
      </div>

      {/* Status Bar */}
      <StatusBar
        connected={connected}
        events={events}
        cycle={nomicState?.cycle || 0}
        phase={currentPhase}
      />
    </main>
    </>
  );
}
