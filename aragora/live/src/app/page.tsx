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
import { DebateExportModal } from '@/components/DebateExportModal';
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
import { CollapsibleSection } from '@/components/CollapsibleSection';
import { FeaturesProvider } from '@/context/FeaturesContext';
import { FeatureGuard } from '@/components/FeatureGuard';
import { DashboardModeToggle } from '@/components/DashboardModeToggle';
import { useDashboardPreferences } from '@/hooks/useDashboardPreferences';
import { OnboardingWizard } from '@/components/OnboardingWizard';
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

const BreakpointsPanel = dynamic(() => import('@/components/BreakpointsPanel').then(m => ({ default: m.BreakpointsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-40 bg-surface rounded" /></div>,
});

const MetricsPanel = dynamic(() => import('@/components/MetricsPanel').then(m => ({ default: m.MetricsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-40 bg-surface rounded" /></div>,
});

// Secondary panels - dynamically loaded for code splitting
const TournamentPanel = dynamic(() => import('@/components/TournamentPanel').then(m => ({ default: m.TournamentPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const CruxPanel = dynamic(() => import('@/components/CruxPanel').then(m => ({ default: m.CruxPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const MemoryInspector = dynamic(() => import('@/components/MemoryInspector').then(m => ({ default: m.MemoryInspector })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const LearningDashboard = dynamic(() => import('@/components/LearningDashboard').then(m => ({ default: m.LearningDashboard })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const CitationsPanel = dynamic(() => import('@/components/CitationsPanel').then(m => ({ default: m.CitationsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const CapabilityProbePanel = dynamic(() => import('@/components/CapabilityProbePanel').then(m => ({ default: m.CapabilityProbePanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const OperationalModesPanel = dynamic(() => import('@/components/OperationalModesPanel').then(m => ({ default: m.OperationalModesPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const RedTeamAnalysisPanel = dynamic(() => import('@/components/RedTeamAnalysisPanel').then(m => ({ default: m.RedTeamAnalysisPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const ContraryViewsPanel = dynamic(() => import('@/components/ContraryViewsPanel').then(m => ({ default: m.ContraryViewsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const RiskWarningsPanel = dynamic(() => import('@/components/RiskWarningsPanel').then(m => ({ default: m.RiskWarningsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const AnalyticsPanel = dynamic(() => import('@/components/AnalyticsPanel').then(m => ({ default: m.AnalyticsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const CalibrationPanel = dynamic(() => import('@/components/CalibrationPanel').then(m => ({ default: m.CalibrationPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const TricksterAlertPanel = dynamic(() => import('@/components/TricksterAlertPanel').then(m => ({ default: m.TricksterAlertPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const RhetoricalObserverPanel = dynamic(() => import('@/components/RhetoricalObserverPanel').then(m => ({ default: m.RhetoricalObserverPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const ConsensusKnowledgeBase = dynamic(() => import('@/components/ConsensusKnowledgeBase').then(m => ({ default: m.ConsensusKnowledgeBase })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const DebateListPanel = dynamic(() => import('@/components/DebateListPanel').then(m => ({ default: m.DebateListPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const AgentComparePanel = dynamic(() => import('@/components/AgentComparePanel').then(m => ({ default: m.AgentComparePanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const TrendingTopicsPanel = dynamic(() => import('@/components/TrendingTopicsPanel').then(m => ({ default: m.TrendingTopicsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const ImpasseDetectionPanel = dynamic(() => import('@/components/ImpasseDetectionPanel').then(m => ({ default: m.ImpasseDetectionPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const LearningEvolution = dynamic(() => import('@/components/LearningEvolution').then(m => ({ default: m.LearningEvolution })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const MomentsTimeline = dynamic(() => import('@/components/MomentsTimeline').then(m => ({ default: m.MomentsTimeline })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const ConsensusQualityDashboard = dynamic(() => import('@/components/ConsensusQualityDashboard').then(m => ({ default: m.ConsensusQualityDashboard })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const MemoryAnalyticsPanel = dynamic(() => import('@/components/MemoryAnalyticsPanel').then(m => ({ default: m.MemoryAnalyticsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

// Additional panels - wired for feature exposure
const UncertaintyPanel = dynamic(() => import('@/components/UncertaintyPanel').then(m => ({ default: m.UncertaintyPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const MoodTrackerPanel = dynamic(() => import('@/components/MoodTrackerPanel').then(m => ({ default: m.MoodTrackerPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const GauntletPanel = dynamic(() => import('@/components/GauntletPanel').then(m => ({ default: m.GauntletPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const ReviewsPanel = dynamic(() => import('@/components/ReviewsPanel').then(m => ({ default: m.ReviewsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const TournamentViewerPanel = dynamic(() => import('@/components/TournamentViewerPanel').then(m => ({ default: m.TournamentViewerPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const PluginMarketplacePanel = dynamic(() => import('@/components/PluginMarketplacePanel').then(m => ({ default: m.PluginMarketplacePanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const MemoryExplorerPanel = dynamic(() => import('@/components/MemoryExplorerPanel').then(m => ({ default: m.MemoryExplorerPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const EvidenceVisualizerPanel = dynamic(() => import('@/components/EvidenceVisualizerPanel').then(m => ({ default: m.EvidenceVisualizerPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const BatchDebatePanel = dynamic(() => import('@/components/BatchDebatePanel').then(m => ({ default: m.BatchDebatePanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const SettingsPanel = dynamic(() => import('@/components/SettingsPanel').then(m => ({ default: m.SettingsPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const ApiExplorerPanel = dynamic(() => import('@/components/ApiExplorerPanel').then(m => ({ default: m.ApiExplorerPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const CheckpointPanel = dynamic(() => import('@/components/CheckpointPanel').then(m => ({ default: m.CheckpointPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const ProofVisualizerPanel = dynamic(() => import('@/components/ProofVisualizerPanel').then(m => ({ default: m.ProofVisualizerPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const EvolutionPanel = dynamic(() => import('@/components/EvolutionPanel').then(m => ({ default: m.EvolutionPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const PulseSchedulerControlPanel = dynamic(() => import('@/components/PulseSchedulerControlPanel').then(m => ({ default: m.PulseSchedulerControlPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const EvidencePanel = dynamic(() => import('@/components/EvidencePanel').then(m => ({ default: m.EvidencePanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

const BroadcastPanel = dynamic(() => import('@/components/broadcast/BroadcastPanel').then(m => ({ default: m.BroadcastPanel })), {
  ssr: false,
  loading: () => <div className="card p-4 animate-pulse"><div className="h-32 bg-surface rounded" /></div>,
});

type ViewMode = 'tabs' | 'stream' | 'deep-audit';
type SiteMode = 'landing' | 'dashboard' | 'loading';

// Mobile sidebar toggle state
const useMobileSidebar = () => {
  const [showSidebar, setShowSidebar] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return { showSidebar, setShowSidebar, isMobile };
};

export default function Home() {
  const router = useRouter();

  // Dashboard preferences (Focus vs Explorer mode)
  const {
    preferences,
    setMode,
    isFocusMode,
    isLoaded: prefsLoaded,
    markOnboardingComplete,
  } = useDashboardPreferences();

  // Onboarding wizard state
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);

  // Mobile sidebar state
  const { showSidebar, setShowSidebar, isMobile } = useMobileSidebar();

  // Boot sequence state (must be declared before useEffect that uses it)
  const [showBoot, setShowBoot] = useState(true);
  const [skipBoot, setSkipBoot] = useState(false);

  // Show onboarding for new users (after boot sequence completes)
  useEffect(() => {
    if (prefsLoaded && !preferences.hasSeenOnboarding && !showBoot) {
      setShowOnboarding(true);
    }
  }, [prefsLoaded, preferences.hasSeenOnboarding, showBoot]);

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

  // Domain detection - show dashboard on live.aragora.ai, landing elsewhere
  const [siteMode, setSiteMode] = useState<SiteMode>('loading');
  // Only show ASCII art in header on live.aragora.ai
  const [showHeaderAscii, setShowHeaderAscii] = useState(false);

  useEffect(() => {
    const hostname = window.location.hostname;
    // Show monitoring dashboard only on live.aragora.ai
    if (hostname === 'live.aragora.ai') {
      setSiteMode('dashboard');
    } else {
      // aragora.ai, www.aragora.ai, localhost all show landing page
      setSiteMode('landing');
    }
    // Show ASCII art on live.aragora.ai (monitoring dashboard)
    setShowHeaderAscii(hostname === 'live.aragora.ai');
  }, []);

  // Handle debate started from landing page - navigate to debate viewer
  const handleDebateStarted = useCallback((debateId: string) => {
    // Navigate to the dedicated debate viewer page
    router.push(`/debate/${debateId}`);
  }, [router]);

  // Handle starting a debate from a trending topic
  const handleStartDebateFromTrend = useCallback(async (topic: string, source: string) => {
    try {
      const response = await fetch(`${apiBase}/api/debate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: topic,
          agents: 'grok,anthropic-api,openai-api,deepseek',
          rounds: 3,
          metadata: { source, from_trending: true },
        }),
      });

      const data = await response.json();
      if (data.success && data.debate_id) {
        router.push(`/debate/${data.debate_id}`);
      } else {
        setError(data.error || 'Failed to start debate from trend');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start debate');
    }
  }, [apiBase, router]);
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
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportDebateId, setExportDebateId] = useState<string | null>(null);

  // Track current debate for impasse detection and broadcast
  const [currentDebateId, setCurrentDebateId] = useState<string | null>(null);
  const [debateTitle, setDebateTitle] = useState<string | null>(null);

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

  // Onboarding completion handler
  const handleOnboardingComplete = useCallback((persona: string, startWithPrompt?: string) => {
    markOnboardingComplete();
    setShowOnboarding(false);
    if (startWithPrompt) {
      setPendingPrompt(startWithPrompt);
    }
  }, [markOnboardingComplete]);

  const handleOnboardingSkip = useCallback(() => {
    markOnboardingComplete();
    setShowOnboarding(false);
  }, [markOnboardingComplete]);

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

  // Track current debate ID and title from events
  useEffect(() => {
    const debateEvent = events.find(
      (e) => e.type === 'debate_start' || (e.data && 'debate_id' in e.data)
    );
    if (debateEvent?.data && 'debate_id' in debateEvent.data) {
      setCurrentDebateId(debateEvent.data.debate_id as string);
      // Extract title from event data if available
      const eventData = debateEvent.data as Record<string, unknown>;
      const title = eventData.title || eventData.topic || eventData.question;
      if (typeof title === 'string') {
        setDebateTitle(title);
      }
    }
  }, [events]);

  // Export modal handlers
  const handleExportDebate = useCallback((debateId: string) => {
    setExportDebateId(debateId);
    setShowExportModal(true);
  }, []);

  const handleCloseExport = useCallback(() => {
    setShowExportModal(false);
    setExportDebateId(null);
  }, []);

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

  // Landing page for www.aragora.ai
  if (siteMode === 'landing') {
    return (
      <LandingPage
        apiBase={apiBase}
        wsUrl={wsUrl}
        onDebateStarted={handleDebateStarted}
      />
    );
  }

  // Dashboard for aragora.ai, localhost, and other hosts
  return (
    <FeaturesProvider apiBase={apiBase}>
      {/* Boot Sequence */}
      {showBoot && <BootSequence onComplete={handleBootComplete} skip={skipBoot} />}

      {/* Onboarding Wizard - shows for new users after boot */}
      {showOnboarding && (
        <OnboardingWizard
          onComplete={handleOnboardingComplete}
          onSkip={handleOnboardingSkip}
        />
      )}

      {/* CRT Effects */}
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Compare Modal */}
        {showCompare && (
          <CompareView events={events} onClose={() => setShowCompare(false)} />
        )}

        {/* Export Modal */}
        {showExportModal && exportDebateId && (
          <DebateExportModal
            debateId={exportDebateId}
            isOpen={showExportModal}
            onClose={handleCloseExport}
            apiBase={apiBase}
          />
        )}

        {/* Header - Responsive */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-screen-2xl mx-auto px-3 sm:px-4 lg:px-6 py-2 sm:py-3">
            <div className="flex items-center justify-between gap-2">
              {/* ASCII Logo - only show art on aragora.ai */}
              <AsciiBannerCompact connected={connected} showAsciiArt={showHeaderAscii} />

              {/* Mobile: Minimal controls */}
              <div className="flex items-center gap-1 sm:gap-2 lg:gap-3">
                {/* Mobile Sidebar Toggle */}
                {isMobile && (
                  <button
                    onClick={() => setShowSidebar(!showSidebar)}
                    className="px-2 py-1 border border-acid-green/30 text-xs font-mono text-acid-green hover:bg-acid-green/10 transition-colors"
                    aria-label="Toggle sidebar"
                  >
                    {showSidebar ? '[HIDE PANELS]' : '[PANELS]'}
                  </button>
                )}
                {/* View Mode Toggle - Hidden on mobile */}
                <div className="hidden sm:flex items-center gap-0.5 bg-bg border border-acid-green/30 p-0.5 font-mono text-xs">
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

                {/* Dashboard Mode Toggle - Focus vs Explorer */}
                <div className="hidden sm:block">
                  <DashboardModeToggle
                    mode={preferences.mode}
                    onModeChange={setMode}
                    compact
                  />
                </div>

                {/* Deep Audit Toggle - Hidden on small screens */}
                <div className="hidden md:block">
                  <DeepAuditToggle
                    isActive={viewMode === 'deep-audit'}
                    onToggle={() => setViewMode(viewMode === 'deep-audit' ? 'tabs' : 'deep-audit')}
                  />
                </div>

                {/* Compare Button - Hidden on mobile */}
                <div className="hidden lg:block">
                  <CompareButton onClick={() => setShowCompare(true)} />
                </div>

                {/* Status Pill - Always visible */}
                <StatusPill connected={connected} phase={currentPhase} />

                {/* Backend Selector - Hidden on mobile */}
                <div className="hidden sm:block">
                  <BackendSelector compact />
                </div>

                {/* Theme Toggle - Always visible */}
                <ThemeToggle />

                {/* Loop Selector - Only show if multiple loops, hidden on mobile */}
                {activeLoops.length > 1 && (
                  <div className="hidden md:flex items-center gap-2">
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
                {/* Single loop indicator - Hidden on mobile */}
                {activeLoops.length === 1 && (
                  <span className="hidden sm:inline text-acid-cyan text-xs font-mono">
                    {activeLoops[0].name}
                  </span>
                )}
              </div>
            </div>
          </div>
        </header>

      {/* Main Content - Wider container */}
      <div className="max-w-screen-2xl mx-auto px-3 sm:px-4 lg:px-6 py-4 sm:py-6 space-y-4 sm:space-y-6">
        {/* Error Banner */}
        {error && (
          <div className="bg-warning/10 border border-warning/30 p-3 sm:p-4 flex items-center justify-between text-sm">
            <div className="flex items-center gap-2 sm:gap-3">
              <span className="text-warning">⚠</span>
              <span className="text-warning">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-warning hover:text-warning/80 px-2"
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

        {/* Mobile Progress Summary - shown when sidebar hidden */}
        {isMobile && !showSidebar && currentPhase !== 'idle' && (
          <div className="p-3 border border-acid-green/30 rounded-lg bg-surface/50">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-acid-green">{currentPhase.toUpperCase()}</span>
              <button
                onClick={() => setShowSidebar(true)}
                className="text-acid-cyan hover:text-acid-green transition-colors"
              >
                [VIEW DETAILS]
              </button>
            </div>
          </div>
        )}

        {/* Main Panels - Responsive grid with wider main panel */}
        <div className="grid grid-cols-1 xl:grid-cols-5 gap-4 lg:gap-6">
          {/* Agent Activity - Takes more space on wider screens */}
          <div className="xl:col-span-3 min-h-[400px] sm:min-h-[500px]">
            {viewMode === 'deep-audit' ? (
              <PanelErrorBoundary panelName="Deep Audit">
                <DeepAuditView
                  events={events}
                  isActive={true}
                  onToggle={() => setViewMode('tabs')}
                />
              </PanelErrorBoundary>
            ) : viewMode === 'tabs' ? (
              <PanelErrorBoundary panelName="Debate Dashboard">
                <AgentTabs events={events} />
              </PanelErrorBoundary>
            ) : (
              <PanelErrorBoundary panelName="Agent Stream">
                <AgentPanel events={events} />
              </PanelErrorBoundary>
            )}
          </div>

          {/* Side Panel - Hidden on mobile unless toggled */}
          <div className={`xl:col-span-2 space-y-2 ${isMobile && !showSidebar ? 'hidden' : ''}`}>
            {/* Section 1: Core Debate - expanded by default */}
            <CollapsibleSection
              id="core-debate"
              title="CORE DEBATE"
              defaultOpen={true}
              priority="core"
              description="Essential debate controls: input, voting, citations"
            >
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
              <PanelErrorBoundary panelName="Trickster Alerts">
                <TricksterAlertPanel events={events} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Rhetorical Observer">
                <RhetoricalObserverPanel events={events} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Citations">
                <CitationsPanel events={events} debateId={currentDebateId || undefined} apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="History">
                <HistoryPanel />
              </PanelErrorBoundary>
            </CollapsibleSection>

            {/* Section 2: Browse & Discover */}
            <CollapsibleSection
              id="browse-discover"
              title="BROWSE & DISCOVER"
              defaultOpen={!isFocusMode}
              forceOpen={isFocusMode ? false : undefined}
              description="Find debates, trending topics, and replays"
            >
              <FeatureGuard featureId="pulse">
                <PanelErrorBoundary panelName="Trending Topics">
                  <TrendingTopicsPanel apiBase={apiBase} onStartDebate={handleStartDebateFromTrend} />
                </PanelErrorBoundary>
              </FeatureGuard>
              <PanelErrorBoundary panelName="Debate List">
                <DebateListPanel />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Debate Browser">
                <DebateBrowser />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Replay Browser">
                <ReplayBrowser />
              </PanelErrorBoundary>
            </CollapsibleSection>

            {/* Section 3: Agent Analysis */}
            <CollapsibleSection
              id="agent-analysis"
              title="AGENT ANALYSIS"
              defaultOpen={!isFocusMode}
              forceOpen={isFocusMode ? false : undefined}
              description="Compare agents, view rankings, and tournaments"
            >
              <PanelErrorBoundary panelName="Agent Compare">
                <AgentComparePanel />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Agent Network">
                <AgentNetworkPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Mood Tracker">
                <MoodTrackerPanel events={events} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Leaderboard">
                <LeaderboardPanel wsMessages={events} loopId={effectiveLoopId} apiBase={apiBase} />
              </PanelErrorBoundary>
              <FeatureGuard featureId="calibration">
                <PanelErrorBoundary panelName="Calibration">
                  <CalibrationPanel apiBase={apiBase} events={events} />
                </PanelErrorBoundary>
              </FeatureGuard>
              <FeatureGuard featureId="tournaments">
                <PanelErrorBoundary panelName="Tournament">
                  <TournamentPanel apiBase={apiBase} events={events} />
                </PanelErrorBoundary>
                <PanelErrorBoundary panelName="Tournament Viewer">
                  <TournamentViewerPanel backendConfig={{ apiUrl: apiBase, wsUrl: wsUrl }} />
                </PanelErrorBoundary>
              </FeatureGuard>
            </CollapsibleSection>

            {/* Section 4: Insights & Learning */}
            <CollapsibleSection
              id="insights-learning"
              title="INSIGHTS & LEARNING"
              defaultOpen={!isFocusMode}
              forceOpen={isFocusMode ? false : undefined}
              priority="secondary"
              description="Deep analysis, consensus quality, and learning evolution"
            >
              <PanelErrorBoundary panelName="Moments Timeline">
                <MomentsTimeline apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Uncertainty Analysis">
                <UncertaintyPanel events={events} debateId={currentDebateId || undefined} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Evidence Visualizer">
                <EvidenceVisualizerPanel backendConfig={{ apiUrl: apiBase, wsUrl: wsUrl }} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Consensus Quality">
                <ConsensusQualityDashboard apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Cross-Cycle Learning">
                <LearningDashboard apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Insights">
                <InsightsPanel wsMessages={events} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Crux Analysis">
                <CruxPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Contrary Views">
                <ContraryViewsPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Risk Warnings">
                <RiskWarningsPanel apiBase={apiBase} events={events} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Learning Evolution">
                <LearningEvolution />
              </PanelErrorBoundary>
              <FeatureGuard featureId="evolution">
                <PanelErrorBoundary panelName="Prompt Evolution">
                  <EvolutionPanel backendConfig={{ apiUrl: apiBase, wsUrl: wsUrl }} />
                </PanelErrorBoundary>
              </FeatureGuard>
              {currentDebateId && (
                <PanelErrorBoundary panelName="Evidence">
                  <EvidencePanel debateId={currentDebateId} />
                </PanelErrorBoundary>
              )}
            </CollapsibleSection>

            {/* Section 5: System Tools - Collapsed in Focus Mode */}
            <CollapsibleSection
              id="system-tools"
              title="SYSTEM TOOLS"
              defaultOpen={false}
              forceOpen={isFocusMode ? false : undefined}
              priority="secondary"
              description="Red team analysis, gauntlet, code reviews, and batch operations"
            >
              <PanelErrorBoundary panelName="Capability Probes">
                <CapabilityProbePanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Operational Modes">
                <OperationalModesPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Red Team">
                <RedTeamAnalysisPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Gauntlet Results">
                <GauntletPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Code Reviews">
                <ReviewsPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <FeatureGuard featureId="plugins">
                <PanelErrorBoundary panelName="Plugin Marketplace">
                  <PluginMarketplacePanel backendConfig={{ apiUrl: apiBase, wsUrl: wsUrl }} />
                </PanelErrorBoundary>
              </FeatureGuard>
              <FeatureGuard featureId="laboratory">
                <PanelErrorBoundary panelName="Laboratory">
                  <LaboratoryPanel apiBase={apiBase} events={events} />
                </PanelErrorBoundary>
              </FeatureGuard>
              <PanelErrorBoundary panelName="Breakpoints">
                <BreakpointsPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Batch Debates">
                <BatchDebatePanel />
              </PanelErrorBoundary>
              <FeatureGuard featureId="pulse">
                <PanelErrorBoundary panelName="Pulse Scheduler">
                  <PulseSchedulerControlPanel />
                </PanelErrorBoundary>
              </FeatureGuard>
              {currentDebateId && debateTitle && (
                <PanelErrorBoundary panelName="Broadcast">
                  <BroadcastPanel debateId={currentDebateId} debateTitle={debateTitle} />
                </PanelErrorBoundary>
              )}
            </CollapsibleSection>

            {/* Section 6: Advanced/Debug */}
            <CollapsibleSection
              id="advanced-debug"
              title="ADVANCED / DEBUG"
              defaultOpen={false}
              forceOpen={isFocusMode ? false : undefined}
              priority="advanced"
              description="Analytics, memory inspection, API explorer, and debug tools"
            >
              <PanelErrorBoundary panelName="Analytics">
                <AnalyticsPanel apiBase={apiBase} events={events} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Server Metrics">
                <MetricsPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Consensus KB">
                <ConsensusKnowledgeBase apiBase={apiBase} events={events} />
              </PanelErrorBoundary>
              <FeatureGuard featureId="memory">
                <PanelErrorBoundary panelName="Memory Inspector">
                  <MemoryInspector apiBase={apiBase} />
                </PanelErrorBoundary>
                <PanelErrorBoundary panelName="Memory Explorer">
                  <MemoryExplorerPanel backendConfig={{ apiUrl: apiBase, wsUrl: wsUrl }} />
                </PanelErrorBoundary>
              </FeatureGuard>
              <PanelErrorBoundary panelName="Memory Analytics">
                <MemoryAnalyticsPanel apiBase={apiBase} />
              </PanelErrorBoundary>
              {currentDebateId && (
                <PanelErrorBoundary panelName="Impasse Detection">
                  <ImpasseDetectionPanel debateId={currentDebateId} apiBase={apiBase} />
                </PanelErrorBoundary>
              )}
              <PanelErrorBoundary panelName="API Explorer">
                <ApiExplorerPanel />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Checkpoints">
                <CheckpointPanel backendConfig={{ apiUrl: apiBase, wsUrl: wsUrl }} debateId={currentDebateId || undefined} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Proof Visualizer">
                <ProofVisualizerPanel backendConfig={{ apiUrl: apiBase, wsUrl: wsUrl }} debateId={currentDebateId || undefined} />
              </PanelErrorBoundary>
              <PanelErrorBoundary panelName="Settings">
                <SettingsPanel />
              </PanelErrorBoundary>
            </CollapsibleSection>

            {/* Focus Mode Hint */}
            {isFocusMode && (
              <div className="mt-4 p-3 border border-acid-green/20 rounded-lg bg-surface/20 text-center">
                <p className="text-xs font-mono text-text-muted mb-2">
                  Some sections are collapsed in Focus Mode
                </p>
                <button
                  onClick={() => setMode('explorer')}
                  className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
                >
                  [SWITCH TO EXPLORER MODE]
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Footer - Simplified */}
        <footer className="text-center text-xs font-mono py-6 sm:py-8 border-t border-border mt-6 sm:mt-8">
          <p className="text-text-muted">
            ARAGORA // MULTI-AGENT DEBATE
          </p>
          <p className="text-accent mt-2">
            <a
              href="https://aragora.ai"
              className="hover:opacity-80 transition-opacity"
              target="_blank"
              rel="noopener noreferrer"
            >
              aragora.ai
            </a>
          </p>
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
    </FeaturesProvider>
  );
}
