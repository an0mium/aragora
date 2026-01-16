'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import Link from 'next/link';
import { fetchDebateById, type DebateArtifact } from '@/utils/supabase';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';
import { UserParticipation } from '@/components/UserParticipation';
import { CitationsPanel } from '@/components/CitationsPanel';
import { MoodTrackerPanel } from '@/components/MoodTrackerPanel';
import { UncertaintyPanel } from '@/components/UncertaintyPanel';
import { TokenStreamViewer } from '@/components/TokenStreamViewer';
import { getAgentColors } from '@/utils/agentColors';
import { useDebateWebSocket, type TranscriptMessage } from '@/hooks/useDebateWebSocket';
import { logger } from '@/utils/logger';
import { API_BASE_URL } from '@/config';

// Type for crux claims from belief network API
interface CruxClaim {
  claim_id: string;
  statement: string;
  author: string;
  crux_score?: number;
}

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.aragora.ai/ws';

interface DebateViewerProps {
  debateId: string;
  wsUrl?: string;
}

export function DebateViewer({ debateId, wsUrl = DEFAULT_WS_URL }: DebateViewerProps) {
  const [debate, setDebate] = useState<DebateArtifact | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // UI state
  const [showParticipation, setShowParticipation] = useState(true);
  const [showCitations, setShowCitations] = useState(false);
  const [userScrolled, setUserScrolled] = useState(false);
  const [showCruxHighlighting, setShowCruxHighlighting] = useState(true);
  const [cruxes, setCruxes] = useState<CruxClaim[]>([]);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Determine if this is a live debate
  const isLiveDebate = debateId.startsWith('adhoc_');

  // Use WebSocket hook for live debates
  const {
    status: liveStatus,
    task: liveTask,
    agents: liveAgents,
    messages: liveMessages,
    streamingMessages,
    streamEvents,
    hasCitations,
    error: liveError,
    errorDetails: liveErrorDetails,
    sendVote,
    sendSuggestion,
    registerAckCallback,
    registerErrorCallback,
    reconnect,
  } = useDebateWebSocket({
    debateId,
    wsUrl,
    enabled: isLiveDebate,
  });

  // State for fallback debate data (when WebSocket misses events)
  const [fallbackDebate, setFallbackDebate] = useState<{
    task: string;
    agents: string[];
    finalAnswer: string;
  } | null>(null);

  // Fetch complete debate data as fallback when WebSocket misses events
  // This happens when the debate completes before the WebSocket connects
  useEffect(() => {
    if (!isLiveDebate) return;
    if (liveStatus !== 'complete') return;
    if (liveMessages.length > 0) return; // Already have messages
    if (fallbackDebate) return; // Already fetched

    const fetchCompletedDebate = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/debates/${debateId}`);
        if (response.ok) {
          const data = await response.json();
          if (data.final_answer || data.conclusion) {
            setFallbackDebate({
              task: data.task || 'Debate completed',
              agents: data.agents || [],
              finalAnswer: data.final_answer || data.conclusion || '',
            });
          }
        }
      } catch (err) {
        logger.debug('Failed to fetch completed debate:', err);
      }
    };

    fetchCompletedDebate();
  }, [isLiveDebate, liveStatus, liveMessages.length, debateId, fallbackDebate]);

  // Use fallback data when available and no live messages
  const effectiveTask = liveTask || fallbackDebate?.task || '';
  const effectiveAgents = liveAgents.length > 0 ? liveAgents : (fallbackDebate?.agents || []);
  const effectiveMessages: TranscriptMessage[] = liveMessages.length > 0
    ? liveMessages
    : (fallbackDebate?.finalAnswer ? [{
        agent: 'consensus',
        role: 'synthesis',  // Mark as synthesis for special styling
        content: fallbackDebate.finalAnswer,
        timestamp: Date.now() / 1000,
      }] : []);

  // Auto-show citations when they arrive
  useEffect(() => {
    if (hasCitations) {
      setShowCitations(true);
    }
  }, [hasCitations]);

  // Fetch cruxes when debate has messages (for highlighting)
  useEffect(() => {
    // Only fetch for non-adhoc debates that have at least 3 messages
    // Ad-hoc debates don't have belief network traces, so skip to avoid 404s
    const isAdhocDebate = debateId.startsWith('adhoc_');
    if (liveMessages.length < 3 || cruxes.length > 0 || !isLiveDebate || isAdhocDebate) return;

    const fetchCruxes = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/belief-network/${debateId}/cruxes?top_k=5`);
        if (response.ok) {
          const data = await response.json();
          if (data.cruxes && data.cruxes.length > 0) {
            setCruxes(data.cruxes.map((c: { claim_id?: string; statement?: string; author?: string; crux_score?: number }) => ({
              claim_id: c.claim_id || '',
              statement: c.statement || '',
              author: c.author || '',
              crux_score: c.crux_score,
            })));
          }
        }
      } catch (err) {
        // Silently ignore - crux highlighting is optional
        logger.debug('Failed to fetch cruxes:', err);
      }
    };

    // Debounce the fetch - wait for debate to settle
    const timer = setTimeout(fetchCruxes, 2000);
    return () => clearTimeout(timer);
  }, [liveMessages.length, cruxes.length, debateId, isLiveDebate]);

  // Detect when user manually scrolls up
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setUserScrolled(!isNearBottom);
  }, []);

  // Resume auto-scroll handler
  const handleResumeAutoScroll = useCallback(() => {
    setUserScrolled(false);
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, []);

  // Smart auto-scroll: only scroll if user hasn't manually scrolled up
  useEffect(() => {
    if (!userScrolled && scrollContainerRef.current && liveStatus === 'streaming') {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [liveMessages, streamingMessages, userScrolled, liveStatus]);

  // Auto-scroll to synthesis when debate completes
  useEffect(() => {
    if (liveStatus === 'complete' && scrollContainerRef.current) {
      // Find synthesis message element and scroll to it
      const synthesisEl = document.getElementById('synthesis-message');
      if (synthesisEl) {
        // Smooth scroll to synthesis with some offset from top
        setTimeout(() => {
          synthesisEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 300); // Small delay to let DOM settle
      } else {
        // If no synthesis element, scroll to bottom
        scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
      }
    }
  }, [liveStatus]);

  // Fetch completed debate from Supabase (non-live debates)
  useEffect(() => {
    if (isLiveDebate) {
      setLoading(false);
      return;
    }

    const loadDebate = async () => {
      try {
        const data = await fetchDebateById(debateId);
        if (data) {
          setDebate(data);
        } else {
          setError('Debate not found');
        }
      } catch {
        setError('Failed to load debate');
      } finally {
        setLoading(false);
      }
    };

    loadDebate();
  }, [debateId, isLiveDebate]);

  const handleShare = async () => {
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      logger.error('Failed to copy link:', err);
    }
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
                [BACK TO LIVE]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          {loading && (
            <div className="flex items-center justify-center py-20">
              <div className="text-acid-green font-mono animate-pulse">
                {'>'} LOADING DEBATE...
              </div>
            </div>
          )}

          {error && !isLiveDebate && (
            <div className="bg-warning/10 border border-warning/30 rounded-lg p-6 text-center">
              <div className="text-warning text-2xl mb-2">{'>'} ERROR</div>
              <div className="text-text-muted">{error}</div>
              <Link
                href="/"
                className="inline-block mt-4 text-acid-green hover:underline font-mono"
              >
                [RETURN HOME]
              </Link>
            </div>
          )}

          {/* Live Debate View */}
          {isLiveDebate && (
            <LiveDebateView
              debateId={debateId}
              status={liveStatus}
              task={effectiveTask}
              agents={effectiveAgents}
              messages={effectiveMessages}
              streamingMessages={streamingMessages}
              streamEvents={streamEvents}
              hasCitations={hasCitations}
              showCitations={showCitations}
              setShowCitations={setShowCitations}
              showParticipation={showParticipation}
              setShowParticipation={setShowParticipation}
              onShare={handleShare}
              copied={copied}
              onVote={sendVote}
              onSuggest={sendSuggestion}
              onAck={registerAckCallback}
              onError={registerErrorCallback}
              scrollContainerRef={scrollContainerRef}
              onScroll={handleScroll}
              userScrolled={userScrolled}
              onResumeAutoScroll={handleResumeAutoScroll}
              cruxes={cruxes}
              showCruxHighlighting={showCruxHighlighting}
              setShowCruxHighlighting={setShowCruxHighlighting}
              error={liveError}
              errorDetails={liveErrorDetails}
              onRetry={reconnect}
            />
          )}

          {/* Archived Debate View */}
          {debate && (
            <ArchivedDebateView
              debate={debate}
              onShare={handleShare}
              copied={copied}
            />
          )}
        </div>

        {/* Footer - Hide during active streaming to maximize content area */}
        {(!isLiveDebate || liveStatus === 'complete' || liveStatus === 'error') && (
          <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
            <div className="text-acid-green/50 mb-2">
              {'═'.repeat(40)}
            </div>
            <button
              onClick={handleShare}
              className="text-text-muted hover:text-acid-green transition-colors cursor-pointer"
              title="Click to copy permalink"
            >
              {'>'} PERMALINK: {debateId} {copied ? '[COPIED!]' : '[CLICK TO COPY]'}
            </button>
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
            <div className="text-acid-green/50 mt-4">
              {'═'.repeat(40)}
            </div>
          </footer>
        )}
      </main>
    </>
  );
}

// Helper to extract actionable info from error messages
function parseErrorDetails(errorDetails: string | null | undefined): {
  problemType: string;
  suggestion: string;
  invalidValues?: string[];
} {
  if (!errorDetails) {
    return {
      problemType: 'unknown',
      suggestion: 'Please try starting a new debate from the home page.',
    };
  }

  // Check for invalid agent type error
  const agentMatch = errorDetails.match(/Invalid agent type[s]?:\s*([^.]+)/i);
  if (agentMatch) {
    const invalidAgents = agentMatch[1].split(',').map(s => s.trim());
    return {
      problemType: 'invalid_agent',
      suggestion: 'The selected agent configuration is invalid. Please go back and select different agents.',
      invalidValues: invalidAgents,
    };
  }

  // Check for validation errors
  if (errorDetails.toLowerCase().includes('validation')) {
    return {
      problemType: 'validation',
      suggestion: 'There was a problem with the debate configuration. Please check your settings and try again.',
    };
  }

  // Check for timeout errors
  if (errorDetails.toLowerCase().includes('timeout') || errorDetails.toLowerCase().includes('time out')) {
    return {
      problemType: 'timeout',
      suggestion: 'The debate took too long to start. The server may be overloaded. Please try again.',
    };
  }

  return {
    problemType: 'unknown',
    suggestion: 'An unexpected error occurred. Please try starting a new debate.',
  };
}

// Error view component for failed debates
interface DebateErrorViewProps {
  debateId: string;
  error: string;
  errorDetails?: string | null;
  onRetry?: () => void;
}

function DebateErrorView({ debateId, error, errorDetails, onRetry }: DebateErrorViewProps) {
  const { problemType, suggestion, invalidValues } = parseErrorDetails(errorDetails);

  return (
    <div className="space-y-6">
      {/* Error Header */}
      <div className="bg-surface border-2 border-red-500/50 p-6">
        <div className="flex items-start gap-4">
          <div className="text-red-400 text-3xl">{'⚠'}</div>
          <div className="flex-1">
            <div className="flex items-center gap-2 text-xs font-mono mb-2">
              <span className="w-2 h-2 rounded-full bg-red-400" />
              <span className="text-red-400 uppercase">DEBATE FAILED</span>
            </div>
            <h1 className="text-lg font-mono text-red-400 mb-2">
              {error}
            </h1>
            <div className="text-xs text-text-muted font-mono">
              ID: {debateId}
            </div>
          </div>
        </div>
      </div>

      {/* Error Details */}
      <div className="bg-surface border border-acid-green/30 p-6">
        <div className="text-xs font-mono text-acid-green uppercase mb-4">
          {'>'} ERROR DETAILS
        </div>

        {errorDetails && (
          <div className="bg-bg/50 border border-red-500/30 p-4 mb-4 font-mono text-sm text-red-400">
            <code>{errorDetails}</code>
          </div>
        )}

        {problemType === 'invalid_agent' && invalidValues && (
          <div className="mb-4">
            <div className="text-xs text-text-muted mb-2">Invalid agent type(s):</div>
            <div className="flex flex-wrap gap-2">
              {invalidValues.map((agent) => (
                <span
                  key={agent}
                  className="px-2 py-1 text-xs font-mono bg-red-500/20 text-red-400 border border-red-500/40"
                >
                  {agent}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="text-text-muted text-sm">
          {suggestion}
        </div>
      </div>

      {/* Actions */}
      <div className="bg-surface border border-acid-green/30 p-6">
        <div className="text-xs font-mono text-acid-green uppercase mb-4">
          {'>'} WHAT TO DO
        </div>

        <div className="space-y-3">
          {onRetry && (
            <button
              onClick={onRetry}
              className="w-full py-3 px-4 bg-acid-cyan/20 text-acid-cyan border border-acid-cyan/40 hover:bg-acid-cyan/30 transition-colors font-mono text-sm text-left flex items-center gap-3"
            >
              <span className="text-lg">{'↻'}</span>
              <div>
                <div className="font-bold">[RETRY CONNECTION]</div>
                <div className="text-xs text-text-muted">Attempt to reconnect to the debate</div>
              </div>
            </button>
          )}

          <Link
            href="/"
            className="w-full py-3 px-4 bg-acid-green/20 text-acid-green border border-acid-green/40 hover:bg-acid-green/30 transition-colors font-mono text-sm text-left flex items-center gap-3 block"
          >
            <span className="text-lg">{'←'}</span>
            <div>
              <div className="font-bold">[START NEW DEBATE]</div>
              <div className="text-xs text-text-muted">Return to home and configure a new debate</div>
            </div>
          </Link>

          {problemType === 'invalid_agent' && (
            <div className="mt-4 p-4 bg-acid-yellow/10 border border-acid-yellow/30 text-sm">
              <div className="text-acid-yellow font-mono text-xs uppercase mb-2">{'>'} TIP</div>
              <p className="text-text-muted">
                If you&apos;re seeing this error repeatedly, try clearing your browser cache
                (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows) to ensure you have the latest
                agent configuration.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Sub-component for live debate view
interface LiveDebateViewProps {
  debateId: string;
  status: 'connecting' | 'streaming' | 'complete' | 'error';
  task: string;
  agents: string[];
  messages: TranscriptMessage[];
  streamingMessages: Map<string, { agent: string; taskId?: string; content: string; startTime: number }>;
  streamEvents: import('@/types/events').StreamEvent[];
  hasCitations: boolean;
  showCitations: boolean;
  setShowCitations: (show: boolean) => void;
  showParticipation: boolean;
  setShowParticipation: (show: boolean) => void;
  onShare: () => void;
  copied: boolean;
  onVote: (choice: string, intensity?: number) => void;
  onSuggest: (suggestion: string) => void;
  onAck: (callback: (msgType: string) => void) => () => void;
  onError: (callback: (message: string) => void) => () => void;
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  userScrolled: boolean;
  onResumeAutoScroll: () => void;
  // Crux highlighting props
  cruxes?: CruxClaim[];
  showCruxHighlighting?: boolean;
  setShowCruxHighlighting?: (show: boolean) => void;
  // Error handling props
  error?: string | null;
  errorDetails?: string | null;
  onRetry?: () => void;
}

function LiveDebateView({
  debateId,
  status,
  task,
  agents,
  messages,
  streamingMessages,
  streamEvents,
  hasCitations,
  showCitations,
  setShowCitations,
  showParticipation,
  setShowParticipation,
  onShare,
  copied,
  onVote,
  onSuggest,
  onAck,
  onError,
  scrollContainerRef,
  onScroll,
  userScrolled,
  onResumeAutoScroll,
  cruxes,
  showCruxHighlighting,
  setShowCruxHighlighting,
  error,
  errorDetails,
  onRetry,
}: LiveDebateViewProps) {
  // Show error view if debate failed
  if (status === 'error' && (error || errorDetails)) {
    return (
      <DebateErrorView
        debateId={debateId}
        error={error || 'Debate failed'}
        errorDetails={errorDetails}
        onRetry={onRetry}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Live Debate Header */}
      <div className="bg-surface border border-acid-green/30 p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono mb-2">
              <span className={`w-2 h-2 rounded-full ${
                status === 'streaming' ? 'bg-green-400 animate-pulse' :
                status === 'complete' ? 'bg-blue-400' :
                status === 'error' ? 'bg-red-400' : 'bg-yellow-400'
              }`} />
              <span className="text-text-muted uppercase">
                {status === 'streaming' ? 'LIVE DEBATE' :
                 status === 'complete' ? 'DEBATE COMPLETE' :
                 status === 'error' ? 'CONNECTION ERROR' : 'CONNECTING...'}
              </span>
            </div>
            <h1 className="text-lg font-mono text-acid-green mb-4">
              {task || (status === 'complete' ? 'Debate completed' : status === 'error' ? 'Debate failed' : 'Waiting for debate to start...')}
            </h1>
            <div className="flex flex-wrap gap-2">
              {agents.map((agent) => {
                const colors = getAgentColors(agent);
                return (
                  <span
                    key={agent}
                    className={`px-2 py-1 text-xs font-mono ${colors.bg} ${colors.text} ${colors.border} border`}
                  >
                    {agent}
                  </span>
                );
              })}
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            <button
              onClick={onShare}
              className="px-3 py-1 text-xs font-mono bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
            >
              {copied ? '[COPIED!]' : '[SHARE LINK]'}
            </button>
            <div className="text-xs text-text-muted font-mono">
              ID: {debateId}
            </div>
          </div>
        </div>
      </div>

      {/* Live Transcript + User Participation Grid */}
      <div className={`grid gap-4 ${showParticipation ? 'lg:grid-cols-3' : 'grid-cols-1'}`}>
        {/* Live Transcript */}
        <div className={`bg-surface border border-acid-green/30 ${showParticipation ? 'lg:col-span-2' : ''}`}>
          <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50 flex items-center justify-between">
            <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
              {'>'} LIVE TRANSCRIPT
            </span>
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-text-muted">
                {messages.length} messages
                {streamingMessages.size > 0 && (
                  <span className="ml-2 text-acid-cyan animate-pulse">
                    ({streamingMessages.size} streaming)
                  </span>
                )}
              </span>
              {cruxes && cruxes.length > 0 && setShowCruxHighlighting && (
                <button
                  onClick={() => setShowCruxHighlighting(!showCruxHighlighting)}
                  className={`px-2 py-1 text-xs font-mono border transition-colors ${
                    showCruxHighlighting
                      ? 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40'
                      : 'bg-surface text-text-muted border-border hover:border-acid-yellow/40'
                  }`}
                  title={`${cruxes.length} crux claim${cruxes.length !== 1 ? 's' : ''} detected`}
                >
                  {showCruxHighlighting ? `[HIDE CRUXES: ${cruxes.length}]` : `[SHOW CRUXES: ${cruxes.length}]`}
                </button>
              )}
              <button
                onClick={() => setShowParticipation(!showParticipation)}
                className={`px-2 py-1 text-xs font-mono border transition-colors ${
                  showParticipation
                    ? 'bg-accent/20 text-accent border-accent/40'
                    : 'bg-surface text-text-muted border-border hover:border-accent/40'
                }`}
              >
                {showParticipation ? '[HIDE VOTE]' : '[JOIN]'}
              </button>
            </div>
          </div>
          <div
            ref={scrollContainerRef}
            onScroll={onScroll}
            className="p-4 space-y-4 min-h-[400px]"
          >
            {messages.length === 0 && streamingMessages.size === 0 && status === 'streaming' && (
              <div className="text-center py-8 text-text-muted font-mono">
                <div className="animate-pulse">Waiting for agents to respond...</div>
              </div>
            )}
            {messages.map((msg, idx) => (
              <TranscriptMessageCard
                key={`${msg.agent}-${msg.timestamp}-${idx}`}
                message={msg}
                cruxes={showCruxHighlighting ? cruxes : undefined}
              />
            ))}
            {/* Streaming messages - sorted by agent name for stable visual ordering */}
            {Array.from(streamingMessages.values())
              .sort((a, b) => a.agent.localeCompare(b.agent))
              .map((streamMsg) => (
              <StreamingMessageCard
                key={`streaming-${streamMsg.agent}-${streamMsg.taskId || 'default'}`}
                message={streamMsg}
              />
            ))}
          </div>
        </div>

        {/* User Participation Panel */}
        {showParticipation && status === 'streaming' && (
          <div className="lg:col-span-1">
            <UserParticipation
              events={streamEvents}
              onVote={onVote}
              onSuggest={onSuggest}
              onAck={onAck}
              onError={onError}
            />
          </div>
        )}
      </div>

      {/* Citations Panel */}
      {hasCitations && (
        <div className="bg-surface border border-accent/30">
          <div className="px-4 py-3 border-b border-accent/20 bg-bg/50 flex items-center justify-between">
            <span className="text-xs font-mono text-accent uppercase tracking-wider">
              {'>'} EVIDENCE & CITATIONS
            </span>
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="px-2 py-1 text-xs font-mono border transition-colors bg-surface text-text-muted border-border hover:border-accent/40"
            >
              {showCitations ? '[HIDE]' : '[SHOW]'}
            </button>
          </div>
          {showCitations && (
            <div className="p-4">
              <CitationsPanel events={streamEvents} />
            </div>
          )}
        </div>
      )}

      {/* Live Analysis Panels - Show during streaming */}
      {status === 'streaming' && streamEvents.length > 0 && (
        <LiveAnalysisPanels events={streamEvents} agents={agents} debateId={debateId} />
      )}

      {/* Footer - show when complete */}
      {status === 'complete' && (
        <div className="text-center text-xs font-mono text-text-muted py-2 border-t border-acid-green/20">
          ID: {debateId}
        </div>
      )}

      {/* Resume auto-scroll button - appears when user scrolls up during streaming */}
      {userScrolled && status === 'streaming' && (
        <button
          onClick={onResumeAutoScroll}
          className="fixed bottom-4 right-4 px-3 py-2 bg-acid-green text-bg font-mono text-xs z-50
                     hover:bg-acid-green/80 transition-colors shadow-lg border border-acid-green/50"
        >
          {'>'} RESUME AUTO-SCROLL
        </button>
      )}
    </div>
  );
}

// Sub-component for archived debate view
interface ArchivedDebateViewProps {
  debate: DebateArtifact;
  onShare: () => void;
  copied: boolean;
}

function ArchivedDebateView({ debate, onShare, copied }: ArchivedDebateViewProps) {
  return (
    <div className="space-y-6">
      {/* Debate Header */}
      <div className="bg-surface border border-acid-green/30 p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-xs text-text-muted font-mono mb-2">
              DEBATE {'// '}CYCLE {debate.cycle_number} {'// '}{debate.phase.toUpperCase()}
            </div>
            <h1 className="text-lg font-mono text-acid-green mb-4">
              {debate.task}
            </h1>
            <div className="flex flex-wrap gap-2">
              {debate.agents.map((agent) => {
                const colors = getAgentColors(agent);
                return (
                  <span
                    key={agent}
                    className={`px-2 py-1 text-xs font-mono ${colors.bg} ${colors.text} ${colors.border} border`}
                  >
                    {agent}
                  </span>
                );
              })}
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            <button
              onClick={onShare}
              className="px-3 py-1 text-xs font-mono bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
            >
              {copied ? '[COPIED!]' : '[SHARE LINK]'}
            </button>
            <div className="text-xs text-text-muted font-mono">
              {new Date(debate.created_at).toLocaleString()}
            </div>
          </div>
        </div>

        {/* Consensus Status */}
        <div className="mt-4 pt-4 border-t border-acid-green/20 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                debate.consensus_reached ? 'bg-green-400' : 'bg-yellow-400'
              }`}
            />
            <span className="text-xs font-mono text-text-muted">
              {debate.consensus_reached ? 'CONSENSUS REACHED' : 'NO CONSENSUS'}
            </span>
          </div>
          <div className="text-xs font-mono text-text-muted">
            CONFIDENCE: {Math.round(debate.confidence * 100)}%
          </div>
          {debate.vote_tally && Object.keys(debate.vote_tally).length > 0 && (
            <div className="text-xs font-mono text-text-muted">
              VOTES: {Object.entries(debate.vote_tally).map(([k, v]) => `${k}:${v}`).join(' ')}
            </div>
          )}
        </div>
      </div>

      {/* Winning Proposal */}
      {debate.winning_proposal && (
        <div className="bg-gradient-to-br from-accent/10 to-purple-500/10 border-2 border-accent/50 p-6">
          <div className="text-xs text-accent font-mono mb-2 uppercase tracking-wider">
            Winning Proposal
          </div>
          <div className="text-text whitespace-pre-wrap font-mono text-sm">
            {debate.winning_proposal}
          </div>
        </div>
      )}

      {/* Transcript */}
      <div className="bg-surface border border-acid-green/30">
        <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
          <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
            {'>'} DEBATE TRANSCRIPT
          </span>
        </div>
        <div className="p-4 space-y-4 min-h-[400px]">
          {(debate.transcript as unknown as TranscriptMessage[]).map((msg, idx) => (
            <TranscriptMessageCard key={idx} message={msg} />
          ))}
        </div>
      </div>

      {/* Metadata */}
      <div className="text-center text-xs font-mono text-text-muted py-4 border-t border-acid-green/20">
        <div>DEBATE ID: {debate.id}</div>
        <div>LOOP: {debate.loop_id}</div>
      </div>
    </div>
  );
}

// Crux highlighting helper functions
function findCruxMatches(
  content: string,
  cruxes: CruxClaim[]
): Array<{ start: number; end: number; crux: CruxClaim }> {
  const matches: Array<{ start: number; end: number; crux: CruxClaim }> = [];
  const contentLower = content.toLowerCase();

  for (const crux of cruxes) {
    const statement = crux.statement;
    const minLen = Math.min(30, statement.length);
    const statementLower = statement.toLowerCase();

    // Try exact match first
    let idx = contentLower.indexOf(statementLower);
    if (idx !== -1) {
      matches.push({ start: idx, end: idx + statement.length, crux });
      continue;
    }

    // Try prefix match for truncated statements
    const prefix = statementLower.slice(0, minLen);
    idx = contentLower.indexOf(prefix);
    if (idx !== -1 && minLen >= 30) {
      let endIdx = idx + minLen;
      const remaining = contentLower.slice(endIdx);
      const sentenceEnd = remaining.search(/[.!?\n]/);
      if (sentenceEnd !== -1 && sentenceEnd < 200) {
        endIdx += sentenceEnd + 1;
      } else {
        endIdx = Math.min(idx + 200, content.length);
      }
      matches.push({ start: idx, end: endIdx, crux });
    }
  }

  // Sort and remove overlaps
  matches.sort((a, b) => a.start - b.start);
  const filtered: Array<{ start: number; end: number; crux: CruxClaim }> = [];
  for (const m of matches) {
    if (filtered.length === 0 || m.start >= filtered[filtered.length - 1].end) {
      filtered.push(m);
    }
  }
  return filtered;
}

function HighlightedContent({ content, cruxes }: { content: string; cruxes?: CruxClaim[] }) {
  const parts = useMemo(() => {
    if (!cruxes || cruxes.length === 0) {
      return [{ text: content, isHighlight: false, crux: undefined }];
    }

    const matches = findCruxMatches(content, cruxes);
    if (matches.length === 0) {
      return [{ text: content, isHighlight: false, crux: undefined }];
    }

    const result: Array<{ text: string; isHighlight: boolean; crux?: CruxClaim }> = [];
    let lastEnd = 0;

    for (const match of matches) {
      if (match.start > lastEnd) {
        result.push({ text: content.slice(lastEnd, match.start), isHighlight: false });
      }
      result.push({ text: content.slice(match.start, match.end), isHighlight: true, crux: match.crux });
      lastEnd = match.end;
    }

    if (lastEnd < content.length) {
      result.push({ text: content.slice(lastEnd), isHighlight: false });
    }

    return result;
  }, [content, cruxes]);

  return (
    <>
      {parts.map((part, i) =>
        part.isHighlight ? (
          <span
            key={i}
            className="bg-acid-yellow/20 border-b-2 border-acid-yellow text-acid-yellow relative group cursor-help"
            title={`Crux: ${part.crux?.statement?.slice(0, 100)}${(part.crux?.statement?.length || 0) > 100 ? '...' : ''}`}
          >
            {part.text}
            <span className="absolute -top-1 -right-1 text-[8px] bg-acid-yellow text-bg-dark px-0.5 rounded font-mono opacity-0 group-hover:opacity-100 transition-opacity">
              CRUX
            </span>
          </span>
        ) : (
          <span key={i}>{part.text}</span>
        )
      )}
    </>
  );
}

// Reusable transcript message component
function TranscriptMessageCard({ message, cruxes, isSynthesisMessage }: { message: TranscriptMessage; cruxes?: CruxClaim[]; isSynthesisMessage?: boolean }) {
  const colors = getAgentColors(message.agent || 'system');
  // Detect synthesis messages by role or agent name
  const isSynthesis =
    message.role === 'synthesis' ||
    message.agent === 'synthesis-agent' ||
    message.agent === 'consensus' ||
    isSynthesisMessage;

  // Special rendering for synthesis messages - highly visible final conclusion
  if (isSynthesis) {
    return (
      <div className="relative my-6" id="synthesis-message">
        {/* Glowing border effect */}
        <div className="absolute inset-0 bg-acid-green/10 blur-xl rounded-lg" />
        {/* Synthesis header bar */}
        <div className="relative bg-acid-green/20 border-l-4 border-acid-green px-4 py-3 flex items-center justify-between rounded-t-lg">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{'🎯'}</span>
            <span className="text-acid-green font-bold text-base tracking-wider">
              FINAL SYNTHESIS
            </span>
          </div>
          {message.timestamp && (
            <span className="text-[10px] text-acid-green/70 font-mono">
              {new Date(message.timestamp * 1000).toLocaleTimeString()}
            </span>
          )}
        </div>
        {/* Synthesis content */}
        <div className="relative bg-bg-secondary/90 border-2 border-acid-green/40 border-t-0 p-6 rounded-b-lg">
          <div className="text-sm text-text-primary font-medium leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
          <div className="mt-4 pt-4 border-t border-acid-green/20 flex items-center justify-between">
            <span className="text-xs text-acid-green/70 font-mono">
              Generated by Claude Opus 4.5
            </span>
            <span className="text-xs text-acid-green/50 font-mono">
              DEBATE CONCLUSION
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Standard rendering for non-synthesis messages
  return (
    <div className={`${colors.bg} border ${colors.border} p-4`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`font-mono font-bold text-sm ${colors.text}`}>
            {(message.agent || 'SYSTEM').toUpperCase()}
          </span>
          {message.role && (
            <span className="text-xs text-text-muted border border-text-muted/30 px-1">
              {message.role}
            </span>
          )}
          {message.round !== undefined && message.round > 0 && (
            <span className="text-xs text-text-muted">
              R{message.round}
            </span>
          )}
        </div>
        {message.timestamp && (
          <span className="text-[10px] text-text-muted font-mono">
            {new Date(message.timestamp * 1000).toLocaleTimeString()}
          </span>
        )}
      </div>
      <div className="text-sm text-text whitespace-pre-wrap">
        <HighlightedContent content={message.content} cruxes={cruxes} />
      </div>
    </div>
  );
}

// Streaming message component with animation
function StreamingMessageCard({ message }: { message: { agent: string; content: string; startTime: number } }) {
  const colors = getAgentColors(message.agent);
  return (
    <div className={`${colors.bg} border-2 ${colors.border} p-4 animate-pulse`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`font-mono font-bold text-sm ${colors.text}`}>
            {message.agent.toUpperCase()}
          </span>
          <span className="text-xs text-acid-cyan border border-acid-cyan/30 px-1 animate-pulse">
            STREAMING
          </span>
        </div>
        <span className="text-[10px] text-text-muted font-mono">
          {Math.round((Date.now() - message.startTime) / 1000)}s
        </span>
      </div>
      <div className="text-sm text-text whitespace-pre-wrap">
        {message.content}
        <span className="inline-block w-2 h-4 bg-acid-cyan ml-1 animate-pulse">▌</span>
      </div>
    </div>
  );
}

// Live Analysis Panels - collapsible section for mood, uncertainty, and token tracking
interface LiveAnalysisPanelsProps {
  events: import('@/types/events').StreamEvent[];
  agents: string[];
  debateId: string;
}

function LiveAnalysisPanels({ events, agents, debateId }: LiveAnalysisPanelsProps) {
  const [showAnalysis, setShowAnalysis] = useState(false);

  return (
    <div className="bg-surface border border-purple-500/30">
      <button
        onClick={() => setShowAnalysis(!showAnalysis)}
        className="w-full px-4 py-3 border-b border-purple-500/20 bg-bg/50 flex items-center justify-between hover:bg-purple-500/5 transition-colors"
      >
        <span className="text-xs font-mono text-purple-400 uppercase tracking-wider">
          {'>'} LIVE ANALYSIS
        </span>
        <span className="text-xs font-mono text-text-muted">
          {showAnalysis ? '[-]' : '[+]'}
        </span>
      </button>

      {showAnalysis && (
        <div className="p-4 space-y-4">
          {/* Grid layout for analysis panels */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Mood Tracker */}
            <div className="bg-bg/50 border border-purple-500/20 p-3">
              <div className="text-xs font-mono text-purple-400 mb-2 uppercase">
                Agent Mood
              </div>
              <MoodTrackerPanel events={events} agents={agents} />
            </div>

            {/* Uncertainty Analysis */}
            <div className="bg-bg/50 border border-purple-500/20 p-3">
              <div className="text-xs font-mono text-purple-400 mb-2 uppercase">
                Uncertainty & Cruxes
              </div>
              <UncertaintyPanel events={events} debateId={debateId} />
            </div>

            {/* Token Stream */}
            <div className="bg-bg/50 border border-purple-500/20 p-3">
              <div className="text-xs font-mono text-purple-400 mb-2 uppercase">
                Token Usage
              </div>
              <TokenStreamViewer events={events} agents={agents} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
