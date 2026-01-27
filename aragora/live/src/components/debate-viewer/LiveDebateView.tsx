'use client';

import { useState, useMemo } from 'react';
import { getAgentColors } from '@/utils/agentColors';
import { UserParticipation } from '@/components/UserParticipation';
import { CitationsPanel } from '@/components/CitationsPanel';
import { DebateExportModal } from '@/components/DebateExportModal';
import { TranscriptMessageCard } from './TranscriptMessageCard';
import { StreamingMessageCard } from './StreamingMessageCard';
import { ConsensusMeter } from './ConsensusMeter';
import { CritiqueSeverityMeter } from './CritiqueSeverityMeter';
import { TricksterAlertPanel } from '@/components/TricksterAlertPanel';
import { RhetoricalObservationsPanel } from './RhetoricalObservationsPanel';
import { UncertaintyPanel } from '@/components/UncertaintyPanel';
import { MoodTrackerPanel } from '@/components/MoodTrackerPanel';
import { TokenStreamViewer } from '@/components/TokenStreamViewer';
import { DebateInitializationProgress } from './DebateInitializationProgress';
import { AudioDownloadSection } from './AudioDownloadSection';
import { InlineDownloadPanel } from './InlineDownloadPanel';
import { PhaseIndicator } from './PhaseIndicator';
import { API_BASE_URL } from '@/config';
import type { LiveDebateViewProps } from './types';

const STATUS_CONFIG = {
  connecting: { color: 'bg-yellow-400', label: 'CONNECTING...' },
  streaming: { color: 'bg-green-400 animate-pulse', label: 'LIVE DEBATE' },
  complete: { color: 'bg-blue-400', label: 'DEBATE COMPLETE' },
  error: { color: 'bg-red-400', label: 'CONNECTION ERROR' },
} as const;

export function LiveDebateView({
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
}: LiveDebateViewProps) {
  const statusConfig = STATUS_CONFIG[status];
  const [showExportModal, setShowExportModal] = useState(false);

  const initErrors = useMemo(() => {
    const errors: Array<{ agent: string; message: string }> = [];
    for (const event of streamEvents) {
      if (event.type === 'error') {
        const data = event.data as Record<string, unknown>;
        const phase = data?.phase as string | undefined;
        if (phase === 'initialization' || phase === 'setup') {
          errors.push({
            agent: (data?.agent as string) || 'unknown',
            message:
              (data?.error as string) ||
              (data?.message as string) ||
              'Initialization failed',
          });
        }
      }
      if (event.type === 'agent_error') {
        const data = event.data as Record<string, unknown>;
        const errorType = data?.error_type as string | undefined;
        if (errorType === 'missing_env' || errorType === 'missing_env_fallback') {
          errors.push({
            agent: (event.agent as string) || (data?.agent as string) || 'unknown',
            message:
              (data?.message as string) ||
              (data?.error as string) ||
              'Missing credentials for agent',
          });
        }
      }
    }
    return errors;
  }, [streamEvents]);

  const runtimeErrors = useMemo(() => {
    const errors: Array<{ agent: string; message: string }> = [];
    for (const event of streamEvents) {
      if (event.type === 'agent_error') {
        const data = event.data as Record<string, unknown>;
        const errorType = data?.error_type as string | undefined;
        if (errorType && ['empty', 'timeout', 'exception', 'internal'].includes(errorType)) {
          errors.push({
            agent: (event.agent as string) || (data?.agent as string) || 'unknown',
            message:
              (data?.message as string) ||
              (data?.error as string) ||
              `Agent error: ${errorType}`,
          });
        }
      }
    }
    return errors;
  }, [streamEvents]);

  const consensusStatus = useMemo(() => {
    const consensusEvents = streamEvents.filter(event => event.type === 'consensus');
    if (consensusEvents.length === 0) return null;
    const lastEvent = consensusEvents[consensusEvents.length - 1];
    return lastEvent.data as {
      status?: string;
      agent_failures?: Record<string, Array<{ message?: string }>>;
    };
  }, [streamEvents]);

  const agentFailureAgents = useMemo(() => {
    const agents = new Set<string>();
    if (consensusStatus?.agent_failures) {
      Object.keys(consensusStatus.agent_failures).forEach((agent) => agents.add(agent));
    }
    for (const event of streamEvents) {
      if (event.type === 'agent_error') {
        agents.add(event.agent || 'unknown');
      }
    }
    return Array.from(agents);
  }, [consensusStatus, streamEvents]);

  // Calculate current phase/round from stream events or messages
  const currentPhase = useMemo(() => {
    // Try to get phase from phase_progress events
    const phaseEvents = streamEvents.filter(e => e.type === 'phase_progress');
    if (phaseEvents.length > 0) {
      const lastEvent = phaseEvents[phaseEvents.length - 1];
      const phase = (lastEvent.data as { phase?: number; round?: number })?.phase
                 ?? (lastEvent.data as { phase?: number; round?: number })?.round;
      if (typeof phase === 'number') return phase;
    }
    // Fallback: estimate from messages
    if (messages.length === 0) return 0;
    return Math.max(...messages.map(m => m.round ?? 0));
  }, [streamEvents, messages]);

  return (
    <div className="space-y-6">
      {/* Live Debate Header */}
      <div className="bg-surface border border-acid-green/30 p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            {/* Status indicator - small, inline with label */}
            <div className="flex items-center gap-2 text-xs font-mono mb-2">
              <span className={`w-2 h-2 rounded-full animate-pulse ${statusConfig.color}`} />
              <span className="text-text-muted uppercase tracking-wider">{statusConfig.label}</span>
              {status === 'streaming' && (
                <span className="text-acid-cyan text-xs animate-pulse ml-2">
                  In progress...
                </span>
              )}
            </div>
            {/* Task/Question - always visible and prominent */}
            <h1 className="text-lg font-mono text-acid-green mb-4">
              {task || 'Waiting for debate topic...'}
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
            {initErrors.length > 0 && (
              <div className="mt-4 border border-red-500/30 bg-red-500/5 px-3 py-2 text-xs font-mono text-red-300">
                Missing agents:{' '}
                {initErrors.map((err) => err.agent).join(', ')}. Check API keys or Secrets
                Manager.
              </div>
            )}
            {consensusStatus?.status === 'insufficient_participation' && (
              <div className="mt-4 border border-yellow-500/30 bg-yellow-500/5 px-3 py-2 text-xs font-mono text-yellow-200">
                Insufficient participation: {agentFailureAgents.length} agent
                {agentFailureAgents.length === 1 ? '' : 's'} failed or timed out.
              </div>
            )}
            {runtimeErrors.length > 0 && consensusStatus?.status !== 'insufficient_participation' && (
              <div className="mt-4 border border-yellow-500/30 bg-yellow-500/5 px-3 py-2 text-xs font-mono text-yellow-200">
                Agent errors detected: {runtimeErrors.map((err) => err.agent).join(', ')}.
              </div>
            )}
          </div>

          <div className="flex flex-col items-end gap-2">
            <button
              onClick={onShare}
              className="px-3 py-1 text-xs font-mono bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
            >
              {copied ? '[COPIED!]' : '[SHARE LINK]'}
            </button>
            <div className="text-xs text-text-muted font-mono">ID: {debateId}</div>
          </div>
        </div>
      </div>

      {/* Phase Progress Indicator - visible during streaming */}
      {status === 'streaming' && (
        <div className="bg-surface border border-acid-green/30 p-4">
          <PhaseIndicator
            currentRound={currentPhase}
            totalRounds={9}
            isComplete={false}
            showProgress={true}
          />
        </div>
      )}

      {/* Analytics Meters - visible during streaming */}
      {status === 'streaming' && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <ConsensusMeter events={streamEvents} agents={agents} />
          <CritiqueSeverityMeter events={streamEvents} agents={agents} />
          <MoodTrackerPanel events={streamEvents} agents={agents} />
          <TokenStreamViewer events={streamEvents} agents={agents} />
        </div>
      )}

      {/* Trickster Alerts - visible when hollow consensus detected */}
      <TricksterAlertPanel events={streamEvents} />

      {/* Rhetorical Observations - collapsible analysis */}
      <RhetoricalObservationsPanel events={streamEvents} />

      {/* Uncertainty Analysis - shows after voting completes */}
      <UncertaintyPanel events={streamEvents} />

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
                  <span className="ml-2 text-acid-cyan animate-pulse">({streamingMessages.size} streaming)</span>
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
            ref={scrollContainerRef as React.RefObject<HTMLDivElement>}
            onScroll={onScroll}
            className="p-4 space-y-4 min-h-[400px]"
          >
            {/* Show initialization progress OR classification summary during early streaming */}
            {status === 'streaming' && messages.length === 0 && (
              <DebateInitializationProgress
                task={task}
                agents={agents}
                streamEvents={streamEvents}
              />
            )}
            {messages.map((msg, idx) => (
              <TranscriptMessageCard
                key={`${msg.agent}-${msg.timestamp}-${idx}`}
                message={msg}
                cruxes={showCruxHighlighting ? cruxes : undefined}
              />
            ))}
            {Array.from(streamingMessages.values())
              .sort((a, b) => a.agent.localeCompare(b.agent))
              .map((streamMsg) => (
              <StreamingMessageCard
                key={`streaming-${streamMsg.agent}-${streamMsg.taskId || 'default'}`}
                message={streamMsg}
              />
            ))}
            {/* Download panel - appears at bottom of transcript when debate is complete */}
            {status === 'complete' && (
              <InlineDownloadPanel debateId={debateId} />
            )}
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

      {/* Export & Share Panel - show when complete */}
      {status === 'complete' && (
        <div className="bg-surface border border-acid-green/30">
          <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
            <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
              {'>'} DOWNLOAD & SHARE
            </span>
          </div>
          <div className="p-4 space-y-4">
            {/* Transcript Downloads */}
            <div>
              <div className="text-xs font-mono text-text-muted mb-2 uppercase">Download Transcript</div>
              <div className="flex flex-wrap gap-2">
                <a
                  href={`${API_BASE_URL}/api/debates/${debateId}/export/txt`}
                  download
                  className="px-3 py-2 text-xs font-mono bg-bg border border-acid-green/40 text-acid-green hover:bg-acid-green/10 transition-colors"
                >
                  [TXT]
                </a>
                <a
                  href={`${API_BASE_URL}/api/debates/${debateId}/export/md`}
                  download
                  className="px-3 py-2 text-xs font-mono bg-bg border border-acid-green/40 text-acid-green hover:bg-acid-green/10 transition-colors"
                >
                  [MARKDOWN]
                </a>
                <a
                  href={`${API_BASE_URL}/api/debates/${debateId}/export/json`}
                  download
                  className="px-3 py-2 text-xs font-mono bg-bg border border-border text-text-muted hover:border-acid-green/40 transition-colors"
                >
                  [JSON]
                </a>
                <a
                  href={`${API_BASE_URL}/api/debates/${debateId}/export/html`}
                  download
                  className="px-3 py-2 text-xs font-mono bg-bg border border-border text-text-muted hover:border-acid-green/40 transition-colors"
                >
                  [HTML]
                </a>
                <a
                  href={`${API_BASE_URL}/api/debates/${debateId}/export/csv?table=messages`}
                  download
                  className="px-3 py-2 text-xs font-mono bg-bg border border-border text-text-muted hover:border-acid-green/40 transition-colors"
                >
                  [CSV]
                </a>
              </div>
            </div>

            {/* Audio Generation */}
            <div>
              <div className="text-xs font-mono text-text-muted mb-2 uppercase">Audio</div>
              <AudioDownloadSection debateId={debateId} />
            </div>

            {/* Advanced Options */}
            <div>
              <button
                onClick={() => setShowExportModal(true)}
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors cursor-pointer"
              >
                [MORE EXPORT OPTIONS...]
              </button>
            </div>

            {/* Permalink */}
            <div className="pt-2 border-t border-acid-green/20">
              <button
                onClick={onShare}
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors cursor-pointer"
                title="Click to copy permalink"
              >
                {'>'} PERMALINK: {debateId} {copied ? '[COPIED!]' : '[CLICK TO COPY]'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Modal for advanced options */}
      <DebateExportModal
        debateId={debateId}
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
      />

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
