'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { fetchDebateById, type DebateArtifact } from '@/utils/supabase';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';
import { UserParticipation } from '@/components/UserParticipation';
import { CitationsPanel } from '@/components/CitationsPanel';
import { getAgentColors } from '@/utils/agentColors';
import { useDebateWebSocket, type TranscriptMessage } from '@/hooks/useDebateWebSocket';

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
    sendVote,
    sendSuggestion,
    registerAckCallback,
    registerErrorCallback,
  } = useDebateWebSocket({
    debateId,
    wsUrl,
    enabled: isLiveDebate,
  });

  // Auto-show citations when they arrive
  useEffect(() => {
    if (hasCitations) {
      setShowCitations(true);
    }
  }, [hasCitations]);

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
      console.error('Failed to copy link:', err);
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
              task={liveTask}
              agents={liveAgents}
              messages={liveMessages}
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
            <p className="text-text-muted">
              {'>'} AGORA DEBATE VIEWER // PERMALINK
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
            <div className="text-acid-green/50 mt-4">
              {'═'.repeat(40)}
            </div>
          </footer>
        )}
      </main>
    </>
  );
}

// Sub-component for live debate view
interface LiveDebateViewProps {
  debateId: string;
  status: 'connecting' | 'streaming' | 'complete' | 'error';
  task: string;
  agents: string[];
  messages: TranscriptMessage[];
  streamingMessages: Map<string, { agent: string; content: string; startTime: number }>;
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
  scrollContainerRef: React.RefObject<HTMLDivElement>;
  onScroll: () => void;
  userScrolled: boolean;
  onResumeAutoScroll: () => void;
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
}: LiveDebateViewProps) {
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
              {task || 'Waiting for debate to start...'}
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
            className={`p-4 space-y-4 overflow-y-auto ${
              status === 'streaming'
                ? 'h-[calc(100vh-200px)]'  // Fixed height during streaming for better scroll control
                : 'max-h-[calc(100vh-280px)]'
            }`}
          >
            {messages.length === 0 && streamingMessages.size === 0 && status === 'streaming' && (
              <div className="text-center py-8 text-text-muted font-mono">
                <div className="animate-pulse">Waiting for agents to respond...</div>
              </div>
            )}
            {messages.map((msg, idx) => (
              <TranscriptMessageCard key={`${msg.agent}-${msg.timestamp}-${idx}`} message={msg} />
            ))}
            {/* Streaming messages */}
            {Array.from(streamingMessages.values()).map((streamMsg) => (
              <StreamingMessageCard key={`streaming-${streamMsg.agent}`} message={streamMsg} />
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
              DEBATE // CYCLE {debate.cycle_number} // {debate.phase.toUpperCase()}
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
        <div className="p-4 space-y-4 max-h-[600px] overflow-y-auto">
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

// Reusable transcript message component
function TranscriptMessageCard({ message }: { message: TranscriptMessage }) {
  const colors = getAgentColors(message.agent || 'system');
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
        {message.content}
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
