'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { fetchDebateById } from '@/utils/supabase';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useDebateWebSocket } from '@/hooks/useDebateWebSocket';
import { LiveDebateView } from './LiveDebateView';
import { ArchivedDebateView } from './ArchivedDebateView';
import type { DebateViewerProps, DebateArtifact, StreamingMessage } from './types';
import { logger } from '@/utils/logger';

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.aragora.ai/ws';

export function DebateViewer({ debateId, wsUrl = DEFAULT_WS_URL }: DebateViewerProps) {
  const [debate, setDebate] = useState<DebateArtifact | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showParticipation, setShowParticipation] = useState(true);
  const [showCitations, setShowCitations] = useState(false);
  const [userScrolled, setUserScrolled] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const isLiveDebate = debateId.startsWith('adhoc_');

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

  // Scroll to synthesis message when debate completes
  useEffect(() => {
    if (liveStatus === 'complete') {
      // Small delay to ensure synthesis message is rendered
      const timer = setTimeout(() => {
        const synthesisEl = document.getElementById('synthesis-message');
        if (synthesisEl) {
          synthesisEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else if (scrollContainerRef.current) {
          // Fallback: scroll to bottom
          scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [liveStatus]);

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
        <Header />

        <div className="container mx-auto px-4 py-6">
          {loading && <LoadingState />}

          {error && !isLiveDebate && <ErrorState error={error} />}

          {isLiveDebate && (
            <LiveDebateView
              debateId={debateId}
              status={liveStatus}
              task={liveTask}
              agents={liveAgents}
              messages={liveMessages}
              streamingMessages={streamingMessages as Map<string, StreamingMessage>}
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

          {debate && <ArchivedDebateView debate={debate} onShare={handleShare} copied={copied} />}
        </div>

        {/* Hide footer during active streaming to maximize content area */}
        {(!isLiveDebate || liveStatus === 'complete' || liveStatus === 'error') && <Footer />}
      </main>
    </>
  );
}

function Header() {
  return (
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
  );
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="text-acid-green font-mono animate-pulse">{'>'} LOADING DEBATE...</div>
    </div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <div className="bg-warning/10 border border-warning/30 rounded-lg p-6 text-center">
      <div className="text-warning text-2xl mb-2">{'>'} ERROR</div>
      <div className="text-text-muted">{error}</div>
      <Link href="/" className="inline-block mt-4 text-acid-green hover:underline font-mono">
        [RETURN HOME]
      </Link>
    </div>
  );
}

function Footer() {
  return (
    <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
      <div className="text-acid-green/50 mb-2">{'═'.repeat(40)}</div>
      <p className="text-text-muted">{'>'} AGORA DEBATE VIEWER // PERMALINK</p>
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
      <div className="text-acid-green/50 mt-4">{'═'.repeat(40)}</div>
    </footer>
  );
}
