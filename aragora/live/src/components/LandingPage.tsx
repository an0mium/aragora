'use client';

import { useState, useCallback, useRef, useEffect, useMemo, FormEvent } from 'react';
import Link from 'next/link';
import { DebateResultPreview, RETURN_URL_KEY, PENDING_DEBATE_KEY, type DebateResponse } from './DebateResultPreview';
import {
  useSpectate,
  type SpectateEvent,
  type SpectateLiveDebateSummary,
  type SpectateStatus,
} from '@/hooks/useSpectate';
import { getCurrentReturnUrl, normalizeReturnUrl } from '@/utils/returnUrl';

interface LandingPageProps {
  apiBase?: string;
  wsUrl?: string;
  onDebateStarted?: (debateId: string) => void;
  onEnterDashboard?: () => void;
}

const PROGRESS_MESSAGES = [
  'Assembling analyst panel...',
  'Agents debating your question...',
  'Analyzing arguments...',
  'Building consensus...',
  'Generating verdict...',
];

function parseRetryAfterSeconds(retryAfter: string | null): number {
  if (!retryAfter) return 60;

  const deltaSeconds = Number.parseInt(retryAfter, 10);
  if (Number.isFinite(deltaSeconds) && deltaSeconds >= 0) {
    return deltaSeconds;
  }

  const retryTime = Date.parse(retryAfter);
  if (Number.isNaN(retryTime)) return 60;

  return Math.max(1, Math.ceil((retryTime - Date.now()) / 1000));
}

function toEpochMs(timestamp: string | null | undefined): number | null {
  if (!timestamp) return null;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : parsed;
}

function isRecentEvent(event: SpectateEvent, windowSeconds: number): boolean {
  const epochMs = toEpochMs(event.timestamp);
  if (epochMs === null) return false;
  return Date.now() - epochMs <= windowSeconds * 1000;
}

function formatRelativeAge(timestamp: string | null | undefined): string {
  const epochMs = toEpochMs(timestamp);
  if (epochMs === null) return 'just now';

  const ageSeconds = Math.max(0, Math.round((Date.now() - epochMs) / 1000));
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  const ageHours = Math.round(ageMinutes / 60);
  if (ageHours < 24) return `${ageHours}h ago`;

  const ageDays = Math.round(ageHours / 24);
  return `${ageDays}d ago`;
}

function formatEventType(eventType: string): string {
  return eventType.replace(/_/g, ' ').toUpperCase();
}

function getEventSummary(event: SpectateEvent): string {
  if (typeof event.data.details === 'string' && event.data.details.trim()) {
    return event.data.details;
  }
  if (typeof event.data.summary === 'string' && event.data.summary.trim()) {
    return event.data.summary;
  }
  if (typeof event.data.message === 'string' && event.data.message.trim()) {
    return event.data.message;
  }

  switch (event.event_type) {
    case 'proposal':
      return 'An agent opened a new proposal.';
    case 'critique':
      return 'Another agent challenged the current line of reasoning.';
    case 'vote':
      return 'The panel is registering a vote.';
    case 'consensus':
      return 'The panel is converging on a verdict.';
    default:
      return 'The live bridge captured a new debate event.';
  }
}

function getDebateHeadline(
  debateId: string | null,
  recentEvents: SpectateEvent[],
): string | null {
  if (!debateId) return null;

  for (const event of recentEvents) {
    if (event.debate_id !== debateId) continue;

    const candidates = [
      event.data.title,
      event.data.topic,
      event.data.question,
      event.data.prompt,
    ];
    for (const candidate of candidates) {
      if (typeof candidate === 'string' && candidate.trim()) {
        return candidate.trim();
      }
    }
  }

  return null;
}

function getLiveBridgeState(
  loaded: boolean,
  connected: boolean,
  status: SpectateStatus | null,
  hasLiveDebate: boolean,
):
  | SpectateStatus['bridge_state']
  | 'checking'
  | 'status_unavailable'
  | 'unreachable' {
  if (!loaded) return 'checking';
  if (status) return status.bridge_state;
  if (hasLiveDebate) return 'live_debates_available';
  return connected ? 'status_unavailable' : 'unreachable';
}

function getLiveBridgeLabel(
  state:
    | SpectateStatus['bridge_state']
    | 'checking'
    | 'status_unavailable'
    | 'unreachable',
): string {
  switch (state) {
    case 'live_debates_available':
      return 'LIVE';
    case 'activity_unattributed':
      return 'PARTIAL';
    case 'idle':
      return 'IDLE';
    case 'inactive':
      return 'OFF';
    case 'status_unavailable':
      return 'RECENT FEED';
    case 'unreachable':
      return 'API OFFLINE';
    case 'checking':
    default:
      return 'CHECKING';
  }
}

export function LandingPage({ apiBase, onEnterDashboard }: LandingPageProps) {
  const [question, setQuestion] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<DebateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastTopic, setLastTopic] = useState('');
  const [progressMsg, setProgressMsg] = useState(PROGRESS_MESSAGES[0]);
  const abortRef = useRef<AbortController | null>(null);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const resolvedApiBase = apiBase || 'https://api.aragora.ai';
  const {
    events: spectateEvents,
    connected: spectateConnected,
    loaded: spectateLoaded,
    status: spectateStatus,
  } = useSpectate(undefined, undefined, {
    apiBase: resolvedApiBase,
    pollInterval: 3000,
    maxEvents: 24,
  });

  const activityWindowSeconds =
    spectateStatus?.recent_activity_window_seconds ?? 120;
  const recentBridgeEvents = useMemo(
    () => spectateEvents.filter((event) => isRecentEvent(event, activityWindowSeconds)),
    [activityWindowSeconds, spectateEvents],
  );

  const fallbackDiscoverableDebates = useMemo(() => {
    const grouped = new Map<
      string,
      {
        debate_id: string;
        recent_event_count: number;
        last_event_at: string | null;
        event_types: Set<string>;
      }
    >();

    for (const event of recentBridgeEvents) {
      if (!event.debate_id) continue;

      const existing = grouped.get(event.debate_id);
      if (!existing) {
        grouped.set(event.debate_id, {
          debate_id: event.debate_id,
          recent_event_count: 1,
          last_event_at: event.timestamp,
          event_types: new Set([event.event_type]),
        });
        continue;
      }

      existing.recent_event_count += 1;
      existing.event_types.add(event.event_type);

      const existingTs = toEpochMs(existing.last_event_at);
      const eventTs = toEpochMs(event.timestamp);
      if (eventTs !== null && (existingTs === null || eventTs >= existingTs)) {
        existing.last_event_at = event.timestamp;
      }
    }

    return Array.from(grouped.values())
      .map((debate) => ({
        debate_id: debate.debate_id,
        recent_event_count: debate.recent_event_count,
        last_event_at: debate.last_event_at,
        event_types: Array.from(debate.event_types).sort(),
      }))
      .sort(
        (left, right) =>
          (toEpochMs(right.last_event_at) ?? 0) - (toEpochMs(left.last_event_at) ?? 0),
      );
  }, [recentBridgeEvents]);

  const discoverableDebates: SpectateLiveDebateSummary[] =
    spectateStatus?.live_debates ?? fallbackDiscoverableDebates;
  const liveDebate = discoverableDebates[0] ?? null;
  const liveBridgeState = getLiveBridgeState(
    spectateLoaded,
    spectateConnected,
    spectateStatus,
    liveDebate !== null,
  );
  const liveBridgeLabel = getLiveBridgeLabel(liveBridgeState);
  const unattributedRecentEvents =
    spectateStatus?.unattributed_recent_event_count ??
    recentBridgeEvents.filter((event) => !event.debate_id).length;

  const liveDebateEvents = useMemo(() => {
    if (!liveDebate) return [];
    return recentBridgeEvents
      .filter((event) => event.debate_id === liveDebate.debate_id)
      .slice(-5)
      .reverse();
  }, [liveDebate, recentBridgeEvents]);

  const liveDebateHeadline = useMemo(
    () => getDebateHeadline(liveDebate?.debate_id ?? null, recentBridgeEvents),
    [liveDebate?.debate_id, recentBridgeEvents],
  );

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (progressRef.current) {
        clearInterval(progressRef.current);
      }
    };
  }, []);

  const saveDebateBeforeLogin = useCallback(() => {
    if (result) {
      sessionStorage.setItem(PENDING_DEBATE_KEY, JSON.stringify(result));
      const debateDestination = result.id ? `/debates/${encodeURIComponent(result.id)}` : getCurrentReturnUrl();
      sessionStorage.setItem(RETURN_URL_KEY, normalizeReturnUrl(debateDestination));
    }
  }, [result]);

  async function runDebate(topic: string) {
    abortRef.current?.abort();
    if (progressRef.current) {
      clearInterval(progressRef.current);
    }

    setIsRunning(true);
    setError(null);
    setResult(null);
    setLastTopic(topic);
    setProgressMsg(PROGRESS_MESSAGES[0]);

    // Rotate progress messages
    let progressIdx = 0;
    progressRef.current = setInterval(() => {
      progressIdx = (progressIdx + 1) % PROGRESS_MESSAGES.length;
      setProgressMsg(PROGRESS_MESSAGES[progressIdx]);
    }, 4000);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${resolvedApiBase}/api/v1/playground/debate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, question: topic, rounds: 2, agents: 3, source: 'landing' }),
        signal: controller.signal,
      });

      if (res.status === 429) {
        const retryAfter = parseRetryAfterSeconds(res.headers.get('Retry-After'));
        const waitText = retryAfter > 60 ? `${Math.ceil(retryAfter / 60)} minutes` : `${retryAfter} seconds`;
        setError(`Rate limit reached. Please try again in ${waitText}.`);
        return;
      }

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setError(data?.error || `Something went wrong (${res.status}). Please try again.`);
        return;
      }

      setResult(await res.json());
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      if (err instanceof Error && err.message.includes('Failed to fetch')) {
        setError('Could not connect to the server. Check your connection and try again.');
        return;
      }
      setError('Network error. Please try again.');
    } finally {
      if (progressRef.current) {
        clearInterval(progressRef.current);
        progressRef.current = null;
      }
      setIsRunning(false);
      setProgressMsg('');
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (question.trim()) {
      runDebate(question.trim());
    }
  }

  return (
    <main className="min-h-screen bg-bg text-text">
      {/* Nav */}
      <nav className="border-b border-border bg-surface/80 backdrop-blur-sm shadow-[0_1px_0_var(--border-glow)] sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <span className="font-mono text-acid-green font-bold text-sm tracking-wider">
            ARAGORA
          </span>
          <div className="flex items-center gap-4">
            <a href="#how-it-works" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block">
              How it works
            </a>
            <Link href="/oracle" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block">
              Oracle
            </Link>
            {onEnterDashboard ? (
              <button
                onClick={() => { saveDebateBeforeLogin(); onEnterDashboard(); }}
                className="text-xs font-mono px-3 py-1.5 border border-acid-green/40 text-text-muted hover:text-acid-green hover:border-acid-green transition-colors"
              >
                Log in
              </button>
            ) : (
              <Link
                href="/login"
                onClick={saveDebateBeforeLogin}
                className="text-xs font-mono px-3 py-1.5 border border-acid-green/40 text-text-muted hover:text-acid-green hover:border-acid-green transition-colors"
              >
                Log in
              </Link>
            )}
            <Link
              href="/signup"
              onClick={saveDebateBeforeLogin}
              className="text-xs font-mono px-3 py-1.5 bg-acid-green text-bg hover:bg-acid-green/80 transition-colors font-bold"
            >
              Sign up free
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="py-20 sm:py-32 px-4">
        <div className="max-w-2xl mx-auto text-center">
          <h1 className="font-mono text-3xl sm:text-5xl text-text mb-6 leading-tight">
            Don&apos;t trust one AI.
            <br />
            <span className="text-acid-green">Make them argue.</span>
          </h1>
          <p className="font-mono text-sm text-text-muted max-w-lg mx-auto mb-12 leading-relaxed">
            Multiple AI models debate your question, stress-test each answer,
            and deliver an audit-ready verdict you can actually defend.
          </p>

          <form onSubmit={handleSubmit} className="text-left max-w-xl mx-auto">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What decision are you facing?"
              disabled={isRunning}
              rows={2}
              className="w-full bg-surface border border-border text-text px-4 py-3 font-mono text-sm placeholder:text-text-muted/50 focus:outline-none focus:border-acid-green transition-colors resize-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isRunning || !question.trim()}
              className="w-full mt-3 font-mono text-sm px-8 py-3 bg-acid-green text-bg font-bold hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isRunning ? 'Agents debating...' : 'Run a free debate'}
            </button>
          </form>

          {/* Example topics — reduce blank-page friction */}
          {!result && !isRunning && (
            <div className="max-w-xl mx-auto mt-4">
              <p className="text-xs font-mono text-text-muted/60 mb-2 text-center">Or try an example:</p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  'Should we build or buy our analytics platform?',
                  'Is remote work better for a 50-person company?',
                  'Should we adopt microservices or keep our monolith?',
                ].map((topic) => (
                  <button
                    key={topic}
                    onClick={() => { setQuestion(topic); runDebate(topic); }}
                    className="text-xs font-mono px-3 py-1.5 border border-border text-text-muted hover:border-acid-green hover:text-acid-green transition-colors"
                  >
                    {topic}
                  </button>
                ))}
              </div>
            </div>
          )}

          {isRunning && (
            <div className="flex flex-col items-center py-8 gap-3">
              <div className="flex items-center gap-3 text-acid-green">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-sm font-mono">{progressMsg}</span>
              </div>
              <span className="text-xs font-mono text-text-muted/60">Usually takes 10-20 seconds</span>
            </div>
          )}

          {error && (
            <div className="border border-crimson/40 bg-crimson/5 p-4 mt-6 text-left max-w-xl mx-auto">
              <p className="text-sm text-crimson font-mono mb-3">{error}</p>
              {lastTopic && (
                <button
                  onClick={() => { setError(null); runDebate(lastTopic); }}
                  className="font-mono text-xs px-4 py-2 border border-crimson/40 text-crimson hover:bg-crimson/10 transition-colors"
                >
                  Try again
                </button>
              )}
            </div>
          )}

          {result && <DebateResultPreview result={result} />}
        </div>
      </section>

      <section
        className="py-16 px-4 border-t border-border bg-[radial-gradient(circle_at_top,_rgba(34,197,94,0.08),_transparent_55%)]"
        data-testid="landing-live-debate"
      >
        <div className="max-w-5xl mx-auto grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
          <div className="border border-acid-green/20 bg-surface/40 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div
                className={`w-2.5 h-2.5 rounded-full ${
                  liveBridgeState === 'live_debates_available'
                    ? 'bg-acid-green animate-pulse'
                    : liveBridgeState === 'activity_unattributed' || liveBridgeState === 'status_unavailable'
                      ? 'bg-acid-cyan animate-pulse'
                      : liveBridgeState === 'checking' || liveBridgeState === 'idle'
                        ? 'bg-acid-yellow animate-pulse'
                        : 'bg-red-500'
                }`}
              />
              <span className="font-mono text-xs text-acid-cyan uppercase tracking-[0.3em]">
                Live Spectate
              </span>
              <span
                className={`px-2 py-0.5 text-[10px] font-mono border ${
                  liveBridgeState === 'live_debates_available'
                    ? 'border-acid-green/30 text-acid-green bg-acid-green/10'
                    : liveBridgeState === 'activity_unattributed' || liveBridgeState === 'status_unavailable'
                      ? 'border-acid-cyan/30 text-acid-cyan bg-acid-cyan/10'
                      : liveBridgeState === 'checking' || liveBridgeState === 'idle'
                        ? 'border-acid-yellow/30 text-acid-yellow bg-acid-yellow/10'
                        : 'border-red-500/30 text-red-400 bg-red-500/10'
                }`}
              >
                {liveBridgeLabel}
              </span>
            </div>

            <h2 className="font-mono text-2xl sm:text-3xl text-text mb-3">
              Watch agents argue in real time.
            </h2>
            <p className="font-mono text-sm text-text-muted leading-relaxed max-w-2xl">
              {liveDebate
                ? 'A live debate is happening right now. Open the viewer to watch the panel challenge proposals, critique reasoning, and converge on a verdict.'
                : liveBridgeState === 'activity_unattributed'
                  ? 'The bridge is seeing fresh debate traffic, but the current events are not tagged with a debate ID yet, so this page waits before showing a watcher.'
                  : liveBridgeState === 'status_unavailable'
                    ? 'Recent bridge events are still arriving, but the readiness endpoint is unavailable. This surface only promotes a watcher when the event feed itself can prove a live debate.'
                    : liveBridgeState === 'checking'
                      ? 'Checking the live bridge for a debate visitors can watch right now.'
                      : 'No live debate is discoverable at this moment. Spectate mode stays available so visitors can jump in as soon as the bridge sees one.'}
            </p>

            <div className="grid gap-3 sm:grid-cols-3 mt-6">
              <div className="border border-border bg-bg/50 px-3 py-3">
                <div className="text-[10px] font-mono text-text-muted uppercase mb-1">
                  Live Debate IDs
                </div>
                <div className="font-mono text-sm text-acid-green">
                  {discoverableDebates.length}
                </div>
              </div>
              <div className="border border-border bg-bg/50 px-3 py-3">
                <div className="text-[10px] font-mono text-text-muted uppercase mb-1">
                  Recent Events
                </div>
                <div className="font-mono text-sm text-acid-cyan">
                  {spectateStatus?.recent_event_count ?? recentBridgeEvents.length}
                </div>
              </div>
              <div className="border border-border bg-bg/50 px-3 py-3">
                <div className="text-[10px] font-mono text-text-muted uppercase mb-1">
                  Unattributed
                </div>
                <div className="font-mono text-sm text-text">
                  {unattributedRecentEvents}
                </div>
              </div>
            </div>

            {liveDebate ? (
              <div className="mt-6 border border-acid-green/30 bg-acid-green/5 p-4" data-testid="landing-live-debate-card">
                <div className="text-[10px] font-mono text-acid-cyan uppercase tracking-[0.25em] mb-2">
                  Happening Now
                </div>
                <h3 className="font-mono text-lg text-text leading-snug">
                  {liveDebateHeadline || `Live debate ${liveDebate.debate_id}`}
                </h3>
                <div className="flex flex-wrap gap-2 mt-3">
                  {liveDebate.event_types.map((eventType) => (
                    <span
                      key={`${liveDebate.debate_id}-${eventType}`}
                      className="px-2 py-1 text-[10px] font-mono border border-acid-cyan/30 bg-acid-cyan/10 text-acid-cyan"
                    >
                      {formatEventType(eventType)}
                    </span>
                  ))}
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center gap-3 mt-4">
                  <Link
                    href={`/spectate/${liveDebate.debate_id}`}
                    className="inline-flex items-center justify-center font-mono text-sm px-4 py-2 bg-acid-green text-bg font-bold hover:bg-acid-green/80 transition-colors"
                  >
                    Watch live debate
                  </Link>
                  <Link
                    href="/spectate"
                    className="inline-flex items-center justify-center font-mono text-sm px-4 py-2 border border-acid-green/30 text-acid-green hover:border-acid-green transition-colors"
                  >
                    Open spectate mode
                  </Link>
                  <span className="font-mono text-xs text-text-muted">
                    Last seen {formatRelativeAge(liveDebate.last_event_at)}
                  </span>
                </div>
              </div>
            ) : (
              <div className="mt-6 border border-border bg-bg/40 p-4">
                <p className="font-mono text-xs text-text-muted leading-relaxed">
                  {liveBridgeState === 'activity_unattributed'
                    ? 'Waiting for the bridge to tag the current debate before we show a live watcher.'
                    : 'Open spectate mode to monitor the bridge and jump into the next live debate as soon as it becomes discoverable.'}
                </p>
                <Link
                  href="/spectate"
                  className="inline-flex items-center justify-center font-mono text-sm px-4 py-2 border border-acid-green/30 text-acid-green hover:border-acid-green transition-colors mt-4"
                >
                  Go to spectate mode
                </Link>
              </div>
            )}
          </div>

          <div className="border border-acid-cyan/20 bg-surface/30 p-6">
            <div className="flex items-center justify-between gap-4 mb-4">
              <h3 className="font-mono text-sm text-acid-cyan uppercase tracking-[0.25em]">
                Recent Exchange
              </h3>
              <span className="font-mono text-[10px] text-text-muted">
                Refreshes every few seconds
              </span>
            </div>

            {liveDebateEvents.length > 0 ? (
              <div className="space-y-3" data-testid="landing-live-debate-feed">
                {liveDebateEvents.map((event, index) => (
                  <div
                    key={`${event.timestamp}-${index}`}
                    className="border border-border bg-bg/40 px-3 py-3"
                  >
                    <div className="flex items-center gap-2 flex-wrap mb-2">
                      <span className="font-mono text-[10px] text-acid-green">
                        {formatEventType(event.event_type)}
                      </span>
                      {event.agent_name && (
                        <span className="font-mono text-[10px] text-acid-cyan">
                          {event.agent_name}
                        </span>
                      )}
                      {event.round_number != null && (
                        <span className="font-mono text-[10px] text-text-muted">
                          Round {event.round_number}
                        </span>
                      )}
                    </div>
                    <p className="font-mono text-sm text-text leading-relaxed">
                      {getEventSummary(event)}
                    </p>
                    <div className="font-mono text-[10px] text-text-muted mt-2">
                      {formatRelativeAge(event.timestamp)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="border border-dashed border-border bg-bg/30 px-4 py-6">
                <p className="font-mono text-sm text-text">
                  {liveDebate
                    ? 'Live debate detected. Waiting for the first spectator messages to land.'
                    : 'No argument feed is available yet.'}
                </p>
                <p className="font-mono text-xs text-text-muted mt-2 leading-relaxed">
                  {spectateLoaded
                    ? 'As soon as the bridge emits attributed debate events, this panel will show the latest back-and-forth.'
                    : 'The bridge is still warming up. This panel fills in once the first poll completes.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="py-20 px-4 border-t border-border">
        <div className="max-w-3xl mx-auto">
          <h2 className="font-mono text-sm text-text-muted text-center mb-12 tracking-widest uppercase">
            How it works
          </h2>
          <div className="space-y-12">
            {[
              { step: '01', title: 'You ask a question', desc: 'Any decision, strategy, or architecture question you need vetted.' },
              { step: '02', title: 'AI agents debate it', desc: 'Claude, GPT, Gemini, Mistral, and others argue every angle. Different models catch different blind spots.' },
              { step: '03', title: 'You get a decision receipt', desc: 'An audit-ready verdict with evidence chains, confidence scores, and dissenting views preserved.' },
            ].map((item) => (
              <div key={item.step} className="flex gap-6 items-start">
                <span className="font-mono text-acid-green text-sm mt-0.5 flex-shrink-0">{item.step}</span>
                <div>
                  <h3 className="font-mono text-base text-text mb-1">{item.title}</h3>
                  <p className="font-mono text-sm text-text-muted leading-relaxed">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Why debate */}
      <section className="py-20 px-4 border-t border-border">
        <div className="max-w-3xl mx-auto">
          <h2 className="font-mono text-sm text-text-muted text-center mb-4 tracking-widest uppercase">
            Why this matters
          </h2>
          <p className="font-mono text-lg text-center text-text mb-12 max-w-xl mx-auto leading-relaxed">
            A single AI hallucinates, agrees with you, and contradicts itself.
            Adversarial debate fixes all three.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { problem: 'Hallucination', fix: 'Cross-model verification catches fabrications before they reach you.' },
              { problem: 'Sycophancy', fix: 'Agents are structurally incentivized to disagree and find flaws.' },
              { problem: 'Inconsistency', fix: 'Debate convergence produces stable, defensible positions.' },
            ].map((item) => (
              <div key={item.problem}>
                <h3 className="font-mono text-sm text-acid-green mb-2">{item.problem}</h3>
                <p className="font-mono text-xs text-text-muted leading-relaxed">{item.fix}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="py-20 px-4 border-t border-border">
        <div className="max-w-2xl mx-auto text-center">
          <p className="font-mono text-sm text-text-muted mb-6">
            No signup required. First result in under 30 seconds.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="font-mono text-sm px-8 py-3 bg-acid-green text-bg font-bold hover:opacity-90 transition-opacity"
            >
              Try it now
            </button>
            <Link
              href="/signup"
              className="font-mono text-sm px-8 py-3 border border-border text-text-muted hover:border-acid-green hover:text-acid-green transition-colors"
            >
              Create an account
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-6 px-4 border-t border-border">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="font-mono text-xs text-text-muted/50">
            Aragora
          </span>
          <div className="flex items-center gap-6">
            <a href="/about" className="font-mono text-xs text-text-muted/50 hover:text-text-muted transition-colors">About</a>
            <a href="/pricing" className="font-mono text-xs text-text-muted/50 hover:text-text-muted transition-colors">Pricing</a>
            <a href="mailto:support@aragora.ai" className="font-mono text-xs text-text-muted/50 hover:text-text-muted transition-colors">Support</a>
          </div>
        </div>
      </footer>
    </main>
  );
}
