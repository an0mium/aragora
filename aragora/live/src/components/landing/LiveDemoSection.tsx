'use client';

import { useEffect, useMemo, useRef } from 'react';
import { useTheme } from '@/context/ThemeContext';
import { useDebateWebSocket } from '@/hooks/useDebateWebSocket';
import { useSpectate, type SpectateEvent } from '@/hooks/useSpectate';
import { getAgentColors } from '@/utils/agentColors';

interface DiscoverableDebate {
  debateId: string;
  recentEventCount: number;
  lastEventAt: string | null;
  agents: string[];
  eventTypes: string[];
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

function summarizeLiveDebates(events: SpectateEvent[]): DiscoverableDebate[] {
  const grouped = new Map<
    string,
    {
      debateId: string;
      recentEventCount: number;
      lastEventAt: string | null;
      agents: Set<string>;
      eventTypes: Set<string>;
    }
  >();

  for (const event of events) {
    if (!event.debate_id) continue;

    const existing = grouped.get(event.debate_id);
    if (!existing) {
      grouped.set(event.debate_id, {
        debateId: event.debate_id,
        recentEventCount: 1,
        lastEventAt: event.timestamp,
        agents: new Set(event.agent_name ? [event.agent_name] : []),
        eventTypes: new Set([event.event_type]),
      });
      continue;
    }

    existing.recentEventCount += 1;
    existing.eventTypes.add(event.event_type);
    if (event.agent_name) {
      existing.agents.add(event.agent_name);
    }

    const existingTs = toEpochMs(existing.lastEventAt);
    const eventTs = toEpochMs(event.timestamp);
    if (eventTs !== null && (existingTs === null || eventTs >= existingTs)) {
      existing.lastEventAt = event.timestamp;
    }
  }

  return Array.from(grouped.values())
    .map((entry) => ({
      debateId: entry.debateId,
      recentEventCount: entry.recentEventCount,
      lastEventAt: entry.lastEventAt,
      agents: Array.from(entry.agents).sort(),
      eventTypes: Array.from(entry.eventTypes).sort(),
    }))
    .sort((left, right) => {
      const rightTs = toEpochMs(right.lastEventAt) ?? 0;
      const leftTs = toEpochMs(left.lastEventAt) ?? 0;
      if (rightTs !== leftTs) return rightTs - leftTs;
      return right.recentEventCount - left.recentEventCount;
    });
}

function formatRelativeAge(timestamp: string | null | undefined): string {
  const epochMs = toEpochMs(timestamp);
  if (epochMs === null) return 'just now';

  const ageSeconds = Math.max(0, Math.round((Date.now() - epochMs) / 1000));
  if (ageSeconds < 5) return 'just now';
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  const ageHours = Math.round(ageMinutes / 60);
  return `${ageHours}h ago`;
}

function formatEventLabel(eventType: string): string {
  return eventType.replace(/[._]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTranscriptTimestamp(timestamp?: number): string {
  if (!timestamp) return 'Live';

  return new Date(timestamp * 1000).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function extractEventDetails(event: SpectateEvent): string {
  const details = event.data.details;
  if (typeof details === 'string' && details.trim().length > 0) {
    return details;
  }

  if (event.agent_name) {
    return `${event.agent_name} ${formatEventLabel(event.event_type).toLowerCase()}`;
  }

  return formatEventLabel(event.event_type);
}

function LiveTranscriptMessage({
  agent,
  content,
  timestamp,
  round,
  role,
}: {
  agent: string;
  content: string;
  timestamp?: number;
  round?: number;
  role?: string;
}) {
  const colors = getAgentColors(agent || 'system');

  return (
    <article
      className={`rounded-2xl border p-4 transition-colors ${colors.border} ${colors.bg}`}
      data-testid="landing-live-message"
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className={`text-xs font-bold uppercase tracking-[0.24em] ${colors.text}`}>
          {agent || 'System'}
        </span>
        {role && (
          <span
            className="rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
            style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
          >
            {role}
          </span>
        )}
        {round !== undefined && round > 0 && (
          <span
            className="rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
            style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
          >
            Round {round}
          </span>
        )}
        <span className="ml-auto text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {formatTranscriptTimestamp(timestamp)}
        </span>
      </div>
      <p
        className="whitespace-pre-wrap"
        style={{
          color: 'var(--text)',
          fontSize: '14px',
          lineHeight: '1.7',
          fontFamily: 'var(--font-landing)',
        }}
      >
        {content}
      </p>
    </article>
  );
}

function LiveStreamingMessage({
  agent,
  content,
  reasoningPhase,
  confidence,
}: {
  agent: string;
  content: string;
  reasoningPhase?: string;
  confidence?: number | null;
}) {
  const colors = getAgentColors(agent || 'system');

  return (
    <article
      className={`rounded-2xl border p-4 shadow-[0_0_20px_rgba(57,255,20,0.08)] ${colors.border} ${colors.bg}`}
      data-testid="landing-live-streaming-message"
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className={`text-xs font-bold uppercase tracking-[0.24em] ${colors.text}`}>
          {agent || 'System'}
        </span>
        <span
          className="rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
          style={{ borderColor: 'var(--accent)', color: 'var(--accent)' }}
        >
          Streaming
        </span>
        {reasoningPhase && (
          <span
            className="rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
            style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
          >
            {reasoningPhase}
          </span>
        )}
        {confidence !== null && confidence !== undefined && (
          <span className="ml-auto text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {Math.round(confidence * 100)}% confidence
          </span>
        )}
      </div>
      <p
        className="whitespace-pre-wrap"
        style={{
          color: 'var(--text)',
          fontSize: '14px',
          lineHeight: '1.7',
          fontFamily: 'var(--font-landing)',
        }}
      >
        {content}
        <span className="ml-1 inline-block h-4 w-2 animate-pulse rounded-sm" style={{ backgroundColor: 'var(--accent)' }} />
      </p>
    </article>
  );
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const scrollRef = useRef<HTMLDivElement>(null);

  const { events, loaded, status: spectateStatus } = useSpectate(undefined, undefined, {
    pollInterval: 3000,
    maxEvents: 80,
  });

  const activityWindowSeconds = spectateStatus?.recent_activity_window_seconds ?? 120;

  const recentEvents = useMemo(
    () => events.filter((event) => isRecentEvent(event, activityWindowSeconds)),
    [activityWindowSeconds, events],
  );

  const discoverableDebates = useMemo(
    () => summarizeLiveDebates(recentEvents),
    [recentEvents],
  );

  const featuredDebate = discoverableDebates[0] ?? null;
  const featuredDebateId = featuredDebate?.debateId ?? '';

  const debateEvents = useMemo(
    () => recentEvents.filter((event) => event.debate_id === featuredDebateId).slice(-6),
    [featuredDebateId, recentEvents],
  );

  const {
    status,
    task,
    agents,
    messages,
    streamingMessages,
    error,
  } = useDebateWebSocket({
    debateId: featuredDebateId,
    enabled: Boolean(featuredDebateId),
  });

  const activeStreams = useMemo(
    () =>
      Array.from(streamingMessages.values())
        .filter((message) => !message.isComplete && message.content.trim().length > 0)
        .sort((left, right) => left.startTime - right.startTime),
    [streamingMessages],
  );

  const visibleMessages = useMemo(() => messages.slice(-8), [messages]);

  const visibleAgents = useMemo(() => {
    if (agents.length > 0) return agents;
    if (featuredDebate) return featuredDebate.agents;
    return [];
  }, [agents, featuredDebate]);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [activeStreams, visibleMessages]);

  const hasLiveDebate = Boolean(featuredDebateId);
  const hasTranscript = visibleMessages.length > 0 || activeStreams.length > 0;
  const viewerHref = hasLiveDebate ? `/spectate/${encodeURIComponent(featuredDebateId)}` : '/spectate';

  const statusLabel = (() => {
    if (!hasLiveDebate) return 'Standby';
    if (status === 'streaming') return 'Live';
    if (status === 'complete') return 'Settled';
    if (status === 'error') return 'Reconnecting';
    if (status === 'polling') return 'Polling';
    return 'Attaching';
  })();

  return (
    <section
      className="px-4"
      style={{
        paddingTop: '120px',
        paddingBottom: '120px',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-6xl mx-auto" data-testid="landing-live-demo">
        <p
          className="text-center uppercase tracking-widest"
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)', marginBottom: '20px' }}
        >
          {isDark ? '> SEE IT IN ACTION' : 'SEE IT IN ACTION'}
        </p>
        <p
          className="text-center"
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text)', fontFamily: 'var(--font-landing)', marginBottom: '48px' }}
        >
          Watch the newest active panel as soon as the live bridge sees it.
        </p>

        <div
          style={{
            backgroundColor: 'var(--surface)',
            borderRadius: 'var(--radius-card)',
            border: '1px solid var(--border)',
            borderTopColor: 'var(--accent)',
            borderTopWidth: '3px',
            boxShadow: 'var(--shadow-card)',
            overflow: 'hidden',
          }}
        >
          <div
            className="flex flex-wrap items-center gap-3"
            style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}
          >
            <span
              className="font-bold px-2 py-0.5 uppercase tracking-wider"
              style={{
                fontSize: '10px',
                backgroundColor: hasLiveDebate ? 'var(--accent)' : 'transparent',
                color: hasLiveDebate ? 'var(--bg)' : 'var(--text-muted)',
                borderRadius: 'var(--radius-button)',
                border: hasLiveDebate ? 'none' : '1px solid var(--border)',
              }}
            >
              {statusLabel}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {task || (hasLiveDebate ? `Tracking debate ${featuredDebateId.slice(0, 14)}` : 'Waiting for a live debate to surface')}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {hasLiveDebate
                ? `${featuredDebate?.recentEventCount ?? 0} recent events · ${formatRelativeAge(featuredDebate?.lastEventAt)}`
                : 'Auto-attaches when fresh debate activity appears'}
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.6fr)_minmax(280px,0.95fr)]">
            <div
              className="min-h-[480px] border-b lg:border-b-0 lg:border-r"
              style={{ borderColor: 'var(--border)' }}
            >
              <div className="flex flex-wrap gap-2 px-5 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
                {(visibleAgents.length > 0 ? visibleAgents : ['Bridge']).map((agent) => {
                  const colors = getAgentColors(agent);
                  return (
                    <span
                      key={agent}
                      className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] ${colors.border} ${colors.bg} ${colors.text}`}
                    >
                      {agent}
                    </span>
                  );
                })}
              </div>

              <div
                ref={scrollRef}
                className="max-h-[420px] space-y-4 overflow-y-auto px-5 py-5"
                aria-live="polite"
                data-testid="landing-live-debate-stream"
              >
                {!hasLiveDebate && !loaded && (
                  <div
                    className="flex h-full min-h-[320px] items-center justify-center text-center"
                    data-testid="landing-live-scanning"
                  >
                    <div className="max-w-md">
                      <p
                        className="mb-3 text-sm font-semibold uppercase tracking-[0.28em]"
                        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                      >
                        Scanning
                      </p>
                      <p
                        style={{
                          color: 'var(--text)',
                          fontSize: '16px',
                          lineHeight: '1.7',
                          fontFamily: 'var(--font-landing)',
                        }}
                      >
                        Checking the spectate bridge for the newest active debate stream.
                      </p>
                    </div>
                  </div>
                )}

                {!hasLiveDebate && loaded && (
                  <div
                    className="flex h-full min-h-[320px] items-center justify-center text-center"
                    data-testid="landing-live-standby"
                  >
                    <div className="max-w-md">
                      <p
                        className="mb-3 text-sm font-semibold uppercase tracking-[0.28em]"
                        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                      >
                        Standby
                      </p>
                      <p
                        style={{
                          color: 'var(--text)',
                          fontSize: '16px',
                          lineHeight: '1.7',
                          fontFamily: 'var(--font-landing)',
                        }}
                      >
                        No public live debate is discoverable right now. This panel automatically follows the next debate the spectate bridge sees.
                      </p>
                    </div>
                  </div>
                )}

                {hasLiveDebate && !hasTranscript && (
                  <div className="space-y-4 py-6">
                    <div
                      className="rounded-2xl border p-5"
                      style={{ borderColor: 'var(--border)', backgroundColor: 'color-mix(in srgb, var(--surface) 86%, transparent)' }}
                    >
                      <p
                        className="mb-2 text-xs font-bold uppercase tracking-[0.28em]"
                        style={{ color: 'var(--accent)' }}
                      >
                        Attaching to debate {featuredDebateId.slice(0, 14)}
                      </p>
                      <p
                        style={{
                          color: 'var(--text)',
                          fontSize: '15px',
                          lineHeight: '1.7',
                          fontFamily: 'var(--font-landing)',
                        }}
                      >
                        {error
                          ? error
                          : 'The live bridge has seen debate activity. The transcript stream is connecting and will render agent messages as soon as the opening statements arrive.'}
                      </p>
                    </div>

                    {debateEvents.map((event) => (
                      <div
                        key={`${event.debate_id}-${event.timestamp}-${event.event_type}`}
                        className="rounded-2xl border p-4"
                        style={{ borderColor: 'var(--border)', backgroundColor: 'transparent' }}
                      >
                        <div className="mb-2 flex items-center gap-2">
                          <span
                            className="rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
                            style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
                          >
                            {formatEventLabel(event.event_type)}
                          </span>
                          {event.agent_name && (
                            <span className="text-xs font-semibold uppercase tracking-[0.2em]" style={{ color: 'var(--accent)' }}>
                              {event.agent_name}
                            </span>
                          )}
                          <span className="ml-auto text-[11px]" style={{ color: 'var(--text-muted)' }}>
                            {formatRelativeAge(event.timestamp)}
                          </span>
                        </div>
                        <p
                          style={{
                            color: 'var(--text)',
                            fontSize: '14px',
                            lineHeight: '1.7',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {extractEventDetails(event)}
                        </p>
                      </div>
                    ))}
                  </div>
                )}

                {visibleMessages.map((message, index) => (
                  <LiveTranscriptMessage
                    key={`${message.agent}-${message.timestamp ?? index}-${message.content.slice(0, 24)}`}
                    agent={message.agent}
                    content={message.content}
                    timestamp={message.timestamp}
                    round={message.round}
                    role={message.role}
                  />
                ))}

                {activeStreams.map((message) => (
                  <LiveStreamingMessage
                    key={`${message.agent}-${message.taskId}`}
                    agent={message.agent}
                    content={message.content}
                    reasoningPhase={message.reasoningPhase}
                    confidence={message.confidence}
                  />
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-0">
              <div className="p-5" style={{ borderBottom: '1px solid var(--border)' }}>
                <p
                  className="mb-3 text-xs font-bold uppercase tracking-[0.28em]"
                  style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                >
                  Debate Pulse
                </p>
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Tracking</span>
                    <span style={{ color: 'var(--text)', fontSize: '12px', fontWeight: 600 }}>
                      {hasLiveDebate ? featuredDebateId.slice(0, 18) : 'Awaiting live debate'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Transport</span>
                    <span style={{ color: hasLiveDebate ? 'var(--accent)' : 'var(--text)', fontSize: '12px', fontWeight: 600 }}>
                      {hasLiveDebate ? 'Debate WebSocket' : 'Spectate bridge'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Visible agents</span>
                    <span style={{ color: 'var(--text)', fontSize: '12px', fontWeight: 600 }}>
                      {visibleAgents.length}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Live debates</span>
                    <span style={{ color: 'var(--text)', fontSize: '12px', fontWeight: 600 }}>
                      {discoverableDebates.length}
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-5" style={{ borderBottom: '1px solid var(--border)' }}>
                <p
                  className="mb-3 text-xs font-bold uppercase tracking-[0.28em]"
                  style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                >
                  Recent Event Flow
                </p>
                <div className="space-y-3">
                  {(debateEvents.length > 0 ? debateEvents : recentEvents.slice(-4)).map((event) => (
                    <div key={`${event.debate_id ?? 'bridge'}-${event.timestamp}-${event.event_type}`}>
                      <div className="mb-1 flex items-center gap-2">
                        <span
                          className="rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em]"
                          style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
                        >
                          {formatEventLabel(event.event_type)}
                        </span>
                        <span className="ml-auto text-[11px]" style={{ color: 'var(--text-muted)' }}>
                          {formatRelativeAge(event.timestamp)}
                        </span>
                      </div>
                      <p
                        style={{
                          color: 'var(--text)',
                          fontSize: '13px',
                          lineHeight: '1.6',
                          fontFamily: 'var(--font-landing)',
                        }}
                      >
                        {extractEventDetails(event)}
                      </p>
                    </div>
                  ))}

                  {recentEvents.length === 0 && (
                    <p
                      style={{
                        color: 'var(--text-muted)',
                        fontSize: '13px',
                        lineHeight: '1.7',
                        fontFamily: 'var(--font-landing)',
                      }}
                    >
                      Fresh bridge events will appear here before the transcript connects.
                    </p>
                  )}
                </div>
              </div>

              <div className="p-5">
                <p
                  className="mb-3 text-xs font-bold uppercase tracking-[0.28em]"
                  style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                >
                  Viewer Actions
                </p>
                <div className="flex flex-col gap-3">
                  <a
                    href={viewerHref}
                    className="text-center text-sm font-semibold transition-all hover:scale-[1.01]"
                    style={{
                      border: '1px solid var(--accent)',
                      borderRadius: 'var(--radius-button)',
                      color: 'var(--bg)',
                      backgroundColor: 'var(--accent)',
                      fontFamily: 'var(--font-landing)',
                      padding: '14px 18px',
                    }}
                  >
                    {hasLiveDebate ? 'Open full live viewer' : 'Open spectate bridge'}
                  </a>
                  <a
                    href="/try"
                    className="text-center text-sm font-semibold transition-all hover:scale-[1.01]"
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-button)',
                      color: 'var(--text)',
                      backgroundColor: 'transparent',
                      fontFamily: 'var(--font-landing)',
                      padding: '14px 18px',
                    }}
                  >
                    Run your own debate
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
