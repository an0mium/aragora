'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { WS_URL } from '@/config';
import { useTheme } from '@/context/ThemeContext';
import { useScrollReveal } from '@/hooks/useScrollReveal';
import {
  useDebateWebSocket,
  type DebateConnectionStatus,
  type StreamingMessage,
  type TranscriptMessage,
} from '@/hooks/useDebateWebSocket';
import {
  useSpectate,
  type SpectateEvent,
  type SpectateLiveDebateSummary,
  type SpectateStatus,
} from '@/hooks/useSpectate';
import { getAgentColors } from '@/utils/agentColors';

const FALLBACK_DISCOVERY_WINDOW_SECONDS = 120;

const STREAM_STATUS: Record<DebateConnectionStatus, { label: string; tone: string }> = {
  idle: { label: 'Ready', tone: 'var(--text-muted)' },
  connecting: { label: 'Connecting', tone: 'var(--accent)' },
  streaming: { label: 'Streaming live', tone: 'var(--accent)' },
  polling: { label: 'Polling fallback', tone: '#f59e0b' },
  complete: { label: 'Debate complete', tone: '#2563eb' },
  error: { label: 'Connection error', tone: '#dc2626' },
};

function toEpochMs(timestamp: string | null | undefined): number | null {
  if (!timestamp) return null;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : parsed;
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

function isRecentEvent(event: SpectateEvent, windowSeconds: number): boolean {
  const epochMs = toEpochMs(event.timestamp);
  if (epochMs === null) return false;
  return Date.now() - epochMs <= windowSeconds * 1000;
}

function deriveDiscoverableDebates(
  events: SpectateEvent[],
  status: SpectateStatus | null,
): SpectateLiveDebateSummary[] {
  if (status?.live_debates?.length) {
    return status.live_debates;
  }

  const activityWindowSeconds = status?.recent_activity_window_seconds ?? FALLBACK_DISCOVERY_WINDOW_SECONDS;
  const recentEvents = events.filter((event) => isRecentEvent(event, activityWindowSeconds));
  const grouped = new Map<string, SpectateLiveDebateSummary>();

  for (const event of recentEvents) {
    if (!event.debate_id) continue;

    const existing = grouped.get(event.debate_id);
    if (!existing) {
      grouped.set(event.debate_id, {
        debate_id: event.debate_id,
        recent_event_count: 1,
        last_event_at: event.timestamp,
        event_types: [event.event_type],
      });
      continue;
    }

    existing.recent_event_count += 1;
    if (!existing.event_types.includes(event.event_type)) {
      existing.event_types.push(event.event_type);
      existing.event_types.sort();
    }

    const existingTs = toEpochMs(existing.last_event_at);
    const eventTs = toEpochMs(event.timestamp);
    if (eventTs !== null && (existingTs === null || eventTs >= existingTs)) {
      existing.last_event_at = event.timestamp;
    }
  }

  return Array.from(grouped.values()).sort(
    (left, right) => (toEpochMs(right.last_event_at) ?? 0) - (toEpochMs(left.last_event_at) ?? 0),
  );
}

function createAgentSummary(
  agent: string,
  messages: TranscriptMessage[],
  activeStreams: StreamingMessage[],
) {
  let latestMessage: TranscriptMessage | null = null;
  let messageCount = 0;

  for (const message of messages) {
    if (message.agent !== agent) continue;
    latestMessage = message;
    messageCount += 1;
  }

  const activeStream = activeStreams.find((stream) => stream.agent === agent) ?? null;

  return {
    agent,
    messageCount,
    latestMessage,
    activeStream,
  };
}

function AgentSummaryCard({
  agent,
  latestMessage,
  activeStream,
  messageCount,
}: {
  agent: string;
  latestMessage: TranscriptMessage | null;
  activeStream: StreamingMessage | null;
  messageCount: number;
}) {
  const colors = getAgentColors(agent);
  const preview = activeStream?.content || latestMessage?.content || 'Waiting for this agent to enter the round.';

  return (
    <div
      className="rounded-[20px] border px-4 py-4"
      style={{
        borderColor: 'color-mix(in srgb, var(--border) 80%, transparent)',
        background: 'color-mix(in srgb, var(--surface) 88%, transparent)',
      }}
      data-testid={`live-debate-agent-${agent}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`inline-flex h-2.5 w-2.5 rounded-full ${colors.tab ?? 'bg-acid-green'}`} />
          <span className={`text-xs font-semibold uppercase tracking-[0.18em] ${colors.text}`}>
            {agent}
          </span>
        </div>
        <span
          className="rounded-full px-2 py-1 text-[10px] uppercase tracking-[0.18em]"
          style={{
            color: activeStream ? 'var(--accent)' : 'var(--text-muted)',
            background: activeStream
              ? 'color-mix(in srgb, var(--accent) 14%, transparent)'
              : 'color-mix(in srgb, var(--surface) 78%, transparent)',
          }}
        >
          {activeStream ? 'typing' : `${messageCount} turns`}
        </span>
      </div>
      <p
        className="mt-3"
        style={{
          fontSize: '13px',
          lineHeight: 1.65,
          color: 'var(--text-muted)',
        }}
      >
        {preview}
      </p>
    </div>
  );
}

function TranscriptCard({
  message,
  accent,
  isStreaming = false,
}: {
  message: { agent: string; content: string; round?: number; role?: string; timestamp?: number };
  accent: string;
  isStreaming?: boolean;
}) {
  return (
    <article
      className="rounded-[24px] border px-5 py-4"
      style={{
        borderColor: `color-mix(in srgb, ${accent} 28%, transparent)`,
        background: isStreaming
          ? `linear-gradient(135deg, color-mix(in srgb, ${accent} 12%, var(--surface)) 0%, var(--surface) 100%)`
          : 'color-mix(in srgb, var(--surface) 92%, transparent)',
        boxShadow: isStreaming ? '0 22px 48px rgba(15, 23, 42, 0.12)' : 'none',
      }}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="text-[11px] font-semibold uppercase tracking-[0.2em]"
          style={{ color: accent }}
        >
          {message.agent}
        </span>
        {message.role ? (
          <span
            className="rounded-full px-2 py-1 text-[10px] uppercase tracking-[0.16em]"
            style={{
              background: `color-mix(in srgb, ${accent} 12%, transparent)`,
              color: accent,
            }}
          >
            {message.role}
          </span>
        ) : null}
        {typeof message.round === 'number' ? (
          <span
            className="rounded-full px-2 py-1 text-[10px] uppercase tracking-[0.16em]"
            style={{
              background: 'color-mix(in srgb, var(--surface) 78%, transparent)',
              color: 'var(--text-muted)',
            }}
          >
            Round {message.round}
          </span>
        ) : null}
        <span className="ml-auto text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {isStreaming ? 'live now' : message.timestamp ? new Date(message.timestamp * 1000).toLocaleTimeString() : 'update'}
        </span>
      </div>
      <p
        className="mt-3 whitespace-pre-wrap"
        style={{
          fontSize: '14px',
          lineHeight: 1.75,
          color: 'var(--text)',
        }}
      >
        {message.content}
        {isStreaming ? (
          <span
            aria-hidden="true"
            className="ml-1 inline-block h-4 w-2 animate-pulse"
            style={{ backgroundColor: accent }}
          />
        ) : null}
      </p>
    </article>
  );
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const sectionRef = useScrollReveal<HTMLElement>();
  const {
    events,
    connected,
    loaded,
    status: spectateStatus,
  } = useSpectate(undefined, undefined, { pollInterval: 3000, maxEvents: 80 });

  const discoverableDebates = deriveDiscoverableDebates(events, spectateStatus);
  const [selectedDebateId, setSelectedDebateId] = useState<string | null>(null);

  useEffect(() => {
    if (discoverableDebates.length === 0) {
      setSelectedDebateId(null);
      return;
    }

    setSelectedDebateId((current) => {
      if (current && discoverableDebates.some((debate) => debate.debate_id === current)) {
        return current;
      }
      return discoverableDebates[0].debate_id;
    });
  }, [discoverableDebates]);

  const activeDebateId = selectedDebateId;
  const selectedDebate = discoverableDebates.find((debate) => debate.debate_id === activeDebateId) ?? null;

  const {
    status,
    error,
    task,
    agents,
    messages,
    streamingMessages,
    reconnectAttempt,
    connectionQuality,
    isPolling,
  } = useDebateWebSocket({
    debateId: activeDebateId ?? '__landing_live_demo__',
    enabled: Boolean(activeDebateId),
    wsUrl: WS_URL,
  });

  const liveStatus = STREAM_STATUS[status];
  const activeStreams = Array.from(streamingMessages.values())
    .filter((stream) => stream.content.trim().length > 0)
    .sort((left, right) => right.startTime - left.startTime);
  const transcriptMessages = messages
    .filter((message) => message.content.trim().length > 0 && message.role !== 'system')
    .slice(-6);
  const visibleAgents = agents.length > 0
    ? agents
    : Array.from(new Set([
      ...messages.map((message) => message.agent).filter(Boolean),
      ...activeStreams.map((stream) => stream.agent).filter(Boolean),
    ]));
  const agentSummaries = visibleAgents.map((agent) => createAgentSummary(agent, messages, activeStreams));

  return (
    <section
      ref={sectionRef}
      className="px-4 animate-on-scroll"
      id="live-debate"
      style={{
        paddingTop: '120px',
        paddingBottom: '120px',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-6xl mx-auto" data-testid="live-debate-section">
        <p
          className="text-center uppercase tracking-widest"
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)', marginBottom: '20px' }}
        >
          {isDark ? '> WATCH IT LIVE' : 'WATCH IT LIVE'}
        </p>
        <div className="mx-auto max-w-2xl text-center" style={{ marginBottom: '48px' }}>
          <p
            style={{
              fontSize: isDark ? '30px' : '34px',
              lineHeight: 1.15,
              color: 'var(--text)',
              fontFamily: 'var(--font-landing)',
              marginBottom: '16px',
            }}
          >
            Visitors can watch a live Aragora debate as it unfolds.
          </p>
          <p
            style={{
              fontSize: isDark ? '15px' : '17px',
              color: 'var(--text-muted)',
              lineHeight: 1.8,
            }}
          >
            The landing page now follows the current public debate feed, discovers a live debate, and
            subscribes over WebSocket so arguments update as agents speak.
          </p>
        </div>

        <div
          style={{
            background: 'linear-gradient(180deg, color-mix(in srgb, var(--surface) 92%, transparent) 0%, color-mix(in srgb, var(--bg) 88%, transparent) 100%)',
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
            style={{ padding: '18px 22px', borderBottom: '1px solid var(--border)' }}
          >
            <span
              className="font-bold px-2 py-0.5 uppercase tracking-wider"
              style={{
                fontSize: '10px',
                backgroundColor: connected ? 'var(--accent)' : 'var(--surface)',
                color: connected ? 'var(--bg)' : 'var(--text-muted)',
                borderRadius: 'var(--radius-button)',
                border: connected ? 'none' : '1px solid var(--border)',
              }}
            >
              {connected ? 'Live feed connected' : 'Bridge unreachable'}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {selectedDebate ? `Debate ${selectedDebate.debate_id}` : 'Scanning for a public live debate'}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {selectedDebate
                ? `${selectedDebate.recent_event_count} recent events · last seen ${formatRelativeAge(selectedDebate.last_event_at)}`
                : loaded
                  ? 'No discoverable debate ID right now'
                  : 'Checking bridge readiness'}
            </span>
          </div>

          {!loaded ? (
            <div
              className="grid gap-6 lg:grid-cols-[1.5fr_0.9fr]"
              style={{ padding: '28px 22px' }}
            >
              <div
                style={{
                  padding: '28px',
                  borderRadius: '24px',
                  border: '1px solid var(--border)',
                  background: 'color-mix(in srgb, var(--surface) 88%, transparent)',
                }}
              >
                <div className="flex items-center gap-3" style={{ marginBottom: '18px' }}>
                  <div
                    className="h-3 w-3 animate-pulse rounded-full"
                    style={{ backgroundColor: 'var(--accent)' }}
                  />
                  <span style={{ fontSize: '12px', letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--accent)' }}>
                    Connecting to live debate stream
                  </span>
                </div>
                <p
                  style={{ fontSize: '15px', color: 'var(--text)', lineHeight: '1.8' }}
                >
                  Aragora is checking the spectate bridge for an active public debate and will open the first
                  discoverable stream automatically.
                </p>
              </div>
              <div className="grid gap-4">
                {[0, 1, 2].map((item) => (
                  <div
                    key={item}
                    style={{
                      height: '104px',
                      borderRadius: '20px',
                      border: '1px solid var(--border)',
                      background: 'color-mix(in srgb, var(--surface) 82%, transparent)',
                    }}
                  />
                ))}
              </div>
            </div>
          ) : !activeDebateId ? (
            <div style={{ padding: '28px 22px' }} data-testid="live-debate-empty">
              <div
                style={{
                  borderRadius: '24px',
                  border: '1px solid var(--border)',
                  background: 'color-mix(in srgb, var(--surface) 88%, transparent)',
                  padding: '28px',
                }}
              >
                <p
                  style={{
                    fontSize: '12px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.18em',
                    color: 'var(--accent)',
                    marginBottom: '14px',
                  }}
                >
                  Honest fallback
                </p>
                <p
                  style={{
                    fontSize: '22px',
                    lineHeight: 1.35,
                    color: 'var(--text)',
                    marginBottom: '12px',
                  }}
                >
                  No live debate is discoverable right now.
                </p>
                <p
                  style={{
                    fontSize: '15px',
                    lineHeight: 1.8,
                    color: 'var(--text-muted)',
                    maxWidth: '720px',
                  }}
                >
                  The landing page now checks the public spectate bridge and refuses to invent a stream when there
                  is no active debate ID. As soon as a public debate appears, this section will lock onto it and
                  render live arguments over WebSocket.
                </p>
                <div className="flex flex-wrap gap-3" style={{ marginTop: '22px' }}>
                  <Link
                    href="/spectate"
                    className="text-sm font-semibold transition-all hover:scale-[1.02]"
                    style={{
                      border: '1px solid var(--accent)',
                      borderRadius: 'var(--radius-button)',
                      color: 'var(--accent)',
                      padding: '14px 22px',
                    }}
                  >
                    Open spectate mode
                  </Link>
                  <button
                    onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                    className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-button)',
                      color: 'var(--text)',
                      backgroundColor: 'transparent',
                      padding: '14px 22px',
                    }}
                  >
                    Run your own debate
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid gap-6 lg:grid-cols-[1.4fr_0.9fr]" style={{ padding: '28px 22px' }}>
              <div>
                <div
                  style={{
                    borderRadius: '24px',
                    border: '1px solid var(--border)',
                    background: 'color-mix(in srgb, var(--surface) 88%, transparent)',
                    padding: '24px',
                    marginBottom: '20px',
                  }}
                >
                  <div className="flex flex-wrap items-start gap-3" style={{ marginBottom: '16px' }}>
                    <span
                      className="rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.18em]"
                      style={{
                        background: `color-mix(in srgb, ${liveStatus.tone} 12%, transparent)`,
                        color: liveStatus.tone,
                      }}
                    >
                      {liveStatus.label}
                    </span>
                    <span
                      className="rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.18em]"
                      style={{
                        background: 'color-mix(in srgb, var(--surface) 78%, transparent)',
                        color: 'var(--text-muted)',
                      }}
                    >
                      {visibleAgents.length} agents
                    </span>
                    {isPolling ? (
                      <span
                        className="rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.18em]"
                        style={{
                          background: 'rgba(245, 158, 11, 0.12)',
                          color: '#f59e0b',
                        }}
                      >
                        HTTP fallback
                      </span>
                    ) : null}
                    <span className="ml-auto text-[11px]" style={{ color: 'var(--text-muted)' }}>
                      {connectionQuality?.avgLatencyMs ? `${Math.round(connectionQuality.avgLatencyMs)} ms latency` : `${reconnectAttempt} reconnect attempts`}
                    </span>
                  </div>

                  <p
                    style={{
                      fontSize: isDark ? '26px' : '30px',
                      lineHeight: 1.18,
                      color: 'var(--text)',
                      marginBottom: '16px',
                    }}
                  >
                    {task || 'Waiting for the first debate prompt...'}
                  </p>

                  <div className="flex flex-wrap gap-2">
                    {discoverableDebates.map((debate) => {
                      const isSelected = debate.debate_id === activeDebateId;
                      return (
                        <button
                          key={debate.debate_id}
                          type="button"
                          onClick={() => setSelectedDebateId(debate.debate_id)}
                          className="rounded-full px-3 py-2 text-[11px] uppercase tracking-[0.18em] transition-all"
                          style={{
                            border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                            background: isSelected
                              ? 'color-mix(in srgb, var(--accent) 14%, transparent)'
                              : 'transparent',
                            color: isSelected ? 'var(--accent)' : 'var(--text-muted)',
                          }}
                          data-testid={`live-debate-picker-${debate.debate_id}`}
                        >
                          {debate.debate_id.slice(0, 18)}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {error ? (
                  <div
                    style={{
                      borderRadius: '20px',
                      border: '1px solid rgba(220, 38, 38, 0.2)',
                      background: 'rgba(220, 38, 38, 0.08)',
                      color: '#991b1b',
                      padding: '16px 18px',
                      marginBottom: '20px',
                    }}
                  >
                    {error}
                  </div>
                ) : null}

                <div
                  className="space-y-4"
                  style={{
                    maxHeight: '720px',
                    overflowY: 'auto',
                    paddingRight: '4px',
                  }}
                  data-testid="live-debate-transcript"
                >
                  {activeStreams.map((stream) => {
                    const colors = getAgentColors(stream.agent);
                    const accent = colors.text.includes('cyan')
                      ? '#06b6d4'
                      : colors.text.includes('gold')
                        ? '#d97706'
                        : colors.text.includes('crimson')
                          ? '#dc2626'
                          : colors.text.includes('purple')
                            ? '#9333ea'
                            : '#10b981';

                    return (
                      <TranscriptCard
                        key={`${stream.agent}-${stream.taskId}-streaming`}
                        message={{
                          agent: stream.agent,
                          content: stream.content,
                          role: stream.reasoningPhase,
                        }}
                        accent={accent}
                        isStreaming={true}
                      />
                    );
                  })}

                  {transcriptMessages.length === 0 && activeStreams.length === 0 ? (
                    <div
                      style={{
                        borderRadius: '24px',
                        border: '1px dashed var(--border)',
                        padding: '28px',
                        color: 'var(--text-muted)',
                        background: 'color-mix(in srgb, var(--surface) 86%, transparent)',
                      }}
                    >
                      Waiting for agents to publish the first argument.
                    </div>
                  ) : null}

                  {transcriptMessages.map((message, index) => {
                    const colors = getAgentColors(message.agent || `agent-${index}`);
                    const accent = colors.text.includes('cyan')
                      ? '#06b6d4'
                      : colors.text.includes('gold')
                        ? '#d97706'
                        : colors.text.includes('crimson')
                          ? '#dc2626'
                          : colors.text.includes('purple')
                            ? '#9333ea'
                            : '#10b981';

                    return (
                      <TranscriptCard
                        key={`${message.agent}-${message.timestamp ?? index}-${message.round ?? 0}`}
                        message={message}
                        accent={accent}
                      />
                    );
                  })}
                </div>
              </div>

              <div className="space-y-4">
                <div
                  style={{
                    borderRadius: '24px',
                    border: '1px solid var(--border)',
                    background: 'color-mix(in srgb, var(--surface) 88%, transparent)',
                    padding: '24px',
                  }}
                >
                  <p
                    style={{
                      fontSize: '12px',
                      textTransform: 'uppercase',
                      letterSpacing: '0.18em',
                      color: 'var(--text-muted)',
                      marginBottom: '14px',
                    }}
                  >
                    Agent positions
                  </p>
                  <div className="space-y-3">
                    {agentSummaries.map((summary) => (
                      <AgentSummaryCard
                        key={summary.agent}
                        agent={summary.agent}
                        latestMessage={summary.latestMessage}
                        activeStream={summary.activeStream}
                        messageCount={summary.messageCount}
                      />
                    ))}
                  </div>
                </div>

                <div
                  style={{
                    borderRadius: '24px',
                    border: '1px solid var(--border)',
                    background: 'color-mix(in srgb, var(--surface) 88%, transparent)',
                    padding: '24px',
                  }}
                >
                  <p
                    style={{
                      fontSize: '12px',
                      textTransform: 'uppercase',
                      letterSpacing: '0.18em',
                      color: 'var(--text-muted)',
                      marginBottom: '14px',
                    }}
                  >
                    Stream telemetry
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <div
                      style={{
                        borderRadius: '18px',
                        background: 'color-mix(in srgb, var(--bg) 52%, transparent)',
                        padding: '14px',
                      }}
                    >
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Bridge state</div>
                      <div style={{ fontSize: '16px', color: 'var(--text)' }}>
                        {connected ? 'reachable' : 'offline'}
                      </div>
                    </div>
                    <div
                      style={{
                        borderRadius: '18px',
                        background: 'color-mix(in srgb, var(--bg) 52%, transparent)',
                        padding: '14px',
                      }}
                    >
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Last bridge event</div>
                      <div style={{ fontSize: '16px', color: 'var(--text)' }}>
                        {selectedDebate ? formatRelativeAge(selectedDebate.last_event_at) : 'n/a'}
                      </div>
                    </div>
                    <div
                      style={{
                        borderRadius: '18px',
                        background: 'color-mix(in srgb, var(--bg) 52%, transparent)',
                        padding: '14px',
                      }}
                    >
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Transcript turns</div>
                      <div style={{ fontSize: '16px', color: 'var(--text)' }}>{messages.length}</div>
                    </div>
                    <div
                      style={{
                        borderRadius: '18px',
                        background: 'color-mix(in srgb, var(--bg) 52%, transparent)',
                        padding: '14px',
                      }}
                    >
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Streaming agents</div>
                      <div style={{ fontSize: '16px', color: 'var(--text)' }}>{activeStreams.length}</div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3" style={{ marginTop: '22px' }}>
                    <Link
                      href={`/spectate/${activeDebateId}`}
                      className="text-sm font-semibold transition-all hover:scale-[1.02]"
                      style={{
                        border: '1px solid var(--accent)',
                        borderRadius: 'var(--radius-button)',
                        color: 'var(--accent)',
                        padding: '14px 22px',
                      }}
                    >
                      Open full spectator view
                    </Link>
                    <button
                      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                      className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
                      style={{
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-button)',
                        color: 'var(--text)',
                        backgroundColor: 'transparent',
                        padding: '14px 22px',
                      }}
                    >
                      Run your own debate
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
