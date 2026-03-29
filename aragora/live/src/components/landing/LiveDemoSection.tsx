'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { WS_URL } from '@/config';
import { useTheme } from '@/context/ThemeContext';
import { useSpectate, type SpectateEvent } from '@/hooks/useSpectate';

type LiveConnectionState =
  | 'discovering'
  | 'connecting'
  | 'live'
  | 'disconnected'
  | 'error';

interface LandingLiveEvent {
  signature: string;
  debateId: string;
  eventType: string;
  timestampMs: number;
  agentName: string | null;
  roundNumber: number | null;
  details: string;
}

interface SpectateSocketMessage {
  type?: unknown;
  timestamp?: unknown;
  debate_id?: unknown;
  agent?: unknown;
  details?: unknown;
  round?: unknown;
  task?: unknown;
  agents?: unknown;
}

const EVENT_STYLES: Record<string, { label: string; accent: string; background: string }> = {
  debate_start: { label: 'Debate Start', accent: 'var(--accent)', background: 'var(--accent-glow)' },
  round_start: { label: 'Round Start', accent: 'var(--accent)', background: 'var(--accent-glow)' },
  proposal: { label: 'Proposal', accent: '#2563eb', background: 'rgba(37, 99, 235, 0.12)' },
  critique: { label: 'Critique', accent: '#dc2626', background: 'rgba(220, 38, 38, 0.12)' },
  refine: { label: 'Refine', accent: '#7c3aed', background: 'rgba(124, 58, 237, 0.12)' },
  vote: { label: 'Vote', accent: '#d97706', background: 'rgba(217, 119, 6, 0.12)' },
  judge: { label: 'Judge', accent: '#ca8a04', background: 'rgba(202, 138, 4, 0.12)' },
  consensus: { label: 'Consensus', accent: '#059669', background: 'rgba(5, 150, 105, 0.12)' },
  debate_end: { label: 'Debate End', accent: '#059669', background: 'rgba(5, 150, 105, 0.12)' },
};

function parseTimestamp(value: string | number | null | undefined): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value > 1_000_000_000_000 ? value : value * 1000;
  }

  if (typeof value === 'string' && value.trim()) {
    const parsed = Date.parse(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  return Date.now();
}

function textValue(value: unknown): string {
  if (typeof value === 'string') return value.trim();
  if (value == null) return '';

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function defaultEventCopy(eventType: string): string {
  switch (eventType) {
    case 'round_start':
      return 'A new round has started.';
    case 'vote':
      return 'An agent cast a vote.';
    case 'consensus':
      return 'The panel reached consensus.';
    case 'debate_end':
      return 'The live debate finished.';
    default:
      return 'A live debate event arrived.';
  }
}

function eventSignature(
  debateId: string,
  eventType: string,
  timestampMs: number,
  agentName: string | null,
  roundNumber: number | null,
  details: string,
): string {
  return [
    debateId,
    eventType,
    Math.round(timestampMs / 1000),
    agentName ?? '',
    roundNumber ?? '',
    details,
  ].join('::');
}

function normalizeRecentEvent(event: SpectateEvent): LandingLiveEvent | null {
  if (!event.debate_id) return null;

  const details = textValue(event.data?.details) || defaultEventCopy(event.event_type);
  const timestampMs = parseTimestamp(event.timestamp);

  return {
    signature: eventSignature(
      event.debate_id,
      event.event_type,
      timestampMs,
      event.agent_name,
      event.round_number,
      details,
    ),
    debateId: event.debate_id,
    eventType: event.event_type,
    timestampMs,
    agentName: event.agent_name,
    roundNumber: event.round_number,
    details,
  };
}

function normalizeSocketEvent(
  debateId: string,
  payload: SpectateSocketMessage,
): LandingLiveEvent | null {
  if (typeof payload.type !== 'string' || payload.type === 'metadata') {
    return null;
  }

  const resolvedDebateId =
    typeof payload.debate_id === 'string' && payload.debate_id
      ? payload.debate_id
      : debateId;
  const timestampMs = parseTimestamp(payload.timestamp);
  const agentName = typeof payload.agent === 'string' && payload.agent ? payload.agent : null;
  const roundNumber = typeof payload.round === 'number' ? payload.round : null;
  const details = textValue(payload.details) || defaultEventCopy(payload.type);

  return {
    signature: eventSignature(
      resolvedDebateId,
      payload.type,
      timestampMs,
      agentName,
      roundNumber,
      details,
    ),
    debateId: resolvedDebateId,
    eventType: payload.type,
    timestampMs,
    agentName,
    roundNumber,
    details,
  };
}

function mergeEvents(...groups: LandingLiveEvent[][]): LandingLiveEvent[] {
  const deduped = new Map<string, LandingLiveEvent>();

  for (const group of groups) {
    for (const event of group) {
      deduped.set(event.signature, event);
    }
  }

  return Array.from(deduped.values()).sort((left, right) => left.timestampMs - right.timestampMs);
}

function formatClock(timestampMs: number): string {
  return new Date(timestampMs).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

function formatRelativeAge(timestampMs: number): string {
  const ageSeconds = Math.max(0, Math.round((Date.now() - timestampMs) / 1000));
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  return `${Math.round(ageMinutes / 60)}h ago`;
}

function socketStatusLabel(state: LiveConnectionState): string {
  switch (state) {
    case 'live':
      return 'Live socket';
    case 'connecting':
      return 'Connecting';
    case 'error':
      return 'Reconnect pending';
    case 'disconnected':
      return 'Polling recent events';
    case 'discovering':
    default:
      return 'Scanning bridge';
  }
}

function socketStatusTone(state: LiveConnectionState): string {
  switch (state) {
    case 'live':
      return 'var(--accent)';
    case 'connecting':
      return '#2563eb';
    case 'error':
      return '#dc2626';
    case 'disconnected':
      return '#d97706';
    case 'discovering':
    default:
      return 'var(--text-muted)';
  }
}

function socketStatusBackground(state: LiveConnectionState): string {
  switch (state) {
    case 'live':
      return 'var(--accent-glow)';
    case 'connecting':
      return 'rgba(37, 99, 235, 0.12)';
    case 'error':
      return 'rgba(220, 38, 38, 0.12)';
    case 'disconnected':
      return 'rgba(217, 119, 6, 0.12)';
    case 'discovering':
    default:
      return 'rgba(148, 163, 184, 0.12)';
  }
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const { status, loaded, events } = useSpectate(undefined, undefined, {
    pollInterval: 4000,
    maxEvents: 40,
  });
  const isDark = theme === 'dark';
  const recentEventCount = status?.recent_event_count ?? 0;
  const recentActivityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const activityWindowMinutes = Math.max(1, Math.round(recentActivityWindowSeconds / 60));
  const activityAgeSeconds = status?.activity_age_seconds;
  const [liveEvents, setLiveEvents] = useState<LandingLiveEvent[]>([]);
  const [liveTask, setLiveTask] = useState('');
  const [liveAgents, setLiveAgents] = useState<string[]>([]);
  const [connectionState, setConnectionState] = useState<LiveConnectionState>('discovering');
  const [socketError, setSocketError] = useState<string | null>(null);

  const activeDebateId = useMemo(() => {
    const latestKnownDebate = status?.live_debates?.[0]?.debate_id;
    if (latestKnownDebate) return latestKnownDebate;

    const latestDebateEvent = [...events]
      .filter((event) => event.debate_id)
      .sort((left, right) => parseTimestamp(right.timestamp) - parseTimestamp(left.timestamp))[0];

    return latestDebateEvent?.debate_id ?? null;
  }, [events, status?.live_debates]);

  const recentLiveEvents = useMemo(
    () =>
      events
        .filter((event) => event.debate_id === activeDebateId)
        .map(normalizeRecentEvent)
        .filter((event): event is LandingLiveEvent => event !== null),
    [activeDebateId, events],
  );

  const visibleEvents = useMemo(
    () => mergeEvents(recentLiveEvents, liveEvents).slice(-10),
    [liveEvents, recentLiveEvents],
  );

  const derivedTask = useMemo(() => {
    if (liveTask) return liveTask;

    for (const event of [...events].reverse()) {
      if (event.debate_id !== activeDebateId) continue;
      const task = event.data?.task;
      if (typeof task === 'string' && task.trim()) {
        return task.trim();
      }
    }

    return activeDebateId
      ? 'Agents are actively debating in public right now.'
      : '';
  }, [activeDebateId, events, liveTask]);

  const derivedAgents = useMemo(() => {
    const agents = new Set<string>(liveAgents);

    for (const event of events) {
      if (event.debate_id !== activeDebateId) continue;
      if (event.agent_name) {
        agents.add(event.agent_name);
      }
      const eventAgents = event.data?.agents;
      if (Array.isArray(eventAgents)) {
        for (const agent of eventAgents) {
          if (typeof agent === 'string' && agent.trim()) {
            agents.add(agent);
          }
        }
      }
    }

    return Array.from(agents);
  }, [activeDebateId, events, liveAgents]);

  useEffect(() => {
    if (!activeDebateId) {
      setLiveEvents([]);
      setLiveTask('');
      setLiveAgents([]);
      setSocketError(null);
      setConnectionState(loaded ? 'disconnected' : 'discovering');
      return;
    }

    setLiveEvents([]);
    setLiveTask('');
    setLiveAgents([]);
    setSocketError(null);

    if (typeof WebSocket === 'undefined') {
      setConnectionState('disconnected');
      return;
    }

    const wsBaseUrl = WS_URL.replace(/\/ws\/?$/, '');
    const socketUrl = `${wsBaseUrl}/spectate/${encodeURIComponent(activeDebateId)}`;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (stopped) return;

      setConnectionState('connecting');
      socket = new WebSocket(socketUrl);

      socket.onopen = () => {
        setConnectionState('live');
        setSocketError(null);
      };

      socket.onmessage = (message) => {
        let payload: SpectateSocketMessage;

        try {
          payload = JSON.parse(message.data as string) as SpectateSocketMessage;
        } catch {
          return;
        }

        if (payload.type === 'metadata') {
          if (typeof payload.task === 'string' && payload.task.trim()) {
            setLiveTask(payload.task.trim());
          }
          if (Array.isArray(payload.agents)) {
            setLiveAgents(
              payload.agents.filter(
                (agent): agent is string => typeof agent === 'string' && agent.trim(),
              ),
            );
          }
          return;
        }

        const normalized = normalizeSocketEvent(activeDebateId, payload);
        if (!normalized) return;

        setLiveEvents((current) => mergeEvents(current, [normalized]));
      };

      socket.onerror = () => {
        if (stopped) return;
        setConnectionState('error');
        setSocketError('Live stream dropped. Falling back to recent events while reconnecting.');
      };

      socket.onclose = () => {
        if (stopped) return;
        setConnectionState('disconnected');
        reconnectTimer = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      stopped = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [activeDebateId, loaded]);

  let bridgeBadge = 'Checking public bridge';
  let bridgeSummary = 'Checking public live bridge before showing recent activity.';

  if (loaded) {
    if (!status?.active) {
      bridgeBadge = 'Bridge offline';
      bridgeSummary = 'Public spectate is offline right now, so this panel waits instead of inventing a debate.';
    } else if (recentEventCount > 0) {
      bridgeBadge = 'Bridge active';
      bridgeSummary = `${recentEventCount} recent event${recentEventCount === 1 ? '' : 's'} in the last ${activityWindowMinutes} minute${activityWindowMinutes === 1 ? '' : 's'}.`;
    } else {
      bridgeBadge = 'Bridge ready';
      bridgeSummary = 'Public spectate is online, but no recent live debate activity is visible yet.';
    }
  }

  let activityAgeLabel: string | null = null;
  if (typeof activityAgeSeconds === 'number') {
    if (activityAgeSeconds < 60) {
      activityAgeLabel = `Last activity ${Math.round(activityAgeSeconds)}s ago`;
    } else if (activityAgeSeconds < 3600) {
      activityAgeLabel = `Last activity ${Math.round(activityAgeSeconds / 60)}m ago`;
    } else {
      activityAgeLabel = `Last activity ${Math.round(activityAgeSeconds / 3600)}h ago`;
    }
  }

  return (
    <section
      data-testid="live-demo-section"
      className="px-4"
      style={{
        paddingTop: '120px',
        paddingBottom: '120px',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-4xl mx-auto">
        <p
          className="text-center uppercase tracking-widest"
          style={{
            fontSize: isDark ? '16px' : '18px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
            marginBottom: '20px',
          }}
        >
          {isDark ? '> SEE IT IN ACTION' : 'SEE IT IN ACTION'}
        </p>
        <p
          className="text-center"
          style={{
            fontSize: isDark ? '16px' : '18px',
            color: 'var(--text)',
            fontFamily: 'var(--font-landing)',
            marginBottom: '48px',
          }}
        >
          Watch a real public debate as agents argue, critique, and converge in real time.
        </p>

        <div
          data-testid="live-demo-bridge-status"
          className="flex flex-wrap items-center gap-3"
          style={{
            backgroundColor: 'var(--surface)',
            borderRadius: 'var(--radius-card)',
            border: '1px solid var(--border)',
            boxShadow: 'var(--shadow-card)',
            padding: '16px 20px',
            margin: '0 24px 20px',
          }}
        >
          <span
            className="font-bold px-2 py-0.5 uppercase tracking-wider"
            style={{
              fontSize: '10px',
              backgroundColor: status?.active ? 'var(--accent)' : 'var(--border)',
              color: status?.active ? 'var(--bg)' : 'var(--text)',
              borderRadius: 'var(--radius-button)',
            }}
          >
            {bridgeBadge}
          </span>
          <span
            style={{
              fontSize: isDark ? '13px' : '14px',
              color: 'var(--text)',
              fontFamily: 'var(--font-landing)',
            }}
          >
            {bridgeSummary}
          </span>
          {activityAgeLabel ? (
            <span
              className="ml-auto"
              style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-landing)',
              }}
            >
              {activityAgeLabel}
            </span>
          ) : null}
        </div>

        <div
          data-testid="landing-live-feed"
          style={{
            backgroundColor: 'var(--surface)',
            borderRadius: 'var(--radius-card)',
            border: '1px solid var(--border)',
            borderTopColor: 'var(--accent)',
            borderTopWidth: '3px',
            boxShadow: 'var(--shadow-card)',
            overflow: 'hidden',
            margin: '0 24px',
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
                backgroundColor: 'var(--accent)',
                color: 'var(--bg)',
                borderRadius: 'var(--radius-button)',
              }}
            >
              {activeDebateId ? 'Live public debate' : 'Waiting for live debate'}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {derivedTask || 'This panel switches to the most recent public debate as soon as one starts.'}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {socketStatusLabel(connectionState)}
            </span>
          </div>

          <div style={{ padding: '20px' }}>
            <div className="flex flex-wrap items-center gap-3" style={{ marginBottom: '16px' }}>
              <span
                className="font-bold px-2 py-0.5 uppercase tracking-wider"
                style={{
                  fontSize: '10px',
                  color: socketStatusTone(connectionState),
                  backgroundColor: socketStatusBackground(connectionState),
                  borderRadius: 'var(--radius-button)',
                }}
              >
                {activeDebateId ? 'Live now' : 'Standby'}
              </span>
              {activeDebateId ? (
                <span
                  style={{
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-landing)',
                  }}
                >
                  Debate {activeDebateId.slice(0, 12)} • {visibleEvents.length} event{visibleEvents.length === 1 ? '' : 's'} • Last update{' '}
                  {visibleEvents.length > 0
                    ? formatRelativeAge(visibleEvents[visibleEvents.length - 1].timestampMs)
                    : 'waiting'}
                </span>
              ) : (
                <span
                  style={{
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-landing)',
                  }}
                >
                  Visitors see real proposals, critiques, votes, and consensus events here as soon as a public debate starts.
                </span>
              )}
            </div>

            {derivedAgents.length > 0 && (
              <div className="flex flex-wrap gap-2" style={{ marginBottom: '16px' }}>
                {derivedAgents.map((agent) => (
                  <span
                    key={agent}
                    className="px-2 py-1"
                    style={{
                      fontSize: '10px',
                      color: 'var(--text)',
                      backgroundColor: 'var(--bg)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-button)',
                      fontFamily: 'var(--font-landing)',
                    }}
                  >
                    {agent}
                  </span>
                ))}
              </div>
            )}

            {socketError && (
              <div
                style={{
                  marginBottom: '16px',
                  padding: '12px 14px',
                  border: '1px solid rgba(220, 38, 38, 0.3)',
                  backgroundColor: 'rgba(220, 38, 38, 0.08)',
                  color: '#dc2626',
                  borderRadius: 'var(--radius-card)',
                  fontSize: '12px',
                  fontFamily: 'var(--font-landing)',
                }}
              >
                {socketError}
              </div>
            )}

            {activeDebateId && visibleEvents.length > 0 ? (
              <div
                aria-live="polite"
                className="space-y-3"
                style={{ maxHeight: '520px', overflowY: 'auto', paddingRight: '4px' }}
              >
                {visibleEvents.map((event) => {
                  const eventStyle = EVENT_STYLES[event.eventType] ?? {
                    label: event.eventType.replace(/_/g, ' '),
                    accent: 'var(--text-muted)',
                    background: 'rgba(148, 163, 184, 0.12)',
                  };

                  return (
                    <div
                      key={event.signature}
                      style={{
                        padding: '16px',
                        border: '1px solid var(--border)',
                        borderLeft: `4px solid ${eventStyle.accent}`,
                        borderRadius: 'var(--radius-card)',
                        backgroundColor: eventStyle.background,
                      }}
                    >
                      <div className="flex flex-wrap items-center gap-2" style={{ marginBottom: '10px' }}>
                        <span
                          className="font-bold px-2 py-0.5 uppercase tracking-wider"
                          style={{
                            fontSize: '10px',
                            color: eventStyle.accent,
                            backgroundColor: 'rgba(255, 255, 255, 0.55)',
                            borderRadius: 'var(--radius-button)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {eventStyle.label}
                        </span>
                        {event.roundNumber !== null && (
                          <span
                            style={{
                              fontSize: '10px',
                              color: 'var(--text-muted)',
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            Round {event.roundNumber}
                          </span>
                        )}
                        {event.agentName && (
                          <span
                            style={{
                              fontSize: '10px',
                              color: eventStyle.accent,
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {event.agentName}
                          </span>
                        )}
                        <span
                          className="ml-auto"
                          style={{
                            fontSize: '10px',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {formatClock(event.timestampMs)}
                        </span>
                      </div>
                      <p
                        style={{
                          fontSize: isDark ? '13px' : '14px',
                          color: 'var(--text)',
                          fontFamily: 'var(--font-landing)',
                          lineHeight: '1.7',
                        }}
                      >
                        {event.details}
                      </p>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div
                style={{
                  padding: '24px',
                  border: '1px dashed var(--border)',
                  borderRadius: 'var(--radius-card)',
                  backgroundColor: 'var(--bg)',
                }}
              >
                <p
                  style={{
                    fontSize: '14px',
                    color: 'var(--text)',
                    fontFamily: 'var(--font-landing)',
                    marginBottom: '10px',
                  }}
                >
                  No public debate is discoverable right now.
                </p>
                <p
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-landing)',
                    lineHeight: '1.7',
                  }}
                >
                  This section now stays truthful: it waits for real public debate events instead of fabricating a transcript.
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4 mt-12">
          {activeDebateId && (
            <Link
              href={`/spectate/${encodeURIComponent(activeDebateId)}`}
              className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
              style={{
                display: 'inline-block',
                border: '1px solid var(--accent)',
                borderRadius: 'var(--radius-button)',
                color: 'var(--bg)',
                backgroundColor: 'var(--accent)',
                fontFamily: 'var(--font-landing)',
                padding: '18px 32px',
              }}
            >
              Open full spectate
            </Link>
          )}
          <Link
            href="/demo"
            className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
            style={{
              display: 'inline-block',
              border: '1px solid var(--accent)',
              borderRadius: 'var(--radius-button)',
              color: activeDebateId ? 'var(--accent)' : 'var(--bg)',
              backgroundColor: activeDebateId ? 'transparent' : 'var(--accent)',
              fontFamily: 'var(--font-landing)',
              padding: '18px 32px',
            }}
          >
            Run your own debate
          </Link>
        </div>
      </div>
    </section>
  );
}
