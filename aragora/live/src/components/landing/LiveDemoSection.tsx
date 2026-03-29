'use client';

import Link from 'next/link';
import { useMemo } from 'react';
import { useTheme } from '@/context/ThemeContext';
import { useSpectate, type SpectateEvent } from '@/hooks/useSpectate';

const DEMO_AGENTS = [
  {
    name: 'Strategic Analyst',
    accent: '#059669',
    content: 'Microservices make sense at your scale (50+ engineers), but only if you invest in service mesh and observability first. The organizational cost of splitting prematurely exceeds the technical debt of a well-structured monolith.',
  },
  {
    name: "Devil's Advocate",
    accent: '#dc2626',
    content: "The industry push toward microservices is survivorship bias. Most teams that succeed with them had strong platform engineering before the migration. Your team's current deployment cadence suggests the monolith isn't actually the bottleneck.",
  },
  {
    name: 'Implementation Expert',
    accent: '#2563eb',
    content: 'Start with the strangler fig pattern: extract the 2-3 domains with the highest change frequency first. Keep shared authentication and data access in the monolith until you have proven service boundaries.',
  },
];

const LIVE_EVENT_STYLES: Record<string, { label: string; accent: string; border: string }> = {
  proposal: {
    label: 'Opening case',
    accent: 'var(--accent)',
    border: 'rgba(34, 197, 94, 0.35)',
  },
  critique: {
    label: 'Counterpoint',
    accent: '#dc2626',
    border: 'rgba(220, 38, 38, 0.35)',
  },
  refine: {
    label: 'Revision',
    accent: '#2563eb',
    border: 'rgba(37, 99, 235, 0.35)',
  },
  vote: {
    label: 'Vote',
    accent: '#d97706',
    border: 'rgba(217, 119, 6, 0.35)',
  },
  judge: {
    label: 'Synthesis',
    accent: '#7c3aed',
    border: 'rgba(124, 58, 237, 0.35)',
  },
  consensus: {
    label: 'Consensus',
    accent: '#059669',
    border: 'rgba(5, 150, 105, 0.35)',
  },
  converged: {
    label: 'Converged',
    accent: '#059669',
    border: 'rgba(5, 150, 105, 0.35)',
  },
  convergence: {
    label: 'Convergence',
    accent: '#0891b2',
    border: 'rgba(8, 145, 178, 0.35)',
  },
  debate_start: {
    label: 'Debate start',
    accent: '#7c3aed',
    border: 'rgba(124, 58, 237, 0.35)',
  },
  round_start: {
    label: 'Round start',
    accent: '#0891b2',
    border: 'rgba(8, 145, 178, 0.35)',
  },
  system: {
    label: 'System',
    accent: 'var(--text-muted)',
    border: 'rgba(148, 163, 184, 0.3)',
  },
};

const STREAMABLE_EVENT_TYPES = new Set([
  'debate_start',
  'round_start',
  'proposal',
  'critique',
  'refine',
  'vote',
  'judge',
  'consensus',
  'converged',
  'convergence',
  'system',
]);

interface LiveDebateSnapshot {
  debateId: string;
  topic: string | null;
  agents: string[];
  round: number | null;
  recentEventCount: number;
  lastEventAt: string | null;
  events: SpectateEvent[];
}

function toEpochMs(timestamp: string | null | undefined): number | null {
  if (!timestamp) return null;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : parsed;
}

function formatRelativeAge(timestamp: string | null | undefined): string {
  const epochMs = toEpochMs(timestamp);
  if (epochMs === null) return 'now';

  const ageSeconds = Math.max(0, Math.round((Date.now() - epochMs) / 1000));
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  const ageHours = Math.round(ageMinutes / 60);
  return `${ageHours}h ago`;
}

function isRecentEvent(event: SpectateEvent, windowSeconds: number): boolean {
  const epochMs = toEpochMs(event.timestamp);
  if (epochMs === null) return false;
  return Date.now() - epochMs <= windowSeconds * 1000;
}

function extractEventDetails(event: SpectateEvent): string | null {
  const details = event.data?.details;
  if (typeof details === 'string' && details.trim()) {
    return details.trim();
  }

  const task = event.data?.task;
  if (event.event_type === 'debate_start' && typeof task === 'string' && task.trim()) {
    return task.trim();
  }

  if (event.event_type === 'round_start' && typeof event.round_number === 'number') {
    return `Round ${event.round_number} is underway.`;
  }

  if (event.event_type === 'convergence') {
    const metric = event.data?.metric;
    if (typeof metric === 'number') {
      return `Convergence moved to ${Math.round(metric * 100)}%.`;
    }
  }

  if (event.event_type === 'vote') return 'An agent cast a vote.';
  if (event.event_type === 'consensus' || event.event_type === 'converged') {
    return 'Agents reached a shared conclusion.';
  }
  if (event.event_type === 'judge') return 'The judge is synthesizing the strongest arguments.';

  return null;
}

function pickActiveLiveDebate(events: SpectateEvent[], windowSeconds: number): LiveDebateSnapshot | null {
  const recentEvents = events.filter(
    (event) => Boolean(event.debate_id) && isRecentEvent(event, windowSeconds),
  );
  const grouped = new Map<string, SpectateEvent[]>();

  for (const event of recentEvents) {
    const debateId = event.debate_id;
    if (!debateId) continue;
    const debateEvents = grouped.get(debateId) ?? [];
    debateEvents.push(event);
    grouped.set(debateId, debateEvents);
  }

  const candidates = Array.from(grouped.entries())
    .map(([debateId, debateEvents]) => {
      const streamableEvents = debateEvents
        .filter(
          (event) =>
            STREAMABLE_EVENT_TYPES.has(event.event_type) || Boolean(extractEventDetails(event)),
        )
        .sort(
          (left, right) =>
            (toEpochMs(left.timestamp) ?? 0) - (toEpochMs(right.timestamp) ?? 0),
        );

      const topic =
        [...debateEvents]
          .reverse()
          .map((event) => event.data?.task)
          .find((task): task is string => typeof task === 'string' && task.trim().length > 0) ??
        null;

      const agents = Array.from(
        new Set(
          debateEvents
            .map((event) => event.agent_name)
            .filter((agentName): agentName is string => Boolean(agentName)),
        ),
      );

      const round = debateEvents.reduce<number | null>((highestRound, event) => {
        if (typeof event.round_number !== 'number') return highestRound;
        if (highestRound === null) return event.round_number;
        return Math.max(highestRound, event.round_number);
      }, null);

      const lastEventAt = debateEvents[debateEvents.length - 1]?.timestamp ?? null;

      return {
        debateId,
        topic,
        agents,
        round,
        recentEventCount: debateEvents.length,
        lastEventAt,
        events: streamableEvents.slice(-6),
      };
    })
    .filter((candidate) => candidate.events.length > 0)
    .sort((left, right) => {
      if (right.recentEventCount !== left.recentEventCount) {
        return right.recentEventCount - left.recentEventCount;
      }
      return (toEpochMs(right.lastEventAt) ?? 0) - (toEpochMs(left.lastEventAt) ?? 0);
    });

  return candidates[0] ?? null;
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const { status, loaded, events: recentEvents } = useSpectate(undefined, undefined, {
    pollInterval: 3000,
    maxEvents: 80,
  });
  const activityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const publicLiveDebate = useMemo(
    () => pickActiveLiveDebate(recentEvents, activityWindowSeconds),
    [activityWindowSeconds, recentEvents],
  );
  const { events: focusedDebateEvents } = useSpectate(publicLiveDebate?.debateId, undefined, {
    pollInterval: 1500,
    maxEvents: 24,
    enabled: Boolean(publicLiveDebate?.debateId),
  });
  const isDark = theme === 'dark';
  const recentEventCount = status?.recent_event_count ?? 0;
  const recentActivityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const activityWindowMinutes = Math.max(1, Math.round(recentActivityWindowSeconds / 60));
  const activityAgeSeconds = status?.activity_age_seconds;
  const liveDebate = useMemo(() => {
    if (!publicLiveDebate) return null;

    const refreshedEvents = focusedDebateEvents
      .filter((event) => isRecentEvent(event, recentActivityWindowSeconds))
      .filter(
        (event) =>
          STREAMABLE_EVENT_TYPES.has(event.event_type) || Boolean(extractEventDetails(event)),
      )
      .sort(
        (left, right) =>
          (toEpochMs(left.timestamp) ?? 0) - (toEpochMs(right.timestamp) ?? 0),
      );

    if (refreshedEvents.length === 0) {
      return publicLiveDebate;
    }

    const agents = Array.from(
      new Set(
        refreshedEvents
          .map((event) => event.agent_name)
          .filter((agentName): agentName is string => Boolean(agentName)),
      ),
    );

    const round = refreshedEvents.reduce<number | null>((highestRound, event) => {
      if (typeof event.round_number !== 'number') return highestRound;
      if (highestRound === null) return event.round_number;
      return Math.max(highestRound, event.round_number);
    }, publicLiveDebate.round);

    const topic =
      [...refreshedEvents]
        .reverse()
        .map((event) => event.data?.task)
        .find((task): task is string => typeof task === 'string' && task.trim().length > 0) ??
      publicLiveDebate.topic;

    return {
      ...publicLiveDebate,
      topic,
      agents: agents.length > 0 ? agents : publicLiveDebate.agents,
      round,
      lastEventAt: refreshedEvents[refreshedEvents.length - 1]?.timestamp ?? publicLiveDebate.lastEventAt,
      recentEventCount: Math.max(publicLiveDebate.recentEventCount, refreshedEvents.length),
      events: refreshedEvents.slice(-6),
    };
  }, [focusedDebateEvents, publicLiveDebate, recentActivityWindowSeconds]);
  const hasLiveTranscript = Boolean(liveDebate && liveDebate.events.length > 0);

  let bridgeBadge = 'Checking public bridge';
  let bridgeSummary = 'Checking public live bridge before showing recent activity.';

  if (loaded) {
    if (!status?.active) {
      bridgeBadge = 'Bridge offline';
      bridgeSummary = 'Public spectate is offline right now, so the sample debate below stays illustrative.';
    } else if (hasLiveTranscript && liveDebate) {
      bridgeBadge = 'Live debate';
      bridgeSummary = `${liveDebate.recentEventCount} recent public event${liveDebate.recentEventCount === 1 ? '' : 's'} streaming from a real debate right now.`;
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
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)', marginBottom: '20px' }}
        >
          {isDark ? '> SEE IT IN ACTION' : 'SEE IT IN ACTION'}
        </p>
        <p
          className="text-center"
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text)', fontFamily: 'var(--font-landing)', marginBottom: '48px' }}
        >
          Every debate produces a defensible, auditable result.
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
          data-testid="live-debate-card"
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
              {hasLiveTranscript ? 'Live public debate' : 'Sample decision trace'}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {liveDebate?.topic ?? 'Should we adopt microservices or keep our monolith?'}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {hasLiveTranscript && liveDebate
                ? `Streaming now | ${liveDebate.agents.length || 1} agents | ${liveDebate.round ? `round ${liveDebate.round}` : 'live now'}`
                : 'Example transcript | 6 agents | 3 rounds'}
            </span>
          </div>

          {hasLiveTranscript && liveDebate ? (
            <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_240px]">
              <div
                data-testid="live-debate-stream"
                style={{
                  padding: '20px',
                  borderBottom: '1px solid var(--border)',
                  borderRight: '1px solid var(--border)',
                }}
              >
                <div
                  className="flex items-center gap-2"
                  style={{ marginBottom: '16px', color: 'var(--text-muted)', fontSize: '11px' }}
                >
                  <span
                    className="w-2 h-2 rounded-full animate-pulse"
                    style={{ backgroundColor: 'var(--accent)' }}
                  />
                  <span style={{ fontFamily: 'var(--font-landing)' }}>
                    Watching agents argue in public with a {recentActivityWindowSeconds}-second activity window.
                  </span>
                </div>

                <div className="space-y-3">
                  {liveDebate.events.map((event) => {
                    const style = LIVE_EVENT_STYLES[event.event_type] ?? LIVE_EVENT_STYLES.system;
                    const detail = extractEventDetails(event) ?? 'A new debate event arrived.';
                    return (
                      <article
                        key={`${event.debate_id ?? 'debate'}-${event.timestamp}-${event.event_type}-${event.agent_name ?? 'system'}`}
                        data-testid="live-debate-event"
                        style={{
                          padding: '14px 16px',
                          border: '1px solid var(--border)',
                          borderLeft: `3px solid ${style.accent}`,
                          borderRadius: '16px',
                          backgroundColor: 'rgba(15, 23, 42, 0.14)',
                        }}
                      >
                        <div
                          className="flex flex-wrap items-center gap-2"
                          style={{ marginBottom: '8px' }}
                        >
                          <span
                            className="text-xs font-bold uppercase tracking-wider"
                            style={{ color: style.accent, fontFamily: 'var(--font-landing)' }}
                          >
                            {style.label}
                          </span>
                          <span
                            style={{
                              fontSize: '11px',
                              color: 'var(--text)',
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {event.agent_name ?? 'System'}
                          </span>
                          <span
                            className="ml-auto"
                            style={{
                              fontSize: '10px',
                              color: 'var(--text-muted)',
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {formatRelativeAge(event.timestamp)}
                          </span>
                        </div>
                        <p
                          style={{
                            fontSize: '13px',
                            color: 'var(--text)',
                            fontFamily: 'var(--font-landing)',
                            lineHeight: '1.65',
                          }}
                        >
                          {detail}
                        </p>
                        {(typeof event.round_number === 'number' || style.border) && (
                          <div
                            className="flex flex-wrap items-center gap-3"
                            style={{ marginTop: '10px', fontSize: '10px', color: 'var(--text-muted)' }}
                          >
                            {typeof event.round_number === 'number' ? (
                              <span style={{ fontFamily: 'var(--font-landing)' }}>
                                Round {event.round_number}
                              </span>
                            ) : null}
                            <span style={{ fontFamily: 'var(--font-landing)' }}>
                              {event.event_type.replace(/_/g, ' ')}
                            </span>
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              </div>

              <aside
                style={{
                  padding: '20px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '18px',
                }}
              >
                <div>
                  <p
                    className="uppercase tracking-widest"
                    style={{
                      fontSize: '10px',
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-landing)',
                      marginBottom: '8px',
                    }}
                  >
                    Live signal
                  </p>
                  <p
                    style={{
                      fontSize: isDark ? '18px' : '20px',
                      color: 'var(--text)',
                      fontFamily: 'var(--font-landing)',
                    }}
                  >
                    Public bridge is carrying a real debate right now.
                  </p>
                </div>

                <div className="space-y-3">
                  <div>
                    <div
                      className="uppercase tracking-widest"
                      style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                    >
                      Debate ID
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}>
                      {liveDebate.debateId}
                    </div>
                  </div>
                  <div>
                    <div
                      className="uppercase tracking-widest"
                      style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                    >
                      Last event
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}>
                      {formatRelativeAge(liveDebate.lastEventAt)}
                    </div>
                  </div>
                  <div>
                    <div
                      className="uppercase tracking-widest"
                      style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                    >
                      Active agents
                    </div>
                    <div className="flex flex-wrap gap-2" style={{ marginTop: '8px' }}>
                      {liveDebate.agents.map((agentName) => (
                        <span
                          key={agentName}
                          style={{
                            fontSize: '10px',
                            color: 'var(--text)',
                            border: '1px solid var(--border)',
                            borderRadius: '999px',
                            padding: '6px 10px',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {agentName}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </aside>
            </div>
          ) : (
            <div data-testid="live-debate-fallback" className="grid grid-cols-1 md:grid-cols-3">
              {DEMO_AGENTS.map((agent, i) => (
                <div
                  key={agent.name}
                  style={{
                    padding: '20px',
                    borderRight: i < DEMO_AGENTS.length - 1 ? '1px solid var(--border)' : 'none',
                    borderBottom: '1px solid var(--border)',
                  }}
                >
                  <div className="flex items-center gap-2" style={{ marginBottom: '12px' }}>
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: agent.accent }}
                    />
                    <span
                      className="text-xs font-bold uppercase tracking-wider"
                      style={{ color: agent.accent, fontFamily: 'var(--font-landing)' }}
                    >
                      {agent.name}
                    </span>
                  </div>
                  <p
                    className="leading-relaxed"
                    style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)', lineHeight: '1.7' }}
                  >
                    {agent.content}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="text-center mt-12">
          <Link
            href={liveDebate?.debateId ? `/spectate/${liveDebate.debateId}` : '/demo'}
            className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
            style={{
              display: 'inline-block',
              border: '1px solid var(--accent)',
              borderRadius: 'var(--radius-button)',
              color: 'var(--accent)',
              backgroundColor: 'transparent',
              fontFamily: 'var(--font-landing)',
              padding: '18px 48px',
            }}
          >
            {liveDebate?.debateId ? 'Watch full debate' : 'Run your own debate'}
          </Link>
        </div>
      </div>
    </section>
  );
}
