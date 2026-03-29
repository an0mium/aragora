'use client';

import Link from 'next/link';
import { useTheme } from '@/context/ThemeContext';
import {
  useSpectate,
  type SpectateEvent,
  type SpectateLiveDebateSummary,
} from '@/hooks/useSpectate';

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

const EVENT_ACCENTS: Record<string, string> = {
  debate_start: '#f59e0b',
  round_start: '#0891b2',
  proposal: '#059669',
  critique: '#dc2626',
  refine: '#2563eb',
  vote: '#7c3aed',
  consensus: '#16a34a',
  debate_end: '#14b8a6',
};

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

function summarizeRecentDebates(events: SpectateEvent[]): SpectateLiveDebateSummary[] {
  const grouped = new Map<
    string,
    {
      debate_id: string;
      recent_event_count: number;
      last_event_at: string | null;
      event_types: Set<string>;
    }
  >();

  for (const event of events) {
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
}

function formatActivityAge(seconds: number | null | undefined): string | null {
  if (typeof seconds !== 'number') return null;
  if (seconds < 60) return `Last activity ${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `Last activity ${Math.round(seconds / 60)}m ago`;
  return `Last activity ${Math.round(seconds / 3600)}h ago`;
}

function formatEventTypeLabel(eventType: string): string {
  return eventType.replace(/_/g, ' ').toUpperCase();
}

function formatEventTime(timestamp: string): string {
  const epochMs = toEpochMs(timestamp);
  if (epochMs === null) return 'now';

  return new Date(epochMs).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

function formatEventSummary(event: SpectateEvent): string {
  const details = event.data?.details;
  if (typeof details === 'string' && details.trim().length > 0) {
    return details;
  }

  switch (event.event_type) {
    case 'debate_start':
      return 'Agents joined the room and the debate stream is now live.';
    case 'round_start':
      return `Round ${event.round_number ?? 1} is underway.`;
    case 'proposal':
      return 'An agent posted a new proposal.';
    case 'critique':
      return 'An agent challenged the current proposal.';
    case 'refine':
      return 'An agent revised its position after critique.';
    case 'vote':
      return 'Agents are voting on the strongest direction.';
    case 'consensus':
      return 'The panel has converged on a shared answer.';
    case 'debate_end':
      return 'The live debate has ended.';
    default:
      return 'The debate stream published a live update.';
  }
}

function SampleTranscript() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3" data-testid="sample-transcript">
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
  );
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const { status, loaded, connected, events } = useSpectate(undefined, undefined, {
    pollInterval: 3000,
    maxEvents: 60,
  });
  const isDark = theme === 'dark';
  const recentActivityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const recentEvents = events.filter((event) =>
    isRecentEvent(event, recentActivityWindowSeconds),
  );
  const discoverableDebates = status?.live_debates?.length
    ? status.live_debates
    : summarizeRecentDebates(recentEvents);
  const activeDebate = discoverableDebates[0] ?? null;
  const liveEvents = activeDebate
    ? recentEvents
        .filter((event) => event.debate_id === activeDebate.debate_id)
        .slice(-8)
    : [];
  const liveAgents = Array.from(
    new Set(
      liveEvents.flatMap((event) => {
        const agents = Array.isArray(event.data?.agents)
          ? event.data.agents.filter(
              (agent): agent is string =>
                typeof agent === 'string' && agent.length > 0,
            )
          : [];
        return event.agent_name ? [...agents, event.agent_name] : agents;
      }),
    ),
  );
  const liveAgentCount = liveAgents.length || 1;
  const liveTaskEvent = liveEvents.find(
    (event) => typeof event.data?.task === 'string',
  );
  const liveTask =
    typeof liveTaskEvent?.data.task === 'string'
      ? liveTaskEvent.data.task
      : 'Watch a public Aragora debate unfold live on the landing page.';
  const liveRound = liveEvents.reduce(
    (maxRound, event) => Math.max(maxRound, event.round_number ?? 0),
    0,
  );
  const recentEventCount = status?.recent_event_count ?? recentEvents.length;
  const activityWindowMinutes = Math.max(1, Math.round(recentActivityWindowSeconds / 60));
  const derivedActivityAgeSeconds = (() => {
    const lastEventAt = activeDebate?.last_event_at;
    const epochMs = toEpochMs(lastEventAt);
    if (epochMs === null) return null;
    return Math.max((Date.now() - epochMs) / 1000, 0);
  })();
  const activityAgeLabel = formatActivityAge(
    status?.activity_age_seconds ?? derivedActivityAgeSeconds,
  );
  const hasLiveDebate = Boolean(activeDebate && liveEvents.length > 0);

  let bridgeBadge = 'Checking public bridge';
  let bridgeSummary = 'Checking public live bridge before showing recent activity.';

  if (loaded) {
    if (hasLiveDebate) {
      bridgeBadge = 'Streaming live';
      bridgeSummary = `Showing ${liveEvents.length} live update${liveEvents.length === 1 ? '' : 's'} from ${liveAgentCount} agent${liveAgentCount === 1 ? '' : 's'} in the last ${activityWindowMinutes} minute${activityWindowMinutes === 1 ? '' : 's'}.`;
    } else if (status?.active === false && !connected) {
      bridgeBadge = 'Bridge offline';
      bridgeSummary = 'Public spectate is offline right now, so the sample debate below stays illustrative.';
    } else if (recentEventCount > 0) {
      bridgeBadge = 'Bridge active';
      bridgeSummary = `${recentEventCount} recent event${recentEventCount === 1 ? '' : 's'} in the last ${activityWindowMinutes} minute${activityWindowMinutes === 1 ? '' : 's'}.`;
    } else if (status?.active || connected) {
      bridgeBadge = 'Bridge ready';
      bridgeSummary = 'Public spectate is online, but no recent live debate activity is visible yet.';
    } else {
      bridgeBadge = 'Bridge offline';
      bridgeSummary = 'Public spectate is offline right now, so the sample debate below stays illustrative.';
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
          Watch the public debate feed when agents are actively arguing, then jump into the full viewer.
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
              backgroundColor: hasLiveDebate || status?.active || connected ? 'var(--accent)' : 'var(--border)',
              color: hasLiveDebate || status?.active || connected ? 'var(--bg)' : 'var(--text)',
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
              {hasLiveDebate ? 'Live public debate' : 'Example transcript'}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {hasLiveDebate ? liveTask : 'Should we adopt microservices or keep our monolith?'}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {hasLiveDebate
                ? `${activeDebate.recent_event_count} recent updates · ${liveRound > 0 ? `Round ${liveRound}` : 'Opening phase'}`
                : 'Example transcript · 6 agents · 3 rounds'}
            </span>
          </div>

          {hasLiveDebate ? (
            <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.5fr)_minmax(280px,0.7fr)]">
              <div
                data-testid="live-debate-stream"
                style={{
                  borderRight: '1px solid var(--border)',
                  borderBottom: '1px solid var(--border)',
                }}
              >
                <div
                  className="space-y-3"
                  style={{ padding: '20px' }}
                  aria-live="polite"
                >
                  {liveEvents.map((event, index) => {
                    const accent = EVENT_ACCENTS[event.event_type] ?? 'var(--accent)';
                    return (
                      <article
                        key={`${event.timestamp}-${event.event_type}-${index}`}
                        data-testid="live-debate-event"
                        style={{
                          border: '1px solid var(--border)',
                          borderLeft: `3px solid ${accent}`,
                          borderRadius: '14px',
                          padding: '14px 16px',
                          backgroundColor: 'var(--surface)',
                        }}
                      >
                        <div className="flex flex-wrap items-center gap-2" style={{ marginBottom: '8px' }}>
                          <span
                            className="font-bold uppercase tracking-wider"
                            style={{
                              color: accent,
                              fontSize: '10px',
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {formatEventTypeLabel(event.event_type)}
                          </span>
                          <span
                            style={{
                              color: 'var(--text)',
                              fontSize: '11px',
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {event.agent_name ?? 'SYSTEM'}
                          </span>
                          <span
                            className="ml-auto"
                            style={{
                              color: 'var(--text-muted)',
                              fontSize: '10px',
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {formatEventTime(event.timestamp)}
                          </span>
                        </div>
                        <p
                          style={{
                            color: 'var(--text-muted)',
                            fontSize: '12px',
                            fontFamily: 'var(--font-landing)',
                            lineHeight: '1.7',
                          }}
                        >
                          {formatEventSummary(event)}
                        </p>
                      </article>
                    );
                  })}
                </div>
              </div>

              <aside
                style={{
                  padding: '20px',
                  backgroundColor: 'color-mix(in srgb, var(--surface) 92%, var(--accent) 8%)',
                }}
              >
                <div
                  className="inline-flex items-center gap-2 font-bold uppercase tracking-wider"
                  style={{ color: 'var(--accent)', fontSize: '11px', fontFamily: 'var(--font-landing)', marginBottom: '16px' }}
                >
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: 'var(--accent)' }}
                  />
                  <span>Streaming live now</span>
                </div>

                <div style={{ marginBottom: '18px' }}>
                  <p
                    style={{ color: 'var(--text-muted)', fontSize: '11px', fontFamily: 'var(--font-landing)', marginBottom: '6px' }}
                  >
                    Debate ID
                  </p>
                  <p
                    style={{ color: 'var(--text)', fontSize: '14px', fontFamily: 'var(--font-landing)', wordBreak: 'break-word' }}
                  >
                    {activeDebate.debate_id}
                  </p>
                </div>

                <div style={{ marginBottom: '18px' }}>
                  <p
                    style={{ color: 'var(--text-muted)', fontSize: '11px', fontFamily: 'var(--font-landing)', marginBottom: '6px' }}
                  >
                    Watching
                  </p>
                  <p
                    style={{ color: 'var(--text)', fontSize: '13px', fontFamily: 'var(--font-landing)' }}
                  >
                    {liveAgentCount} agent{liveAgentCount === 1 ? '' : 's'} trading proposals, critiques, and votes in real time.
                  </p>
                </div>

                <div className="flex flex-wrap gap-2" style={{ marginBottom: '20px' }}>
                  {liveAgents.map((agent) => (
                    <span
                      key={agent}
                      className="font-bold uppercase tracking-wider"
                      style={{
                        fontSize: '10px',
                        color: 'var(--accent)',
                        border: '1px solid var(--border)',
                        borderRadius: '999px',
                        padding: '6px 10px',
                        fontFamily: 'var(--font-landing)',
                      }}
                    >
                      {agent}
                    </span>
                  ))}
                </div>

                <Link
                  href={`/debate/${activeDebate.debate_id}`}
                  className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
                  style={{
                    display: 'inline-block',
                    border: '1px solid var(--accent)',
                    borderRadius: 'var(--radius-button)',
                    color: 'var(--bg)',
                    backgroundColor: 'var(--accent)',
                    fontFamily: 'var(--font-landing)',
                    padding: '14px 22px',
                  }}
                >
                  Watch full live debate
                </Link>
              </aside>
            </div>
          ) : (
            <SampleTranscript />
          )}
        </div>

        <div className="text-center mt-12 flex flex-wrap items-center justify-center gap-4">
          {hasLiveDebate ? (
            <Link
              href={`/debate/${activeDebate.debate_id}`}
              className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
              style={{
                display: 'inline-block',
                border: '1px solid var(--accent)',
                borderRadius: 'var(--radius-button)',
                color: 'var(--accent)',
                backgroundColor: 'transparent',
                fontFamily: 'var(--font-landing)',
                padding: '18px 32px',
              }}
            >
              Open the live viewer
            </Link>
          ) : null}
          <Link
            href="/demo"
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
            {hasLiveDebate ? 'Start your own debate' : 'Run your own debate'}
          </Link>
        </div>
      </div>
    </section>
  );
}
