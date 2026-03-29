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

export function LiveDemoSection() {
  const { theme } = useTheme();
  const {
    status,
    loaded,
    connected,
    events: recentBridgeEvents,
  } = useSpectate(undefined, undefined, {
    pollInterval: 3000,
    maxEvents: 80,
  });
  const recentActivityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const visibleBridgeEvents = recentBridgeEvents.filter((event) =>
    isRecentEvent(event.timestamp, recentActivityWindowSeconds),
  );
  const discoverableDebates = status?.live_debates?.length
    ? status.live_debates
    : deriveDiscoverableDebates(visibleBridgeEvents);
  const activeDebate = discoverableDebates[0] ?? null;
  const {
    events: liveDebateEvents,
    loaded: liveDebateLoaded,
  } = useSpectate(activeDebate?.debate_id, undefined, {
    enabled: Boolean(activeDebate?.debate_id),
    pollInterval: 2500,
    maxEvents: 18,
  });
  const isDark = theme === 'dark';
  const recentEventCount = status?.recent_event_count ?? visibleBridgeEvents.length;
  const activityWindowMinutes = Math.max(1, Math.round(recentActivityWindowSeconds / 60));
  const activityAgeSeconds = status?.activity_age_seconds;
  const bridgeActive = Boolean(activeDebate) || Boolean(status?.active) || connected;

  let bridgeBadge = 'Checking public bridge';
  let bridgeSummary = 'Checking public live bridge before showing recent activity.';

  if (loaded) {
    if (activeDebate) {
      bridgeBadge = 'Live debate on air';
      bridgeSummary = 'Watching a public spectate feed update every few seconds.';
    } else if (!bridgeActive) {
      bridgeBadge = 'Bridge offline';
      bridgeSummary = 'Public spectate is offline right now, so the sample debate below stays illustrative.';
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
              backgroundColor: bridgeActive ? 'var(--accent)' : 'var(--border)',
              color: bridgeActive ? 'var(--bg)' : 'var(--text)',
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
              {activeDebate ? 'Live public debate' : 'Sample decision trace'}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {activeDebate
                ? `Debate ${truncateDebateId(activeDebate.debate_id)}`
                : 'Should we adopt microservices or keep our monolith?'}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {activeDebate
                ? `${activeDebate.recent_event_count} recent event${activeDebate.recent_event_count === 1 ? '' : 's'} · updated ${formatRelativeAge(activeDebate.last_event_at)}`
                : 'Example transcript · 6 agents · 3 rounds'}
            </span>
          </div>

          {activeDebate ? (
            <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.5fr)_minmax(280px,0.9fr)]">
              <div
                style={{
                  padding: '20px',
                  borderRight: '1px solid var(--border)',
                  borderBottom: '1px solid var(--border)',
                }}
              >
                <div className="flex flex-wrap gap-2" style={{ marginBottom: '16px' }}>
                  {activeDebate.event_types.map((eventType) => (
                    <span
                      key={`${activeDebate.debate_id}-${eventType}`}
                      className="font-bold uppercase tracking-wider"
                      style={{
                        fontSize: '10px',
                        padding: '4px 8px',
                        borderRadius: '999px',
                        border: '1px solid var(--border)',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-landing)',
                      }}
                    >
                      {eventType.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>

                <div
                  aria-live="polite"
                  data-testid="live-debate-feed"
                  className="space-y-3"
                  style={{ maxHeight: '360px', overflowY: 'auto', paddingRight: '4px' }}
                >
                  {liveDebateLoaded && liveDebateEvents.length > 0 ? (
                    liveDebateEvents.map((event, index) => {
                      const style = getEventStyle(event.event_type);
                      const details = getEventSummary(event);

                      return (
                        <article
                          key={`${event.timestamp}-${event.event_type}-${index}`}
                          style={{
                            border: '1px solid var(--border)',
                            borderRadius: 'var(--radius-card)',
                            padding: '14px 16px',
                            backgroundColor: 'color-mix(in srgb, var(--surface) 88%, white 12%)',
                          }}
                        >
                          <div
                            className="flex flex-wrap items-center gap-2"
                            style={{ marginBottom: details ? '8px' : '0' }}
                          >
                            <span
                              className="font-bold uppercase tracking-wider"
                              style={{
                                fontSize: '10px',
                                color: style.accent,
                                fontFamily: 'var(--font-landing)',
                              }}
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
                              {event.agent_name || 'System'}
                            </span>
                            {event.round_number != null ? (
                              <span
                                style={{
                                  fontSize: '10px',
                                  color: 'var(--text-muted)',
                                  fontFamily: 'var(--font-landing)',
                                }}
                              >
                                Round {event.round_number}
                              </span>
                            ) : null}
                            <span
                              className="ml-auto"
                              style={{
                                fontSize: '10px',
                                color: 'var(--text-muted)',
                                fontFamily: 'var(--font-landing)',
                              }}
                            >
                              {formatClockTime(event.timestamp)}
                            </span>
                          </div>

                          {details ? (
                            <p
                              style={{
                                fontSize: '12px',
                                color: 'var(--text-muted)',
                                fontFamily: 'var(--font-landing)',
                                lineHeight: '1.7',
                              }}
                            >
                              {details}
                            </p>
                          ) : null}
                        </article>
                      );
                    })
                  ) : (
                    <div
                      style={{
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-card)',
                        padding: '18px 16px',
                        backgroundColor: 'color-mix(in srgb, var(--surface) 92%, white 8%)',
                      }}
                    >
                      <p
                        style={{
                          fontSize: '12px',
                          color: 'var(--text-muted)',
                          fontFamily: 'var(--font-landing)',
                        }}
                      >
                        Waiting for the next live event from the public spectate bridge.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <div style={{ padding: '20px', borderBottom: '1px solid var(--border)' }}>
                <div style={{ marginBottom: '20px' }}>
                  <p
                    className="uppercase tracking-widest"
                    style={{
                      fontSize: '10px',
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-landing)',
                      marginBottom: '8px',
                    }}
                  >
                    Live debate status
                  </p>
                  <p
                    style={{
                      fontSize: '14px',
                      color: 'var(--text)',
                      fontFamily: 'var(--font-landing)',
                      lineHeight: '1.6',
                      marginBottom: '12px',
                    }}
                  >
                    Visitors can watch agents challenge each other in real-time without starting a debate.
                  </p>
                  <p
                    style={{
                      fontSize: '12px',
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-landing)',
                      lineHeight: '1.7',
                    }}
                  >
                    This stream only appears when the public event buffer exposes an attributable debate ID. If the bridge goes quiet, the landing page falls back to the labeled sample below.
                  </p>
                </div>

                <dl className="grid grid-cols-2 gap-3" style={{ marginBottom: '20px' }}>
                  <StatCell label="Debate ID" value={truncateDebateId(activeDebate.debate_id)} />
                  <StatCell label="Recent events" value={String(activeDebate.recent_event_count)} />
                  <StatCell label="Last activity" value={formatRelativeAge(activeDebate.last_event_at)} />
                  <StatCell label="More live debates" value={String(Math.max(0, discoverableDebates.length - 1))} />
                </dl>

                <div className="flex flex-col gap-3">
                  <Link
                    href={`/spectate/${encodeURIComponent(activeDebate.debate_id)}`}
                    className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer text-center"
                    style={{
                      display: 'inline-block',
                      border: '1px solid var(--accent)',
                      borderRadius: 'var(--radius-button)',
                      color: 'var(--bg)',
                      backgroundColor: 'var(--accent)',
                      fontFamily: 'var(--font-landing)',
                      padding: '14px 20px',
                    }}
                  >
                    Open live feed
                  </Link>
                  <Link
                    href="/spectate"
                    className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer text-center"
                    style={{
                      display: 'inline-block',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-button)',
                      color: 'var(--text)',
                      backgroundColor: 'transparent',
                      fontFamily: 'var(--font-landing)',
                      padding: '14px 20px',
                    }}
                  >
                    Browse all live debates
                  </Link>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3">
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
            href={activeDebate ? `/spectate/${encodeURIComponent(activeDebate.debate_id)}` : '/demo'}
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
            {activeDebate ? 'Watch this debate live' : 'Run your own debate'}
          </Link>
        </div>
      </div>
    </section>
  );
}

function deriveDiscoverableDebates(events: SpectateEvent[]): SpectateLiveDebateSummary[] {
  const debates = new Map<string, SpectateLiveDebateSummary>();

  for (const event of events) {
    if (!event.debate_id) continue;

    const existing = debates.get(event.debate_id);
    if (!existing) {
      debates.set(event.debate_id, {
        debate_id: event.debate_id,
        recent_event_count: 1,
        last_event_at: event.timestamp,
        event_types: [event.event_type],
      });
      continue;
    }

    existing.recent_event_count += 1;
    if ((toEpochMs(event.timestamp) ?? 0) >= (toEpochMs(existing.last_event_at) ?? 0)) {
      existing.last_event_at = event.timestamp;
    }
    if (!existing.event_types.includes(event.event_type)) {
      existing.event_types = [...existing.event_types, event.event_type].sort();
    }
  }

  return Array.from(debates.values()).sort(
    (left, right) =>
      (toEpochMs(right.last_event_at) ?? 0) - (toEpochMs(left.last_event_at) ?? 0),
  );
}

function getEventStyle(eventType: string): { label: string; accent: string } {
  switch (eventType) {
    case 'proposal':
      return { label: 'Proposal', accent: '#2563eb' };
    case 'critique':
      return { label: 'Critique', accent: '#dc2626' };
    case 'vote':
      return { label: 'Vote', accent: '#d97706' };
    case 'consensus':
    case 'converged':
      return { label: 'Consensus', accent: '#059669' };
    case 'round_start':
    case 'round_end':
      return { label: 'Round', accent: '#0891b2' };
    default:
      return {
        label: eventType.replace(/_/g, ' '),
        accent: 'var(--accent)',
      };
  }
}

function getEventSummary(event: SpectateEvent): string | null {
  const detailKeys = ['details', 'summary', 'message', 'content'];
  for (const key of detailKeys) {
    const value = event.data[key];
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }

  if (event.agent_name) {
    return `${event.agent_name} posted a ${event.event_type.replace(/_/g, ' ')} update.`;
  }

  return null;
}

function toEpochMs(timestamp: string | null | undefined): number | null {
  if (!timestamp) return null;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : parsed;
}

function formatRelativeAge(timestamp: string | null | undefined): string {
  const epochMs = toEpochMs(timestamp);
  if (epochMs === null) return '—';

  const ageSeconds = Math.max(0, Math.round((Date.now() - epochMs) / 1000));
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  const ageHours = Math.round(ageMinutes / 60);
  if (ageHours < 24) return `${ageHours}h ago`;

  const ageDays = Math.round(ageHours / 24);
  return `${ageDays}d ago`;
}

function formatClockTime(timestamp: string): string {
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) return 'Unknown time';
  return parsed.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

function isRecentEvent(timestamp: string | null | undefined, windowSeconds: number): boolean {
  const epochMs = toEpochMs(timestamp);
  if (epochMs === null) return false;
  return Date.now() - epochMs <= windowSeconds * 1000;
}

function truncateDebateId(debateId: string): string {
  if (debateId.length <= 22) return debateId;
  return `${debateId.slice(0, 10)}…${debateId.slice(-8)}`;
}

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-card)',
        padding: '12px',
        backgroundColor: 'color-mix(in srgb, var(--surface) 92%, white 8%)',
      }}
    >
      <dt
        className="uppercase tracking-widest"
        style={{
          fontSize: '10px',
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-landing)',
          marginBottom: '6px',
        }}
      >
        {label}
      </dt>
      <dd
        style={{
          fontSize: '12px',
          color: 'var(--text)',
          fontFamily: 'var(--font-landing)',
          lineHeight: '1.5',
        }}
      >
        {value}
      </dd>
    </div>
  );
}
