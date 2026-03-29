'use client';

import Link from 'next/link';
import { useTheme } from '@/context/ThemeContext';
import {
  useSpectate,
  type SpectateEvent,
  type SpectateLiveDebateSummary,
} from '@/hooks/useSpectate';

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
  if (epochMs === null) return 'Waiting for timestamps';

  const ageSeconds = Math.max(0, Math.round((Date.now() - epochMs) / 1000));
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  const ageHours = Math.round(ageMinutes / 60);
  if (ageHours < 24) return `${ageHours}h ago`;

  const ageDays = Math.round(ageHours / 24);
  return `${ageDays}d ago`;
}

function formatEventLabel(eventType: string): string {
  return eventType
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function getEventAccent(eventType: string): string {
  switch (eventType) {
    case 'proposal':
      return '#2563eb';
    case 'critique':
      return '#dc2626';
    case 'consensus':
    case 'judge':
    case 'vote':
      return '#059669';
    case 'round_start':
    case 'round_end':
      return '#9333ea';
    default:
      return 'var(--accent)';
  }
}

function getEventDetails(event: SpectateEvent): string {
  const details = event.data?.details;
  if (typeof details === 'string' && details.trim()) {
    return details;
  }

  const content = event.data?.content;
  if (typeof content === 'string' && content.trim()) {
    return content;
  }

  const label = formatEventLabel(event.event_type).toLowerCase();
  if (event.agent_name) {
    return `${event.agent_name} pushed the debate forward with a ${label}.`;
  }
  return `The live bridge emitted a ${label}.`;
}

function summarizeFallbackDebates(
  events: SpectateEvent[],
): SpectateLiveDebateSummary[] {
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

export function LiveDemoSection() {
  const { theme } = useTheme();
  const {
    events: bridgeEvents,
    status,
    loaded,
    connected,
  } = useSpectate(undefined, undefined, {
    pollInterval: 4000,
    maxEvents: 40,
  });
  const isDark = theme === 'dark';
  const activityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const recentBridgeEvents = bridgeEvents.filter((event) =>
    isRecentEvent(event, activityWindowSeconds),
  );
  const fallbackDiscoverableDebates = summarizeFallbackDebates(recentBridgeEvents);
  const discoverableDebates =
    status?.live_debates && status.live_debates.length > 0
      ? status.live_debates
      : fallbackDiscoverableDebates;
  const featuredDebate = discoverableDebates[0] ?? null;
  const featuredDebateId = featuredDebate?.debate_id;
  const {
    events: featuredDebateEvents,
  } = useSpectate(featuredDebateId, undefined, {
    pollInterval: 2500,
    maxEvents: 12,
    enabled: Boolean(featuredDebateId),
  });
  const recentEventCount = status?.recent_event_count ?? recentBridgeEvents.length;
  const activityWindowMinutes = Math.max(1, Math.round(activityWindowSeconds / 60));
  const activityAgeSeconds = status?.activity_age_seconds;
  const liveDebateEvents = featuredDebateId
    ? featuredDebateEvents.filter((event) => event.debate_id === featuredDebateId)
    : [];
  const visibleEvents =
    liveDebateEvents.length > 0
      ? liveDebateEvents
      : featuredDebateId
        ? recentBridgeEvents.filter((event) => event.debate_id === featuredDebateId)
        : recentBridgeEvents;
  const streamEvents = visibleEvents.slice(-8);
  const otherLiveDebateCount = Math.max(discoverableDebates.length - 1, 0);

  let bridgeBadge = 'Checking public bridge';
  let bridgeSummary = 'Checking public live bridge before showing recent activity.';

  if (loaded) {
    if (featuredDebate) {
      bridgeBadge = 'Live debate detected';
      bridgeSummary = `Streaming ${featuredDebate.recent_event_count} recent event${featuredDebate.recent_event_count === 1 ? '' : 's'} from debate ${featuredDebate.debate_id}.`;
    } else if (!status?.active && !connected) {
      bridgeBadge = 'Bridge offline';
      bridgeSummary = 'Public spectate is offline right now, so the landing page cannot mirror a live debate yet.';
    } else if (recentEventCount > 0) {
      bridgeBadge = 'Bridge active';
      bridgeSummary = `${recentEventCount} recent event${recentEventCount === 1 ? '' : 's'} in the last ${activityWindowMinutes} minute${activityWindowMinutes === 1 ? '' : 's'}, but no debate ID is safely discoverable yet.`;
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
          Watch real agents propose, critique, and converge as the public bridge updates.
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
                backgroundColor: featuredDebate || status?.active ? 'var(--accent)' : 'var(--border)',
                color: featuredDebate || status?.active ? 'var(--bg)' : 'var(--text)',
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
              {featuredDebate ? 'Live debate stream' : 'Waiting on a live debate'}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {featuredDebate
                ? `Debate ${featuredDebate.debate_id}`
                : 'The landing page only shows a debate once the bridge can attribute real activity.'}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {featuredDebate
                ? `${featuredDebate.recent_event_count} recent events${otherLiveDebateCount > 0 ? ` · +${otherLiveDebateCount} more live debate${otherLiveDebateCount === 1 ? '' : 's'}` : ''}`
                : 'Auto-refreshing every few seconds'}
            </span>
          </div>

          <div
            data-testid="live-debate-stream"
            aria-live="polite"
            style={{
              padding: '8px 0',
              maxHeight: '420px',
              overflowY: 'auto',
            }}
          >
            {streamEvents.length > 0 ? (
              streamEvents.map((event, index) => (
                <div
                  key={`${event.timestamp}-${event.event_type}-${index}`}
                  data-testid="live-debate-event"
                  className="flex items-start gap-4"
                  style={{
                    padding: '18px 20px',
                    borderBottom:
                      index < streamEvents.length - 1 ? '1px solid var(--border)' : 'none',
                  }}
                >
                  <div
                    style={{
                      width: '10px',
                      height: '10px',
                      borderRadius: '999px',
                      backgroundColor: getEventAccent(event.event_type),
                      marginTop: '6px',
                      flexShrink: 0,
                    }}
                  />

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      className="flex flex-wrap items-center gap-2"
                      style={{ marginBottom: '6px' }}
                    >
                      <span
                        className="text-xs font-bold uppercase tracking-wider"
                        style={{
                          color: getEventAccent(event.event_type),
                          fontFamily: 'var(--font-landing)',
                        }}
                      >
                        {formatEventLabel(event.event_type)}
                      </span>
                      {event.agent_name ? (
                        <span
                          style={{
                            fontSize: '12px',
                            color: 'var(--text)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {event.agent_name}
                        </span>
                      ) : null}
                      {event.round_number !== null ? (
                        <span
                          style={{
                            fontSize: '11px',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          Round {event.round_number}
                        </span>
                      ) : null}
                    </div>
                    <p
                      style={{
                        fontSize: '13px',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-landing)',
                        lineHeight: '1.7',
                        margin: 0,
                      }}
                    >
                      {getEventDetails(event)}
                    </p>
                    {event.debate_id ? (
                      <p
                        style={{
                          fontSize: '11px',
                          color: 'var(--text-muted)',
                          fontFamily: 'var(--font-landing)',
                          marginTop: '8px',
                          marginBottom: 0,
                        }}
                      >
                        Debate ID: {event.debate_id}
                      </p>
                    ) : null}
                  </div>

                  <span
                    style={{
                      fontSize: '11px',
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-landing)',
                      flexShrink: 0,
                    }}
                  >
                    {formatRelativeAge(event.timestamp)}
                  </span>
                </div>
              ))
            ) : (
              <div style={{ padding: '28px 20px' }}>
                <p
                  style={{
                    fontSize: '14px',
                    color: 'var(--text)',
                    fontFamily: 'var(--font-landing)',
                    marginBottom: '8px',
                  }}
                >
                  {featuredDebate
                    ? 'The bridge has identified a live debate and is waiting for the next event burst.'
                    : 'No live debate is discoverable yet.'}
                </p>
                <p
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-landing)',
                    margin: 0,
                    lineHeight: '1.7',
                  }}
                >
                  {featuredDebate
                    ? 'This panel refreshes automatically as soon as the next agent message arrives.'
                    : 'As soon as proposal, critique, or consensus events appear with a debate ID, this panel turns into a real-time stream.'}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="text-center mt-12">
          <Link
            href={featuredDebateId ? `/spectate/${featuredDebateId}` : '/spectate'}
            className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
            style={{
              display: 'inline-block',
              border: '1px solid var(--accent)',
              borderRadius: 'var(--radius-button)',
              color: 'var(--accent)',
              backgroundColor: 'transparent',
              fontFamily: 'var(--font-landing)',
              padding: '18px 32px',
              marginRight: '12px',
            }}
          >
            {featuredDebateId ? 'Open live arena' : 'See spectate bridge'}
          </Link>
          <Link
            href="/demo"
            className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
            style={{
              display: 'inline-block',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-button)',
              color: 'var(--text)',
              backgroundColor: 'transparent',
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
