'use client';

import Link from 'next/link';
import { useTheme } from '@/context/ThemeContext';
import { type SpectateEvent, useSpectate } from '@/hooks/useSpectate';

const SAMPLE_AGENTS = [
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

const LIVE_EVENT_TYPES = new Set([
  'agent_critique',
  'agent_message',
  'agent_synthesis',
  'consensus',
  'critique',
  'message',
  'proposal',
  'turn',
  'vote',
]);

const EVENT_LABELS: Record<string, string> = {
  agent_critique: 'Critique',
  agent_message: 'Argument',
  agent_synthesis: 'Synthesis',
  consensus: 'Consensus',
  critique: 'Critique',
  debate_start: 'Debate start',
  message: 'Message',
  proposal: 'Proposal',
  round_start: 'Round start',
  turn: 'Turn',
  vote: 'Vote',
};

const EVENT_ACCENTS = ['#059669', '#2563eb', '#dc2626', '#d97706', '#7c3aed'];

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

function firstString(values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function getEventLabel(eventType: string): string {
  return EVENT_LABELS[eventType] ?? eventType.replace(/[_.-]+/g, ' ');
}

function getEventSummary(event: SpectateEvent): string {
  const data = event.data ?? {};
  const structuredSummary = firstString([
    data.details,
    data.content,
    data.message,
    data.summary,
    data.verdict,
    data.text,
    data.output,
    data.reasoning,
    data.proposal,
    data.critique,
  ]);

  if (structuredSummary) {
    return structuredSummary;
  }

  if (event.event_type === 'round_start' && event.round_number != null) {
    return `Round ${event.round_number} opened for fresh arguments.`;
  }
  if (event.event_type === 'debate_start') {
    return 'The public bridge detected a new debate starting.';
  }
  if (event.event_type === 'vote') {
    return 'An agent cast a vote in the live debate.';
  }
  if (event.event_type === 'consensus') {
    return 'The panel reached a synthesis and is closing the loop.';
  }

  return 'Live debate event received from the public spectate bridge.';
}

function formatEventClock(timestamp: string | null): string {
  if (!timestamp) return 'LIVE';

  const directIsoTime = timestamp.match(/T(\d{2}:\d{2}:\d{2})/);
  if (directIsoTime) {
    return `${directIsoTime[1]} UTC`;
  }

  const parsed = new Date(timestamp);
  if (!Number.isNaN(parsed.getTime())) {
    return `${parsed.toISOString().slice(11, 19)} UTC`;
  }

  return 'LIVE';
}

function getAgentAccent(agentName: string | null): string {
  if (!agentName) {
    return 'var(--accent)';
  }

  let hash = 0;
  for (const char of agentName) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }

  return EVENT_ACCENTS[hash % EVENT_ACCENTS.length];
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const { status, loaded, connected, events } = useSpectate(undefined, undefined, {
    pollInterval: 2000,
    maxEvents: 40,
  });
  const isDark = theme === 'dark';
  const recentEventCount = status?.recent_event_count ?? 0;
  const recentActivityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const recentEvents = events.filter((event) => isRecentEvent(event, recentActivityWindowSeconds));
  const activityWindowMinutes = Math.max(1, Math.round(recentActivityWindowSeconds / 60));
  const activityAgeSeconds = status?.activity_age_seconds;
  const bridgeOnline = status?.active ?? connected;
  const focusedDebateId =
    status?.live_debates[0]?.debate_id ??
    [...recentEvents].reverse().find((event) => event.debate_id)?.debate_id ??
    null;
  const focusedDebateEvents = focusedDebateId
    ? recentEvents.filter((event) => event.debate_id === focusedDebateId)
    : [];
  const liveFeedEvents =
    focusedDebateEvents.length > 0
      ? focusedDebateEvents
      : recentEvents.filter(
          (event) =>
            LIVE_EVENT_TYPES.has(event.event_type) ||
            event.agent_name !== null ||
            typeof event.data.details === 'string',
        );
  const displayedEvents = liveFeedEvents.slice(-6);
  const focusedFeedEvents = focusedDebateEvents.length > 0 ? focusedDebateEvents : liveFeedEvents;
  const liveTask =
    focusedFeedEvents
      .map((event) => event.data.task)
      .find((task): task is string => typeof task === 'string' && task.trim().length > 0) ??
    null;
  const liveDebateCount =
    status?.live_debate_count ??
    (focusedDebateId ? 1 : 0);

  const agentSet = new Set<string>();
  for (const event of focusedFeedEvents) {
    if (event.agent_name) {
      agentSet.add(event.agent_name);
    }
    const agents = event.data.agents;
    if (Array.isArray(agents)) {
      for (const agent of agents) {
        if (typeof agent === 'string' && agent.trim()) {
          agentSet.add(agent);
        }
      }
    }
  }
  const liveAgentCount = agentSet.size;
  const showingAttributedLiveDebate = Boolean(focusedDebateId && focusedDebateEvents.length > 0);
  const showingLiveBridgeFeed = !showingAttributedLiveDebate && displayedEvents.length > 0;

  let bridgeBadge = 'Checking public bridge';
  let bridgeSummary = 'Checking public live bridge before showing recent activity.';

  if (loaded) {
    if (showingAttributedLiveDebate) {
      bridgeBadge = 'Debate live';
      bridgeSummary = `Streaming ${focusedDebateId} with ${displayedEvents.length} recent argument update${displayedEvents.length === 1 ? '' : 's'}.`;
    } else if (showingLiveBridgeFeed) {
      bridgeBadge = 'Bridge active';
      bridgeSummary = `${displayedEvents.length} recent live update${displayedEvents.length === 1 ? '' : 's'} are visible while the bridge waits for a debate ID attribution.`;
    } else if (!bridgeOnline) {
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
              backgroundColor: bridgeOnline ? 'var(--accent)' : 'var(--border)',
              color: bridgeOnline ? 'var(--bg)' : 'var(--text)',
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

        {showingAttributedLiveDebate ? (
          <div
            data-testid="live-debate-stream"
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
                Live public debate
              </span>
              <span
                className="font-medium"
                style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
              >
                {liveTask ?? `Watching debate ${focusedDebateId}`}
              </span>
              <span
                className="ml-auto"
                style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
              >
                {liveAgentCount || 'Unknown'} agent{liveAgentCount === 1 ? '' : 's'} · {liveDebateCount} live debate{liveDebateCount === 1 ? '' : 's'}
              </span>
            </div>

            <div
              className="space-y-0"
              style={{
                maxHeight: '420px',
                overflowY: 'auto',
                background:
                  'linear-gradient(180deg, color-mix(in srgb, var(--surface) 94%, var(--accent) 6%) 0%, var(--surface) 100%)',
              }}
            >
              {displayedEvents.map((event, index) => {
                const accent = getAgentAccent(event.agent_name);

                return (
                  <div
                    key={`${event.debate_id ?? 'live'}-${event.timestamp}-${index}`}
                    className="flex gap-4"
                    style={{
                      padding: '18px 20px',
                      borderBottom: index < displayedEvents.length - 1 ? '1px solid var(--border)' : 'none',
                    }}
                  >
                    <div
                      className="shrink-0"
                      style={{
                        width: '10px',
                        minWidth: '10px',
                        borderRadius: '999px',
                        backgroundColor: accent,
                        boxShadow: `0 0 12px ${accent}`,
                      }}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2" style={{ marginBottom: '10px' }}>
                        <span
                          className="font-bold uppercase tracking-wider"
                          style={{
                            fontSize: '11px',
                            color: accent,
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {event.agent_name ?? 'Bridge'}
                        </span>
                        <span
                          className="uppercase tracking-wider"
                          style={{
                            fontSize: '10px',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {getEventLabel(event.event_type)}
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
                          {formatEventClock(event.timestamp)}
                        </span>
                      </div>
                      <p
                        className="leading-relaxed"
                        style={{
                          fontSize: '13px',
                          color: 'var(--text)',
                          fontFamily: 'var(--font-landing)',
                          lineHeight: '1.7',
                        }}
                      >
                        {getEventSummary(event)}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : showingLiveBridgeFeed ? (
          <div
            data-testid="live-bridge-feed"
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
                Live bridge activity
              </span>
              <span
                className="font-medium"
                style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
              >
                Real argument updates are flowing, but this buffer has not attributed a debate ID yet.
              </span>
            </div>

            <div className="space-y-0" style={{ maxHeight: '420px', overflowY: 'auto' }}>
              {displayedEvents.map((event, index) => {
                const accent = getAgentAccent(event.agent_name);

                return (
                  <div
                    key={`${event.timestamp}-${index}`}
                    className="flex gap-4"
                    style={{
                      padding: '18px 20px',
                      borderBottom: index < displayedEvents.length - 1 ? '1px solid var(--border)' : 'none',
                    }}
                  >
                    <div
                      className="shrink-0"
                      style={{
                        width: '10px',
                        minWidth: '10px',
                        borderRadius: '999px',
                        backgroundColor: accent,
                        boxShadow: `0 0 12px ${accent}`,
                      }}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2" style={{ marginBottom: '10px' }}>
                        <span
                          className="font-bold uppercase tracking-wider"
                          style={{
                            fontSize: '11px',
                            color: accent,
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {event.agent_name ?? 'Bridge'}
                        </span>
                        <span
                          className="uppercase tracking-wider"
                          style={{
                            fontSize: '10px',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {getEventLabel(event.event_type)}
                        </span>
                        <span
                          className="ml-auto"
                          style={{
                            fontSize: '10px',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {formatEventClock(event.timestamp)}
                        </span>
                      </div>
                      <p
                        className="leading-relaxed"
                        style={{
                          fontSize: '13px',
                          color: 'var(--text)',
                          fontFamily: 'var(--font-landing)',
                          lineHeight: '1.7',
                        }}
                      >
                        {getEventSummary(event)}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div
            data-testid="sample-debate-trace"
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
                Sample decision trace
              </span>
              <span
                className="font-medium"
                style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
              >
                Should we adopt microservices or keep our monolith?
              </span>
              <span
                className="ml-auto"
                style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
              >
                Example transcript · 6 agents · 3 rounds
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3">
              {SAMPLE_AGENTS.map((agent, i) => (
                <div
                  key={agent.name}
                  style={{
                    padding: '20px',
                    borderRight: i < SAMPLE_AGENTS.length - 1 ? '1px solid var(--border)' : 'none',
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
          </div>
        )}

        <div className="text-center mt-12">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            {showingAttributedLiveDebate ? (
              <Link
                href={`/spectate/${focusedDebateId}`}
                className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
                style={{
                  display: 'inline-block',
                  border: '1px solid var(--accent)',
                  borderRadius: 'var(--radius-button)',
                  color: 'var(--bg)',
                  backgroundColor: 'var(--accent)',
                  fontFamily: 'var(--font-landing)',
                  padding: '18px 48px',
                }}
              >
                Watch this debate live
              </Link>
            ) : (
              <Link
                href="/spectate"
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
                Open the live feed
              </Link>
            )}
            <Link
              href="/demo"
              className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
              style={{
                display: 'inline-block',
                border: '1px solid var(--accent)',
                borderRadius: 'var(--radius-button)',
                color: showingAttributedLiveDebate ? 'var(--accent)' : 'var(--bg)',
                backgroundColor: showingAttributedLiveDebate ? 'transparent' : 'var(--accent)',
                fontFamily: 'var(--font-landing)',
                padding: '18px 48px',
              }}
            >
              Run your own debate
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
