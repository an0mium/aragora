'use client';

import { useMemo } from 'react';
import Link from 'next/link';
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

const EVENT_APPEARANCE: Record<string, { label: string; accent: string; tint: string }> = {
  debate_start: { label: 'Debate start', accent: '#22c55e', tint: 'rgba(34, 197, 94, 0.14)' },
  round_start: { label: 'Round start', accent: '#06b6d4', tint: 'rgba(6, 182, 212, 0.14)' },
  proposal: { label: 'Proposal', accent: '#3b82f6', tint: 'rgba(59, 130, 246, 0.14)' },
  critique: { label: 'Critique', accent: '#ef4444', tint: 'rgba(239, 68, 68, 0.14)' },
  refine: { label: 'Refine', accent: '#8b5cf6', tint: 'rgba(139, 92, 246, 0.14)' },
  vote: { label: 'Vote', accent: '#f59e0b', tint: 'rgba(245, 158, 11, 0.14)' },
  judge: { label: 'Judge', accent: '#f97316', tint: 'rgba(249, 115, 22, 0.14)' },
  consensus: { label: 'Consensus', accent: '#10b981', tint: 'rgba(16, 185, 129, 0.14)' },
  round_end: { label: 'Round end', accent: '#14b8a6', tint: 'rgba(20, 184, 166, 0.14)' },
  debate_end: { label: 'Debate end', accent: '#22c55e', tint: 'rgba(34, 197, 94, 0.14)' },
  agent_thinking: { label: 'Thinking', accent: '#a855f7', tint: 'rgba(168, 85, 247, 0.14)' },
  agent_reasoning: { label: 'Reasoning', accent: '#8b5cf6', tint: 'rgba(139, 92, 246, 0.14)' },
  agent_message: { label: 'Argument', accent: '#2563eb', tint: 'rgba(37, 99, 235, 0.14)' },
};

const AGENT_ACCENTS = ['#22c55e', '#3b82f6', '#ef4444', '#f59e0b', '#a855f7', '#14b8a6'];

interface LiveDebateSnapshot {
  debateId: string;
  events: SpectateEvent[];
  task: string | null;
  agents: string[];
  lastEventAt: string | null;
  highestRound: number;
}

function toEpochMs(timestamp: string | null | undefined): number {
  if (!timestamp) return 0;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function truncateCopy(value: string, maxLength = 220): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1)}…`;
}

function getStringField(data: Record<string, unknown>, key: string): string | null {
  const value = data[key];
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
}

function getEventSummary(event: SpectateEvent): string {
  const details =
    getStringField(event.data, 'details') ??
    getStringField(event.data, 'content') ??
    getStringField(event.data, 'reasoning_chunk') ??
    getStringField(event.data, 'step') ??
    getStringField(event.data, 'argument_summary') ??
    getStringField(event.data, 'crux_description') ??
    getStringField(event.data, 'context_summary') ??
    getStringField(event.data, 'content_summary');

  if (details) {
    return truncateCopy(details);
  }

  switch (event.event_type) {
    case 'proposal':
      return 'Shared a fresh proposal.';
    case 'critique':
      return 'Pushed back on another agent’s argument.';
    case 'refine':
      return 'Revised their position after critique.';
    case 'vote':
      return 'Cast a vote on the strongest argument.';
    case 'judge':
      return 'Synthesized the strongest ideas into a ruling.';
    case 'consensus':
      return 'Consensus was reached across the panel.';
    case 'round_start':
      return `Round ${event.round_number ?? '?'} is now live.`;
    case 'round_end':
      return `Round ${event.round_number ?? '?'} closed.`;
    case 'debate_start':
      return 'A public debate is starting.';
    case 'debate_end':
      return 'The public debate has finished.';
    case 'agent_thinking':
      return 'Analyzing the current argument graph.';
    case 'agent_reasoning':
      return 'Streaming internal reasoning.';
    case 'agent_message':
      return 'Published a fresh argument.';
    default:
      return 'Live debate activity detected.';
  }
}

function extractTask(events: SpectateEvent[]): string | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const task = getStringField(events[index].data, 'task');
    if (task) {
      return task;
    }
  }

  const debateStart = events.find((event) => event.event_type === 'debate_start');
  const details = debateStart ? getStringField(debateStart.data, 'details') : null;
  if (!details) {
    return null;
  }

  if (details.startsWith('Task: ')) {
    return details.slice(6).replace(/\.\.\.$/, '').trim();
  }

  return details;
}

function selectLiveDebate(events: SpectateEvent[]): LiveDebateSnapshot | null {
  const debates = new Map<
    string,
    {
      debateId: string;
      events: SpectateEvent[];
      agents: Set<string>;
      lastEventAt: string | null;
      highestRound: number;
    }
  >();

  for (const event of events) {
    if (!event.debate_id) {
      continue;
    }

    const group = debates.get(event.debate_id) ?? {
      debateId: event.debate_id,
      events: [],
      agents: new Set<string>(),
      lastEventAt: event.timestamp,
      highestRound: 0,
    };

    group.events.push(event);
    if (event.agent_name) {
      group.agents.add(event.agent_name);
    }
    if (event.round_number && event.round_number > group.highestRound) {
      group.highestRound = event.round_number;
    }
    if (toEpochMs(event.timestamp) >= toEpochMs(group.lastEventAt)) {
      group.lastEventAt = event.timestamp;
    }

    debates.set(event.debate_id, group);
  }

  const [selected] = Array.from(debates.values()).sort((left, right) => {
    const timeDiff = toEpochMs(right.lastEventAt) - toEpochMs(left.lastEventAt);
    if (timeDiff !== 0) {
      return timeDiff;
    }
    return right.events.length - left.events.length;
  });

  if (!selected) {
    return null;
  }

  return {
    debateId: selected.debateId,
    events: selected.events.sort((left, right) => toEpochMs(left.timestamp) - toEpochMs(right.timestamp)),
    task: extractTask(selected.events),
    agents: Array.from(selected.agents),
    lastEventAt: selected.lastEventAt,
    highestRound: selected.highestRound,
  };
}

function formatEventTime(timestamp: string): string {
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return 'LIVE';
  }

  return parsed.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function getAgentAccent(agentName: string): string {
  let hash = 0;
  for (let index = 0; index < agentName.length; index += 1) {
    hash = (hash * 31 + agentName.charCodeAt(index)) >>> 0;
  }

  return AGENT_ACCENTS[hash % AGENT_ACCENTS.length];
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const { status, loaded, connected, events } = useSpectate(undefined, undefined, {
    pollInterval: 3000,
    maxEvents: 40,
  });
  const isDark = theme === 'dark';
  const recentEventCount = status?.recent_event_count ?? 0;
  const recentActivityWindowSeconds = status?.recent_activity_window_seconds ?? 120;
  const activityWindowMinutes = Math.max(1, Math.round(recentActivityWindowSeconds / 60));
  const liveDebate = useMemo(() => selectLiveDebate(events), [events]);
  const liveDebateEvents = liveDebate?.events.slice(-8) ?? [];
  const liveDebateAvailable = liveDebateEvents.length > 0;
  const activityAgeSeconds =
    status?.activity_age_seconds ??
    (liveDebate?.lastEventAt ? Math.max(0, Math.round((Date.now() - toEpochMs(liveDebate.lastEventAt)) / 1000)) : null);

  let bridgeBadge = 'Checking public bridge';
  let bridgeSummary = 'Checking public live bridge before showing recent activity.';

  if (loaded) {
    if (liveDebateAvailable) {
      bridgeBadge = 'Live public debate';
      bridgeSummary = `Watching ${liveDebate.agents.length || 'multiple'} agent${liveDebate.agents.length === 1 ? '' : 's'} argue in real time. Refreshing from the public bridge every 3 seconds.`;
    } else if (!(status?.active || connected)) {
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
              backgroundColor: status?.active || connected || liveDebateAvailable ? 'var(--accent)' : 'var(--border)',
              color: status?.active || connected || liveDebateAvailable ? 'var(--bg)' : 'var(--text)',
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

        {liveDebateAvailable ? (
          <div
            data-testid="live-debate-feed"
            style={{
              backgroundColor: 'var(--surface)',
              borderRadius: 'var(--radius-card)',
              border: '1px solid var(--border)',
              borderTopColor: '#22c55e',
              borderTopWidth: '3px',
              boxShadow: 'var(--shadow-card)',
              overflow: 'hidden',
              margin: '0 24px',
            }}
          >
            <div
              className="flex flex-wrap items-start gap-3"
              style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}
            >
              <div className="flex items-center gap-3">
                <span
                  className="font-bold px-2 py-0.5 uppercase tracking-wider"
                  style={{
                    fontSize: '10px',
                    backgroundColor: '#22c55e',
                    color: '#041310',
                    borderRadius: 'var(--radius-button)',
                  }}
                >
                  Live public debate
                </span>
                <span className="flex items-center gap-2" style={{ color: '#22c55e', fontSize: '12px' }}>
                  <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: '#22c55e' }} />
                  Streaming now
                </span>
              </div>
              <div className="min-w-0" style={{ flex: 1 }}>
                <div
                  className="font-medium"
                  style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
                >
                  {liveDebate.task ?? 'Public agents are debating live right now.'}
                </div>
                <div
                  className="flex flex-wrap items-center gap-2"
                  style={{ marginTop: '8px', fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                >
                  <span>{liveDebateEvents.length} recent event{liveDebateEvents.length === 1 ? '' : 's'}</span>
                  <span>&middot;</span>
                  <span>{liveDebate.agents.length || 'Multiple'} agent{liveDebate.agents.length === 1 ? '' : 's'}</span>
                  {liveDebate.highestRound > 0 ? (
                    <>
                      <span>&middot;</span>
                      <span>Round {liveDebate.highestRound}</span>
                    </>
                  ) : null}
                </div>
              </div>
            </div>

            {liveDebate.agents.length > 0 ? (
              <div
                className="flex flex-wrap gap-2"
                style={{
                  padding: '14px 20px',
                  borderBottom: '1px solid var(--border)',
                  backgroundColor: 'rgba(255, 255, 255, 0.02)',
                }}
              >
                {liveDebate.agents.map((agentName) => {
                  const accent = getAgentAccent(agentName);
                  return (
                    <span
                      key={agentName}
                      className="inline-flex items-center gap-2"
                      style={{
                        fontSize: '10px',
                        color: accent,
                        border: `1px solid ${accent}33`,
                        borderRadius: '999px',
                        padding: '6px 10px',
                        fontFamily: 'var(--font-landing)',
                        backgroundColor: `${accent}14`,
                      }}
                    >
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: accent }} />
                      {agentName}
                    </span>
                  );
                })}
              </div>
            ) : null}

            <div className="space-y-0" style={{ maxHeight: '440px', overflowY: 'auto' }}>
              {liveDebateEvents.map((event, index) => {
                const appearance = EVENT_APPEARANCE[event.event_type] ?? {
                  label: event.event_type.replace(/_/g, ' '),
                  accent: '#64748b',
                  tint: 'rgba(100, 116, 139, 0.14)',
                };

                return (
                  <div
                    key={`${event.debate_id}-${event.timestamp}-${index}`}
                    className="grid grid-cols-[84px_1fr] gap-4"
                    style={{
                      padding: '16px 20px',
                      borderBottom: index < liveDebateEvents.length - 1 ? '1px solid var(--border)' : 'none',
                    }}
                  >
                    <div
                      style={{
                        fontSize: '10px',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-landing)',
                        paddingTop: '2px',
                      }}
                    >
                      {formatEventTime(event.timestamp)}
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2" style={{ marginBottom: '8px' }}>
                        <span
                          style={{
                            fontSize: '10px',
                            fontWeight: 700,
                            letterSpacing: '0.08em',
                            textTransform: 'uppercase',
                            color: appearance.accent,
                            backgroundColor: appearance.tint,
                            borderRadius: '999px',
                            padding: '4px 8px',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {appearance.label}
                        </span>
                        {event.agent_name ? (
                          <span
                            style={{
                              fontSize: '11px',
                              color: 'var(--text)',
                              fontWeight: 600,
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {event.agent_name}
                          </span>
                        ) : null}
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
                      </div>
                      <p
                        style={{
                          fontSize: isDark ? '12px' : '13px',
                          color: 'var(--text-muted)',
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
          </div>
        )}

        <div className="text-center mt-12">
          <Link
            href={liveDebateAvailable ? '/spectate' : '/demo'}
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
            {liveDebateAvailable ? 'Open live feed' : 'Run your own debate'}
          </Link>
        </div>
      </div>
    </section>
  );
}
