'use client';

import Link from 'next/link';
import { useMemo } from 'react';
import { useTheme } from '@/context/ThemeContext';
import {
  useSpectate,
  type SpectateEvent,
  type SpectateStatus,
} from '@/hooks/useSpectate';

const LIVE_ARGUMENT_EVENTS = new Set([
  'debate_start',
  'round_start',
  'proposal',
  'critique',
  'agent_message',
  'vote',
  'judge',
  'consensus',
  'convergence',
  'round_end',
  'debate_end',
]);

const EVENT_LABELS: Record<string, string> = {
  debate_start: 'Debate opened',
  round_start: 'Round opened',
  proposal: 'Proposal',
  critique: 'Critique',
  agent_message: 'Argument',
  vote: 'Vote',
  judge: 'Judgment',
  consensus: 'Consensus',
  convergence: 'Convergence',
  round_end: 'Round closed',
  debate_end: 'Debate ended',
};

const EVENT_TONES: Record<string, { border: string; pill: string }> = {
  debate_start: { border: 'rgba(20, 184, 166, 0.4)', pill: '#14b8a6' },
  round_start: { border: 'rgba(14, 165, 233, 0.4)', pill: '#0ea5e9' },
  proposal: { border: 'rgba(59, 130, 246, 0.4)', pill: '#2563eb' },
  critique: { border: 'rgba(239, 68, 68, 0.4)', pill: '#dc2626' },
  agent_message: { border: 'rgba(168, 85, 247, 0.4)', pill: '#a855f7' },
  vote: { border: 'rgba(245, 158, 11, 0.4)', pill: '#f59e0b' },
  judge: { border: 'rgba(245, 158, 11, 0.4)', pill: '#f59e0b' },
  consensus: { border: 'rgba(16, 185, 129, 0.4)', pill: '#10b981' },
  convergence: { border: 'rgba(16, 185, 129, 0.4)', pill: '#10b981' },
  round_end: { border: 'rgba(14, 165, 233, 0.4)', pill: '#0ea5e9' },
  debate_end: { border: 'rgba(16, 185, 129, 0.4)', pill: '#10b981' },
};

function getBridgeState(
  loaded: boolean,
  connected: boolean,
  status: SpectateStatus | null,
): SpectateStatus['bridge_state'] | 'checking' | 'offline' {
  if (!loaded) return 'checking';
  if (status) return status.bridge_state;
  return connected ? 'idle' : 'offline';
}

function formatRelativeAge(timestamp: string | null | undefined): string {
  if (!timestamp) return 'just now';

  const epochMs = Date.parse(timestamp);
  if (Number.isNaN(epochMs)) return 'just now';

  const ageSeconds = Math.max(0, Math.round((Date.now() - epochMs) / 1000));
  if (ageSeconds < 15) return 'just now';
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  const ageHours = Math.round(ageMinutes / 60);
  return `${ageHours}h ago`;
}

function getEventSummary(event: SpectateEvent): string {
  const data = event.data || {};
  const candidates = [
    data['details'],
    data['content'],
    data['message'],
    data['summary'],
    data['reasoning'],
  ];

  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate.trim();
    }
  }

  switch (event.event_type) {
    case 'vote':
      return `${event.agent_name || 'An agent'} cast a vote.`;
    case 'consensus':
      return 'The panel reached a shared position.';
    case 'debate_start':
      return 'The debate is now live.';
    case 'debate_end':
      return 'The current debate has concluded.';
    default:
      return 'New live debate activity arrived.';
  }
}

function pickLeadDebateId(
  status: SpectateStatus | null,
  events: SpectateEvent[],
): string | null {
  const fromStatus = status?.live_debates[0]?.debate_id;
  if (fromStatus) return fromStatus;

  const grouped = new Map<string, { count: number; lastSeen: number }>();
  for (const event of events) {
    if (!event.debate_id) continue;
    const seenAt = Date.parse(event.timestamp);
    const current = grouped.get(event.debate_id) || { count: 0, lastSeen: 0 };
    grouped.set(event.debate_id, {
      count: current.count + 1,
      lastSeen: Number.isNaN(seenAt) ? current.lastSeen : Math.max(current.lastSeen, seenAt),
    });
  }

  return Array.from(grouped.entries())
    .sort((left, right) => {
      if (right[1].count !== left[1].count) {
        return right[1].count - left[1].count;
      }
      return right[1].lastSeen - left[1].lastSeen;
    })[0]?.[0] || null;
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const { events, connected, loaded, status, refresh } = useSpectate(
    undefined,
    undefined,
    { pollInterval: 2000, maxEvents: 24 }
  );

  const leadDebateId = useMemo(
    () => pickLeadDebateId(status, events),
    [events, status]
  );

  const liveEvents = useMemo(() => {
    const scopedEvents = leadDebateId
      ? events.filter((event) => event.debate_id === leadDebateId)
      : events;

    return scopedEvents
      .filter((event) => LIVE_ARGUMENT_EVENTS.has(event.event_type))
      .slice(-8);
  }, [events, leadDebateId]);

  const liveAgents = useMemo(() => {
    const seen = new Set<string>();
    const agents: string[] = [];

    for (const event of liveEvents) {
      if (!event.agent_name || seen.has(event.agent_name)) continue;
      seen.add(event.agent_name);
      agents.push(event.agent_name);
      if (agents.length === 4) break;
    }

    return agents;
  }, [liveEvents]);

  const bridgeState = getBridgeState(loaded, connected, status);
  const hasLiveFeed = liveEvents.length > 0;
  const latestTimestamp = liveEvents[liveEvents.length - 1]?.timestamp ?? status?.last_event_at ?? null;
  const readinessCopy =
    bridgeState === 'live_debates_available' || hasLiveFeed
      ? 'Agents are actively arguing on the landing page right now.'
      : bridgeState === 'activity_unattributed'
        ? 'Live events are arriving, but the bridge has not attached a debate ID yet.'
        : bridgeState === 'idle'
          ? 'The bridge is healthy and standing by for the next live debate.'
          : bridgeState === 'offline'
            ? 'The live bridge is unreachable from this page right now.'
            : 'Connecting to the live bridge...';
  const pollSummary = `Polling every 2s${status?.recent_activity_window_seconds ? ` | ${status.recent_activity_window_seconds}s activity window` : ''}`;

  return (
    <section
      className="px-4"
      id="live-stream"
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
          Watch agents argue back and forth in real time as the spectate bridge streams live events onto the homepage.
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
                backgroundColor: hasLiveFeed ? 'var(--accent)' : 'transparent',
                color: hasLiveFeed ? 'var(--bg)' : 'var(--accent)',
                borderRadius: 'var(--radius-button)',
                border: hasLiveFeed ? 'none' : '1px solid var(--accent)',
              }}
            >
              {hasLiveFeed ? 'Live stream active' : loaded ? 'Awaiting live debate' : 'Connecting'}
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {leadDebateId ? `Debate ${leadDebateId}` : 'Landing page live debate stream'}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {latestTimestamp ? `Updated ${formatRelativeAge(latestTimestamp)}` : pollSummary}
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr]">
            <div
              className="border-b border-[var(--border)] lg:border-b-0 lg:border-r"
              style={{
                padding: '20px',
              }}
            >
              <p
                className="uppercase tracking-widest"
                style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '10px' }}
              >
                Spectate bridge
              </p>
              <p
                style={{
                  fontSize: '18px',
                  color: 'var(--text)',
                  fontFamily: 'var(--font-display, var(--font-landing))',
                  marginBottom: '10px',
                }}
              >
                {hasLiveFeed ? 'Visitors are watching a real debate.' : 'The homepage is ready for the next live debate.'}
              </p>
              <p
                style={{
                  fontSize: '13px',
                  color: 'var(--text-muted)',
                  lineHeight: '1.7',
                  marginBottom: '18px',
                }}
              >
                {readinessCopy}
              </p>

              <div className="space-y-3">
                <div>
                  <p className="uppercase tracking-widest" style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                    Agents live
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {liveAgents.length > 0 ? (
                      liveAgents.map((agent) => (
                        <span
                          key={agent}
                          className="px-2 py-1"
                          style={{
                            fontSize: '11px',
                            borderRadius: '999px',
                            border: '1px solid var(--border)',
                            color: 'var(--text)',
                          }}
                        >
                          {agent}
                        </span>
                      ))
                    ) : (
                      <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Waiting for agent activity</span>
                    )}
                  </div>
                </div>

                <div>
                  <p className="uppercase tracking-widest" style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                    Feed health
                  </p>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.7' }}>
                    {pollSummary}
                  </p>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.7' }}>
                    {status?.recent_event_count ?? events.length} recent event{(status?.recent_event_count ?? events.length) === 1 ? '' : 's'} observed
                  </p>
                </div>
              </div>
            </div>

            <div style={{ padding: '20px' }}>
              {hasLiveFeed ? (
                <div className="space-y-3" aria-label="Live debate event feed">
                  {liveEvents.map((event, index) => {
                    const tone = EVENT_TONES[event.event_type] || {
                      border: 'var(--border)',
                      pill: 'var(--accent)',
                    };

                    return (
                      <article
                        key={`${event.timestamp}-${event.event_type}-${index}`}
                        style={{
                          border: `1px solid ${tone.border}`,
                          borderLeftWidth: '3px',
                          borderRadius: '14px',
                          padding: '14px 16px',
                          backgroundColor: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
                        }}
                      >
                        <div className="flex flex-wrap items-center gap-2" style={{ marginBottom: '8px' }}>
                          <span
                            className="font-bold uppercase tracking-wider px-2 py-0.5"
                            style={{
                              fontSize: '10px',
                              borderRadius: '999px',
                              backgroundColor: tone.pill,
                              color: '#041218',
                            }}
                          >
                            {EVENT_LABELS[event.event_type] || event.event_type}
                          </span>
                          {event.agent_name && (
                            <span style={{ fontSize: '11px', color: 'var(--text)' }}>
                              {event.agent_name}
                            </span>
                          )}
                          {typeof event.round_number === 'number' && (
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                              Round {event.round_number}
                            </span>
                          )}
                          <span className="ml-auto" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                            {formatRelativeAge(event.timestamp)}
                          </span>
                        </div>
                        <p
                          style={{
                            fontSize: '13px',
                            color: 'var(--text-muted)',
                            lineHeight: '1.7',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {getEventSummary(event)}
                        </p>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div
                  className="flex h-full min-h-[260px] items-center justify-center"
                  style={{
                    border: '1px dashed var(--border)',
                    borderRadius: '18px',
                    backgroundColor: isDark ? 'rgba(255,255,255,0.015)' : 'rgba(0,0,0,0.02)',
                    padding: '28px',
                  }}
                >
                  <div className="text-center max-w-md">
                    <p
                      style={{
                        fontSize: '18px',
                        color: 'var(--text)',
                        fontFamily: 'var(--font-display, var(--font-landing))',
                        marginBottom: '10px',
                      }}
                    >
                      {bridgeState === 'offline' ? 'Live debate feed offline' : 'Waiting for the next live debate'}
                    </p>
                    <p
                      style={{
                        fontSize: '13px',
                        color: 'var(--text-muted)',
                        lineHeight: '1.7',
                        marginBottom: '18px',
                      }}
                    >
                      {readinessCopy}
                    </p>
                    <button
                      type="button"
                      onClick={() => {
                        void refresh();
                      }}
                      className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
                      style={{
                        border: '1px solid var(--accent)',
                        borderRadius: 'var(--radius-button)',
                        color: 'var(--accent)',
                        backgroundColor: 'transparent',
                        fontFamily: 'var(--font-landing)',
                        padding: '12px 22px',
                      }}
                    >
                      Refresh live feed
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="text-center mt-12">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            {leadDebateId ? (
              <Link
                href={`/debate/${encodeURIComponent(leadDebateId)}`}
                className="text-sm font-semibold transition-all hover:scale-[1.02]"
                style={{
                  border: '1px solid var(--accent)',
                  borderRadius: 'var(--radius-button)',
                  color: 'var(--bg)',
                  backgroundColor: 'var(--accent)',
                  fontFamily: 'var(--font-landing)',
                  padding: '18px 32px',
                }}
              >
                Open the live debate
              </Link>
            ) : (
              <button
                type="button"
                onClick={() => {
                  void refresh();
                }}
                className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
                style={{
                  border: '1px solid var(--accent)',
                  borderRadius: 'var(--radius-button)',
                  color: 'var(--bg)',
                  backgroundColor: 'var(--accent)',
                  fontFamily: 'var(--font-landing)',
                  padding: '18px 32px',
                }}
              >
                Check for a live debate
              </button>
            )}
            <button
              type="button"
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
              style={{
                border: '1px solid var(--accent)',
                borderRadius: 'var(--radius-button)',
                color: 'var(--accent)',
                backgroundColor: 'transparent',
                fontFamily: 'var(--font-landing)',
                padding: '18px 32px',
              }}
            >
              Run your own debate
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
