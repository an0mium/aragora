'use client';

import Link from 'next/link';
import { useMemo } from 'react';
import {
  useSpectate,
  type SpectateEvent,
  type SpectateLiveDebateSummary,
  type SpectateStatus,
} from '@/hooks/useSpectate';

type BridgeState =
  | SpectateStatus['bridge_state']
  | 'checking'
  | 'status_unavailable'
  | 'unreachable';

const EVENT_LABELS: Record<string, string> = {
  debate_start: 'Debate started',
  debate_end: 'Debate ended',
  round_start: 'Round started',
  round_end: 'Round ended',
  proposal: 'Proposal',
  critique: 'Critique',
  refine: 'Refinement',
  vote: 'Vote',
  judge: 'Judge',
  consensus: 'Consensus',
  convergence: 'Convergence',
  converged: 'Converged',
  memory_recall: 'Memory recall',
  breakpoint: 'Breakpoint',
  breakpoint_resolved: 'Breakpoint resolved',
  system: 'System',
  error: 'Error',
};

function toEpochMs(timestamp: string | null | undefined): number | null {
  if (!timestamp) return null;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : parsed;
}

function formatRelativeAge(timestamp: string | null | undefined): string {
  const epochMs = toEpochMs(timestamp);
  if (epochMs === null) return 'unknown';

  const ageSeconds = Math.max(0, Math.round((Date.now() - epochMs) / 1000));
  if (ageSeconds < 60) return `${ageSeconds}s ago`;

  const ageMinutes = Math.round(ageSeconds / 60);
  if (ageMinutes < 60) return `${ageMinutes}m ago`;

  const ageHours = Math.round(ageMinutes / 60);
  if (ageHours < 24) return `${ageHours}h ago`;

  const ageDays = Math.round(ageHours / 24);
  return `${ageDays}d ago`;
}

function getBridgeState(
  loaded: boolean,
  connected: boolean,
  status: SpectateStatus | null,
): BridgeState {
  if (!loaded) return 'checking';
  if (status) return status.bridge_state;
  return connected ? 'status_unavailable' : 'unreachable';
}

function getBridgeLabel(state: BridgeState): string {
  switch (state) {
    case 'live_debates_available':
      return 'LIVE';
    case 'activity_unattributed':
      return 'PARTIAL';
    case 'idle':
      return 'IDLE';
    case 'inactive':
      return 'OFF';
    case 'status_unavailable':
      return 'UNKNOWN';
    case 'unreachable':
      return 'OFFLINE';
    case 'checking':
    default:
      return 'CHECKING';
  }
}

function getBridgeTone(state: BridgeState): string {
  switch (state) {
    case 'live_debates_available':
      return 'border-acid-green/30 bg-acid-green/10 text-acid-green';
    case 'activity_unattributed':
    case 'status_unavailable':
      return 'border-acid-cyan/30 bg-acid-cyan/10 text-acid-cyan';
    case 'idle':
    case 'checking':
      return 'border-acid-yellow/30 bg-acid-yellow/10 text-acid-yellow';
    case 'inactive':
    case 'unreachable':
    default:
      return 'border-red-500/30 bg-red-500/10 text-red-400';
  }
}

function getEmptyStateCopy(state: BridgeState): { title: string; body: string } {
  switch (state) {
    case 'activity_unattributed':
      return {
        title: 'Recent activity is visible.',
        body: 'The bridge is seeing debate events, but they are not attached to a public debate ID yet.',
      };
    case 'idle':
      return {
        title: 'No public debate is live right now.',
        body: 'This panel refreshes automatically and will light up as soon as a live debate becomes discoverable.',
      };
    case 'inactive':
      return {
        title: 'Spectate bridge is offline.',
        body: 'Landing-page visitors cannot watch a debate until the public spectate bridge is active again.',
      };
    case 'status_unavailable':
      return {
        title: 'Bridge status is unavailable.',
        body: 'The panel avoids claiming a live debate until the readiness endpoint responds again.',
      };
    case 'unreachable':
      return {
        title: 'Live feed is unreachable.',
        body: 'The spectate endpoints did not respond, so this panel stays truthful instead of fabricating activity.',
      };
    case 'checking':
      return {
        title: 'Checking for a live debate...',
        body: 'The landing page is polling the spectate bridge for a public debate to show.',
      };
    case 'live_debates_available':
    default:
      return {
        title: 'Waiting for event details...',
        body: 'A debate is marked live, and this panel will fill in as soon as recent events arrive.',
      };
  }
}

function getEventTone(eventType: string): string {
  switch (eventType) {
    case 'proposal':
    case 'refine':
      return 'border-acid-cyan/20 bg-acid-cyan/5 text-acid-cyan';
    case 'critique':
    case 'error':
      return 'border-red-500/20 bg-red-500/5 text-red-400';
    case 'vote':
    case 'judge':
      return 'border-acid-yellow/20 bg-acid-yellow/5 text-acid-yellow';
    case 'consensus':
    case 'converged':
      return 'border-acid-green/20 bg-acid-green/5 text-acid-green';
    default:
      return 'border-border bg-surface text-text-muted';
  }
}

function getEventSummary(event: SpectateEvent): string {
  const candidateKeys = [
    'details',
    'summary',
    'content',
    'message',
    'text',
    'proposal',
    'critique',
    'reason',
    'analysis',
    'verdict',
    'decision',
    'question',
  ];

  for (const key of candidateKeys) {
    const value = event.data[key];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }

  const choice = event.data.choice;
  if (typeof choice === 'string' && choice.trim()) {
    return `Choice: ${choice.trim()}`;
  }

  if (event.round_number !== null) {
    return `Round ${event.round_number}`;
  }

  return EVENT_LABELS[event.event_type] ?? event.event_type.replaceAll('_', ' ');
}

function buildFallbackDebates(events: SpectateEvent[]): SpectateLiveDebateSummary[] {
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

function shortDebateId(debateId: string): string {
  return debateId.length > 12 ? `${debateId.slice(0, 12)}...` : debateId;
}

interface LandingLiveDebatePreviewProps {
  apiBase?: string;
}

export function LandingLiveDebatePreview({
  apiBase,
}: LandingLiveDebatePreviewProps) {
  const { events, connected, loaded, status } = useSpectate(undefined, undefined, {
    pollInterval: 2000,
    maxEvents: 30,
    baseUrl: apiBase,
  });

  const bridgeState = getBridgeState(loaded, connected, status);
  const debates = useMemo(() => {
    if (status?.live_debates?.length) return status.live_debates;
    return buildFallbackDebates(events);
  }, [events, status?.live_debates]);

  const featuredDebate = debates[0] ?? null;
  const featuredEvents = useMemo(() => {
    if (!featuredDebate) return [];

    return events
      .filter((event) => event.debate_id === featuredDebate.debate_id)
      .sort(
        (left, right) =>
          (toEpochMs(right.timestamp) ?? 0) - (toEpochMs(left.timestamp) ?? 0),
      )
      .slice(0, 6);
  }, [events, featuredDebate]);

  const emptyState = getEmptyStateCopy(bridgeState);

  return (
    <section
      className="border-t border-border bg-surface/30 px-4 py-16"
      aria-label="Live debate preview"
      data-testid="landing-live-debate"
    >
      <div className="mx-auto max-w-5xl">
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span
                className={`inline-flex items-center border px-2 py-1 font-mono text-[10px] tracking-[0.2em] ${getBridgeTone(bridgeState)}`}
                data-testid="landing-live-debate-status"
              >
                {getBridgeLabel(bridgeState)}
              </span>
              <span className="font-mono text-[10px] uppercase tracking-[0.3em] text-text-muted">
                Public Spectate Feed
              </span>
            </div>
            <div>
              <h2 className="font-mono text-2xl text-text sm:text-3xl">
                Watch a live debate as it unfolds.
              </h2>
              <p className="mt-3 max-w-2xl font-mono text-sm leading-relaxed text-text-muted">
                The landing page polls the public spectate bridge every 2 seconds.
                When a debate is live, visitors can follow the latest proposals,
                critiques, and consensus moves without signing in.
              </p>
            </div>
          </div>

          <Link
            href={featuredDebate ? `/spectate/${featuredDebate.debate_id}` : '/spectate'}
            className="inline-flex items-center justify-center border border-acid-green/40 px-4 py-3 font-mono text-xs text-acid-green transition-colors hover:border-acid-green hover:bg-acid-green/10"
            data-testid="landing-live-debate-link"
          >
            {featuredDebate ? 'Open full spectate view' : 'Open spectate archive'}
          </Link>
        </div>

        {featuredDebate ? (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(18rem,0.8fr)]">
            <div className="border border-acid-green/20 bg-bg/70">
              <div className="flex flex-col gap-3 border-b border-acid-green/20 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="font-mono text-xs uppercase tracking-[0.28em] text-acid-green">
                    Live now
                  </div>
                  <div className="mt-2 font-mono text-lg text-text">
                    Debate {shortDebateId(featuredDebate.debate_id)}
                  </div>
                </div>
                <div className="font-mono text-xs text-text-muted">
                  <div>{featuredDebate.recent_event_count} recent events</div>
                  <div>Last update {formatRelativeAge(featuredDebate.last_event_at)}</div>
                </div>
              </div>

              <div className="space-y-3 p-4" data-testid="landing-live-debate-events">
                {featuredEvents.length > 0 ? (
                  featuredEvents.map((event, index) => (
                    <article
                      key={`${event.timestamp}-${event.event_type}-${index}`}
                      className="border border-border bg-surface/60 px-4 py-3"
                    >
                      <div className="mb-2 flex flex-wrap items-center gap-2 text-[10px] font-mono uppercase tracking-[0.16em]">
                        <span
                          className={`border px-2 py-1 ${getEventTone(event.event_type)}`}
                        >
                          {EVENT_LABELS[event.event_type] ?? event.event_type.replaceAll('_', ' ')}
                        </span>
                        {event.agent_name && (
                          <span className="text-acid-green">{event.agent_name}</span>
                        )}
                        {event.round_number !== null && (
                          <span className="text-text-muted">Round {event.round_number}</span>
                        )}
                        <span className="text-text-muted">{formatRelativeAge(event.timestamp)}</span>
                      </div>
                      <p className="font-mono text-sm leading-relaxed text-text">
                        {getEventSummary(event)}
                      </p>
                    </article>
                  ))
                ) : (
                  <div className="border border-dashed border-acid-green/20 bg-surface/40 px-4 py-6 font-mono text-sm text-text-muted">
                    A debate is live, but the latest event batch has not arrived yet.
                  </div>
                )}
              </div>
            </div>

            <aside className="space-y-4 border border-border bg-bg/70 p-5">
              <div>
                <div className="font-mono text-xs uppercase tracking-[0.24em] text-text-muted">
                  What you are seeing
                </div>
                <p className="mt-3 font-mono text-sm leading-relaxed text-text-muted">
                  This preview only shows debates that the public spectate bridge
                  can verify right now. It avoids claiming a live debate when the
                  bridge is idle or unreachable.
                </p>
              </div>

              <div>
                <div className="font-mono text-xs uppercase tracking-[0.24em] text-text-muted">
                  Signals in this preview
                </div>
                <ul className="mt-3 space-y-2 font-mono text-sm text-text-muted">
                  <li>Recent proposals and critiques from the active debate.</li>
                  <li>Round markers and consensus steps when the bridge emits them.</li>
                  <li>Automatic fallback to a truthful empty state when no debate is discoverable.</li>
                </ul>
              </div>
            </aside>
          </div>
        ) : (
          <div
            className="border border-dashed border-border bg-bg/70 px-6 py-10"
            data-testid="landing-live-debate-empty"
          >
            <div className="max-w-3xl space-y-3">
              <div className="font-mono text-lg text-text">{emptyState.title}</div>
              <p className="font-mono text-sm leading-relaxed text-text-muted">
                {emptyState.body}
              </p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
