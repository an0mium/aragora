'use client';

import { useEffect, useMemo } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useRightSidebar } from '@/context/RightSidebarContext';
import {
  useSpectate,
  type SpectateEvent,
  type SpectateStatus,
} from '@/hooks/useSpectate';

type BridgeState =
  | SpectateStatus['bridge_state']
  | 'checking'
  | 'status_unavailable'
  | 'unreachable';

function EventTypeIcon({ eventType }: { eventType: string }) {
  const icons: Record<string, string> = {
    proposal: '$',
    critique: '!',
    vote: '#',
    consensus: '*',
    round_start: '>',
    round_end: '<',
    agent_message: '@',
  };
  return <span>{icons[eventType] || '>'}</span>;
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
      return 'STATUS UNKNOWN';
    case 'unreachable':
      return 'API OFFLINE';
    case 'checking':
    default:
      return 'CHECKING';
  }
}

function getBridgeTone(state: BridgeState): string {
  switch (state) {
    case 'live_debates_available':
      return 'bg-acid-green/10 text-acid-green border-acid-green/30';
    case 'activity_unattributed':
    case 'status_unavailable':
      return 'bg-acid-cyan/10 text-acid-cyan border-acid-cyan/30';
    case 'idle':
    case 'checking':
      return 'bg-acid-yellow/10 text-acid-yellow border-acid-yellow/30';
    case 'inactive':
    case 'unreachable':
    default:
      return 'bg-red-500/10 text-red-400 border-red-500/30';
  }
}

function getReadinessTitle(state: BridgeState): string {
  switch (state) {
    case 'live_debates_available':
      return 'Live debate IDs are available from the spectate bridge.';
    case 'activity_unattributed':
      return 'Recent bridge activity is visible, but it is not linked to a debate ID yet.';
    case 'idle':
      return 'The spectate bridge is running, but no recent debate activity is visible.';
    case 'inactive':
      return 'The spectate bridge is installed but currently inactive.';
    case 'status_unavailable':
      return 'Recent events are reachable, but bridge readiness details are unavailable.';
    case 'unreachable':
      return 'The spectate API is unreachable from this surface right now.';
    case 'checking':
    default:
      return 'Checking live debate availability and bridge readiness…';
  }
}

function getReadinessBody(
  state: BridgeState,
  discoverableDebates: number,
  unattributedRecentEvents: number,
): string {
  switch (state) {
    case 'live_debates_available':
      return `${discoverableDebates} debate ID${discoverableDebates === 1 ? '' : 's'} can be opened from this surface.`;
    case 'activity_unattributed':
      return `${unattributedRecentEvents} recent event${unattributedRecentEvents === 1 ? '' : 's'} arrived without a debate ID, so this page stays partial instead of fabricating a live debate list.`;
    case 'idle':
      return 'Nothing recent is being surfaced by the live bridge, so the page does not claim a debate is in progress.';
    case 'inactive':
      return 'Turn on the bridge before expecting this page to reflect live debate readiness.';
    case 'status_unavailable':
      return 'This page only trusts explicit bridge status before showing live readiness details.';
    case 'unreachable':
      return 'No bridge facts can be confirmed until the spectate endpoints respond again.';
    case 'checking':
    default:
      return 'The page will only surface debate links once live activity is observed and attributed.';
  }
}

function getEmptyStateTitle(state: BridgeState): string {
  switch (state) {
    case 'inactive':
      return 'Spectate Bridge Offline';
    case 'idle':
      return 'No Recent Debate Activity';
    case 'status_unavailable':
      return 'Bridge Status Unavailable';
    case 'unreachable':
      return 'Spectate API Unreachable';
    case 'checking':
      return 'Scanning Spectate Bridge';
    default:
      return 'No Discoverable Live Debates';
  }
}

function getEmptyStateBody(state: BridgeState): string {
  switch (state) {
    case 'inactive':
      return 'This surface cannot confirm live debate readiness until the spectate bridge is active.';
    case 'idle':
      return 'The bridge is up, but it has not observed recent live debate events within the current readiness window.';
    case 'status_unavailable':
      return 'Bridge status did not load, so this page avoids presenting debate links as live.';
    case 'unreachable':
      return 'The spectate endpoints did not respond, so no live readiness claim is shown.';
    case 'checking':
      return 'Waiting for the first readiness snapshot…';
    default:
      return 'Only debate IDs seen in recent bridge events are listed here.';
  }
}

function isRecentEvent(event: SpectateEvent, windowSeconds: number): boolean {
  const epochMs = toEpochMs(event.timestamp);
  if (epochMs === null) return false;
  return Date.now() - epochMs <= windowSeconds * 1000;
}

export default function SpectatePage() {
  const { setContext, clearContext } = useRightSidebar();
  const {
    events: spectateEvents,
    connected: spectateConnected,
    loaded: spectateLoaded,
    status: spectateStatus,
  } = useSpectate(undefined, undefined, { pollInterval: 3000 });

  const bridgeState = getBridgeState(
    spectateLoaded,
    spectateConnected,
    spectateStatus,
  );
  const activityWindowSeconds =
    spectateStatus?.recent_activity_window_seconds ?? 120;

  const recentBridgeEvents = useMemo(
    () => spectateEvents.filter((event) => isRecentEvent(event, activityWindowSeconds)),
    [activityWindowSeconds, spectateEvents],
  );

  const fallbackDiscoverableDebates = useMemo(() => {
    const grouped = new Map<
      string,
      {
        debate_id: string;
        recent_event_count: number;
        last_event_at: string | null;
        event_types: Set<string>;
      }
    >();

    for (const event of recentBridgeEvents) {
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
  }, [recentBridgeEvents]);

  const discoverableDebates = spectateStatus?.live_debates ?? fallbackDiscoverableDebates;
  const recentEventCount = spectateStatus?.recent_event_count ?? recentBridgeEvents.length;
  const unattributedRecentEvents =
    spectateStatus?.unattributed_recent_event_count ??
    recentBridgeEvents.filter((event) => !event.debate_id).length;

  useEffect(() => {
    setContext({
      title: 'Spectate Mode',
      subtitle: 'Truthful live readiness',
      statsContent: (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Bridge</span>
            <span
              className={`text-sm font-mono ${
                bridgeState === 'live_debates_available'
                  ? 'text-[var(--acid-green)]'
                  : bridgeState === 'activity_unattributed' || bridgeState === 'status_unavailable'
                    ? 'text-[var(--acid-cyan)]'
                    : bridgeState === 'idle' || bridgeState === 'checking'
                      ? 'text-[var(--acid-yellow)]'
                      : 'text-red-400'
              }`}
            >
              {getBridgeLabel(bridgeState)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Discoverable Debates</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">
              {discoverableDebates.length}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Recent Events</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">
              {recentEventCount}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Last Activity</span>
            <span className="text-sm font-mono text-[var(--text)]">
              {formatRelativeAge(spectateStatus?.last_event_at)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Buffered Events</span>
            <span className="text-sm font-mono text-[var(--text)]">
              {spectateStatus?.buffer_size ?? spectateEvents.length}
            </span>
          </div>
        </div>
      ),
      actionsContent: (
        <div className="space-y-2">
          <Link
            href="/arena"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
          >
            + START DEBATE
          </Link>
          <Link
            href="/debates"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            VIEW ARCHIVE
          </Link>
        </div>
      ),
    });

    return () => clearContext();
  }, [
    bridgeState,
    clearContext,
    discoverableDebates.length,
    recentEventCount,
    setContext,
    spectateEvents.length,
    spectateStatus?.buffer_size,
    spectateStatus?.last_event_at,
  ]);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="max-w-6xl mx-auto px-4 py-8">
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  bridgeState === 'live_debates_available'
                    ? 'bg-acid-green animate-pulse'
                    : bridgeState === 'activity_unattributed'
                      ? 'bg-acid-cyan animate-pulse'
                      : bridgeState === 'idle' || bridgeState === 'checking'
                        ? 'bg-acid-yellow animate-pulse'
                        : 'bg-red-500'
                }`}
              />
              <h1 className="text-2xl font-mono text-acid-green">SPECTATE MODE</h1>
              <span className={`px-2 py-0.5 text-xs font-mono border rounded ${getBridgeTone(bridgeState)}`}>
                {getBridgeLabel(bridgeState)}
              </span>
            </div>
            <p className="text-text-muted text-sm font-mono">
              Watch only what the live bridge can actually confirm.
            </p>
          </div>

          <div className="border border-acid-cyan/20 bg-acid-cyan/5 p-4 mb-6">
            <div className="flex items-start justify-between gap-4 flex-col sm:flex-row">
              <div>
                <h2 className="text-sm font-mono text-acid-cyan uppercase tracking-wider mb-2">
                  Bridge Readiness
                </h2>
                <p className="text-sm font-mono text-text mb-2">
                  {getReadinessTitle(bridgeState)}
                </p>
                <p className="text-xs font-mono text-text-muted max-w-3xl">
                  {getReadinessBody(
                    bridgeState,
                    discoverableDebates.length,
                    unattributedRecentEvents,
                  )}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs font-mono min-w-[240px]">
                <div className="border border-border bg-surface/50 px-3 py-2">
                  <div className="text-text-muted mb-1">Bridge State</div>
                  <div className="text-acid-green break-words">{getBridgeLabel(bridgeState)}</div>
                </div>
                <div className="border border-border bg-surface/50 px-3 py-2">
                  <div className="text-text-muted mb-1">Last Event</div>
                  <div className="text-acid-cyan">{formatRelativeAge(spectateStatus?.last_event_at)}</div>
                </div>
                <div className="border border-border bg-surface/50 px-3 py-2">
                  <div className="text-text-muted mb-1">Recent Events</div>
                  <div className="text-acid-green">{recentEventCount}</div>
                </div>
                <div className="border border-border bg-surface/50 px-3 py-2">
                  <div className="text-text-muted mb-1">Debate IDs</div>
                  <div className="text-acid-cyan">{discoverableDebates.length}</div>
                </div>
              </div>
            </div>
          </div>

          {!spectateLoaded && (
            <div className="border border-acid-green/20 bg-surface/30 p-8 text-center mb-6">
              <div className="w-8 h-8 border-2 border-acid-green/30 border-t-acid-green rounded-full animate-spin mx-auto mb-4" />
              <p className="text-text-muted text-sm font-mono">
                Checking live bridge readiness...
              </p>
            </div>
          )}

          {spectateLoaded && discoverableDebates.length > 0 && (
            <div className="space-y-4 mb-6">
              <h2 className="text-sm font-mono text-acid-cyan uppercase tracking-wider">
                Discoverable Live Debates ({discoverableDebates.length})
              </h2>

              <div className="grid gap-4">
                {discoverableDebates.map((debate) => (
                  <Link
                    key={debate.debate_id}
                    href={`/spectate/${debate.debate_id}`}
                    className="block border border-acid-green/30 bg-surface/50 p-4 hover:border-acid-green/60 hover:bg-surface/80 transition-all group"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-mono text-text-muted mb-2">Debate ID</div>
                        <h3 className="text-sm font-mono text-text break-all group-hover:text-acid-green transition-colors">
                          {debate.debate_id}
                        </h3>
                        <div className="flex flex-wrap gap-2 mt-3">
                          {debate.event_types.map((eventType) => (
                            <span
                              key={`${debate.debate_id}-${eventType}`}
                              className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono bg-acid-cyan/10 text-acid-cyan border border-acid-cyan/30"
                            >
                              <EventTypeIcon eventType={eventType} />
                              {eventType}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-1 text-xs font-mono">
                        <div className="flex items-center gap-2">
                          <span className="text-text-muted">Recent Events</span>
                          <span className="text-acid-green">{debate.recent_event_count}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-text-muted">Last Seen</span>
                          <span className="text-acid-cyan">
                            {formatRelativeAge(debate.last_event_at)}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 text-acid-green">
                          <div className="w-2 h-2 bg-acid-green rounded-full animate-pulse" />
                          OPEN FEED
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          )}

          {spectateLoaded &&
            bridgeState === 'activity_unattributed' &&
            discoverableDebates.length === 0 && (
              <div className="border border-acid-yellow/30 bg-acid-yellow/10 p-4 mb-6">
                <h2 className="text-sm font-mono text-acid-yellow mb-2">
                  Partial Readiness
                </h2>
                <p className="text-xs font-mono text-text-muted">
                  Recent bridge activity is flowing, but the current events are not tagged with
                  a `debate_id`. This surface stays honest and does not invent clickable live
                  debates until attribution is present.
                </p>
              </div>
            )}

          {spectateLoaded && recentBridgeEvents.length > 0 && (
            <div className="mt-6 space-y-4">
              <h2 className="text-sm font-mono text-acid-cyan uppercase tracking-wider">
                Recent Bridge Event Feed ({recentBridgeEvents.length} events)
              </h2>
              <div className="border border-acid-green/20 bg-surface/30 divide-y divide-border max-h-[400px] overflow-y-auto">
                {recentBridgeEvents
                  .slice(-20)
                  .reverse()
                  .map((event, index) => {
                    const details =
                      typeof event.data.details === 'string' ? event.data.details : null;

                    return (
                      <div
                        key={`${event.timestamp}-${index}`}
                        className="px-4 py-2 flex items-start gap-3 text-xs font-mono hover:bg-surface/50 transition-colors"
                      >
                        <span className="text-acid-green mt-0.5">
                          <EventTypeIcon eventType={event.event_type} />
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-acid-cyan">{event.event_type}</span>
                            {event.agent_name && (
                              <span className="text-text-muted">by {event.agent_name}</span>
                            )}
                            {event.round_number != null && (
                              <span className="text-text-muted">R{event.round_number}</span>
                            )}
                          </div>
                          {details && (
                            <span className="text-text-muted/80 truncate block">
                              {details}
                            </span>
                          )}
                          {event.debate_id && (
                            <span className="text-text-muted/60 truncate block">
                              debate: {event.debate_id}
                            </span>
                          )}
                        </div>
                        <span className="text-text-muted/40 flex-shrink-0">
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {spectateLoaded &&
            discoverableDebates.length === 0 &&
            recentBridgeEvents.length === 0 && (
              <div className="border border-acid-green/20 bg-surface/30 p-8 text-center">
                <div className="text-4xl mb-4">👁️</div>
                <h2 className="text-lg font-mono text-acid-green mb-2">
                  {getEmptyStateTitle(bridgeState)}
                </h2>
                <p className="text-text-muted text-sm font-mono mb-6 max-w-md mx-auto">
                  {getEmptyStateBody(bridgeState)}
                </p>
                <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                  <Link
                    href="/arena"
                    className="px-6 py-2 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors"
                  >
                    START DEBATE
                  </Link>
                  <Link
                    href="/debates"
                    className="px-6 py-2 border border-acid-green/30 text-acid-green font-mono hover:border-acid-green transition-colors"
                  >
                    VIEW ARCHIVE
                  </Link>
                </div>
              </div>
            )}

          <div className="mt-8 border border-acid-cyan/20 bg-acid-cyan/5 p-4">
            <h3 className="text-sm font-mono text-acid-cyan mb-2">
              About Spectate Mode
            </h3>
            <ul className="text-xs font-mono text-text-muted space-y-1">
              <li>• This page only lists debates that appear in recent bridge events with a `debate_id`.</li>
              <li>• If activity is real but unattributed, the surface stays partial instead of inventing a live card.</li>
              <li>• Recent bridge status is shown separately from the raw event feed.</li>
              <li>• Read-only: spectators cannot influence debates.</li>
              <li>• Use the archive when no live debate is currently discoverable.</li>
            </ul>
          </div>
        </div>
      </main>
    </>
  );
}
