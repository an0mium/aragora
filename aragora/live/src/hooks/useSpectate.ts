'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { API_BASE_URL } from '@/config';

/**
 * A single spectate event from the SpectatorStream bridge.
 */
export interface SpectateEvent {
  event_type: string;
  timestamp: string;
  data: Record<string, unknown>;
  debate_id: string | null;
  pipeline_id: string | null;
  agent_name: string | null;
  round_number: number | null;
}

export interface SpectateLiveDebateSummary {
  debate_id: string;
  recent_event_count: number;
  last_event_at: string | null;
  event_types: string[];
}

export interface SpectateStatus {
  active: boolean;
  subscribers: number;
  buffer_size: number;
  bridge_state:
    | 'inactive'
    | 'idle'
    | 'activity_unattributed'
    | 'live_debates_available';
  last_event_at: string | null;
  activity_age_seconds: number | null;
  recent_activity_window_seconds: number;
  recent_event_count: number;
  live_debate_count: number;
  live_debate_ids: string[];
  live_debates: SpectateLiveDebateSummary[];
  unattributed_recent_event_count: number;
}

interface UseSpectateOptions {
  /** Poll interval in milliseconds (default: 2000) */
  pollInterval?: number;
  /** Maximum number of events to fetch per poll (default: 50) */
  maxEvents?: number;
  /** Whether polling is enabled (default: true) */
  enabled?: boolean;
}

interface UseSpectateReturn {
  /** Array of spectate events, newest last */
  events: SpectateEvent[];
  /** Whether the polling endpoints are currently reachable */
  connected: boolean;
  /** Whether the hook has completed its first fetch cycle */
  loaded: boolean;
  /** Bridge status (active, subscriber count, buffer size) */
  status: SpectateStatus | null;
  /** Manually trigger a refresh */
  refresh: () => Promise<void>;
}

const SPECTATE_STREAM_EVENT_TYPES = [
  'debate_start',
  'debate_end',
  'round_start',
  'round_end',
  'proposal',
  'critique',
  'refine',
  'vote',
  'judge',
  'consensus',
  'convergence',
  'converged',
  'memory_recall',
  'breakpoint',
  'breakpoint_resolved',
  'system',
  'error',
] as const;

function getSpectateEventKey(event: SpectateEvent): string {
  return [
    event.event_type,
    event.timestamp,
    event.debate_id ?? '',
    event.pipeline_id ?? '',
    event.agent_name ?? '',
    String(event.round_number ?? ''),
    JSON.stringify(event.data ?? {}),
  ].join('|');
}

function toTimestamp(timestamp: string): number {
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function mergeSpectateEvents(
  current: SpectateEvent[],
  incoming: SpectateEvent[],
  maxEvents: number,
): SpectateEvent[] {
  const deduped = new Map<string, SpectateEvent>();

  for (const event of current) {
    deduped.set(getSpectateEventKey(event), event);
  }
  for (const event of incoming) {
    deduped.set(getSpectateEventKey(event), event);
  }

  return Array.from(deduped.values())
    .sort((left, right) => toTimestamp(left.timestamp) - toTimestamp(right.timestamp))
    .slice(-maxEvents);
}

/**
 * React hook for real-time spectate events from the SpectatorStream bridge.
 *
 * Bootstraps from /api/v1/spectate/recent, then prefers the live SSE stream
 * with polling fallback when EventSource is unavailable or drops.
 *
 * @example
 * ```tsx
 * function DebateViewer({ debateId }: { debateId: string }) {
 *   const { events, connected } = useSpectate({ debateId });
 *
 *   return (
 *     <div>
 *       {connected ? 'Live' : 'Disconnected'}
 *       {events.map((e, i) => (
 *         <div key={i}>{e.event_type}: {e.agent_name}</div>
 *       ))}
 *     </div>
 *   );
 * }
 * ```
 */
export function useSpectate(
  debateId?: string,
  pipelineId?: string,
  options: UseSpectateOptions = {},
): UseSpectateReturn {
  const {
    pollInterval = 2000,
    maxEvents = 50,
    enabled = true,
  } = options;

  const [events, setEvents] = useState<SpectateEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [status, setStatus] = useState<SpectateStatus | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const recentPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const statusPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchRecent = useCallback(async () => {
    try {
      const params = new URLSearchParams({ count: String(maxEvents) });
      if (debateId) params.set('debate_id', debateId);
      if (pipelineId) params.set('pipeline_id', pipelineId);

      const res = await fetch(
        `${API_BASE_URL}/api/v1/spectate/recent?${params.toString()}`,
      );
      if (res.ok) {
        const data = await res.json();
        setEvents(data.events || []);
        return true;
      } else {
        setEvents([]);
        return false;
      }
    } catch {
      setEvents([]);
      return false;
    }
  }, [debateId, pipelineId, maxEvents]);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/spectate/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        return true;
      }
    } catch {
      // Status fetch is best-effort
    }

    setStatus(null);
    return false;
  }, []);

  const stopRecentPolling = useCallback(() => {
    if (recentPollRef.current) {
      clearInterval(recentPollRef.current);
      recentPollRef.current = null;
    }
  }, []);

  const stopEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const startRecentPolling = useCallback(() => {
    if (recentPollRef.current) {
      return;
    }

    recentPollRef.current = setInterval(() => {
      void fetchRecent().then((recentOk) => {
        setConnected(recentOk);
      });
    }, pollInterval);
  }, [fetchRecent, pollInterval]);

  const startEventSource = useCallback((): boolean => {
    if (typeof EventSource === 'undefined') {
      return false;
    }

    const params = new URLSearchParams({ count: String(maxEvents), format: 'sse' });
    if (debateId) params.set('debate_id', debateId);
    if (pipelineId) params.set('pipeline_id', pipelineId);

    try {
      const eventSource = new EventSource(
        `${API_BASE_URL}/api/v1/spectate/stream?${params.toString()}`,
      );

      const handleLiveEvent = (message: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(message.data) as SpectateEvent;
          if (!payload?.event_type) {
            return;
          }
          setEvents((current) => mergeSpectateEvents(current, [payload], maxEvents));
          setConnected(true);
        } catch {
          // Ignore malformed stream events and keep the live connection running.
        }
      };

      eventSource.onopen = () => {
        stopRecentPolling();
        setConnected(true);
      };

      eventSource.addEventListener('heartbeat', () => {
        setConnected(true);
      });

      for (const eventType of SPECTATE_STREAM_EVENT_TYPES) {
        eventSource.addEventListener(eventType, handleLiveEvent as EventListener);
      }

      eventSource.onerror = () => {
        if (eventSourceRef.current !== eventSource) {
          return;
        }
        stopEventSource();
        setConnected(false);
        void fetchRecent().then((recentOk) => {
          setConnected(recentOk);
        });
        startRecentPolling();
      };

      eventSourceRef.current = eventSource;
      return true;
    } catch {
      return false;
    }
  }, [
    debateId,
    fetchRecent,
    maxEvents,
    pipelineId,
    startRecentPolling,
    stopEventSource,
    stopRecentPolling,
  ]);

  const refresh = useCallback(async () => {
    const [recentOk] = await Promise.all([
      fetchRecent(),
      fetchStatus(),
    ]);
    setConnected(recentOk);
    setLoaded(true);
  }, [fetchRecent, fetchStatus]);

  useEffect(() => {
    if (!enabled) return;

    let active = true;
    setEvents([]);
    setConnected(false);
    setLoaded(false);

    void refresh().then(() => {
      if (!active) {
        return;
      }
      if (!startEventSource()) {
        startRecentPolling();
      }
    });

    statusPollRef.current = setInterval(() => {
      void fetchStatus();
    }, pollInterval);

    return () => {
      active = false;
      stopEventSource();
      stopRecentPolling();
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current);
        statusPollRef.current = null;
      }
    };
  }, [
    enabled,
    fetchStatus,
    pollInterval,
    refresh,
    startEventSource,
    startRecentPolling,
    stopEventSource,
    stopRecentPolling,
  ]);

  return { events, connected, loaded, status, refresh };
}
