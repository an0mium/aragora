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

/**
 * React hook for real-time spectate events from the SpectatorStream bridge.
 *
 * Polls the /api/v1/spectate/recent endpoint at a configurable interval
 * and optionally filters events by debate or pipeline ID.
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
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

    void refresh();
    intervalRef.current = setInterval(() => {
      void refresh();
    }, pollInterval);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [refresh, pollInterval, enabled]);

  return { events, connected, loaded, status, refresh };
}
