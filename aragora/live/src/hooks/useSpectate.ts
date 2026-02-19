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

interface SpectateStatus {
  active: boolean;
  subscribers: number;
  buffer_size: number;
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
  /** Whether the hook has successfully connected to the API */
  connected: boolean;
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
        setConnected(true);
      } else {
        setConnected(false);
      }
    } catch {
      setConnected(false);
    }
  }, [debateId, pipelineId, maxEvents]);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/spectate/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch {
      // Status fetch is best-effort
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    fetchRecent();
    fetchStatus();
    intervalRef.current = setInterval(() => {
      fetchRecent();
    }, pollInterval);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchRecent, fetchStatus, pollInterval, enabled]);

  return { events, connected, status, refresh: fetchRecent };
}
