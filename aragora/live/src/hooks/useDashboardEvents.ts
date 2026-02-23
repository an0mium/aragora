'use client';

/**
 * Dashboard event hook — listens for global WebSocket events and triggers
 * SWR cache revalidation so the dashboard refreshes instantly when debates
 * complete, start, or update.
 */

import { useCallback, useRef } from 'react';
import { useWebSocketBase } from './useWebSocketBase';
import { invalidateCachePattern } from './useSWRFetch';
import { WS_URL } from '@/config';

interface DashboardEvent {
  type: string;
  debate_id?: string;
  data?: Record<string, unknown>;
}

export interface UseDashboardEventsReturn {
  /** Whether the events WebSocket is connected */
  isConnected: boolean;
  /** Number of live updates received this session */
  updateCount: number;
}

/**
 * Subscribes to global debate lifecycle events via WebSocket and
 * invalidates the SWR cache so dashboard data refreshes immediately.
 *
 * Events handled: debate_end, debate_complete, debate_start, consensus
 */
export function useDashboardEvents(): UseDashboardEventsReturn {
  const updateCountRef = useRef(0);

  const onEvent = useCallback((event: DashboardEvent) => {
    const eventType = event.type;

    // Debate lifecycle events that should trigger a dashboard refresh
    const refreshTriggers = [
      'debate_end',
      'debate_complete',
      'debate_start',
      'consensus',
      'debate_created',
    ];

    if (refreshTriggers.includes(eventType)) {
      updateCountRef.current += 1;
      // Invalidate all debate-related SWR caches
      invalidateCachePattern(/\/api\/v1\/debates/);
      invalidateCachePattern(/\/api\/debates/);
      invalidateCachePattern(/\/api\/health/);
    }
  }, []);

  const { isConnected } = useWebSocketBase<DashboardEvent>({
    wsUrl: WS_URL,
    enabled: true,
    autoReconnect: true,
    subscribeMessage: { type: 'subscribe', channel: 'dashboard' },
    onEvent,
    logPrefix: '[Dashboard]',
  });

  return {
    isConnected,
    updateCount: updateCountRef.current,
  };
}

export default useDashboardEvents;
