'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { logger } from '@/utils/logger';

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.aragora.ai/ws';

// Graph debate event types
export type GraphEventType =
  | 'graph_node_added'
  | 'graph_branch_created'
  | 'graph_branch_merged'
  | 'graph_debate_complete'
  | 'debate_branch'
  | 'debate_merge';

export interface GraphDebateEvent {
  type: GraphEventType;
  data: {
    debate_id?: string;
    node_id?: string;
    node_type?: string;
    agent_id?: string;
    content?: string;
    branch_id?: string;
    branch_name?: string;
    parent_ids?: string[];
    confidence?: number;
    merged_branch_ids?: string[];
    synthesis?: string;
    [key: string]: unknown;
  };
  timestamp: number;
  seq?: number;
}

export type GraphConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface UseGraphDebateWebSocketOptions {
  debateId?: string | null;
  wsUrl?: string;
  enabled?: boolean;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

interface UseGraphDebateWebSocketReturn {
  status: GraphConnectionStatus;
  error: string | null;
  isConnected: boolean;
  events: GraphDebateEvent[];
  lastEvent: GraphDebateEvent | null;
  reconnect: () => void;
  clearEvents: () => void;
}

export function useGraphDebateWebSocket({
  debateId,
  wsUrl = DEFAULT_WS_URL,
  enabled = true,
  autoReconnect = true,
  reconnectInterval = 3000,
}: UseGraphDebateWebSocketOptions): UseGraphDebateWebSocketReturn {
  const [status, setStatus] = useState<GraphConnectionStatus>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<GraphDebateEvent[]>([]);
  const [lastEvent, setLastEvent] = useState<GraphDebateEvent | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const handleEvent = useCallback((event: GraphDebateEvent) => {
    // Filter by debate ID if provided
    if (debateId && event.data.debate_id && event.data.debate_id !== debateId) {
      return;
    }

    // Only handle graph-related events
    const graphEventTypes: GraphEventType[] = [
      'graph_node_added',
      'graph_branch_created',
      'graph_branch_merged',
      'graph_debate_complete',
      'debate_branch',
      'debate_merge',
    ];

    if (!graphEventTypes.includes(event.type)) {
      return;
    }

    logger.debug(`Graph WebSocket event: ${event.type}`, event.data);
    setLastEvent(event);
    setEvents((prev) => [...prev.slice(-99), event]); // Keep last 100 events
  }, [debateId]);

  const connect = useCallback(() => {
    if (!enabled) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      const url = wsUrl.endsWith('/') ? wsUrl.slice(0, -1) : wsUrl;
      logger.debug(`Connecting to Graph WebSocket: ${url}`);

      const ws = new WebSocket(url);
      wsRef.current = ws;
      setStatus('connecting');
      setError(null);

      ws.onopen = () => {
        logger.debug('Graph WebSocket connected');
        setStatus('connected');
        reconnectAttemptsRef.current = 0;

        // Subscribe to graph events if debate ID is provided
        if (debateId) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            channel: 'graph_debate',
            debate_id: debateId,
          }));
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as GraphDebateEvent;
          handleEvent(data);
        } catch (err) {
          logger.error('Failed to parse Graph WebSocket event:', err);
        }
      };

      ws.onerror = (err) => {
        logger.error('Graph WebSocket error:', err);
        setError('Connection error');
        setStatus('error');
      };

      ws.onclose = () => {
        logger.debug('Graph WebSocket closed');
        setStatus('disconnected');

        // Auto-reconnect logic
        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          const delay = reconnectInterval * Math.pow(1.5, reconnectAttemptsRef.current - 1);
          logger.debug(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (err) {
      logger.error('Failed to create Graph WebSocket:', err);
      setError(err instanceof Error ? err.message : 'Connection failed');
      setStatus('error');
    }
  }, [enabled, wsUrl, debateId, autoReconnect, reconnectInterval, handleEvent]);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    connect();
  }, [connect]);

  const clearEvents = useCallback(() => {
    setEvents([]);
    setLastEvent(null);
  }, []);

  // Connect on mount and when debateId changes
  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [connect]);

  return {
    status,
    error,
    isConnected: status === 'connected',
    events,
    lastEvent,
    reconnect,
    clearEvents,
  };
}
