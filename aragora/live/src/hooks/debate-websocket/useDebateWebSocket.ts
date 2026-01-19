'use client';

/**
 * Debate WebSocket Hook
 *
 * Provides real-time WebSocket communication for debates.
 * Handles connection management, event processing, token streaming,
 * reconnection with exponential backoff, and state synchronization.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type { StreamEvent } from '@/types/events';
import { logger } from '@/utils/logger';
import { API_BASE_URL } from '@/config';

// Import types from local module
import type {
  TranscriptMessage,
  StreamingMessage,
  DebateConnectionStatus,
  UseDebateWebSocketOptions,
  UseDebateWebSocketReturn,
  DebateStatus,
} from './types';

// Import constants from local module
import {
  DEFAULT_WS_URL,
  DEBATE_START_TIMEOUT_MS,
  MAX_RECONNECT_ATTEMPTS,
  MAX_RECONNECT_DELAY_MS,
  STREAM_TIMEOUT_MS,
  STALL_WARNING_MS,
  MAX_STREAM_EVENTS,
} from './constants';

// Import event handlers
import { eventHandlerRegistry, type EventHandlerContext, type ParsedEventData } from './eventHandlers';

// Re-export types for convenience
export type {
  TranscriptMessage,
  StreamingMessage,
  DebateConnectionStatus,
  UseDebateWebSocketOptions,
  UseDebateWebSocketReturn,
};

export function useDebateWebSocket({
  debateId,
  wsUrl = DEFAULT_WS_URL,
  enabled = true,
}: UseDebateWebSocketOptions): UseDebateWebSocketReturn {
  const [status, setStatus] = useState<DebateConnectionStatus>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);
  const [task, setTask] = useState<string>('');
  const [agents, setAgents] = useState<string[]>([]);
  const [messages, setMessages] = useState<TranscriptMessage[]>([]);
  const [streamingMessages, setStreamingMessages] = useState<Map<string, StreamingMessage>>(new Map());
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [hasCitations, setHasCitations] = useState(false);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [hasReceivedDebateStart, setHasReceivedDebateStart] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const ackCallbackRef = useRef<((msgType: string) => void) | null>(null);
  const errorCallbackRef = useRef<((message: string) => void) | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const debateStartTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmountedRef = useRef(false);

  // Refs for callbacks to avoid triggering useEffect re-runs
  const handleMessageRef = useRef<((event: MessageEvent) => void) | null>(null);
  const scheduleReconnectRef = useRef<(() => void) | null>(null);
  const handleDebateStartTimeoutRef = useRef<(() => void) | null>(null);

  // Reconnection trigger - increment to force reconnection
  const [reconnectTrigger, setReconnectTrigger] = useState(0);

  // Message deduplication
  const seenMessagesRef = useRef<Set<string>>(new Set());

  // Sequence tracking for gap detection
  const lastSeqRef = useRef<number>(0);

  // Track last event timestamp for stall detection
  const lastEventTimestampRef = useRef<number>(Date.now());

  // Helper to add stream event with size limit
  const addStreamEvent = useCallback((event: StreamEvent) => {
    setStreamEvents(prev => {
      const updated = [...prev, event];
      if (updated.length > MAX_STREAM_EVENTS) {
        return updated.slice(-MAX_STREAM_EVENTS);
      }
      return updated;
    });
  }, []);

  // Orphaned stream cleanup - handles agents that never send token_end
  useEffect(() => {
    const interval = setInterval(() => {
      setStreamingMessages(prev => {
        const now = Date.now();
        const updated = new Map(prev);
        let changed = false;

        for (const [streamKey, msg] of Array.from(updated.entries())) {
          if (now - msg.startTime > STREAM_TIMEOUT_MS) {
            // Convert stale stream to completed message with timeout indicator
            if (msg.content) {
              const timedOutMsg = {
                agent: msg.agent,
                content: msg.content + ' [stream timed out]',
                timestamp: Date.now() / 1000,
              };
              const msgKey = `${timedOutMsg.agent}-${timedOutMsg.content.slice(0, 100)}`;
              if (!seenMessagesRef.current.has(msgKey)) {
                seenMessagesRef.current.add(msgKey);
                setMessages(prevMsgs => [...prevMsgs, timedOutMsg]);
              }
            }
            updated.delete(streamKey);
            changed = true;
          }
        }

        return changed ? updated : prev;
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Clear deduplication set and stream events when debate ends
  useEffect(() => {
    if (status === 'complete' || status === 'error') {
      seenMessagesRef.current.clear();
      setStreamEvents([]);
    }
  }, [status]);

  // Stall detection - warn if no events received for 2 minutes during streaming
  useEffect(() => {
    if (status !== 'streaming') return;

    const checkStall = () => {
      const timeSinceLastEvent = Date.now() - lastEventTimestampRef.current;
      if (timeSinceLastEvent > STALL_WARNING_MS) {
        console.warn(`[WS] Debate may be stalled - no events for ${Math.round(timeSinceLastEvent / 1000)}s`);
        console.warn('[WS] Check server logs for consensus_phase or synthesis_or_hooks_failed errors');
      }
    };

    const interval = setInterval(checkStall, 30000);
    return () => clearInterval(interval);
  }, [status]);

  // Reset all state when debateId changes
  useEffect(() => {
    setMessages([]);
    setStreamingMessages(new Map());
    setStreamEvents([]);
    setTask('');
    setAgents([]);
    setHasCitations(false);
    setStatus('connecting');
    setError(null);
    setErrorDetails(null);
    setHasReceivedDebateStart(false);
    seenMessagesRef.current.clear();
    lastSeqRef.current = 0;
  }, [debateId]);

  // Send vote to server
  const sendVote = useCallback((choice: string, intensity?: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'user_vote',
        debate_id: debateId,
        data: { choice, intensity: intensity ?? 5 },
      }));
    }
  }, [debateId]);

  // Send suggestion to server
  const sendSuggestion = useCallback((suggestion: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'user_suggestion',
        debate_id: debateId,
        data: { suggestion },
      }));
    }
  }, [debateId]);

  // Register callbacks
  const registerAckCallback = useCallback((callback: (msgType: string) => void) => {
    ackCallbackRef.current = callback;
    return () => { ackCallbackRef.current = null; };
  }, []);

  const registerErrorCallback = useCallback((callback: (message: string) => void) => {
    errorCallbackRef.current = callback;
    return () => { errorCallbackRef.current = null; };
  }, []);

  // Clear timeouts
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const clearDebateStartTimeout = useCallback(() => {
    if (debateStartTimeoutRef.current) {
      clearTimeout(debateStartTimeoutRef.current);
      debateStartTimeoutRef.current = null;
    }
  }, []);

  // Fetch debate status from HTTP API
  const fetchDebateStatus = useCallback(async (): Promise<DebateStatus | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/debates/${debateId}`);
      if (response.ok) {
        const data = await response.json();
        return {
          status: data.status || 'unknown',
          error: data.error || data.error_message,
          task: data.task,
          agents: data.agents,
        };
      } else if (response.status === 404) {
        return { status: 'not_found', error: 'Debate not found' };
      }
      return null;
    } catch (e) {
      logger.debug('Failed to fetch debate status:', e);
      return null;
    }
  }, [debateId]);

  // Handle debate start timeout
  const handleDebateStartTimeout = useCallback(async () => {
    if (isUnmountedRef.current || hasReceivedDebateStart) return;

    logger.warn(`[WebSocket] No debate_start received within ${DEBATE_START_TIMEOUT_MS}ms, checking debate status`);

    const debateStatus = await fetchDebateStatus();

    if (isUnmountedRef.current) return;

    if (debateStatus) {
      if (debateStatus.status === 'error' || debateStatus.status === 'failed') {
        setStatus('error');
        setError('Debate failed to start');
        setErrorDetails(debateStatus.error || 'Unknown error occurred while starting the debate');
        errorCallbackRef.current?.(debateStatus.error || 'Debate failed to start');
        return;
      }
      if (debateStatus.status === 'not_found') {
        setStatus('error');
        setError('Debate not found');
        setErrorDetails('The debate could not be found. It may have been deleted or never created.');
        return;
      }
      if (debateStatus.status === 'running' || debateStatus.status === 'active') {
        if (debateStatus.task) setTask(debateStatus.task);
        if (debateStatus.agents) setAgents(debateStatus.agents);
        return;
      }
    }

    if (reconnectAttempt >= 3) {
      setStatus('error');
      setError('Debate failed to start');
      setErrorDetails('The debate did not start within the expected time. This could be due to invalid configuration or server issues.');
    }
  }, [hasReceivedDebateStart, reconnectAttempt, fetchDebateStatus]);

  // Schedule reconnection with exponential backoff
  const scheduleReconnect = useCallback(() => {
    if (isUnmountedRef.current) return;
    if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      setStatus('error');
      setError(`Connection lost. Max reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached.`);
      errorCallbackRef.current?.(`Connection lost after ${MAX_RECONNECT_ATTEMPTS} attempts`);
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, reconnectAttempt), MAX_RECONNECT_DELAY_MS);
    logger.debug(`[WebSocket] Scheduling reconnect attempt ${reconnectAttempt + 1} in ${delay}ms`);

    clearReconnectTimeout();
    reconnectTimeoutRef.current = setTimeout(() => {
      if (!isUnmountedRef.current) {
        setReconnectAttempt(prev => prev + 1);
        setReconnectTrigger(prev => prev + 1);
      }
    }, delay);
  }, [reconnectAttempt, clearReconnectTimeout]);

  // Manual reconnect trigger
  const reconnect = useCallback(() => {
    clearReconnectTimeout();
    clearDebateStartTimeout();
    setReconnectAttempt(0);
    setStatus('connecting');
    setError(null);
    setErrorDetails(null);
    setHasReceivedDebateStart(false);
    setReconnectTrigger(prev => prev + 1);
  }, [clearReconnectTimeout, clearDebateStartTimeout]);

  // Helper to add message with deduplication
  const addMessageIfNew = useCallback((msg: TranscriptMessage) => {
    let msgKey: string;

    if (msg.role === 'critic') {
      msgKey = `${msg.agent}-critic-r${msg.round || 0}`;
    } else {
      msgKey = `${msg.agent}-${msg.content.slice(0, 100)}`;
    }

    if (seenMessagesRef.current.has(msgKey)) return false;
    seenMessagesRef.current.add(msgKey);
    setMessages(prev => [...prev, msg]);
    return true;
  }, []);

  // Create event handler context for extracted handlers
  const createHandlerContext = useCallback((): EventHandlerContext => ({
    debateId,
    setTask,
    setAgents,
    setStatus,
    setError,
    setErrorDetails,
    setHasCitations,
    setHasReceivedDebateStart,
    setStreamingMessages,
    addMessageIfNew,
    addStreamEvent,
    clearDebateStartTimeout,
    errorCallbackRef,
    ackCallbackRef,
    seenMessagesRef,
    lastSeqRef,
  }), [debateId, addMessageIfNew, addStreamEvent, clearDebateStartTimeout]);

  // Process a single event from the WebSocket
  const processEvent = useCallback((data: Record<string, unknown>) => {
    // Update last event timestamp for stall detection
    lastEventTimestampRef.current = Date.now();

    // Debug logging for all WebSocket events
    if (process.env.NODE_ENV === 'development') {
      const eventInfo = data.agent ? ` from ${data.agent}` : '';
      const seqInfo = data.seq ? ` (seq=${data.seq})` : '';
      logger.debug(`[WS] Event: ${data.type}${eventInfo}${seqInfo}`);
    }

    // Enhanced logging for debate completion events
    if (data.type === 'consensus' || data.type === 'debate_end') {
      logger.debug(`[WS] DEBATE COMPLETION: ${data.type}`, data);
    }

    // Track sequence numbers for gap detection
    if (data.seq && typeof data.seq === 'number' && data.seq > 0) {
      const isTokenEvent = data.type === 'token_delta';
      if (lastSeqRef.current > 0 && data.seq > lastSeqRef.current + 1 && !isTokenEvent) {
        const gap = data.seq - lastSeqRef.current - 1;
        if (gap <= 2) {
          logger.debug(`Sequence reorder: expected ${lastSeqRef.current + 1}, got ${data.seq}`);
        } else if (gap <= 5) {
          console.warn(`[WebSocket] Sequence gap: ${gap} events (minor, likely reordering)`);
        } else {
          console.error(`[WebSocket] Large sequence gap: ${gap} events missed - may have lost data`);
        }
      }
      lastSeqRef.current = data.seq as number;
    }

    // Check if event belongs to this debate
    const eventData = data.data as Record<string, unknown> | undefined;
    const eventDebateId = data.loop_id || eventData?.debate_id || eventData?.loop_id;
    const isOurDebate = !eventDebateId || eventDebateId === debateId;

    if (!isOurDebate) return;

    // Get handler for this event type
    const eventType = data.type as string;
    const handler = eventHandlerRegistry[eventType];

    if (handler) {
      const ctx = createHandlerContext();
      const parsedEvent: ParsedEventData = {
        type: eventType,
        agent: data.agent as string,
        seq: data.seq as number,
        agent_seq: data.agent_seq as number,
        loop_id: data.loop_id as string,
        task_id: data.task_id as string,
        round: data.round as number,
        timestamp: data.timestamp as number,
        data: eventData,
      };
      handler(parsedEvent, ctx);
    }
  }, [debateId, createHandlerContext]);

  // Handle incoming WebSocket message
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const parsed = JSON.parse(event.data);

      if (Array.isArray(parsed)) {
        if (process.env.NODE_ENV === 'development' && parsed.length > 10) {
          const tokenDeltas = parsed.filter((e: Record<string, unknown>) => e.type === 'token_delta').length;
          logger.debug(`[WS] BATCH: ${parsed.length} events (${tokenDeltas} token_deltas)`);
        }
        for (const evt of parsed) {
          processEvent(evt as Record<string, unknown>);
        }
      } else {
        processEvent(parsed as Record<string, unknown>);
      }
    } catch (e) {
      logger.error('Failed to parse WebSocket message:', e);
    }
  }, [processEvent]);

  // Keep refs updated with latest callbacks
  useEffect(() => {
    handleMessageRef.current = handleMessage;
  }, [handleMessage]);

  useEffect(() => {
    scheduleReconnectRef.current = scheduleReconnect;
  }, [scheduleReconnect]);

  useEffect(() => {
    handleDebateStartTimeoutRef.current = handleDebateStartTimeout;
  }, [handleDebateStartTimeout]);

  // Track unmount state
  useEffect(() => {
    isUnmountedRef.current = false;
    return () => {
      isUnmountedRef.current = true;
      clearReconnectTimeout();
      clearDebateStartTimeout();
    };
  }, [clearReconnectTimeout, clearDebateStartTimeout]);

  // WebSocket connection effect
  useEffect(() => {
    if (!enabled) return;

    setStatus('connecting');
    let ws: WebSocket;

    try {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;
    } catch (e) {
      logger.error('[WebSocket] Failed to create connection:', e);
      setStatus('error');
      setError('Failed to establish WebSocket connection');
      scheduleReconnectRef.current?.();
      return;
    }

    ws.onopen = () => {
      logger.debug('[WebSocket] Connected');
      if (process.env.NODE_ENV === 'development') {
        logger.debug(`[WS] Connected to debate ${debateId}`);
      }
      setStatus('streaming');
      setError(null);
      setErrorDetails(null);
      setReconnectAttempt(0);
      lastSeqRef.current = 0;
      ws.send(JSON.stringify({ type: 'subscribe', debate_id: debateId }));

      clearDebateStartTimeout();
      debateStartTimeoutRef.current = setTimeout(() => {
        handleDebateStartTimeoutRef.current?.();
      }, DEBATE_START_TIMEOUT_MS);
    };

    ws.onmessage = (event) => {
      handleMessageRef.current?.(event);
    };

    ws.onerror = (e) => {
      logger.error('[WebSocket] Connection error:', e);
    };

    ws.onclose = (event) => {
      wsRef.current = null;

      if (event.code === 1000) {
        setStatus('complete');
        return;
      }

      logger.warn(`[WebSocket] Connection closed (code: ${event.code}, reason: ${event.reason || 'none'})`);

      if (!isUnmountedRef.current) {
        setStatus('connecting');
        setError(`Connection lost (code: ${event.code}). Reconnecting...`);
        scheduleReconnectRef.current?.();
      }
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close(1000, 'Component unmounted');
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, wsUrl, debateId, reconnectTrigger, clearDebateStartTimeout]);

  return {
    status,
    error,
    errorDetails,
    isConnected: status === 'streaming',
    reconnectAttempt,
    task,
    agents,
    messages,
    streamingMessages,
    streamEvents,
    hasCitations,
    sendVote,
    sendSuggestion,
    registerAckCallback,
    registerErrorCallback,
    reconnect,
  };
}
