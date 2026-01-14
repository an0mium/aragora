'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { StreamEvent } from '@/types/events';
import { logger } from '@/utils/logger';

import { API_BASE_URL } from '@/config';

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.aragora.ai/ws';

// Timeout for waiting for debate_start event
const DEBATE_START_TIMEOUT_MS = 15000; // 15 seconds

export interface TranscriptMessage {
  agent: string;
  role?: string;
  content: string;
  round?: number;
  timestamp?: number;
}

export interface StreamingMessage {
  agent: string;
  content: string;
  isComplete: boolean;
  startTime: number;
  expectedSeq: number;  // Next expected agent_seq for ordering
  pendingTokens: Map<number, string>;  // Buffer for out-of-order tokens
}

export type DebateConnectionStatus = 'connecting' | 'streaming' | 'complete' | 'error';

// Reconnection configuration
const MAX_RECONNECT_ATTEMPTS = 15;
const MAX_RECONNECT_DELAY_MS = 30000; // 30 seconds cap

interface UseDebateWebSocketOptions {
  debateId: string;
  wsUrl?: string;
  enabled?: boolean;
}

// Debate status from server API
interface DebateStatus {
  status: string;
  error?: string;
  task?: string;
  agents?: string[];
}

interface UseDebateWebSocketReturn {
  // Connection state
  status: DebateConnectionStatus;
  error: string | null;
  errorDetails: string | null;  // Detailed error message from server
  isConnected: boolean;
  reconnectAttempt: number;  // Expose for UI feedback

  // Debate data
  task: string;
  agents: string[];
  messages: TranscriptMessage[];
  streamingMessages: Map<string, StreamingMessage>;
  streamEvents: StreamEvent[];
  hasCitations: boolean;

  // Actions
  sendVote: (choice: string, intensity?: number) => void;
  sendSuggestion: (suggestion: string) => void;
  registerAckCallback: (callback: (msgType: string) => void) => () => void;
  registerErrorCallback: (callback: (message: string) => void) => () => void;
  reconnect: () => void;  // Manual reconnect trigger
}

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

  // Message deduplication
  const seenMessagesRef = useRef<Set<string>>(new Set());

  // Sequence tracking for gap detection
  const lastSeqRef = useRef<number>(0);

  // Stream events buffer limit to prevent unbounded memory growth
  const MAX_STREAM_EVENTS = 500;

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
  const STREAM_TIMEOUT_MS = 60000; // 60 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setStreamingMessages(prev => {
        const now = Date.now();
        const updated = new Map(prev);
        let changed = false;

        for (const [agent, msg] of Array.from(updated.entries())) {
          if (now - msg.startTime > STREAM_TIMEOUT_MS) {
            // Convert stale stream to completed message with timeout indicator
            if (msg.content) {
              const timedOutMsg = {
                agent: msg.agent,
                content: msg.content + ' [stream timed out]',
                timestamp: Date.now() / 1000,
              };
              // Deduplication check
              const msgKey = `${timedOutMsg.agent}-${timedOutMsg.timestamp}-${timedOutMsg.content.slice(0, 50)}`;
              if (!seenMessagesRef.current.has(msgKey)) {
                seenMessagesRef.current.add(msgKey);
                setMessages(prevMsgs => [...prevMsgs, timedOutMsg]);
              }
            }
            updated.delete(agent);
            changed = true;
          }
        }

        return changed ? updated : prev;
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Clear deduplication set and stream events when debate ends to prevent memory leak
  useEffect(() => {
    if (status === 'complete' || status === 'error') {
      seenMessagesRef.current.clear();
      setStreamEvents([]);
    }
  }, [status]);

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

  // Register acknowledgment callback
  const registerAckCallback = useCallback((callback: (msgType: string) => void) => {
    ackCallbackRef.current = callback;
    return () => { ackCallbackRef.current = null; };
  }, []);

  // Register error callback
  const registerErrorCallback = useCallback((callback: (message: string) => void) => {
    errorCallbackRef.current = callback;
    return () => { errorCallbackRef.current = null; };
  }, []);

  // Clear any pending reconnection timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  // Clear debate start timeout
  const clearDebateStartTimeout = useCallback(() => {
    if (debateStartTimeoutRef.current) {
      clearTimeout(debateStartTimeoutRef.current);
      debateStartTimeoutRef.current = null;
    }
  }, []);

  // Fetch debate status from HTTP API to check for errors
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

  // Handle debate start timeout - fetch status and show error if debate failed
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
      // If status is running/active but we haven't received events, could be network issue
      if (debateStatus.status === 'running' || debateStatus.status === 'active') {
        // Set task/agents from API if we have them
        if (debateStatus.task) setTask(debateStatus.task);
        if (debateStatus.agents) setAgents(debateStatus.agents);
        // Keep trying to connect
        return;
      }
    }

    // If we couldn't determine status, show generic error after several reconnect attempts
    if (reconnectAttempt >= 3) {
      setStatus('error');
      setError('Debate failed to start');
      setErrorDetails('The debate did not start within the expected time. This could be due to invalid configuration or server issues.');
    }
  }, [debateId, hasReceivedDebateStart, reconnectAttempt, fetchDebateStatus]);

  // Schedule reconnection with exponential backoff
  const scheduleReconnect = useCallback(() => {
    if (isUnmountedRef.current) return;
    if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      setStatus('error');
      setError(`Connection lost. Max reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached.`);
      errorCallbackRef.current?.(`Connection lost after ${MAX_RECONNECT_ATTEMPTS} attempts`);
      return;
    }

    // Exponential backoff: 1s, 2s, 4s, 8s, 16s (capped at 30s)
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempt), MAX_RECONNECT_DELAY_MS);
    logger.debug(`[WebSocket] Scheduling reconnect attempt ${reconnectAttempt + 1} in ${delay}ms`);

    clearReconnectTimeout();
    reconnectTimeoutRef.current = setTimeout(() => {
      if (!isUnmountedRef.current) {
        setReconnectAttempt(prev => prev + 1);
        // Reconnection will be triggered by the useEffect dependency on reconnectAttempt
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
  }, [clearReconnectTimeout, clearDebateStartTimeout]);

  // Helper to add message with deduplication
  const addMessageIfNew = useCallback((msg: TranscriptMessage) => {
    // Create key from agent + timestamp + content prefix for deduplication
    const msgKey = `${msg.agent}-${msg.timestamp}-${msg.content.slice(0, 50)}`;
    if (seenMessagesRef.current.has(msgKey)) return false;
    seenMessagesRef.current.add(msgKey);
    setMessages(prev => [...prev, msg]);
    return true;
  }, []);

  // Handle incoming WebSocket message
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);

      // Track sequence numbers for gap detection
      if (data.seq && data.seq > 0) {
        if (lastSeqRef.current > 0 && data.seq > lastSeqRef.current + 1) {
          const gap = data.seq - lastSeqRef.current - 1;
          console.warn(`[WebSocket] Sequence gap detected: expected ${lastSeqRef.current + 1}, got ${data.seq} (${gap} events missed)`);
        }
        lastSeqRef.current = data.seq;
      }

      // Check if event belongs to this debate
      const eventDebateId = data.loop_id || data.data?.debate_id || data.data?.loop_id;
      const isOurDebate = !eventDebateId || eventDebateId === debateId;

      if (!isOurDebate) return;

      // Handle queue overflow notifications
      if (data.type === 'error' && data.data?.error_type === 'queue_overflow') {
        console.warn('[WebSocket] Server queue overflow:', data.data.message);
        errorCallbackRef.current?.(`Some updates may be missing (${data.data.dropped_count} events dropped)`);
        return;
      }

      // Debate lifecycle events
      if (data.type === 'debate_start') {
        setTask(data.data.task || 'Debate in progress...');
        setAgents(data.data.agents || []);
        setHasReceivedDebateStart(true);
        clearDebateStartTimeout();
      } else if (data.type === 'debate_end') {
        setStatus('complete');
        clearDebateStartTimeout();
      }

      // Agent message events
      else if (data.type === 'debate_message' || data.type === 'agent_message') {
        const msg: TranscriptMessage = {
          agent: data.agent || data.data?.agent || 'unknown',
          role: data.data?.role,
          content: data.data?.content || '',
          round: data.round || data.data?.round,
          timestamp: data.timestamp || data.data?.timestamp || Date.now() / 1000,
        };
        if (msg.content && addMessageIfNew(msg)) {
          const agentName = msg.agent;
          if (agentName) {
            setAgents(prev => prev.includes(agentName) ? prev : [...prev, agentName]);
          }
        }

        // Also track as stream event
        const streamEvent: StreamEvent = {
          type: 'agent_message',
          data: {
            agent: data.agent || data.data?.agent || 'unknown',
            content: data.data?.content || '',
            role: data.data?.role || '',
          },
          timestamp: data.timestamp || Date.now() / 1000,
          round: data.round || data.data?.round,
          agent: data.agent || data.data?.agent,
        };
        addStreamEvent(streamEvent);
      }

      // Legacy agent_response events
      else if (data.type === 'agent_response') {
        const msg: TranscriptMessage = {
          agent: data.data?.agent || 'unknown',
          role: data.data?.role,
          content: data.data?.content || data.data?.response || '',
          round: data.data?.round,
          timestamp: Date.now() / 1000,
        };
        if (msg.content) {
          addMessageIfNew(msg);
        }
      }

      // Token streaming events with sequence-based ordering
      else if (data.type === 'token_start') {
        const agent = data.agent || data.data?.agent;
        if (agent) {
          setStreamingMessages(prev => {
            const updated = new Map(prev);
            updated.set(agent, {
              agent,
              content: '',
              isComplete: false,
              startTime: Date.now(),
              expectedSeq: 1,  // First token should have agent_seq=1
              pendingTokens: new Map(),
            });
            return updated;
          });
          setAgents(prev => prev.includes(agent) ? prev : [...prev, agent]);
        }
      } else if (data.type === 'token_delta') {
        const agent = data.agent || data.data?.agent;
        const token = data.data?.token || '';
        const agentSeq = data.agent_seq || 0;  // Per-agent sequence number

        if (agent && token) {
          setStreamingMessages(prev => {
            const updated = new Map(prev);
            const existing = updated.get(agent);

            if (existing) {
              // If we have sequence info, use it for ordering
              if (agentSeq > 0) {
                // Check if this is the expected sequence
                if (agentSeq === existing.expectedSeq) {
                  // Token is in order - append it
                  let newContent = existing.content + token;
                  let nextExpected = agentSeq + 1;

                  // Check if we have buffered tokens that can now be appended
                  const pending = new Map(existing.pendingTokens);
                  while (pending.has(nextExpected)) {
                    newContent += pending.get(nextExpected)!;
                    pending.delete(nextExpected);
                    nextExpected++;
                  }

                  updated.set(agent, {
                    ...existing,
                    content: newContent,
                    expectedSeq: nextExpected,
                    pendingTokens: pending,
                  });
                } else if (agentSeq > existing.expectedSeq) {
                  // Token arrived out of order - buffer it
                  const pending = new Map(existing.pendingTokens);
                  pending.set(agentSeq, token);
                  updated.set(agent, {
                    ...existing,
                    pendingTokens: pending,
                  });
                }
                // Ignore tokens with seq < expectedSeq (duplicate/old)
              } else {
                // No sequence info - fall back to simple append (backward compat)
                updated.set(agent, {
                  ...existing,
                  content: existing.content + token,
                });
              }
            } else {
              // First token for this agent (no token_start received)
              updated.set(agent, {
                agent,
                content: token,
                isComplete: false,
                startTime: Date.now(),
                expectedSeq: agentSeq > 0 ? agentSeq + 1 : 1,
                pendingTokens: new Map(),
              });
            }
            return updated;
          });
        }
      } else if (data.type === 'token_end') {
        const agent = data.agent || data.data?.agent;
        if (agent) {
          setStreamingMessages(prev => {
            const updated = new Map(prev);
            const existing = updated.get(agent);
            if (existing) {
              // Flush any remaining buffered tokens in order
              let finalContent = existing.content;
              if (existing.pendingTokens.size > 0) {
                const sortedSeqs = Array.from(existing.pendingTokens.keys()).sort((a, b) => a - b);
                for (const seq of sortedSeqs) {
                  finalContent += existing.pendingTokens.get(seq)!;
                }
              }

              if (finalContent) {
                const msg: TranscriptMessage = {
                  agent,
                  content: finalContent,
                  timestamp: Date.now() / 1000,
                };
                // Deduplication check
                const msgKey = `${msg.agent}-${msg.timestamp}-${msg.content.slice(0, 50)}`;
                if (!seenMessagesRef.current.has(msgKey)) {
                  seenMessagesRef.current.add(msgKey);
                  setMessages(prevMsgs => [...prevMsgs, msg]);
                }
              }
            }
            updated.delete(agent);
            return updated;
          });
        }
      }

      // Critique events
      else if (data.type === 'critique') {
        const msg: TranscriptMessage = {
          agent: data.agent || data.data?.agent || 'unknown',
          role: 'critic',
          content: `[CRITIQUE → ${data.data?.target || 'unknown'}] ${data.data?.issues?.join('; ') || data.data?.content || ''}`,
          round: data.round || data.data?.round,
          timestamp: data.timestamp || Date.now() / 1000,
        };
        if (msg.content) {
          addMessageIfNew(msg);
        }
      }

      // Consensus events
      else if (data.type === 'consensus') {
        const msg: TranscriptMessage = {
          agent: 'system',
          role: 'synthesizer',
          content: `[CONSENSUS ${data.data?.reached ? 'REACHED' : 'NOT REACHED'}] Confidence: ${Math.round((data.data?.confidence || 0) * 100)}%`,
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addMessageIfNew(msg);
      }

      // Acknowledgment events
      else if (data.type === 'ack') {
        const msgType = data.data?.message_type || '';
        if (ackCallbackRef.current) {
          ackCallbackRef.current(msgType);
        }
      }

      // Error events
      else if (data.type === 'error') {
        const errorMsg = data.data?.message || 'Unknown error';
        if (errorCallbackRef.current) {
          errorCallbackRef.current(errorMsg);
        }
        // If this is a fatal error (e.g., invalid agent type), set error state
        if (data.data?.fatal || data.data?.error_type === 'validation_error') {
          setStatus('error');
          setError('Debate failed');
          setErrorDetails(errorMsg);
          clearDebateStartTimeout();
        }
      }

      // Debate error event (sent when debate fails to start)
      else if (data.type === 'debate_error') {
        const errorMsg = data.data?.message || data.data?.error || 'Debate failed to start';
        setStatus('error');
        setError('Debate failed to start');
        setErrorDetails(errorMsg);
        clearDebateStartTimeout();
        if (errorCallbackRef.current) {
          errorCallbackRef.current(errorMsg);
        }
      }

      // Audience events
      else if (data.type === 'audience_summary' || data.type === 'audience_metrics') {
        const event: StreamEvent = {
          type: data.type,
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addStreamEvent(event);
      }

      // Grounded verdict events (citations)
      else if (data.type === 'grounded_verdict') {
        const event: StreamEvent = {
          type: 'grounded_verdict',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addStreamEvent(event);
        setHasCitations(true);
      }

      // Uncertainty analysis events (disagreement detection)
      else if (data.type === 'uncertainty_analysis') {
        const event: StreamEvent = {
          type: 'uncertainty_analysis',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addStreamEvent(event);
      }

      // Vote events (for analytics panels)
      else if (data.type === 'vote') {
        const event: StreamEvent = {
          type: 'vote',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
          agent: data.agent || data.data?.agent,
        };
        addStreamEvent(event);
      }

      // Rhetorical observation events
      else if (data.type === 'rhetorical_observation') {
        const event: StreamEvent = {
          type: 'rhetorical_observation',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
          agent: data.agent || data.data?.agent,
          round: data.round || data.data?.round,
        };
        addStreamEvent(event);
      }

      // Hollow consensus / trickster events
      else if (data.type === 'hollow_consensus' || data.type === 'trickster_intervention') {
        const event: StreamEvent = {
          type: data.type,
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addStreamEvent(event);
      }

      // Memory recall events
      else if (data.type === 'memory_recall') {
        const event: StreamEvent = {
          type: 'memory_recall',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addStreamEvent(event);
      }

      // Flip detected events
      else if (data.type === 'flip_detected') {
        const event: StreamEvent = {
          type: 'flip_detected',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addStreamEvent(event);
      }

      // Evidence found events (real-time evidence collection)
      else if (data.type === 'evidence_found') {
        const event: StreamEvent = {
          type: 'evidence_found',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        addStreamEvent(event);
        // Also mark that we have citations available
        if (data.data?.count > 0) {
          setHasCitations(true);
        }
      }
    } catch (e) {
      logger.error('Failed to parse WebSocket message:', e);
    }
  }, [debateId, addMessageIfNew, addStreamEvent, clearDebateStartTimeout]);

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

    // Don't reconnect if we've reached max attempts
    if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS && status === 'error') {
      return;
    }

    setStatus('connecting');
    let ws: WebSocket;

    try {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;
    } catch (e) {
      logger.error('[WebSocket] Failed to create connection:', e);
      setStatus('error');
      setError('Failed to establish WebSocket connection');
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      logger.debug(`[WebSocket] Connected (attempt ${reconnectAttempt + 1})`);
      setStatus('streaming');
      setError(null);
      setErrorDetails(null);
      setReconnectAttempt(0);  // Reset on successful connection
      lastSeqRef.current = 0;  // Reset sequence tracking
      ws.send(JSON.stringify({ type: 'subscribe', debate_id: debateId }));

      // Start timeout for debate_start event
      // If we don't receive debate_start within timeout, check debate status via API
      clearDebateStartTimeout();
      debateStartTimeoutRef.current = setTimeout(() => {
        handleDebateStartTimeout();
      }, DEBATE_START_TIMEOUT_MS);
    };

    ws.onmessage = handleMessage;

    ws.onerror = (e) => {
      logger.error('[WebSocket] Connection error:', e);
      // Don't set error status here - let onclose handle it
      // This prevents duplicate error handling
    };

    ws.onclose = (event) => {
      wsRef.current = null;

      // Normal closure (code 1000) or debate ended
      if (event.code === 1000 || status === 'complete') {
        setStatus('complete');
        return;
      }

      // Abnormal closure - attempt reconnection
      logger.warn(`[WebSocket] Connection closed (code: ${event.code}, reason: ${event.reason || 'none'})`);

      if (!isUnmountedRef.current) {
        setStatus('connecting');
        setError(`Connection lost (code: ${event.code}). Reconnecting...`);
        scheduleReconnect();
      }
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close(1000, 'Component unmounted');
      }
    };
  }, [enabled, wsUrl, debateId, handleMessage, reconnectAttempt, scheduleReconnect, status, handleDebateStartTimeout, clearDebateStartTimeout]);

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
