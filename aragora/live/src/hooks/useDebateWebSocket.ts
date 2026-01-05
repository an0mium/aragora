'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { StreamEvent } from '@/types/events';

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.aragora.ai/ws';

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
}

export type DebateConnectionStatus = 'connecting' | 'streaming' | 'complete' | 'error';

interface UseDebateWebSocketOptions {
  debateId: string;
  wsUrl?: string;
  enabled?: boolean;
}

interface UseDebateWebSocketReturn {
  // Connection state
  status: DebateConnectionStatus;
  error: string | null;
  isConnected: boolean;

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
}

export function useDebateWebSocket({
  debateId,
  wsUrl = DEFAULT_WS_URL,
  enabled = true,
}: UseDebateWebSocketOptions): UseDebateWebSocketReturn {
  const [status, setStatus] = useState<DebateConnectionStatus>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [task, setTask] = useState<string>('');
  const [agents, setAgents] = useState<string[]>([]);
  const [messages, setMessages] = useState<TranscriptMessage[]>([]);
  const [streamingMessages, setStreamingMessages] = useState<Map<string, StreamingMessage>>(new Map());
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [hasCitations, setHasCitations] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const ackCallbackRef = useRef<((msgType: string) => void) | null>(null);
  const errorCallbackRef = useRef<((message: string) => void) | null>(null);

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

  // Handle incoming WebSocket message
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);

      // Check if event belongs to this debate
      const eventDebateId = data.loop_id || data.data?.debate_id || data.data?.loop_id;
      const isOurDebate = !eventDebateId || eventDebateId === debateId;

      if (!isOurDebate) return;

      // Debate lifecycle events
      if (data.type === 'debate_start') {
        setTask(data.data.task || 'Debate in progress...');
        setAgents(data.data.agents || []);
      } else if (data.type === 'debate_end') {
        setStatus('complete');
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
        if (msg.content) {
          setMessages(prev => [...prev, msg]);
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
        setStreamEvents(prev => [...prev, streamEvent]);
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
          setMessages(prev => [...prev, msg]);
        }
      }

      // Token streaming events
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
            });
            return updated;
          });
          setAgents(prev => prev.includes(agent) ? prev : [...prev, agent]);
        }
      } else if (data.type === 'token_delta') {
        const agent = data.agent || data.data?.agent;
        const token = data.data?.token || '';
        if (agent && token) {
          setStreamingMessages(prev => {
            const updated = new Map(prev);
            const existing = updated.get(agent);
            if (existing) {
              updated.set(agent, {
                ...existing,
                content: existing.content + token,
              });
            } else {
              updated.set(agent, {
                agent,
                content: token,
                isComplete: false,
                startTime: Date.now(),
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
            if (existing && existing.content) {
              const msg: TranscriptMessage = {
                agent,
                content: existing.content,
                timestamp: Date.now() / 1000,
              };
              setMessages(prevMsgs => [...prevMsgs, msg]);
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
          setMessages(prev => [...prev, msg]);
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
        setMessages(prev => [...prev, msg]);
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
      }

      // Audience events
      else if (data.type === 'audience_summary' || data.type === 'audience_metrics') {
        const event: StreamEvent = {
          type: data.type,
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        setStreamEvents(prev => [...prev, event]);
      }

      // Grounded verdict events (citations)
      else if (data.type === 'grounded_verdict') {
        const event: StreamEvent = {
          type: 'grounded_verdict',
          data: data.data || {},
          timestamp: data.timestamp || Date.now() / 1000,
        };
        setStreamEvents(prev => [...prev, event]);
        setHasCitations(true);
      }
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e);
    }
  }, [debateId]);

  // WebSocket connection effect
  useEffect(() => {
    if (!enabled) return;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('streaming');
      ws.send(JSON.stringify({ type: 'subscribe', debate_id: debateId }));
    };

    ws.onmessage = handleMessage;

    ws.onerror = () => {
      setStatus('error');
      setError('WebSocket connection error');
    };

    ws.onclose = () => {
      if (status === 'streaming') {
        setStatus('complete');
      }
    };

    return () => {
      ws.close();
    };
  }, [enabled, wsUrl, debateId, handleMessage, status]);

  return {
    status,
    error,
    isConnected: status === 'streaming',
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
  };
}
