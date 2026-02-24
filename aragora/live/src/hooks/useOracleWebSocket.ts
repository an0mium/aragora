/**
 * useOracleWebSocket — React hook for real-time Oracle streaming.
 *
 * Connects to /ws/oracle and manages the full lifecycle:
 *   - JSON text frames → token/phase/tentacle state updates
 *   - Binary frames → useStreamingAudio hook for progressive playback
 *   - Auto-reconnect with exponential backoff (3 attempts)
 *   - Fallback flag when WebSocket is unavailable
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { WS_URL } from '@/config';
import { useStreamingAudio } from './useStreamingAudio';

export type OraclePhase = 'idle' | 'reflex' | 'deep' | 'tentacles' | 'synthesis';

interface TentacleState {
  text: string;
  done: boolean;
}

export interface UseOracleWebSocket {
  /** Whether the WebSocket is connected. */
  connected: boolean;
  /** Send a question to the Oracle. */
  ask: (question: string, mode: string, options?: { sessionId?: string; summaryDepth?: string }) => void;
  /** Stop the current Oracle response. */
  stop: () => void;
  /** Send an interim speech transcript for think-while-listening. */
  sendInterim: (text: string) => void;
  /** Accumulated text tokens (reflex + deep combined). */
  tokens: string;
  /** Current Oracle phase. */
  phase: OraclePhase;
  /** Per-agent tentacle text. */
  tentacles: Map<string, TentacleState>;
  /** Final synthesis text. */
  synthesis: string;
  /** True if WebSocket failed and component should use fetch fallback. */
  fallbackMode: boolean;
  /** True when the current stream appears stalled. */
  streamStalled: boolean;
  /** Why the stream was marked stalled. */
  stallReason: 'waiting_first_token' | 'stream_inactive' | null;
  /** Client-observed time to first token (ms) for the current/last stream. */
  timeToFirstTokenMs: number | null;
  /** Client-observed end-to-end stream duration (ms) for the current/last stream. */
  streamDurationMs: number | null;
  /** Streaming audio controls. */
  audio: ReturnType<typeof useStreamingAudio>;
}

const MAX_RECONNECT_ATTEMPTS = 3;
const BASE_RECONNECT_DELAY_MS = 1000;
const ORACLE_FIRST_TOKEN_TIMEOUT_MS = parseInt(
  process.env.NEXT_PUBLIC_ORACLE_FIRST_TOKEN_TIMEOUT_MS || '15000',
  10
);
const ORACLE_ACTIVITY_TIMEOUT_MS = parseInt(
  process.env.NEXT_PUBLIC_ORACLE_ACTIVITY_TIMEOUT_MS || '20000',
  10
);

function resolveOracleWsUrl(): string {
  // Derive from WS_URL: replace /ws suffix with /ws/oracle
  const base = WS_URL.replace(/\/ws\/?$/, '');
  return `${base}/ws/oracle`;
}

export function useOracleWebSocket(): UseOracleWebSocket {
  const [connected, setConnected] = useState(false);
  const [tokens, setTokens] = useState('');
  const [phase, setPhase] = useState<OraclePhase>('idle');
  const [tentacles, setTentacles] = useState<Map<string, TentacleState>>(new Map());
  const [synthesis, setSynthesis] = useState('');
  const [fallbackMode, setFallbackMode] = useState(false);
  const [streamStalled, setStreamStalled] = useState(false);
  const [stallReason, setStallReason] = useState<'waiting_first_token' | 'stream_inactive' | null>(null);
  const [timeToFirstTokenMs, setTimeToFirstTokenMs] = useState<number | null>(null);
  const [streamDurationMs, setStreamDurationMs] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const firstTokenTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activityTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const phaseRef = useRef<OraclePhase>('idle');
  const askStartedAtRef = useRef<number | null>(null);
  const firstTokenSeenRef = useRef(false);
  const streamActiveRef = useRef(false);
  const audio = useStreamingAudio();

  const clearStallTimers = useCallback(() => {
    if (firstTokenTimeoutRef.current) {
      clearTimeout(firstTokenTimeoutRef.current);
      firstTokenTimeoutRef.current = null;
    }
    if (activityTimeoutRef.current) {
      clearTimeout(activityTimeoutRef.current);
      activityTimeoutRef.current = null;
    }
  }, []);

  const startFirstTokenTimeout = useCallback(() => {
    if (firstTokenTimeoutRef.current) clearTimeout(firstTokenTimeoutRef.current);
    firstTokenTimeoutRef.current = setTimeout(() => {
      if (!mountedRef.current || !streamActiveRef.current || firstTokenSeenRef.current) return;
      setStreamStalled(true);
      setStallReason('waiting_first_token');
    }, ORACLE_FIRST_TOKEN_TIMEOUT_MS);
  }, []);

  const resetActivityTimeout = useCallback(() => {
    if (!streamActiveRef.current || !firstTokenSeenRef.current) return;
    if (activityTimeoutRef.current) clearTimeout(activityTimeoutRef.current);
    activityTimeoutRef.current = setTimeout(() => {
      if (!mountedRef.current || !streamActiveRef.current) return;
      setStreamStalled(true);
      setStallReason('stream_inactive');
    }, ORACLE_ACTIVITY_TIMEOUT_MS);
  }, []);

  const markStreamProgress = useCallback(() => {
    if (!streamActiveRef.current) return;
    setStreamStalled(false);
    setStallReason(null);
    resetActivityTimeout();
  }, [resetActivityTimeout]);

  const cleanup = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    clearStallTimers();
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      wsRef.current.onopen = null;
      if (wsRef.current.readyState === WebSocket.OPEN ||
          wsRef.current.readyState === WebSocket.CONNECTING) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    cleanup();

    const url = resolveOracleWsUrl();
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      // WebSocket constructor failed
      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current++;
        const delay = BASE_RECONNECT_DELAY_MS * Math.pow(2, reconnectAttempts.current - 1);
        reconnectTimer.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, delay);
      } else {
        clearStallTimers();
        streamActiveRef.current = false;
        setFallbackMode(true);
      }
      return;
    }

    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setConnected(true);
      reconnectAttempts.current = 0;
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);

      // Auto-reconnect with backoff
      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current++;
        const delay = BASE_RECONNECT_DELAY_MS * Math.pow(2, reconnectAttempts.current - 1);
        reconnectTimer.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, delay);
      } else {
        clearStallTimers();
        streamActiveRef.current = false;
        setFallbackMode(true);
      }
    };

    ws.onerror = () => {
      // onclose will fire after this
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!mountedRef.current) return;

      // Binary frame → audio chunk
      if (event.data instanceof ArrayBuffer) {
        audio.appendChunk(event.data);
        return;
      }

      // Text frame → JSON event
      try {
        const data = JSON.parse(event.data as string);
        const type = data.type as string;

        switch (type) {
          case 'connected':
            // Server acknowledged connection
            break;

          case 'reflex_start':
            phaseRef.current = 'reflex';
            setPhase('reflex');
            markStreamProgress();
            setTokens('');
            setTentacles(new Map());
            setSynthesis('');
            break;

          case 'token':
            if (!firstTokenSeenRef.current && askStartedAtRef.current !== null) {
              firstTokenSeenRef.current = true;
              if (firstTokenTimeoutRef.current) {
                clearTimeout(firstTokenTimeoutRef.current);
                firstTokenTimeoutRef.current = null;
              }
              setTimeToFirstTokenMs(Math.round(performance.now() - askStartedAtRef.current));
            }
            markStreamProgress();
            setTokens(prev => prev + (data.text || ''));
            if (data.phase === 'deep' && phaseRef.current !== 'deep') {
              phaseRef.current = 'deep';
              setPhase('deep');
            }
            break;

          case 'sentence_ready':
            // Sentence boundary — useful for display, audio already streaming
            markStreamProgress();
            break;

          case 'phase_done':
            markStreamProgress();
            if (data.phase === 'deep') {
              phaseRef.current = 'tentacles';
              setPhase('tentacles');
            }
            audio.endSegment();
            break;

          case 'tentacle_start':
            phaseRef.current = 'tentacles';
            setPhase('tentacles');
            markStreamProgress();
            setTentacles(prev => {
              const next = new Map(prev);
              next.set(data.agent, { text: '', done: false });
              return next;
            });
            break;

          case 'tentacle_token':
            markStreamProgress();
            setTentacles(prev => {
              const next = new Map(prev);
              const existing = next.get(data.agent);
              if (existing) {
                next.set(data.agent, { text: existing.text + (data.text || ''), done: false });
              } else {
                next.set(data.agent, { text: data.text || '', done: false });
              }
              return next;
            });
            break;

          case 'tentacle_done':
            markStreamProgress();
            setTentacles(prev => {
              const next = new Map(prev);
              next.set(data.agent, { text: data.full_text || '', done: true });
              return next;
            });
            break;

          case 'synthesis':
            phaseRef.current = 'synthesis';
            setPhase('synthesis');
            markStreamProgress();
            setSynthesis(data.text || '');
            if (askStartedAtRef.current !== null) {
              setStreamDurationMs(Math.round(performance.now() - askStartedAtRef.current));
            }
            streamActiveRef.current = false;
            clearStallTimers();
            break;

          case 'error':
            // Surface error but don't crash — let the component handle it
            console.error('[Oracle WS] Server error:', data.message);
            setStreamStalled(true);
            if (!firstTokenSeenRef.current) {
              setStallReason('waiting_first_token');
            } else {
              setStallReason('stream_inactive');
            }
            break;

          case 'pong':
            // Heartbeat response
            markStreamProgress();
            break;
        }
      } catch {
        // Non-JSON text frame — ignore
      }
    };
  }, [cleanup, audio, clearStallTimers, markStreamProgress]);

  // Connect on mount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      cleanup();
      audio.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ask = useCallback((question: string, mode: string, options?: { sessionId?: string; summaryDepth?: string }) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // Reset state for new question
    streamActiveRef.current = true;
    askStartedAtRef.current = performance.now();
    firstTokenSeenRef.current = false;
    setTokens('');
    setStreamStalled(false);
    setStallReason(null);
    setTimeToFirstTokenMs(null);
    setStreamDurationMs(null);
    phaseRef.current = 'idle';
    setPhase('idle');
    setTentacles(new Map());
    setSynthesis('');
    audio.stop();

    const payload: Record<string, string> = { type: 'ask', question, mode };
    if (options?.sessionId) payload.session_id = options.sessionId;
    if (options?.summaryDepth) payload.summary_depth = options.summaryDepth;
    wsRef.current.send(JSON.stringify(payload));
    startFirstTokenTimeout();
  }, [audio, startFirstTokenTimeout]);

  const stop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
    }
    streamActiveRef.current = false;
    clearStallTimers();
    if (askStartedAtRef.current !== null && streamDurationMs === null) {
      setStreamDurationMs(Math.round(performance.now() - askStartedAtRef.current));
    }
    phaseRef.current = 'idle';
    setPhase('idle');
    setStreamStalled(false);
    setStallReason(null);
    audio.stop();
  }, [audio, clearStallTimers, streamDurationMs]);

  const sendInterim = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interim', text }));
    }
  }, []);

  return {
    connected,
    ask,
    stop,
    sendInterim,
    tokens,
    phase,
    tentacles,
    synthesis,
    fallbackMode,
    streamStalled,
    stallReason,
    timeToFirstTokenMs,
    streamDurationMs,
    audio,
  };
}

export default useOracleWebSocket;
