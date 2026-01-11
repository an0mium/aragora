'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { logger } from '@/utils/logger';

const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.aragora.ai/ws';

// Gauntlet event types
export type GauntletEventType =
  | 'gauntlet_start'
  | 'gauntlet_phase'
  | 'gauntlet_agent_active'
  | 'gauntlet_attack'
  | 'gauntlet_finding'
  | 'gauntlet_probe'
  | 'gauntlet_verification'
  | 'gauntlet_risk'
  | 'gauntlet_progress'
  | 'gauntlet_verdict'
  | 'gauntlet_complete';

export interface GauntletEvent {
  type: GauntletEventType;
  data: Record<string, unknown>;
  timestamp: number;
  seq: number;
}

export interface GauntletFinding {
  finding_id: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  category: string;
  title: string;
  description: string;
  source: string;
}

export interface GauntletAgent {
  name: string;
  role: string;
  status: 'idle' | 'active' | 'complete';
  attackCount: number;
  probeCount: number;
}

export interface GauntletVerdict {
  verdict: 'APPROVED' | 'APPROVED_WITH_CONDITIONS' | 'NEEDS_REVIEW' | 'REJECTED';
  confidence: number;
  riskScore: number;
  robustnessScore: number;
  findings: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    total: number;
  };
}

export type GauntletConnectionStatus = 'connecting' | 'streaming' | 'complete' | 'error';

interface UseGauntletWebSocketOptions {
  gauntletId: string;
  wsUrl?: string;
  enabled?: boolean;
}

interface UseGauntletWebSocketReturn {
  // Connection state
  status: GauntletConnectionStatus;
  error: string | null;
  isConnected: boolean;

  // Gauntlet data
  inputType: string;
  inputSummary: string;
  phase: string;
  progress: number;
  agents: Map<string, GauntletAgent>;
  findings: GauntletFinding[];
  events: GauntletEvent[];
  verdict: GauntletVerdict | null;
  elapsedSeconds: number;

  // Actions
  reconnect: () => void;
}

export function useGauntletWebSocket({
  gauntletId,
  wsUrl = DEFAULT_WS_URL,
  enabled = true,
}: UseGauntletWebSocketOptions): UseGauntletWebSocketReturn {
  const [status, setStatus] = useState<GauntletConnectionStatus>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [inputType, setInputType] = useState<string>('');
  const [inputSummary, setInputSummary] = useState<string>('');
  const [phase, setPhase] = useState<string>('init');
  const [progress, setProgress] = useState<number>(0);
  const [agents, setAgents] = useState<Map<string, GauntletAgent>>(new Map());
  const [findings, setFindings] = useState<GauntletFinding[]>([]);
  const [events, setEvents] = useState<GauntletEvent[]>([]);
  const [verdict, setVerdict] = useState<GauntletVerdict | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  const wsRef = useRef<WebSocket | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const elapsedIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!enabled || !gauntletId) return;

    try {
      const url = `${wsUrl}/gauntlet/${gauntletId}`;
      logger.info(`Connecting to Gauntlet WebSocket: ${url}`);

      const ws = new WebSocket(url);
      wsRef.current = ws;
      setStatus('connecting');
      setError(null);

      ws.onopen = () => {
        logger.info('Gauntlet WebSocket connected');
        setStatus('streaming');
        startTimeRef.current = Date.now();

        // Start elapsed time counter
        elapsedIntervalRef.current = setInterval(() => {
          if (startTimeRef.current) {
            setElapsedSeconds((Date.now() - startTimeRef.current) / 1000);
          }
        }, 1000);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as GauntletEvent;
          handleEvent(data);
        } catch (err) {
          logger.error('Failed to parse Gauntlet event:', err);
        }
      };

      ws.onerror = (err) => {
        logger.error('Gauntlet WebSocket error:', err);
        setError('Connection error');
        setStatus('error');
      };

      ws.onclose = () => {
        logger.info('Gauntlet WebSocket closed');
        if (status !== 'complete') {
          setStatus('error');
        }
        // Clean up interval
        if (elapsedIntervalRef.current) {
          clearInterval(elapsedIntervalRef.current);
        }
      };
    } catch (err) {
      logger.error('Failed to create Gauntlet WebSocket:', err);
      setError(err instanceof Error ? err.message : 'Connection failed');
      setStatus('error');
    }
  }, [enabled, gauntletId, wsUrl, status]);

  const handleEvent = useCallback((event: GauntletEvent) => {
    // Add to events list
    setEvents(prev => [...prev, event]);

    switch (event.type) {
      case 'gauntlet_start': {
        const data = event.data as {
          input_type: string;
          input_summary: string;
          agents: string[];
        };
        setInputType(data.input_type);
        setInputSummary(data.input_summary);

        // Initialize agents
        const newAgents = new Map<string, GauntletAgent>();
        data.agents.forEach(name => {
          newAgents.set(name, {
            name,
            role: 'analyst',
            status: 'idle',
            attackCount: 0,
            probeCount: 0,
          });
        });
        setAgents(newAgents);
        break;
      }

      case 'gauntlet_phase': {
        const data = event.data as { phase: string };
        setPhase(data.phase);
        break;
      }

      case 'gauntlet_progress': {
        const data = event.data as { progress: number; elapsed_seconds: number };
        setProgress(data.progress);
        setElapsedSeconds(data.elapsed_seconds);
        break;
      }

      case 'gauntlet_agent_active': {
        const data = event.data as { agent: string; role: string };
        setAgents(prev => {
          const updated = new Map(prev);
          const existing = updated.get(data.agent);
          if (existing) {
            updated.set(data.agent, { ...existing, role: data.role, status: 'active' });
          }
          return updated;
        });
        break;
      }

      case 'gauntlet_attack': {
        const data = event.data as { agent: string };
        setAgents(prev => {
          const updated = new Map(prev);
          const existing = updated.get(data.agent);
          if (existing) {
            updated.set(data.agent, { ...existing, attackCount: existing.attackCount + 1 });
          }
          return updated;
        });
        break;
      }

      case 'gauntlet_probe': {
        const data = event.data as { agent: string };
        setAgents(prev => {
          const updated = new Map(prev);
          const existing = updated.get(data.agent);
          if (existing) {
            updated.set(data.agent, { ...existing, probeCount: existing.probeCount + 1 });
          }
          return updated;
        });
        break;
      }

      case 'gauntlet_finding': {
        const data = event.data as GauntletFinding;
        setFindings(prev => [...prev, data]);
        break;
      }

      case 'gauntlet_verdict': {
        const data = event.data as {
          verdict: string;
          confidence: number;
          risk_score: number;
          robustness_score: number;
          findings: {
            critical: number;
            high: number;
            medium: number;
            low: number;
            total: number;
          };
        };
        setVerdict({
          verdict: data.verdict as GauntletVerdict['verdict'],
          confidence: data.confidence,
          riskScore: data.risk_score,
          robustnessScore: data.robustness_score,
          findings: data.findings,
        });
        break;
      }

      case 'gauntlet_complete': {
        setStatus('complete');
        // Mark all agents as complete
        setAgents(prev => {
          const updated = new Map(prev);
          updated.forEach((agent, name) => {
            updated.set(name, { ...agent, status: 'complete' });
          });
          return updated;
        });
        // Clean up interval
        if (elapsedIntervalRef.current) {
          clearInterval(elapsedIntervalRef.current);
        }
        break;
      }
    }
  }, []);

  const reconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    // Reset state
    setStatus('connecting');
    setError(null);
    setEvents([]);
    setFindings([]);
    setVerdict(null);
    setProgress(0);
    setPhase('init');
    // Reconnect
    connect();
  }, [connect]);

  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (elapsedIntervalRef.current) {
        clearInterval(elapsedIntervalRef.current);
      }
    };
  }, [connect]);

  return {
    status,
    error,
    isConnected: status === 'streaming',
    inputType,
    inputSummary,
    phase,
    progress,
    agents,
    findings,
    events,
    verdict,
    elapsedSeconds,
    reconnect,
  };
}
