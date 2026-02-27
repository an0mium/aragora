'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8765';

export type EngineStage = 'intake' | 'decompose' | 'interrogate' | 'research' | 'spec' | 'validate' | 'handoff';
export type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'error';

export interface QuestionOption {
  label: string;
  description: string;
  tradeoff: string;
}

export interface ClarifyingQuestion {
  id: string;
  question: string;
  why_it_matters: string;
  options: QuestionOption[];
  default_option: string | null;
  impact: string;
}

export interface SessionState {
  session_id: string;
  stage: EngineStage;
  profile: string;
  raw_prompt: string;
  questions: ClarifyingQuestion[];
  answers: Record<string, string>;
  research_sources: string[];
  provenance_hash: string;
  pipeline_id: string | null;
}

export interface ResearchProgress {
  source: string;
  status: 'pending' | 'complete' | 'failed';
  results: number;
}

export interface UsePromptEngineReturn {
  status: ConnectionStatus;
  session: SessionState | null;
  stage: EngineStage | null;
  questions: ClarifyingQuestion[];
  researchProgress: ResearchProgress[];
  spec: Record<string, unknown> | null;
  validationResult: Record<string, unknown> | null;
  error: string | null;
  confidence: number | null;
  createSession: (prompt: string, profile: string, sources?: string[]) => Promise<void>;
  answerQuestion: (questionId: string, answer: string) => void;
  approveSpec: () => void;
  skipValidation: () => void;
  deleteSession: () => Promise<void>;
}

export function usePromptEngine(): UsePromptEngineReturn {
  const [status, setStatus] = useState<ConnectionStatus>('idle');
  const [session, setSession] = useState<SessionState | null>(null);
  const [questions, setQuestions] = useState<ClarifyingQuestion[]>([]);
  const [researchProgress, setResearchProgress] = useState<ResearchProgress[]>([]);
  const [spec, setSpec] = useState<Record<string, unknown> | null>(null);
  const [validationResult, setValidationResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const createSession = useCallback(async (prompt: string, profile: string, sources?: string[]) => {
    try {
      const body: Record<string, unknown> = { prompt, profile };
      if (sources) body.research_sources = sources;

      const res = await fetch(`${API_BASE}/api/v1/prompt-engine/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSession(data.data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create session');
    }
  }, []);

  const connectWebSocket = useCallback((sessionId: string) => {
    const ws = new WebSocket(`${WS_BASE}/ws/prompt-engine/${sessionId}`);
    wsRef.current = ws;
    setStatus('connecting');

    ws.onopen = () => setStatus('connected');
    ws.onerror = () => setStatus('error');
    ws.onclose = () => setStatus('idle');

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case 'stage_transition':
          setSession(prev => prev ? { ...prev, stage: msg.to } : null);
          break;
        case 'question':
          setQuestions(prev => [...prev, msg.question]);
          break;
        case 'research_progress':
          setResearchProgress(prev => {
            const updated = prev.filter(r => r.source !== msg.source);
            return [...updated, { source: msg.source, status: msg.status, results: msg.results }];
          });
          break;
        case 'spec_ready':
          setSpec(msg.spec);
          setConfidence(msg.confidence);
          break;
        case 'validation_result':
          setValidationResult(msg);
          break;
        case 'error':
          setError(msg.message);
          break;
      }
    };
  }, []);

  const answerQuestion = useCallback((questionId: string, answer: string) => {
    wsRef.current?.send(JSON.stringify({ type: 'answer', question_id: questionId, answer }));
  }, []);

  const approveSpec = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'approve_spec' }));
  }, []);

  const skipValidation = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'skip_validation' }));
  }, []);

  const deleteSession = useCallback(async () => {
    if (!session) return;
    wsRef.current?.close();
    await fetch(`${API_BASE}/api/v1/prompt-engine/sessions/${session.session_id}`, {
      method: 'DELETE',
    });
    setSession(null);
    setStatus('idle');
  }, [session]);

  useEffect(() => {
    return () => { wsRef.current?.close(); };
  }, []);

  return {
    status,
    session,
    stage: session?.stage ?? null,
    questions,
    researchProgress,
    spec,
    validationResult,
    error,
    confidence,
    createSession,
    answerQuestion,
    approveSpec,
    skipValidation,
    deleteSession,
  };
}
