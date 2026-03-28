'use client';

import { useCallback } from 'react';
import { LiveDebateStream } from './LiveDebateStream';
import { useDebateWebSocket } from '@/hooks/debate-websocket';
import type { DebateResponse } from '@/components/DebateResultPreview';

interface PublicDebateStreamProps {
  debateId: string;
  apiBase: string;
  wsUrl?: string;
  task: string;
  agents?: string[];
  className?: string;
  onResolved: (result: DebateResponse | null) => void;
}

function buildApiUrl(apiBase: string, path: string): string {
  const base = apiBase.replace(/\/$/, '');
  return base ? `${base}${path}` : path;
}

function isCompletedStatus(value: unknown): boolean {
  if (typeof value !== 'string') {
    return false;
  }
  return ['completed', 'consensus_reached', 'concluded', 'archived'].includes(value);
}

function normalizeDebateResponse(
  payload: unknown,
  fallbackTask: string,
  fallbackAgents: string[],
): DebateResponse | null {
  const raw =
    payload && typeof payload === 'object' && 'data' in payload
      ? (payload as { data?: unknown }).data
      : payload;

  if (!raw || typeof raw !== 'object') {
    return null;
  }

  const data = raw as Record<string, unknown>;
  const status = data.status;
  if (!isCompletedStatus(status)) {
    return null;
  }

  const participants = Array.isArray(data.participants)
    ? data.participants.filter((agent): agent is string => typeof agent === 'string')
    : Array.isArray(data.agents)
      ? data.agents.filter((agent): agent is string => typeof agent === 'string')
      : fallbackAgents;

  const proposals =
    data.proposals && typeof data.proposals === 'object' && !Array.isArray(data.proposals)
      ? (data.proposals as Record<string, string>)
      : {};

  const critiques = Array.isArray(data.critiques)
    ? (data.critiques as DebateResponse['critiques'])
    : [];

  const votes = Array.isArray(data.votes)
    ? (data.votes as DebateResponse['votes'])
    : [];

  const dissentingViews = Array.isArray(data.dissenting_views)
    ? data.dissenting_views.filter((view): view is string => typeof view === 'string')
    : [];

  const consensus =
    typeof data.consensus === 'object' && data.consensus !== null
      ? (data.consensus as Record<string, unknown>)
      : null;

  return {
    id:
      (typeof data.id === 'string' && data.id) ||
      (typeof data.debate_id === 'string' && data.debate_id) ||
      '',
    topic:
      (typeof data.topic === 'string' && data.topic) ||
      (typeof data.task === 'string' && data.task) ||
      fallbackTask,
    status: typeof status === 'string' ? status : 'completed',
    rounds_used:
      typeof data.rounds_used === 'number'
        ? data.rounds_used
        : typeof data.rounds === 'number'
          ? data.rounds
          : 0,
    consensus_reached:
      typeof data.consensus_reached === 'boolean'
        ? data.consensus_reached
        : Boolean(consensus?.reached),
    confidence:
      typeof data.confidence === 'number'
        ? data.confidence
        : typeof data.agreement === 'number'
          ? data.agreement
          : typeof consensus?.confidence === 'number'
            ? consensus.confidence
            : 0,
    verdict: typeof data.verdict === 'string' ? data.verdict : null,
    duration_seconds: typeof data.duration_seconds === 'number' ? data.duration_seconds : 0,
    participants,
    proposals,
    critiques,
    votes,
    dissenting_views: dissentingViews,
    final_answer:
      (typeof data.final_answer === 'string' && data.final_answer) ||
      (typeof data.conclusion === 'string' && data.conclusion) ||
      '',
    receipt:
      data.receipt && typeof data.receipt === 'object'
        ? (data.receipt as DebateResponse['receipt'])
        : null,
    receipt_hash: typeof data.receipt_hash === 'string' ? data.receipt_hash : null,
    is_live: data.is_live === true,
    mock_fallback: data.mock_fallback === true,
    mock_fallback_reason:
      typeof data.mock_fallback_reason === 'string' ? data.mock_fallback_reason : undefined,
    upgrade_cta:
      data.upgrade_cta && typeof data.upgrade_cta === 'object'
        ? (data.upgrade_cta as DebateResponse['upgrade_cta'])
        : undefined,
  };
}

async function fetchCompletedDebate(
  apiBase: string,
  debateId: string,
  fallbackTask: string,
  fallbackAgents: string[],
): Promise<DebateResponse | null> {
  const encodedDebateId = encodeURIComponent(debateId);
  const urls = [
    buildApiUrl(apiBase, `/api/v1/debates/${encodedDebateId}`),
    buildApiUrl(apiBase, `/api/v1/debates/public/${encodedDebateId}`),
    buildApiUrl(apiBase, `/api/v1/playground/debate/${encodedDebateId}`),
  ];

  for (let attempt = 0; attempt < 5; attempt += 1) {
    for (const url of urls) {
      try {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) {
          continue;
        }
        const data = await response.json();
        const debate = normalizeDebateResponse(data, fallbackTask, fallbackAgents);
        if (debate) {
          return debate;
        }
      } catch {
        // Try the next candidate URL.
      }
    }

    if (attempt < 4) {
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
  }

  return null;
}

export function PublicDebateStream({
  debateId,
  apiBase,
  wsUrl,
  task,
  agents = [],
  className,
  onResolved,
}: PublicDebateStreamProps) {
  const ws = useDebateWebSocket({
    debateId,
    wsUrl,
    enabled: Boolean(debateId),
  });

  const handleComplete = useCallback(async () => {
    const debate = await fetchCompletedDebate(apiBase, debateId, task, agents);
    onResolved(debate);
  }, [agents, apiBase, debateId, onResolved, task]);

  return (
    <div className={className}>
      <LiveDebateStream
        status={ws.status}
        error={ws.error}
        errorDetails={ws.errorDetails}
        task={ws.task || task}
        agents={ws.agents.length > 0 ? ws.agents : agents}
        messages={ws.messages}
        streamingMessages={ws.streamingMessages}
        streamEvents={ws.streamEvents}
        reconnectAttempt={ws.reconnectAttempt}
        connectionQuality={ws.connectionQuality}
        isPolling={ws.isPolling}
        onReconnect={ws.reconnect}
        onComplete={handleComplete}
      />
    </div>
  );
}
