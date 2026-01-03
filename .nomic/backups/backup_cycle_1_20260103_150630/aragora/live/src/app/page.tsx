'use client';

import { useState, useEffect } from 'react';
import { useNomicStream } from '@/hooks/useNomicStream';
import { ConnectionStatus } from '@/components/ConnectionStatus';
import { MetricsCards } from '@/components/MetricsCards';
import { PhaseProgress } from '@/components/PhaseProgress';
import { AgentPanel } from '@/components/AgentPanel';
import { HistoryPanel } from '@/components/HistoryPanel';
import { UserParticipation } from '@/components/UserParticipation';
import { ReplayBrowser } from '@/components/ReplayBrowser';
import type { NomicState } from '@/types/events';

// WebSocket URL - can be overridden via environment variable
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.aragora.ai';
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

export default function Home() {
  const { events, connected, activeLoops, selectedLoopId, selectLoop, sendMessage, onAck, onError } = useNomicStream(WS_URL);
  const [nomicState, setNomicState] = useState<NomicState | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch initial nomic state on mount
  useEffect(() => {
    const fetchState = async () => {
      try {
        const response = await fetch(`${API_URL}/api/nomic/state`);
        if (response.ok) {
          const data = await response.json();
          setNomicState(data);
        }
      } catch (err) {
        // API might not be available, that's OK
        console.log('Could not fetch initial state:', err);
      }
    };

    fetchState();
  }, []);

  // Update nomic state from events
  useEffect(() => {
    if (events.length === 0) return;

    const lastEvent = events[events.length - 1];

    // Update state based on event type
    switch (lastEvent.type) {
      case 'cycle_start':
        setNomicState((prev) => ({
          ...prev,
          cycle: lastEvent.data.cycle as number,
          phase: 'debate',
          last_success: undefined,
        }));
        break;
      case 'phase_start':
        setNomicState((prev) => ({
          ...prev,
          phase: lastEvent.data.phase as string,
        }));
        break;
      case 'task_complete':
        setNomicState((prev) => ({
          ...prev,
          completed_tasks: (prev?.completed_tasks || 0) + 1,
        }));
        break;
      case 'cycle_end':
        setNomicState((prev) => ({
          ...prev,
          last_success: lastEvent.data.success as boolean,
        }));
        break;
      case 'error':
        setError(lastEvent.data.error as string);
        break;
    }
  }, [events]);

  // Derive current phase from state or latest phase event
  const currentPhase = nomicState?.phase || 'idle';

  // Format timestamp as relative time
  const formatRelativeTime = (timestamp: number) => {
    const seconds = Math.floor((Date.now() / 1000) - timestamp);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  // User participation handlers
  const handleUserVote = (choice: string) => {
    if (!selectedLoopId) {
      setError('No active debate loop selected. Please wait for a debate to start.');
      return;
    }
    sendMessage({
      type: 'user_vote',
      loop_id: selectedLoopId,
      payload: { choice }
    });
  };

  const handleUserSuggestion = (suggestion: string) => {
    if (!selectedLoopId) {
      setError('No active debate loop selected. Please wait for a debate to start.');
      return;
    }
    sendMessage({
      type: 'user_suggestion',
      loop_id: selectedLoopId,
      payload: { suggestion }
    });
  };

  return (
    <main className="min-h-screen bg-bg text-text">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold">
              <span className="text-accent">aragora</span>
              <span className="text-text-muted font-normal ml-2">live</span>
            </h1>
          </div>
          <div className="flex items-center gap-4">
            {/* Loop Selector - Only show if multiple loops */}
            {activeLoops.length > 1 && (
              <div className="flex items-center gap-2">
                <span className="text-text-muted text-sm">{activeLoops.length} loops active</span>
                <select
                  value={selectedLoopId || ''}
                  onChange={(e) => selectLoop(e.target.value)}
                  className="bg-surface border border-border rounded px-2 py-1 text-sm text-text"
                >
                  {activeLoops.map((loop) => (
                    <option key={loop.loop_id} value={loop.loop_id}>
                      {loop.name} (cycle {loop.cycle}, {formatRelativeTime(loop.started_at)})
                    </option>
                  ))}
                </select>
              </div>
            )}
            {/* Single loop indicator */}
            {activeLoops.length === 1 && (
              <span className="text-text-muted text-sm">
                {activeLoops[0].name}
              </span>
            )}
            <ConnectionStatus connected={connected} />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6 space-y-6">
        {/* Error Banner */}
        {error && (
          <div className="bg-warning/10 border border-warning/30 rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-warning text-xl">⚠️</span>
              <span className="text-warning">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-warning hover:text-warning/80"
            >
              ✕
            </button>
          </div>
        )}

        {/* Phase Progress */}
        <PhaseProgress events={events} currentPhase={currentPhase} />

        {/* Metrics */}
        <MetricsCards nomicState={nomicState} events={events} />

        {/* Main Panels */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Agent Activity */}
          <div className="lg:col-span-2 min-h-[500px]">
            <AgentPanel events={events} />
          </div>

          {/* Side Panel */}
          <div className="lg:col-span-1 space-y-4">
            <UserParticipation
              events={events}
              onVote={handleUserVote}
              onSuggest={handleUserSuggestion}
              onAck={onAck}
              onError={onError}
            />
            <HistoryPanel />
            <ReplayBrowser />
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-text-muted text-sm py-8 border-t border-border">
          <p>
            Watching aragora's self-improving nomic loop in real-time.
            <br />
            <a
              href="https://aragora.ai"
              className="text-accent hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn more about aragora →
            </a>
          </p>
          <p className="mt-4 text-xs">
            Built on ideas from{' '}
            <a href="https://github.com/AI-Counsel/ai-counsel" className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">ai-counsel</a>,{' '}
            <a href="https://github.com/Tsinghua-MARS-Lab/DebateLLM" className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">DebateLLM</a>,{' '}
            <a href="https://github.com/camel-ai/camel" className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">CAMEL-AI</a>,{' '}
            <a href="https://github.com/joonspk-research/generative_agents" className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">Generative Agents</a>, and{' '}
            <a href="https://github.com/composable-models/llm_multiagent_debate" className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">LLM Multi-Agent Debate</a>.
          </p>
        </footer>
      </div>
    </main>
  );
}
