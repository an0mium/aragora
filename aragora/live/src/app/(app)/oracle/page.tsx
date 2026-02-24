'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import {
  useOracleWebSocket,
  type OraclePhase,
  type DebateEvent,
  type DebateAgentState,
} from '@/hooks/useOracleWebSocket';

// ============================================================================
// Types
// ============================================================================

type OracleMode = 'consult' | 'divine' | 'commune';

const MODE_DESCRIPTIONS: Record<OracleMode, { label: string; desc: string; icon: string }> = {
  consult: {
    label: 'CONSULT',
    desc: 'Quick single-agent analysis with streaming response',
    icon: '\u25C9',
  },
  divine: {
    label: 'DIVINE',
    desc: 'Deep multi-perspective reasoning with tentacle exploration',
    icon: '\u2726',
  },
  commune: {
    label: 'COMMUNE',
    desc: 'Full multi-agent debate with consensus-seeking',
    icon: '\u2318',
  },
};

const PHASE_LABELS: Record<OraclePhase, string> = {
  idle: 'IDLE',
  reflex: 'REFLEX',
  deep: 'DEEP REASONING',
  tentacles: 'EXPLORATION',
  synthesis: 'SYNTHESIS',
};

// ============================================================================
// Sub-components
// ============================================================================

function PhaseIndicator({ phase }: { phase: OraclePhase }) {
  const colors: Record<OraclePhase, string> = {
    idle: 'text-[var(--text-muted)]',
    reflex: 'text-yellow-400',
    deep: 'text-[var(--acid-cyan)]',
    tentacles: 'text-purple-400',
    synthesis: 'text-[var(--acid-green)]',
  };

  return (
    <div className="flex items-center gap-3">
      {Object.entries(PHASE_LABELS).map(([key, label]) => {
        const isActive = key === phase;
        const isPast =
          Object.keys(PHASE_LABELS).indexOf(key) <
          Object.keys(PHASE_LABELS).indexOf(phase);
        return (
          <div
            key={key}
            className={`flex items-center gap-1.5 text-xs font-mono transition-colors ${
              isActive
                ? colors[key as OraclePhase]
                : isPast
                  ? 'text-[var(--acid-green)]/50'
                  : 'text-[var(--text-muted)]/30'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isActive
                  ? 'bg-current animate-pulse'
                  : isPast
                    ? 'bg-[var(--acid-green)]/40'
                    : 'bg-[var(--text-muted)]/20'
              }`}
            />
            {label}
          </div>
        );
      })}
    </div>
  );
}

function ModeSelector({
  mode,
  onChange,
  disabled,
}: {
  mode: OracleMode;
  onChange: (m: OracleMode) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex gap-2">
      {(Object.keys(MODE_DESCRIPTIONS) as OracleMode[]).map((m) => {
        const info = MODE_DESCRIPTIONS[m];
        const isSelected = m === mode;
        return (
          <button
            key={m}
            onClick={() => onChange(m)}
            disabled={disabled}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-mono border transition-colors ${
              isSelected
                ? 'border-[var(--acid-green)] bg-[var(--acid-green)]/10 text-[var(--acid-green)]'
                : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-green)]/40 hover:text-[var(--text)]'
            } disabled:opacity-40 disabled:cursor-not-allowed`}
            title={info.desc}
          >
            <span>{info.icon}</span>
            <span>{info.label}</span>
          </button>
        );
      })}
    </div>
  );
}

function TokenStream({ tokens, phase }: { tokens: string; phase: OraclePhase }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [tokens]);

  if (!tokens && phase === 'idle') return null;

  return (
    <div className="border border-[var(--acid-green)]/20 bg-[var(--surface)]/30 p-4 max-h-[400px] overflow-y-auto">
      <div className="text-[10px] font-mono text-[var(--acid-green)] mb-2">
        {'>'}  ORACLE RESPONSE
      </div>
      <div className="font-mono text-sm text-[var(--text)] whitespace-pre-wrap leading-relaxed">
        {tokens}
        {phase !== 'idle' && phase !== 'synthesis' && (
          <span className="inline-block w-2 h-4 bg-[var(--acid-green)] animate-pulse ml-0.5" />
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}

function TentaclePanel({
  tentacles,
}: {
  tentacles: Map<string, { text: string; done: boolean }>;
}) {
  if (tentacles.size === 0) return null;

  return (
    <div className="space-y-3">
      <div className="text-[10px] font-mono text-purple-400">
        {'>'} TENTACLE EXPLORATION ({tentacles.size} agents)
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {Array.from(tentacles.entries()).map(([agent, state]) => (
          <div
            key={agent}
            className={`p-3 border rounded ${
              state.done
                ? 'border-[var(--acid-green)]/30 bg-[var(--acid-green)]/5'
                : 'border-purple-500/30 bg-purple-500/5'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-[var(--acid-cyan)]">
                {agent}
              </span>
              <span
                className={`text-[10px] font-mono ${
                  state.done ? 'text-[var(--acid-green)]' : 'text-purple-400 animate-pulse'
                }`}
              >
                {state.done ? 'DONE' : 'EXPLORING...'}
              </span>
            </div>
            <p className="font-mono text-xs text-[var(--text)] whitespace-pre-wrap line-clamp-6">
              {state.text || '...'}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function SynthesisPanel({ synthesis }: { synthesis: string }) {
  if (!synthesis) return null;

  return (
    <div className="border border-[var(--acid-green)]/40 bg-[var(--acid-green)]/5 rounded p-4">
      <div className="text-[10px] font-mono text-[var(--acid-green)] mb-2">
        {'>'} SYNTHESIS
      </div>
      <p className="font-mono text-sm text-[var(--text)] whitespace-pre-wrap leading-relaxed">
        {synthesis}
      </p>
    </div>
  );
}

function DebateEventLog({ events }: { events: DebateEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  if (events.length === 0) return null;

  const eventColor = (type: string): string => {
    switch (type) {
      case 'agent_message':
        return 'text-[var(--acid-cyan)]';
      case 'critique':
        return 'text-orange-400';
      case 'vote':
        return 'text-yellow-400';
      case 'consensus':
        return 'text-[var(--acid-green)]';
      case 'round_start':
        return 'text-purple-400';
      case 'debate_start':
      case 'debate_end':
        return 'text-[var(--acid-green)]';
      case 'agent_thinking':
        return 'text-[var(--text-muted)]';
      case 'agent_error':
        return 'text-red-400';
      default:
        return 'text-[var(--text-muted)]';
    }
  };

  return (
    <div className="border border-[var(--acid-green)]/20 bg-[var(--surface)]/30 p-4 max-h-[500px] overflow-y-auto">
      <div className="text-[10px] font-mono text-[var(--acid-green)] mb-3">
        {'>'} DEBATE EVENT LOG ({events.length} events)
      </div>
      <div className="space-y-2">
        {events.map((evt, i) => (
          <div key={i} className="flex items-start gap-2 text-xs font-mono">
            <span className="text-[var(--text-muted)]/50 shrink-0 w-16">
              {new Date(evt.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              })}
            </span>
            <span className={`shrink-0 uppercase ${eventColor(evt.type)}`}>
              [{evt.type.replace(/_/g, ' ')}]
            </span>
            {evt.agent && (
              <span className="text-[var(--acid-cyan)] shrink-0">
                {evt.agent}
              </span>
            )}
            {evt.content && (
              <span className="text-[var(--text)] line-clamp-2">
                {evt.content.slice(0, 200)}
                {evt.content.length > 200 ? '...' : ''}
              </span>
            )}
            {evt.type === 'round_start' && (
              <span className="text-purple-400">Round {evt.round}</span>
            )}
            {evt.type === 'consensus' && Boolean(evt.data?.reached) && (
              <span className="text-[var(--acid-green)]">
                Consensus reached (
                {Math.round(Number(evt.data?.confidence ?? 0) * 100)}%
                confidence)
              </span>
            )}
          </div>
        ))}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}

function DebateAgentCards({
  agents,
}: {
  agents: Map<string, DebateAgentState>;
}) {
  if (agents.size === 0) return null;

  return (
    <div className="space-y-3">
      <div className="text-[10px] font-mono text-[var(--acid-cyan)]">
        {'>'} DEBATE AGENTS ({agents.size})
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {Array.from(agents.entries()).map(([name, state]) => (
          <div
            key={name}
            className={`p-3 border rounded ${
              state.done
                ? 'border-[var(--acid-green)]/30'
                : state.thinking
                  ? 'border-yellow-500/30 bg-yellow-500/5'
                  : 'border-[var(--border)]'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-[var(--acid-cyan)]">
                {name}
              </span>
              <span className="text-[10px] font-mono text-[var(--text-muted)]">
                {state.role}
              </span>
            </div>
            {state.thinking && (
              <div className="text-[10px] font-mono text-yellow-400 animate-pulse mb-1">
                {state.thinkingStep || 'Thinking...'}
              </div>
            )}
            {state.streamingTokens && (
              <p className="font-mono text-[10px] text-[var(--text)]/70 line-clamp-3">
                {state.streamingTokens}
              </p>
            )}
            {state.lastMessage && !state.streamingTokens && (
              <p className="font-mono text-[10px] text-[var(--text)] line-clamp-3">
                {state.lastMessage.slice(0, 150)}
                {state.lastMessage.length > 150 ? '...' : ''}
              </p>
            )}
            {state.done && (
              <div className="text-[10px] font-mono text-[var(--acid-green)] mt-1">
                COMPLETE
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ConnectionStatus({
  connected,
  fallbackMode,
  streamStalled,
  stallReason,
  timeToFirstTokenMs,
}: {
  connected: boolean;
  fallbackMode: boolean;
  streamStalled: boolean;
  stallReason: string | null;
  timeToFirstTokenMs: number | null;
}) {
  return (
    <div className="flex items-center gap-4 text-xs font-mono">
      <div className="flex items-center gap-1.5">
        <span
          className={`w-2 h-2 rounded-full ${
            connected
              ? 'bg-[var(--acid-green)] animate-pulse'
              : fallbackMode
                ? 'bg-yellow-400'
                : 'bg-red-400'
          }`}
        />
        <span className="text-[var(--text-muted)]">
          {connected
            ? 'CONNECTED'
            : fallbackMode
              ? 'FALLBACK MODE'
              : 'DISCONNECTED'}
        </span>
      </div>
      {streamStalled && (
        <span className="text-yellow-400">
          STALLED: {stallReason?.replace(/_/g, ' ')}
        </span>
      )}
      {timeToFirstTokenMs !== null && (
        <span className="text-[var(--text-muted)]">
          TTFT: {timeToFirstTokenMs}ms
        </span>
      )}
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function OraclePage() {
  const oracle = useOracleWebSocket();
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState<OracleMode>('consult');
  const inputRef = useRef<HTMLInputElement>(null);

  const isStreaming =
    oracle.phase !== 'idle' && oracle.phase !== 'synthesis';

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!question.trim() || !oracle.connected) return;

      if (mode === 'commune') {
        oracle.debate(question.trim(), mode);
      } else {
        oracle.ask(question.trim(), mode);
      }
    },
    [question, mode, oracle],
  );

  const handleStop = useCallback(() => {
    oracle.stop();
  }, [oracle]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="container mx-auto px-4 py-6 max-w-5xl">
          {/* Header */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h1 className="text-2xl font-mono text-[var(--acid-green)]">
                {'>'} ORACLE
              </h1>
              <div className="flex items-center gap-3">
                <Link
                  href="/arena"
                  className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
                >
                  [ARENA]
                </Link>
                <Link
                  href="/spectate"
                  className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
                >
                  [SPECTATE]
                </Link>
              </div>
            </div>
            <p className="text-[var(--text-muted)] font-mono text-sm mb-3">
              Ask the Oracle a question. Choose a mode: Consult for quick
              analysis, Divine for deep multi-perspective reasoning, or Commune
              for a full multi-agent debate with consensus.
            </p>
            <ConnectionStatus
              connected={oracle.connected}
              fallbackMode={oracle.fallbackMode}
              streamStalled={oracle.streamStalled}
              stallReason={oracle.stallReason}
              timeToFirstTokenMs={oracle.timeToFirstTokenMs}
            />
          </div>

          {/* Mode Selector */}
          <div className="mb-4">
            <ModeSelector
              mode={mode}
              onChange={setMode}
              disabled={isStreaming}
            />
          </div>

          {/* Input Form */}
          <PanelErrorBoundary panelName="Oracle Input">
            <form onSubmit={handleSubmit} className="mb-6">
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask the Oracle anything..."
                  disabled={!oracle.connected && !oracle.fallbackMode}
                  className="flex-1 bg-[var(--surface)] border border-[var(--acid-green)]/30 px-4 py-3 text-sm font-mono text-[var(--text)] placeholder:text-[var(--text-muted)]/50 focus:outline-none focus:border-[var(--acid-green)] disabled:opacity-40 transition-colors"
                />
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={handleStop}
                    className="px-6 py-3 border border-red-500/50 text-red-400 font-mono text-sm hover:bg-red-500/10 transition-colors"
                  >
                    [STOP]
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={
                      !question.trim() ||
                      (!oracle.connected && !oracle.fallbackMode)
                    }
                    className="px-6 py-3 border border-[var(--acid-green)] text-[var(--acid-green)] font-mono text-sm hover:bg-[var(--acid-green)]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    [{MODE_DESCRIPTIONS[mode].label}]
                  </button>
                )}
              </div>
            </form>
          </PanelErrorBoundary>

          {/* Phase Indicator */}
          {oracle.phase !== 'idle' && (
            <div className="mb-4">
              <PhaseIndicator phase={oracle.phase} />
            </div>
          )}

          {/* Results Area */}
          <div className="space-y-6">
            {/* Token Stream (direct mode) */}
            {!oracle.isDebateMode && (
              <PanelErrorBoundary panelName="Oracle Response">
                <TokenStream tokens={oracle.tokens} phase={oracle.phase} />
              </PanelErrorBoundary>
            )}

            {/* Tentacles (divine mode) */}
            {!oracle.isDebateMode && oracle.tentacles.size > 0 && (
              <PanelErrorBoundary panelName="Tentacle Exploration">
                <TentaclePanel tentacles={oracle.tentacles} />
              </PanelErrorBoundary>
            )}

            {/* Debate Agents (commune mode) */}
            {oracle.isDebateMode && (
              <PanelErrorBoundary panelName="Debate Agents">
                <DebateAgentCards agents={oracle.debateAgents} />
              </PanelErrorBoundary>
            )}

            {/* Debate Event Log (commune mode) */}
            {oracle.isDebateMode && oracle.debateEvents.length > 0 && (
              <PanelErrorBoundary panelName="Debate Events">
                <DebateEventLog events={oracle.debateEvents} />
              </PanelErrorBoundary>
            )}

            {/* Synthesis */}
            <PanelErrorBoundary panelName="Synthesis">
              <SynthesisPanel synthesis={oracle.synthesis} />
            </PanelErrorBoundary>

            {/* Debate metadata */}
            {oracle.isDebateMode && oracle.debateId && (
              <div className="flex items-center gap-4 text-xs font-mono text-[var(--text-muted)]">
                <span>
                  Debate:{' '}
                  <span className="text-[var(--acid-cyan)]">
                    {oracle.debateId}
                  </span>
                </span>
                {oracle.debateRound > 0 && (
                  <span>
                    Round:{' '}
                    <span className="text-purple-400">
                      {oracle.debateRound}
                    </span>
                  </span>
                )}
                {oracle.streamDurationMs !== null && (
                  <span>
                    Duration:{' '}
                    <span className="text-[var(--text)]">
                      {(oracle.streamDurationMs / 1000).toFixed(1)}s
                    </span>
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Empty State */}
          {oracle.phase === 'idle' &&
            !oracle.tokens &&
            !oracle.synthesis &&
            oracle.debateEvents.length === 0 && (
              <div className="mt-12 p-12 border border-[var(--border)] rounded bg-[var(--surface)]/30 text-center">
                <div className="text-4xl mb-4 font-mono">{'\u25C9'}</div>
                <h3 className="font-mono text-lg text-[var(--text)] mb-2">
                  The Oracle Awaits
                </h3>
                <p className="font-mono text-sm text-[var(--text-muted)] max-w-md mx-auto mb-6">
                  Ask a question to receive AI-powered analysis. The Oracle
                  connects to the Aragora debate engine for real-time streaming
                  responses, multi-agent exploration, and consensus synthesis.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                  {(Object.keys(MODE_DESCRIPTIONS) as OracleMode[]).map((m) => {
                    const info = MODE_DESCRIPTIONS[m];
                    return (
                      <div
                        key={m}
                        className="p-4 border border-[var(--acid-green)]/20 rounded"
                      >
                        <div className="text-2xl mb-2">{info.icon}</div>
                        <div className="font-mono text-sm text-[var(--acid-green)] mb-1">
                          {info.label}
                        </div>
                        <p className="font-mono text-xs text-[var(--text-muted)]">
                          {info.desc}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

          {/* Footer */}
          <footer className="text-center text-xs font-mono py-8 border-t border-[var(--acid-green)]/20 mt-8">
            <div className="text-[var(--acid-green)]/50 mb-2">
              {'='.repeat(40)}
            </div>
            <p className="text-[var(--text-muted)]">
              {'>'} ARAGORA // ORACLE
            </p>
          </footer>
        </div>
      </main>
    </>
  );
}
