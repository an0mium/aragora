'use client';

import { useState, useCallback } from 'react';
import { API_BASE_URL } from '@/config';

interface InterventionPanelProps {
  debateId: string;
  isActive: boolean;
  isPaused: boolean;
  currentRound: number;
  totalRounds: number;
  agents: string[];
  consensusThreshold: number;
  onPause?: () => void;
  onResume?: () => void;
  onInject?: (content: string) => void;
  onWeightChange?: (agent: string, weight: number) => void;
  onThresholdChange?: (threshold: number) => void;
  apiBase?: string;
}

interface AgentWeight {
  agent: string;
  weight: number;
}

export function InterventionPanel({
  debateId,
  isActive,
  isPaused,
  currentRound,
  totalRounds,
  agents,
  consensusThreshold: initialThreshold,
  onPause,
  onResume,
  onInject,
  onWeightChange,
  onThresholdChange,
  apiBase = API_BASE_URL,
}: InterventionPanelProps) {
  const [injection, setInjection] = useState('');
  const [injecting, setInjecting] = useState(false);
  const [pauseLoading, setPauseLoading] = useState(false);
  const [_showWeights, _setShowWeights] = useState(false);
  const [agentWeights, setAgentWeights] = useState<AgentWeight[]>(
    agents.map((agent) => ({ agent, weight: 1.0 }))
  );
  const [consensusThreshold, setConsensusThreshold] = useState(initialThreshold);
  const [followUpQuestion, setFollowUpQuestion] = useState('');
  const [activeTab, setActiveTab] = useState<'inject' | 'control' | 'weights'>('inject');

  // Handle pause/resume
  const handlePauseToggle = useCallback(async () => {
    setPauseLoading(true);
    try {
      const action = isPaused ? 'resume' : 'pause';
      const response = await fetch(
        `${apiBase}/api/debates/${debateId}/intervention/${action}`,
        { method: 'POST' }
      );

      if (response.ok) {
        if (isPaused) {
          onResume?.();
        } else {
          onPause?.();
        }
      }
    } catch (error) {
      console.error('Failed to toggle pause:', error);
    } finally {
      setPauseLoading(false);
    }
  }, [apiBase, debateId, isPaused, onPause, onResume]);

  // Handle argument injection
  const handleInject = useCallback(async () => {
    if (!injection.trim()) return;

    setInjecting(true);
    try {
      const response = await fetch(
        `${apiBase}/api/debates/${debateId}/intervention/inject`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content: injection,
            type: 'argument',
            source: 'user',
          }),
        }
      );

      if (response.ok) {
        onInject?.(injection);
        setInjection('');
      }
    } catch (error) {
      console.error('Failed to inject argument:', error);
    } finally {
      setInjecting(false);
    }
  }, [apiBase, debateId, injection, onInject]);

  // Handle follow-up question
  const handleFollowUp = useCallback(async () => {
    if (!followUpQuestion.trim()) return;

    setInjecting(true);
    try {
      const response = await fetch(
        `${apiBase}/api/debates/${debateId}/intervention/inject`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content: followUpQuestion,
            type: 'follow_up',
            source: 'user',
          }),
        }
      );

      if (response.ok) {
        onInject?.(followUpQuestion);
        setFollowUpQuestion('');
      }
    } catch (error) {
      console.error('Failed to add follow-up:', error);
    } finally {
      setInjecting(false);
    }
  }, [apiBase, debateId, followUpQuestion, onInject]);

  // Handle weight change
  const handleWeightChange = useCallback(
    async (agent: string, weight: number) => {
      setAgentWeights((prev) =>
        prev.map((w) => (w.agent === agent ? { ...w, weight } : w))
      );

      try {
        await fetch(
          `${apiBase}/api/debates/${debateId}/intervention/weights`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent, weight }),
          }
        );
        onWeightChange?.(agent, weight);
      } catch (error) {
        console.error('Failed to update weight:', error);
      }
    },
    [apiBase, debateId, onWeightChange]
  );

  // Handle threshold change
  const handleThresholdChange = useCallback(
    async (threshold: number) => {
      setConsensusThreshold(threshold);

      try {
        await fetch(
          `${apiBase}/api/debates/${debateId}/intervention/threshold`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ threshold }),
          }
        );
        onThresholdChange?.(threshold);
      } catch (error) {
        console.error('Failed to update threshold:', error);
      }
    },
    [apiBase, debateId, onThresholdChange]
  );

  if (!isActive) {
    return (
      <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
        <div className="text-center text-[var(--text-muted)] text-sm font-mono">
          Intervention controls are only available during active debates
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[var(--surface)] border border-[var(--acid-green)]/30">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <span className="text-lg"></span>
          <h3 className="text-sm font-mono font-bold text-[var(--text)] uppercase">
            Intervention Controls
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-[var(--text-muted)]">
            Round {currentRound}/{totalRounds}
          </span>
          <button
            onClick={handlePauseToggle}
            disabled={pauseLoading}
            className={`px-2 py-1 text-xs font-mono border transition-colors ${
              isPaused
                ? 'bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30'
                : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/30'
            }`}
          >
            {pauseLoading ? '...' : isPaused ? ' RESUME' : ' PAUSE'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[var(--border)]">
        {[
          { id: 'inject', label: 'Inject', icon: '' },
          { id: 'control', label: 'Control', icon: '' },
          { id: 'weights', label: 'Weights', icon: '' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`flex-1 px-3 py-2 text-xs font-mono transition-colors ${
              activeTab === tab.id
                ? 'bg-[var(--acid-green)]/10 text-[var(--acid-green)] border-b-2 border-[var(--acid-green)]'
                : 'text-[var(--text-muted)] hover:bg-[var(--bg)]'
            }`}
          >
            <span className="mr-1">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-3">
        {/* Inject Tab */}
        {activeTab === 'inject' && (
          <div className="space-y-4">
            {/* Argument Injection */}
            <div>
              <label className="block text-xs font-mono text-[var(--text-muted)] mb-2">
                INJECT ARGUMENT
              </label>
              <textarea
                value={injection}
                onChange={(e) => setInjection(e.target.value)}
                placeholder="Add your argument to the debate..."
                className="w-full h-24 bg-[var(--bg)] border border-[var(--border)] text-[var(--text)] font-mono text-sm p-2 resize-none focus:border-[var(--acid-green)] focus:outline-none"
              />
              <button
                onClick={handleInject}
                disabled={!injection.trim() || injecting}
                className="mt-2 w-full px-3 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {injecting ? 'INJECTING...' : ' INJECT ARGUMENT'}
              </button>
            </div>

            {/* Follow-up Question */}
            <div>
              <label className="block text-xs font-mono text-[var(--text-muted)] mb-2">
                ADD FOLLOW-UP QUESTION
              </label>
              <input
                type="text"
                value={followUpQuestion}
                onChange={(e) => setFollowUpQuestion(e.target.value)}
                placeholder="Ask a follow-up question..."
                className="w-full bg-[var(--bg)] border border-[var(--border)] text-[var(--text)] font-mono text-sm p-2 focus:border-[var(--acid-green)] focus:outline-none"
              />
              <button
                onClick={handleFollowUp}
                disabled={!followUpQuestion.trim() || injecting}
                className="mt-2 w-full px-3 py-2 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                 ADD FOLLOW-UP
              </button>
            </div>
          </div>
        )}

        {/* Control Tab */}
        {activeTab === 'control' && (
          <div className="space-y-4">
            {/* Consensus Threshold */}
            <div>
              <label className="block text-xs font-mono text-[var(--text-muted)] mb-2">
                CONSENSUS THRESHOLD: {Math.round(consensusThreshold * 100)}%
              </label>
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.05"
                value={consensusThreshold}
                onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
                className="w-full accent-[var(--acid-green)]"
              />
              <div className="flex justify-between text-[10px] font-mono text-[var(--text-muted)] mt-1">
                <span>50%</span>
                <span>75%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Quick Actions */}
            <div>
              <label className="block text-xs font-mono text-[var(--text-muted)] mb-2">
                QUICK ACTIONS
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  className="px-3 py-2 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
                >
                   Skip Round
                </button>
                <button
                  className="px-3 py-2 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
                >
                   Add Round
                </button>
                <button
                  className="px-3 py-2 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
                >
                   Force Vote
                </button>
                <button
                  className="px-3 py-2 text-xs font-mono bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 transition-colors"
                >
                   End Debate
                </button>
              </div>
            </div>

            {/* Debate Status */}
            <div className="pt-3 border-t border-[var(--border)]">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-[var(--text-muted)]">Status</span>
                <span className={isPaused ? 'text-yellow-400' : 'text-green-400'}>
                  {isPaused ? ' PAUSED' : ' RUNNING'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Weights Tab */}
        {activeTab === 'weights' && (
          <div className="space-y-3">
            <div className="text-xs font-mono text-[var(--text-muted)] mb-3">
              Adjust agent influence on consensus:
            </div>
            {agentWeights.map(({ agent, weight }) => (
              <div key={agent} className="space-y-1">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-[var(--text)]">{agent}</span>
                  <span className="text-[var(--acid-cyan)]">{weight.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={weight}
                  onChange={(e) => handleWeightChange(agent, parseFloat(e.target.value))}
                  className="w-full accent-[var(--acid-cyan)] h-1"
                />
              </div>
            ))}
            <div className="pt-2 text-[10px] font-mono text-[var(--text-muted)]">
              0 = muted | 1 = normal | 2 = double influence
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-[var(--border)] text-[10px] font-mono text-[var(--text-muted)]">
        Interventions are logged in the audit trail
      </div>
    </div>
  );
}

export default InterventionPanel;
