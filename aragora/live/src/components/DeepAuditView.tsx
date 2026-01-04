'use client';

import { useState, useMemo } from 'react';
import type { StreamEvent } from '@/types/events';
import { RoleBadge } from './RoleBadge';
import { CitationBadge } from './CitationsPanel';

interface DeepAuditViewProps {
  events: StreamEvent[];
  isActive: boolean;
  onToggle: () => void;
}

// Deep Audit rounds (Heavy3-inspired 6-round protocol)
const AUDIT_ROUNDS = [
  { round: 1, name: 'Initial Analysis', icon: '🔬', description: 'Agents present initial assessments' },
  { round: 2, name: 'Skeptical Review', icon: '🤔', description: 'Challenge assumptions and identify gaps' },
  { round: 3, name: 'Lateral Exploration', icon: '💡', description: 'Explore alternative perspectives' },
  { round: 4, name: 'Devil\'s Advocacy', icon: '😈', description: 'Argue against the emerging consensus' },
  { round: 5, name: 'Synthesis', icon: '⚖️', description: 'Integrate insights into recommendations' },
  { round: 6, name: 'Cross-Examination', icon: '🎯', description: 'Final probing of conclusions' },
];

interface RoundData {
  round: number;
  messages: Array<{
    agent: string;
    content: string;
    role: string;
    cognitiveRole?: string;
    confidence?: number;
    citations?: number;
    timestamp: number;
  }>;
  status: 'pending' | 'active' | 'complete';
}

export function DeepAuditView({ events, isActive, onToggle }: DeepAuditViewProps) {
  const [expandedRound, setExpandedRound] = useState<number | null>(null);

  // Extract round data from events
  const roundData = useMemo(() => {
    const rounds: Record<number, RoundData> = {};
    let maxRound = 0;

    // Initialize rounds
    AUDIT_ROUNDS.forEach((r) => {
      rounds[r.round] = {
        round: r.round,
        messages: [],
        status: 'pending',
      };
    });

    // Populate from events
    events.forEach((event) => {
      if (event.type === 'agent_message' && event.agent) {
        const round = event.round || 1;
        if (round > maxRound) maxRound = round;

        if (rounds[round]) {
          rounds[round].messages.push({
            agent: event.agent,
            content: event.data?.content as string || '',
            role: event.data?.role as string || 'proposer',
            cognitiveRole: event.data?.cognitive_role as string,
            confidence: event.data?.confidence as number,
            citations: (event.data?.citations as any[])?.length,
            timestamp: event.timestamp,
          });
        }
      }
    });

    // Update statuses
    AUDIT_ROUNDS.forEach((r) => {
      if (r.round < maxRound) {
        rounds[r.round].status = 'complete';
      } else if (r.round === maxRound && rounds[r.round].messages.length > 0) {
        rounds[r.round].status = 'active';
      }
    });

    return Object.values(rounds);
  }, [events]);

  const completedRounds = roundData.filter((r) => r.status === 'complete').length;
  const activeRound = roundData.find((r) => r.status === 'active');

  if (!isActive) {
    return (
      <button
        onClick={onToggle}
        className="px-3 py-1.5 text-sm bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded hover:bg-purple-500/30 flex items-center gap-2"
      >
        <span>🔬</span>
        <span>Deep Audit Mode</span>
      </button>
    );
  }

  return (
    <div className="bg-gradient-to-br from-purple-500/5 to-indigo-500/5 border border-purple-500/30 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-purple-500/10 px-4 py-3 border-b border-purple-500/20 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-lg">🔬</span>
          <div>
            <h3 className="text-sm font-semibold text-purple-400">Deep Audit Mode</h3>
            <p className="text-xs text-text-muted">
              {completedRounds}/6 rounds complete
            </p>
          </div>
        </div>
        <button
          onClick={onToggle}
          className="px-2 py-1 text-xs bg-surface border border-border rounded hover:bg-surface-hover"
        >
          Exit
        </button>
      </div>

      {/* Round Timeline */}
      <div className="p-4">
        <div className="space-y-2">
          {AUDIT_ROUNDS.map((auditRound) => {
            const data = roundData.find((r) => r.round === auditRound.round);
            const isExpanded = expandedRound === auditRound.round;
            const status = data?.status || 'pending';

            const statusConfig = {
              pending: { bg: 'bg-surface', border: 'border-border', text: 'text-text-muted' },
              active: { bg: 'bg-accent/10', border: 'border-accent', text: 'text-accent' },
              complete: { bg: 'bg-success/10', border: 'border-success/50', text: 'text-success' },
            };

            const config = statusConfig[status];

            return (
              <div
                key={auditRound.round}
                className={`rounded-lg border transition-all ${config.bg} ${config.border}`}
              >
                <button
                  onClick={() => setExpandedRound(isExpanded ? null : auditRound.round)}
                  className="w-full p-3 text-left"
                  disabled={status === 'pending'}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-lg">{auditRound.icon}</span>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-medium ${config.text}`}>
                            Round {auditRound.round}: {auditRound.name}
                          </span>
                          {status === 'active' && (
                            <span className="w-2 h-2 bg-accent rounded-full animate-pulse" />
                          )}
                          {status === 'complete' && (
                            <span className="text-success text-xs">✓</span>
                          )}
                        </div>
                        <p className="text-xs text-text-muted">{auditRound.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {data && data.messages.length > 0 && (
                        <span className="text-xs text-text-muted">
                          {data.messages.length} response{data.messages.length !== 1 ? 's' : ''}
                        </span>
                      )}
                      {status !== 'pending' && (
                        <span className="text-text-muted text-xs">
                          {isExpanded ? '▼' : '▶'}
                        </span>
                      )}
                    </div>
                  </div>
                </button>

                {/* Expanded Content */}
                {isExpanded && data && data.messages.length > 0 && (
                  <div className="px-3 pb-3 space-y-2">
                    {data.messages.map((msg, idx) => (
                      <div
                        key={idx}
                        className="p-2 bg-bg rounded border border-border"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-text">{msg.agent}</span>
                            <RoleBadge role={msg.role} cognitiveRole={msg.cognitiveRole} size="sm" />
                          </div>
                          <div className="flex items-center gap-2">
                            {msg.confidence !== undefined && (
                              <span className={`text-xs font-mono ${
                                msg.confidence >= 0.8 ? 'text-green-400' :
                                msg.confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                              }`}>
                                {Math.round(msg.confidence * 100)}%
                              </span>
                            )}
                            {msg.citations !== undefined && msg.citations > 0 && (
                              <CitationBadge count={msg.citations} />
                            )}
                          </div>
                        </div>
                        <p className="text-xs text-text-muted whitespace-pre-wrap break-words max-h-32 overflow-y-auto">
                          {msg.content.slice(0, 500)}
                          {msg.content.length > 500 && '...'}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Active Round Summary */}
      {activeRound && (
        <div className="px-4 pb-4">
          <div className="p-3 bg-accent/10 border border-accent/30 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 bg-accent rounded-full animate-pulse" />
              <span className="text-sm font-medium text-accent">
                Currently: {AUDIT_ROUNDS.find((r) => r.round === activeRound.round)?.name}
              </span>
            </div>
            <p className="text-xs text-text-muted">
              {activeRound.messages.length} agents have responded in this round
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// Toggle button component
export function DeepAuditToggle({ isActive, onToggle }: { isActive: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className={`px-3 py-1.5 text-sm rounded flex items-center gap-2 transition-colors ${
        isActive
          ? 'bg-purple-500 text-white'
          : 'bg-purple-500/20 text-purple-400 border border-purple-500/30 hover:bg-purple-500/30'
      }`}
    >
      <span>🔬</span>
      <span>{isActive ? 'Deep Audit Active' : 'Deep Audit'}</span>
    </button>
  );
}
