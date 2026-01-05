'use client';

import { useState, useEffect } from 'react';

interface Moment {
  type: string;
  description: string;
  timestamp: string;
  significance: number;
  context: string;
}

interface AgentMomentsModalProps {
  agentName: string;
  onClose: () => void;
  apiBase?: string;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

export function AgentMomentsModal({ agentName, onClose, apiBase = DEFAULT_API_BASE }: AgentMomentsModalProps) {
  const [moments, setMoments] = useState<Moment[]>([]);
  const [narrative, setNarrative] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMoments() {
      try {
        setLoading(true);
        const res = await fetch(`${apiBase}/api/agent/${agentName}/moments?limit=20`);
        if (res.ok) {
          const data = await res.json();
          setMoments(data.moments || []);
          setNarrative(data.narrative || '');
        } else {
          setError('Failed to load moments');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch');
      } finally {
        setLoading(false);
      }
    }
    fetchMoments();
  }, [agentName, apiBase]);

  const getMomentIcon = (type: string): string => {
    switch (type.toLowerCase()) {
      case 'breakthrough': return '⚡';
      case 'upset_win': return '🏆';
      case 'consensus_leader': return '🎯';
      case 'streak': return '🔥';
      case 'first_win': return '🌟';
      case 'comeback': return '💪';
      case 'dominant_performance': return '👑';
      default: return '📌';
    }
  };

  const getSignificanceColor = (sig: number): string => {
    if (sig >= 0.8) return 'text-yellow-400';
    if (sig >= 0.6) return 'text-green-400';
    if (sig >= 0.4) return 'text-blue-400';
    return 'text-text-muted';
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-surface border border-border rounded-lg p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text">
            {agentName} - Moments Timeline
          </h3>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text text-xl"
          >
            &times;
          </button>
        </div>

        {/* Narrative Summary */}
        {narrative && (
          <div className="mb-4 p-3 bg-bg border border-border rounded-lg">
            <p className="text-sm text-text-muted italic">{narrative}</p>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-8">Loading moments...</div>
          )}

          {error && (
            <div className="text-center text-red-400 py-8">{error}</div>
          )}

          {!loading && !error && moments.length === 0 && (
            <div className="text-center text-text-muted py-8">
              No significant moments recorded yet.
            </div>
          )}

          {!loading && !error && moments.length > 0 && (
            <div className="space-y-3">
              {moments.map((moment, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-bg border border-border rounded-lg hover:border-accent/50 transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">{getMomentIcon(moment.type)}</span>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-text">
                          {moment.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                        </span>
                        <span className={`text-xs ${getSignificanceColor(moment.significance)}`}>
                          {(moment.significance * 100).toFixed(0)}% significance
                        </span>
                      </div>
                      <p className="text-sm text-text-muted">{moment.description}</p>
                      {moment.context && (
                        <p className="text-xs text-text-muted/70 mt-1">{moment.context}</p>
                      )}
                      <p className="text-xs text-text-muted/50 mt-1">
                        {new Date(moment.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-4 pt-4 border-t border-border flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-surface border border-border rounded text-sm text-text hover:bg-surface-hover"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
