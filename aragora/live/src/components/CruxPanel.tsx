'use client';

import { useState, useCallback } from 'react';
import { API_BASE_URL } from '@/config';

interface Crux {
  claim_id: string;
  statement: string;
  author: string;
  crux_score: number;
  centrality: number;
  entropy: number;
  current_belief: {
    true_prob: number;
    false_prob: number;
    uncertain_prob: number;
    confidence: number;
  };
}

interface LoadBearingClaim {
  claim_id: string;
  statement: string;
  author: string;
  centrality: number;
}

interface CruxPanelProps {
  debateId?: string;
  apiBase?: string;
}

const DEFAULT_API_BASE = API_BASE_URL;

export function CruxPanel({ debateId: initialDebateId, apiBase = DEFAULT_API_BASE }: CruxPanelProps) {
  const [debateId, setDebateId] = useState(initialDebateId || '');
  const [cruxes, setCruxes] = useState<Crux[]>([]);
  const [loadBearingClaims, setLoadBearingClaims] = useState<LoadBearingClaim[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'cruxes' | 'load-bearing'>('cruxes');

  const fetchCruxData = useCallback(async (id: string) => {
    if (!id.trim()) {
      setError('Please enter a debate ID');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [cruxesRes, lbRes] = await Promise.all([
        fetch(`${apiBase}/api/belief-network/${id}/cruxes?top_k=5`),
        fetch(`${apiBase}/api/belief-network/${id}/load-bearing-claims?limit=5`),
      ]);

      if (!cruxesRes.ok) {
        const data = await cruxesRes.json();
        throw new Error(data.error || `HTTP ${cruxesRes.status}`);
      }

      const cruxesData = await cruxesRes.json();
      setCruxes(cruxesData.cruxes || []);

      if (lbRes.ok) {
        const lbData = await lbRes.json();
        setLoadBearingClaims(lbData.load_bearing_claims || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch crux data');
      setCruxes([]);
      setLoadBearingClaims([]);
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchCruxData(debateId);
  };

  const getEntropyColor = (entropy: number): string => {
    if (entropy >= 0.8) return 'text-red-400';
    if (entropy >= 0.5) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getCentralityColor = (centrality: number): string => {
    if (centrality >= 0.3) return 'text-acid-cyan';
    if (centrality >= 0.1) return 'text-text';
    return 'text-text-muted';
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text font-mono">Belief Network Analysis</h3>
        <span className="text-xs text-text-muted font-mono">[CRUXES]</span>
      </div>

      {/* Debate ID Input */}
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={debateId}
            onChange={(e) => setDebateId(e.target.value)}
            placeholder="Enter debate ID..."
            className="flex-1 px-3 py-2 bg-bg border border-border rounded text-sm font-mono text-text focus:border-acid-green focus:outline-none"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-acid-green text-bg font-mono text-sm font-bold hover:bg-acid-green/80 disabled:bg-text-muted transition-colors"
          >
            {loading ? '...' : 'ANALYZE'}
          </button>
        </div>
      </form>

      {error && (
        <div className="mb-4 p-2 bg-warning/10 border border-warning/30 rounded text-sm text-warning font-mono">
          {error}
        </div>
      )}

      {/* Tab Navigation */}
      {(cruxes.length > 0 || loadBearingClaims.length > 0) && (
        <div className="flex space-x-1 bg-bg border border-border rounded p-1 mb-4">
          <button
            onClick={() => setActiveTab('cruxes')}
            className={`px-3 py-1 rounded text-sm font-mono transition-colors flex-1 ${
              activeTab === 'cruxes'
                ? 'bg-acid-green text-bg font-medium'
                : 'text-text-muted hover:text-text'
            }`}
          >
            CRUXES ({cruxes.length})
          </button>
          <button
            onClick={() => setActiveTab('load-bearing')}
            className={`px-3 py-1 rounded text-sm font-mono transition-colors flex-1 ${
              activeTab === 'load-bearing'
                ? 'bg-acid-green text-bg font-medium'
                : 'text-text-muted hover:text-text'
            }`}
          >
            LOAD-BEARING ({loadBearingClaims.length})
          </button>
        </div>
      )}

      {/* Cruxes Tab */}
      {activeTab === 'cruxes' && (
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {cruxes.length === 0 && !loading && !error && (
            <div className="text-center text-text-muted py-4 font-mono text-sm">
              Enter a debate ID to analyze belief network cruxes.
            </div>
          )}

          {cruxes.map((crux, index) => (
            <div
              key={crux.claim_id}
              className="p-3 bg-bg border border-border rounded-lg hover:border-acid-cyan/50 transition-colors"
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <span className="px-2 py-0.5 text-xs bg-acid-cyan/20 text-acid-cyan border border-acid-cyan/30 rounded font-mono">
                  CRUX #{index + 1}
                </span>
                <span className="text-xs font-mono text-text-muted">
                  score: {crux.crux_score.toFixed(3)}
                </span>
              </div>

              <p className="text-sm text-text mb-2 line-clamp-2">
                {crux.statement}
              </p>

              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="text-text-muted">
                  by: <span className="text-text">{crux.author}</span>
                </span>
                <span className={getCentralityColor(crux.centrality)}>
                  centrality: {(crux.centrality * 100).toFixed(1)}%
                </span>
                <span className={getEntropyColor(crux.entropy)}>
                  entropy: {crux.entropy.toFixed(2)}
                </span>
              </div>

              {crux.current_belief && (
                <div className="mt-2 flex gap-2 text-xs font-mono">
                  <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded">
                    T: {(crux.current_belief.true_prob * 100).toFixed(0)}%
                  </span>
                  <span className="px-1.5 py-0.5 bg-red-500/20 text-red-400 rounded">
                    F: {(crux.current_belief.false_prob * 100).toFixed(0)}%
                  </span>
                  <span className="px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">
                    ?: {(crux.current_belief.uncertain_prob * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Load-Bearing Tab */}
      {activeTab === 'load-bearing' && (
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {loadBearingClaims.length === 0 && !loading && !error && (
            <div className="text-center text-text-muted py-4 font-mono text-sm">
              No load-bearing claims found for this debate.
            </div>
          )}

          {loadBearingClaims.map((claim, index) => (
            <div
              key={claim.claim_id}
              className="p-3 bg-bg border border-border rounded-lg hover:border-acid-green/50 transition-colors"
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <span className="px-2 py-0.5 text-xs bg-acid-green/20 text-acid-green border border-acid-green/30 rounded font-mono">
                  #{index + 1} STRUCTURAL
                </span>
                <span className={`text-xs font-mono ${getCentralityColor(claim.centrality)}`}>
                  centrality: {(claim.centrality * 100).toFixed(1)}%
                </span>
              </div>

              <p className="text-sm text-text mb-2 line-clamp-2">
                {claim.statement}
              </p>

              <div className="text-xs font-mono text-text-muted">
                by: <span className="text-text">{claim.author}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Help text */}
      <div className="mt-4 text-xs text-text-muted font-mono border-t border-border pt-3">
        <p><span className="text-acid-cyan">Cruxes:</span> Claims with high uncertainty and high centrality - resolving these would most impact the debate outcome.</p>
        <p className="mt-1"><span className="text-acid-green">Load-bearing:</span> Claims that many other claims depend on - foundational to the argument structure.</p>
      </div>
    </div>
  );
}
