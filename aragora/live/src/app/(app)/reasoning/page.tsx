'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useBackend } from '@/components/BackendSelector';

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface CruxClaim {
  claim_id?: string;
  claim: string;
  probability: number;
  sensitivity: number;
  source_agent?: string;
}

interface LoadBearingClaim {
  claim_id?: string;
  claim: string;
  centrality: number;
  source_agent?: string;
}

interface ClaimNode {
  id: string;
  text: string;
  type: string;
  agent?: string;
  evidence_count?: number;
  contradicts?: string[];
}

interface PositionRecord {
  agent: string;
  round: number;
  stance: string;
  confidence: number;
  key_argument: string;
}

interface PivotPoint {
  agent: string;
  from_round: number;
  to_round: number;
  from_stance: string;
  to_stance: string;
  pivot_magnitude: number;
  pivot_type: string;
}

interface PositionEvolution {
  debate_id: string;
  topic: string;
  positions: Record<string, PositionRecord[]>;
  pivots: PivotPoint[];
  summary: {
    convergence_score: number;
    total_pivots: number;
    reversals: number;
    stability_scores: Record<string, number>;
    influencers: Record<string, number>;
  };
}

type TabType = 'belief' | 'claims' | 'positions';

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const STANCE_COLORS: Record<string, string> = {
  strongly_agree: '#22c55e',
  agree: '#4ade80',
  lean_agree: '#86efac',
  neutral: '#a3a3a3',
  lean_disagree: '#fca5a5',
  disagree: '#f87171',
  strongly_disagree: '#ef4444',
};

const STANCE_LABELS: Record<string, string> = {
  strongly_agree: 'STRONG YES',
  agree: 'AGREE',
  lean_agree: 'LEAN YES',
  neutral: 'NEUTRAL',
  lean_disagree: 'LEAN NO',
  disagree: 'DISAGREE',
  strongly_disagree: 'STRONG NO',
};

function stanceColor(stance: string): string {
  return STANCE_COLORS[stance] ?? '#a3a3a3';
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function ReasoningPage() {
  const { config } = useBackend();
  const backendUrl = config.api;

  const [activeTab, setActiveTab] = useState<TabType>('belief');
  const [debateId, setDebateId] = useState('');
  const [loadedId, setLoadedId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Belief Network tab
  const [cruxes, setCruxes] = useState<CruxClaim[]>([]);
  const [loadBearing, setLoadBearing] = useState<LoadBearingClaim[]>([]);

  // Claims tab
  const [claims, setClaims] = useState<ClaimNode[]>([]);

  // Positions tab
  const [positions, setPositions] = useState<PositionEvolution | null>(null);

  const loadData = useCallback(async () => {
    if (!debateId.trim()) return;
    setLoading(true);
    setError(null);
    const id = encodeURIComponent(debateId.trim());

    try {
      const [cruxRes, lbRes, graphRes, posRes] = await Promise.allSettled([
        fetch(`${backendUrl}/api/belief-network/${id}/cruxes`),
        fetch(`${backendUrl}/api/belief-network/${id}/load-bearing-claims`),
        fetch(`${backendUrl}/api/belief-network/${id}/graph`),
        fetch(`${backendUrl}/api/v1/debates/${id}/positions`),
      ]);

      let anyOk = false;

      if (cruxRes.status === 'fulfilled' && cruxRes.value.ok) {
        const d = await cruxRes.value.json();
        setCruxes(d.cruxes ?? d.data?.cruxes ?? []);
        anyOk = true;
      } else {
        setCruxes([]);
      }

      if (lbRes.status === 'fulfilled' && lbRes.value.ok) {
        const d = await lbRes.value.json();
        setLoadBearing(d.load_bearing_claims ?? d.claims ?? d.data?.load_bearing_claims ?? []);
        anyOk = true;
      } else {
        setLoadBearing([]);
      }

      if (graphRes.status === 'fulfilled' && graphRes.value.ok) {
        const d = await graphRes.value.json();
        setClaims(d.nodes ?? d.graph?.nodes ?? d.data?.nodes ?? []);
        anyOk = true;
      } else {
        setClaims([]);
      }

      if (posRes.status === 'fulfilled' && posRes.value.ok) {
        const d = await posRes.value.json();
        setPositions(d);
        anyOk = true;
      } else {
        setPositions(null);
      }

      if (!anyOk) {
        const firstFailed =
          cruxRes.status === 'fulfilled'
            ? `HTTP ${cruxRes.value.status}`
            : 'Network error';
        setError(`Failed to load reasoning data: ${firstFailed}`);
      }

      setLoadedId(debateId.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [backendUrl, debateId]);

  /* ---- Belief Network tab ---- */
  const renderBeliefNetwork = () => (
    <div className="space-y-6">
      {/* Crux Claims */}
      <div>
        <h2 className="text-lg font-mono font-bold text-acid-green mb-3">Crux Claims</h2>
        {cruxes.length === 0 ? (
          <p className="text-text-muted font-mono text-sm">No crux claims found for this debate.</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {cruxes.map((c, i) => (
              <div key={i} className="p-4 bg-surface border border-border rounded-lg">
                <p className="text-sm font-mono text-text mb-3">{c.claim}</p>
                <div className="mb-2">
                  <div className="flex items-center justify-between text-xs text-text-muted font-mono mb-1">
                    <span>Probability</span>
                    <span>{(c.probability * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-bg rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${c.probability * 100}%`,
                        backgroundColor: 'var(--acid-green)',
                      }}
                    />
                  </div>
                </div>
                <div className="text-xs text-text-muted font-mono">
                  Sensitivity: <span className="text-text">{c.sensitivity.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Load-Bearing Claims */}
      <div>
        <h2 className="text-lg font-mono font-bold text-acid-green mb-3">Load-Bearing Claims</h2>
        {loadBearing.length === 0 ? (
          <p className="text-text-muted font-mono text-sm">No load-bearing claims found.</p>
        ) : (
          <div className="space-y-2">
            {loadBearing.map((lb, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-3 bg-surface border border-border rounded-lg"
              >
                <span className="text-sm font-mono text-text flex-1 mr-4">{lb.claim}</span>
                <span className="text-xs font-mono text-acid-green whitespace-nowrap">
                  centrality: {lb.centrality.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  /* ---- Claims tab ---- */
  const renderClaims = () => (
    <div className="space-y-4">
      <h2 className="text-lg font-mono font-bold text-acid-green mb-3">Claim Graph</h2>
      {claims.length === 0 ? (
        <p className="text-text-muted font-mono text-sm">No claims found for this debate.</p>
      ) : (
        <div className="space-y-2">
          {claims.map((c, i) => {
            const hasContradiction = c.contradicts && c.contradicts.length > 0;
            return (
              <div
                key={c.id || i}
                className={`p-3 bg-surface border rounded-lg ${
                  hasContradiction ? 'border-red-500/60' : 'border-border'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span
                    className="px-2 py-0.5 text-xs font-mono rounded border whitespace-nowrap"
                    style={{
                      borderColor: hasContradiction ? '#ef4444' : 'var(--border)',
                      color: hasContradiction ? '#ef4444' : 'var(--acid-green)',
                      backgroundColor: hasContradiction
                        ? 'rgba(239,68,68,0.1)'
                        : 'rgba(0,255,65,0.1)',
                    }}
                  >
                    {c.type}
                  </span>
                  <div className="flex-1">
                    <p className="text-sm font-mono text-text">{c.text}</p>
                    <div className="flex gap-4 mt-1 text-xs text-text-muted font-mono">
                      {c.agent && <span>Agent: {c.agent}</span>}
                      {c.evidence_count !== undefined && (
                        <span>Evidence: {c.evidence_count}</span>
                      )}
                      {hasContradiction && (
                        <span className="text-red-400">
                          Contradicts: {c.contradicts!.join(', ')}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  /* ---- Positions tab ---- */
  const renderPositions = () => {
    if (!positions) {
      return (
        <p className="text-text-muted font-mono text-sm">No position data found for this debate.</p>
      );
    }

    const agents = Object.keys(positions.positions);
    const maxRound = Math.max(
      ...agents.flatMap((a) => positions.positions[a].map((p) => p.round)),
      0,
    );
    const rounds = Array.from({ length: maxRound + 1 }, (_, i) => i);

    return (
      <div className="space-y-6">
        <h2 className="text-lg font-mono font-bold text-acid-green mb-3">
          Position Evolution
        </h2>

        {/* Summary stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3 bg-surface border border-border rounded-lg">
            <div className="text-xs text-text-muted font-mono">Convergence</div>
            <div className="text-xl font-mono font-bold text-acid-green">
              {(positions.summary.convergence_score * 100).toFixed(0)}%
            </div>
          </div>
          <div className="p-3 bg-surface border border-border rounded-lg">
            <div className="text-xs text-text-muted font-mono">Total Pivots</div>
            <div className="text-xl font-mono font-bold text-text">
              {positions.summary.total_pivots}
            </div>
          </div>
          <div className="p-3 bg-surface border border-border rounded-lg">
            <div className="text-xs text-text-muted font-mono">Reversals</div>
            <div className="text-xl font-mono font-bold text-red-400">
              {positions.summary.reversals}
            </div>
          </div>
          <div className="p-3 bg-surface border border-border rounded-lg">
            <div className="text-xs text-text-muted font-mono">Agents</div>
            <div className="text-xl font-mono font-bold text-text">{agents.length}</div>
          </div>
        </div>

        {/* Per-agent timelines */}
        <div className="space-y-4">
          {agents.map((agent) => {
            const records = positions.positions[agent];
            const agentPivots = positions.pivots.filter((p) => p.agent === agent);
            const pivotRounds = new Set(agentPivots.filter((p) => p.pivot_magnitude >= 0.3).map((p) => p.to_round));
            const stability = positions.summary.stability_scores[agent] ?? 1;

            return (
              <div key={agent} className="p-4 bg-surface border border-border rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-mono font-bold text-text">{agent}</span>
                  <span className="text-xs font-mono text-text-muted">
                    stability: {(stability * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Horizontal timeline */}
                <div className="flex gap-1">
                  {rounds.map((r) => {
                    const rec = records.find((p) => p.round === r);
                    const isPivot = pivotRounds.has(r);
                    const bg = rec ? stanceColor(rec.stance) : 'var(--border)';

                    return (
                      <div
                        key={r}
                        className="relative flex-1 group"
                        title={rec ? `R${r}: ${rec.stance} (${(rec.confidence * 100).toFixed(0)}%)` : `R${r}: -`}
                      >
                        <div
                          className="h-6 rounded-sm"
                          style={{ backgroundColor: bg, opacity: rec ? 1 : 0.3 }}
                        />
                        {isPivot && (
                          <div
                            className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full border border-bg"
                            style={{ backgroundColor: '#facc15' }}
                          />
                        )}
                        <div className="text-center text-[9px] text-text-muted font-mono mt-0.5">
                          {r}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Stance legend */}
        <div className="flex flex-wrap gap-3 text-xs font-mono text-text-muted">
          {Object.entries(STANCE_COLORS).map(([stance, color]) => (
            <div key={stance} className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
              <span>{stance.replace(/_/g, ' ')}</span>
            </div>
          ))}
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#facc15' }} />
            <span>pivot point</span>
          </div>
        </div>
      </div>
    );
  };

  /* ---- Main render ---- */
  return (
    <div className="min-h-screen bg-bg text-text relative overflow-hidden">
      <Scanlines />
      <CRTVignette />

      <div className="max-w-6xl mx-auto px-4 py-8 relative z-10">
        {/* Breadcrumb + Title */}
        <div className="mb-6">
          <div className="text-xs font-mono text-text-muted mb-1">
            DASHBOARD / REASONING ENGINE
          </div>
          <h1 className="text-3xl font-mono font-bold text-acid-green">
            Reasoning Engine
          </h1>
        </div>

        {/* Debate ID input */}
        <div className="flex gap-2 mb-6">
          <input
            type="text"
            value={debateId}
            onChange={(e) => setDebateId(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && loadData()}
            placeholder="Enter debate ID..."
            className="flex-1 px-3 py-2 bg-surface border border-border rounded font-mono text-sm text-text placeholder:text-text-muted focus:outline-none focus:border-acid-green/50"
          />
          <button
            onClick={loadData}
            disabled={loading || !debateId.trim()}
            className="px-4 py-2 font-mono text-sm bg-acid-green/10 border border-acid-green/30 text-acid-green rounded hover:bg-acid-green/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Loading...' : 'Load'}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6">
            <ErrorWithRetry error={error} onRetry={loadData} />
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-border pb-2">
          {(['belief', 'claims', 'positions'] as TabType[]).map((tab) => {
            const labels: Record<TabType, string> = {
              belief: 'BELIEF NETWORK',
              claims: 'CLAIMS',
              positions: 'POSITIONS',
            };
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 font-mono text-sm rounded-t transition-colors ${
                  activeTab === tab
                    ? 'bg-acid-green/10 text-acid-green border-b-2 border-acid-green'
                    : 'text-text-muted hover:text-text'
                }`}
              >
                {labels[tab]}
              </button>
            );
          })}
        </div>

        {/* Content */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-acid-green font-mono animate-pulse">Loading...</div>
          </div>
        ) : !loadedId ? (
          <div className="text-center py-12 text-text-muted font-mono text-sm">
            Enter a debate ID above to explore reasoning data.
          </div>
        ) : (
          <div>
            {activeTab === 'belief' && renderBeliefNetwork()}
            {activeTab === 'claims' && renderClaims()}
            {activeTab === 'positions' && renderPositions()}
          </div>
        )}
      </div>
    </div>
  );
}
