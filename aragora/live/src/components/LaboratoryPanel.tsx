'use client';

import { useState, useEffect, useCallback } from 'react';

interface EmergentTrait {
  agent: string;
  trait: string;
  domain: string;
  confidence: number;
  evidence: string[];
  detected_at: string;
}

interface CrossPollination {
  source_agent: string;
  target_agent: string;
  trait: string;
  expected_improvement: number;
  rationale: string;
}

interface LaboratoryPanelProps {
  apiBase?: string;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

export function LaboratoryPanel({ apiBase = DEFAULT_API_BASE }: LaboratoryPanelProps) {
  const [traits, setTraits] = useState<EmergentTrait[]>([]);
  const [pollinations, setPollinations] = useState<CrossPollination[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'traits' | 'pollinations'>('traits');
  const [expanded, setExpanded] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [traitsRes, pollinationsRes] = await Promise.all([
        fetch(`${apiBase}/api/laboratory/emergent-traits?min_confidence=0.3&limit=10`),
        fetch(`${apiBase}/api/laboratory/cross-pollinations/suggest`),
      ]);

      if (traitsRes.ok) {
        const data = await traitsRes.json();
        setTraits(data.emergent_traits || []);
      }

      if (pollinationsRes.ok) {
        const data = await pollinationsRes.json();
        setPollinations(data.suggestions || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch laboratory data');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchData();
    // Refresh every 5 minutes
    const interval = setInterval(fetchData, 300000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.5) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const getDomainColor = (domain: string): string => {
    const colors: Record<string, string> = {
      technical: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      ethics: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      creative: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
      analytical: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
      general: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    };
    return colors[domain] || colors.general;
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text font-mono">Persona Laboratory</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchData}
            disabled={loading}
            className="text-xs font-mono text-text-muted hover:text-acid-green disabled:opacity-50"
          >
            [REFRESH]
          </button>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs font-mono text-text-muted hover:text-text"
          >
            [{expanded ? '-' : '+'}]
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="flex items-center gap-4 text-xs font-mono text-text-muted mb-4 border-b border-border pb-3">
        <span>
          Emergent Traits: <span className="text-acid-cyan">{traits.length}</span>
        </span>
        <span>
          Pollinations: <span className="text-acid-green">{pollinations.length}</span>
        </span>
      </div>

      {error && (
        <div className="mb-4 p-2 bg-warning/10 border border-warning/30 rounded text-sm text-warning font-mono">
          {error}
        </div>
      )}

      {expanded && (
        <>
          {/* Tab Navigation */}
          <div className="flex space-x-1 bg-bg border border-border rounded p-1 mb-4">
            <button
              onClick={() => setActiveTab('traits')}
              className={`px-3 py-1 rounded text-sm font-mono transition-colors flex-1 ${
                activeTab === 'traits'
                  ? 'bg-acid-cyan text-bg font-medium'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              EMERGENT TRAITS
            </button>
            <button
              onClick={() => setActiveTab('pollinations')}
              className={`px-3 py-1 rounded text-sm font-mono transition-colors flex-1 ${
                activeTab === 'pollinations'
                  ? 'bg-acid-green text-bg font-medium'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              CROSS-POLLINATIONS
            </button>
          </div>

          {/* Traits Tab */}
          {activeTab === 'traits' && (
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {loading && traits.length === 0 && (
                <div className="text-center text-text-muted py-4 font-mono text-sm">
                  Detecting emergent traits...
                </div>
              )}

              {!loading && traits.length === 0 && (
                <div className="text-center text-text-muted py-4 font-mono text-sm">
                  No emergent traits detected yet. Run more debates to discover agent specializations.
                </div>
              )}

              {traits.map((trait, index) => (
                <div
                  key={`${trait.agent}-${trait.trait}-${index}`}
                  className="p-3 bg-bg border border-border rounded-lg hover:border-acid-cyan/50 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-acid-cyan font-bold">
                        {trait.agent}
                      </span>
                      <span className={`px-2 py-0.5 text-xs rounded border ${getDomainColor(trait.domain)}`}>
                        {trait.domain}
                      </span>
                    </div>
                    <span className={`text-xs font-mono ${getConfidenceColor(trait.confidence)}`}>
                      {(trait.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  <p className="text-sm text-text font-medium mb-2">
                    {trait.trait}
                  </p>

                  {trait.evidence && trait.evidence.length > 0 && (
                    <div className="space-y-1">
                      {trait.evidence.slice(0, 2).map((e, i) => (
                        <p key={i} className="text-xs text-text-muted line-clamp-1">
                          {e}
                        </p>
                      ))}
                      {trait.evidence.length > 2 && (
                        <p className="text-xs text-text-muted">
                          +{trait.evidence.length - 2} more evidence
                        </p>
                      )}
                    </div>
                  )}

                  <div className="mt-2 text-xs text-text-muted font-mono">
                    Detected: {new Date(trait.detected_at).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Pollinations Tab */}
          {activeTab === 'pollinations' && (
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {loading && pollinations.length === 0 && (
                <div className="text-center text-text-muted py-4 font-mono text-sm">
                  Analyzing cross-pollination opportunities...
                </div>
              )}

              {!loading && pollinations.length === 0 && (
                <div className="text-center text-text-muted py-4 font-mono text-sm">
                  No cross-pollination suggestions yet. Lab needs more trait data.
                </div>
              )}

              {pollinations.map((pollination, index) => (
                <div
                  key={`${pollination.source_agent}-${pollination.target_agent}-${index}`}
                  className="p-3 bg-bg border border-border rounded-lg hover:border-acid-green/50 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2 font-mono text-sm">
                    <span className="text-acid-cyan">{pollination.source_agent}</span>
                    <span className="text-text-muted">-&gt;</span>
                    <span className="text-acid-green">{pollination.target_agent}</span>
                  </div>

                  <p className="text-sm text-text font-medium mb-1">
                    Transfer: {pollination.trait}
                  </p>

                  <p className="text-xs text-text-muted mb-2">
                    {pollination.rationale}
                  </p>

                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-text-muted">Expected improvement:</span>
                    <span className="text-acid-green">
                      +{(pollination.expected_improvement * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Help text when collapsed */}
      {!expanded && (
        <div className="text-xs font-mono text-text-muted">
          <p>
            <span className="text-acid-cyan">Emergent traits:</span> Specializations discovered from performance patterns
          </p>
          <p className="mt-1">
            <span className="text-acid-green">Cross-pollinations:</span> Suggested trait transfers between agents
          </p>
        </div>
      )}
    </div>
  );
}
