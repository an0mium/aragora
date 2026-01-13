'use client';

import { useState, useCallback } from 'react';
import { API_BASE_URL } from '@/config';

interface RedTeamFinding {
  attack_type: string;
  severity: string;
  description: string;
  recommendation?: string;
}

interface RedTeamResult {
  session_id: string;
  findings: RedTeamFinding[];
  robustness_score: number;
  attack_summary?: Record<string, number>;
}

interface RedTeamAnalysisPanelProps {
  debateId?: string;
  apiBase?: string;
  onComplete?: (result: RedTeamResult) => void;
}

const DEFAULT_API_BASE = API_BASE_URL;

const ATTACK_TYPES = [
  { value: 'logical_fallacy', label: 'Logical Fallacy', description: 'Test for flawed reasoning' },
  { value: 'edge_case', label: 'Edge Cases', description: 'Find boundary condition issues' },
  { value: 'unstated_assumption', label: 'Unstated Assumptions', description: 'Expose hidden assumptions' },
  { value: 'counterexample', label: 'Counterexamples', description: 'Find contradicting cases' },
  { value: 'scalability', label: 'Scalability', description: 'Test at scale limitations' },
  { value: 'security', label: 'Security', description: 'Security vulnerability analysis' },
];

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-red-100 dark:bg-red-900/30 border-red-300 dark:border-red-700 text-red-700 dark:text-red-400',
  high: 'bg-orange-100 dark:bg-orange-900/30 border-orange-300 dark:border-orange-700 text-orange-700 dark:text-orange-400',
  medium: 'bg-yellow-100 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700 text-yellow-700 dark:text-yellow-400',
  low: 'bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-400',
  info: 'bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-400',
};

export function RedTeamAnalysisPanel({
  debateId,
  apiBase = DEFAULT_API_BASE,
  onComplete,
}: RedTeamAnalysisPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedAttacks, setSelectedAttacks] = useState<string[]>(['logical_fallacy', 'edge_case']);
  const [maxRounds, setMaxRounds] = useState(3);
  const [focusProposal, setFocusProposal] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RedTeamResult | null>(null);

  const toggleAttackType = (type: string) => {
    setSelectedAttacks((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const runAnalysis = useCallback(async () => {
    if (!debateId) {
      setError('No debate selected for red team analysis');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${apiBase}/api/debates/${debateId}/red-team`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          attack_types: selectedAttacks.length > 0 ? selectedAttacks : undefined,
          max_rounds: maxRounds,
          focus_proposal: focusProposal || undefined,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
      onComplete?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [debateId, selectedAttacks, maxRounds, focusProposal, apiBase, onComplete]);

  const getRobustnessColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    if (score >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  // Collapsed view
  if (!isExpanded) {
    return (
      <div
        className="panel panel-compact cursor-pointer"
        onClick={() => setIsExpanded(true)}
      >
        <div className="flex items-center justify-between">
          <h3 className="panel-title-sm flex items-center gap-2">
            <span className="text-accent">{'>'}</span>
            RED_TEAM_ANALYSIS {result ? `[${Math.round(result.robustness_score * 100)}% robust]` : ''}
          </h3>
          <div className="flex items-center gap-2">
            {result && (
              <span className="text-xs font-mono text-text-muted">
                {result.findings?.length || 0} findings
              </span>
            )}
            <span className="panel-toggle">[EXPAND]</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header mb-4">
        <h3 className="panel-title-sm flex items-center gap-2">
          <span>🛡️</span> RED_TEAM_ANALYSIS
        </h3>
        <button
          onClick={() => setIsExpanded(false)}
          className="panel-toggle hover:text-accent"
        >
          [COLLAPSE]
        </button>
      </div>

      {/* Configuration */}
      <div className="bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg p-4 mb-4">
        <h4 className="text-sm font-medium text-zinc-500 dark:text-zinc-400 mb-3">Attack Types</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
          {ATTACK_TYPES.map((attack) => (
            <button
              key={attack.value}
              onClick={() => toggleAttackType(attack.value)}
              className={`p-2 rounded border text-left text-sm ${
                selectedAttacks.includes(attack.value)
                  ? 'border-red-500 bg-red-500/10 text-red-600 dark:text-red-400'
                  : 'border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600 text-zinc-600 dark:text-zinc-400'
              }`}
            >
              <div className="font-medium">{attack.label}</div>
              <div className="text-xs opacity-70">{attack.description}</div>
            </button>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-zinc-500 dark:text-zinc-400 mb-1">Max Rounds</label>
            <input
              type="number"
              min={1}
              max={5}
              value={maxRounds}
              onChange={(e) => setMaxRounds(Math.min(5, Math.max(1, parseInt(e.target.value) || 1)))}
              className="w-full bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded px-3 py-2 text-zinc-700 dark:text-zinc-300"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-500 dark:text-zinc-400 mb-1">Focus Proposal (optional)</label>
            <input
              type="text"
              value={focusProposal}
              onChange={(e) => setFocusProposal(e.target.value)}
              placeholder="Specific proposal to analyze"
              className="w-full bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded px-3 py-2 text-zinc-700 dark:text-zinc-300"
            />
          </div>
        </div>
      </div>

      {/* Run Button */}
      <button
        onClick={runAnalysis}
        disabled={loading || selectedAttacks.length === 0}
        className="w-full py-3 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white rounded-lg font-medium mb-4"
      >
        {loading ? 'Running Analysis...' : 'Run Red Team Analysis'}
      </button>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Robustness Score */}
          <div className="bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <span className="text-zinc-500 dark:text-zinc-400">Robustness Score</span>
              <span className={`text-2xl font-bold ${getRobustnessColor(result.robustness_score)}`}>
                {(result.robustness_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-2 bg-zinc-200 dark:bg-zinc-900 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full ${
                  result.robustness_score >= 0.8
                    ? 'bg-green-500'
                    : result.robustness_score >= 0.5
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                }`}
                style={{ width: `${result.robustness_score * 100}%` }}
              />
            </div>
          </div>

          {/* Findings */}
          <div className="bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg p-4">
            <h4 className="text-sm font-medium text-zinc-500 dark:text-zinc-400 mb-3">
              Findings ({result.findings?.length || 0})
            </h4>
            {result.findings && result.findings.length > 0 ? (
              <div className="space-y-2">
                {result.findings.map((finding, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded border ${SEVERITY_COLORS[finding.severity] || SEVERITY_COLORS.info}`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium">{finding.attack_type}</span>
                      <span className="text-xs uppercase px-2 py-0.5 rounded bg-black/20">
                        {finding.severity}
                      </span>
                    </div>
                    <p className="text-sm opacity-90">{finding.description}</p>
                    {finding.recommendation && (
                      <p className="text-xs mt-2 opacity-70">
                        💡 {finding.recommendation}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-zinc-500 text-center py-4">
                No vulnerabilities found
              </div>
            )}
          </div>

          {/* Session Info */}
          <div className="text-xs text-zinc-500 text-center">
            Session ID: {result.session_id}
          </div>
        </div>
      )}

      <div className="mt-3 text-[10px] text-text-muted font-mono">
        Adversarial red team analysis for debate robustness
      </div>
    </div>
  );
}

export default RedTeamAnalysisPanel;
