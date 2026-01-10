'use client';

import { useState, useCallback } from 'react';
import { getAgentColors } from '@/utils/agentColors';

// Types from the backend
interface ScenarioResult {
  scenario_name: string;
  parameters: Record<string, unknown>;
  constraints: string[];
  is_baseline: boolean;
  winner: string | null;
  final_answer: string;
  confidence: number;
  consensus_reached: boolean;
  rounds_used: number;
}

interface ConditionalConclusion {
  condition: string;
  parameters: Record<string, unknown>;
  conclusion: string;
  confidence: number;
}

interface ComparisonMatrix {
  scenarios: string[];
  consensus_rate: number;
  avg_confidence: number;
  avg_rounds: number;
}

interface MatrixDebateResult {
  matrix_id: string;
  task: string;
  scenario_count: number;
  results: ScenarioResult[];
  universal_conclusions: string[];
  conditional_conclusions: ConditionalConclusion[];
  comparison_matrix: ComparisonMatrix;
}

interface ScenarioInput {
  name: string;
  parameters: Record<string, string>;
  constraints: string[];
  is_baseline: boolean;
}

function MetricCard({
  label,
  value,
  color = 'text-acid-green',
}: {
  label: string;
  value: string | number;
  color?: string;
}) {
  return (
    <div className="bg-bg/50 border border-acid-green/20 p-3 text-center">
      <div className="text-xs font-mono text-text-muted mb-1">{label}</div>
      <div className={`text-lg font-mono ${color}`}>{value}</div>
    </div>
  );
}

function ScenarioCard({
  result,
  isExpanded,
  onToggle,
}: {
  result: ScenarioResult;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const winnerColors = result.winner ? getAgentColors(result.winner) : null;

  return (
    <div
      className={`bg-surface border transition-colors ${
        result.is_baseline
          ? 'border-gold/40'
          : result.consensus_reached
          ? 'border-acid-green/40'
          : 'border-crimson/40'
      }`}
    >
      <div
        className="px-4 py-3 cursor-pointer hover:bg-bg/50 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`text-xs font-mono ${
                result.is_baseline ? 'text-gold' : 'text-acid-cyan'
              }`}
            >
              {result.is_baseline ? '[BASELINE]' : '[SCENARIO]'}
            </span>
            <span className="text-sm font-mono text-text">{result.scenario_name}</span>
          </div>

          <div className="flex items-center gap-3">
            {result.winner && winnerColors && (
              <span className={`px-2 py-0.5 ${winnerColors.bg} ${winnerColors.text} text-xs font-mono`}>
                {result.winner}
              </span>
            )}
            <span
              className={`w-2 h-2 rounded-full ${
                result.consensus_reached ? 'bg-acid-green' : 'bg-crimson'
              }`}
            />
            <span className="text-xs font-mono text-text-muted">
              {(result.confidence * 100).toFixed(0)}%
            </span>
            <span className="text-xs font-mono text-text-muted">
              {isExpanded ? '[-]' : '[+]'}
            </span>
          </div>
        </div>

        {/* Parameters preview */}
        {Object.keys(result.parameters).length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(result.parameters).map(([key, value]) => (
              <span
                key={key}
                className="px-2 py-0.5 bg-acid-cyan/10 text-acid-cyan text-[10px] font-mono"
              >
                {key}={String(value)}
              </span>
            ))}
          </div>
        )}
      </div>

      {isExpanded && (
        <div className="px-4 pb-4 border-t border-border space-y-3 pt-3">
          {/* Final answer */}
          <div>
            <div className="text-xs font-mono text-text-muted mb-1">CONCLUSION</div>
            <div className="text-sm font-mono text-text bg-bg/50 p-3 border border-border">
              {result.final_answer || 'No conclusion reached'}
            </div>
          </div>

          {/* Constraints */}
          {result.constraints.length > 0 && (
            <div>
              <div className="text-xs font-mono text-text-muted mb-1">CONSTRAINTS</div>
              <ul className="space-y-1">
                {result.constraints.map((c, i) => (
                  <li key={i} className="text-xs font-mono text-text-muted pl-2 border-l border-gold/30">
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border">
            <div className="text-xs font-mono">
              <span className="text-text-muted">Rounds: </span>
              <span className="text-text">{result.rounds_used}</span>
            </div>
            <div className="text-xs font-mono">
              <span className="text-text-muted">Consensus: </span>
              <span className={result.consensus_reached ? 'text-acid-green' : 'text-crimson'}>
                {result.consensus_reached ? 'YES' : 'NO'}
              </span>
            </div>
            <div className="text-xs font-mono">
              <span className="text-text-muted">Confidence: </span>
              <span className="text-acid-cyan">{(result.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ScenarioBuilder({
  scenarios,
  onAdd,
  onRemove,
  onUpdate,
}: {
  scenarios: ScenarioInput[];
  onAdd: () => void;
  onRemove: (index: number) => void;
  onUpdate: (index: number, scenario: ScenarioInput) => void;
}) {
  return (
    <div className="space-y-3">
      {scenarios.map((scenario, idx) => (
        <div key={idx} className="bg-bg/50 border border-acid-green/20 p-3">
          <div className="flex items-center gap-2 mb-3">
            <input
              type="text"
              value={scenario.name}
              onChange={(e) =>
                onUpdate(idx, { ...scenario, name: e.target.value })
              }
              placeholder="Scenario name..."
              className="flex-1 px-2 py-1 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green"
            />
            <label className="flex items-center gap-1 text-xs font-mono text-text-muted">
              <input
                type="checkbox"
                checked={scenario.is_baseline}
                onChange={(e) =>
                  onUpdate(idx, { ...scenario, is_baseline: e.target.checked })
                }
                className="accent-gold"
              />
              Baseline
            </label>
            <button
              onClick={() => onRemove(idx)}
              className="px-2 py-1 text-xs font-mono text-crimson hover:bg-crimson/10"
            >
              [X]
            </button>
          </div>

          {/* Parameters */}
          <div className="mb-2">
            <div className="text-[10px] font-mono text-text-muted mb-1">
              PARAMETERS (key=value, comma separated)
            </div>
            <input
              type="text"
              value={Object.entries(scenario.parameters)
                .map(([k, v]) => `${k}=${v}`)
                .join(', ')}
              onChange={(e) => {
                const params: Record<string, string> = {};
                e.target.value.split(',').forEach((pair) => {
                  const [key, value] = pair.split('=').map((s) => s.trim());
                  if (key && value) params[key] = value;
                });
                onUpdate(idx, { ...scenario, parameters: params });
              }}
              placeholder="e.g., budget=high, timeline=short"
              className="w-full px-2 py-1 bg-bg border border-acid-cyan/30 text-text font-mono text-xs focus:outline-none focus:border-acid-cyan"
            />
          </div>

          {/* Constraints */}
          <div>
            <div className="text-[10px] font-mono text-text-muted mb-1">
              CONSTRAINTS (comma separated)
            </div>
            <input
              type="text"
              value={scenario.constraints.join(', ')}
              onChange={(e) =>
                onUpdate(idx, {
                  ...scenario,
                  constraints: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                })
              }
              placeholder="e.g., must be scalable, no external deps"
              className="w-full px-2 py-1 bg-bg border border-gold/30 text-text font-mono text-xs focus:outline-none focus:border-gold"
            />
          </div>
        </div>
      ))}

      <button
        onClick={onAdd}
        className="w-full py-2 border border-dashed border-acid-green/40 text-acid-green text-xs font-mono hover:bg-acid-green/10 transition-colors"
      >
        + ADD SCENARIO
      </button>
    </div>
  );
}

export function ScenarioMatrixView() {
  const [task, setTask] = useState('');
  const [scenarios, setScenarios] = useState<ScenarioInput[]>([
    { name: 'Baseline', parameters: {}, constraints: [], is_baseline: true },
  ]);
  const [result, setResult] = useState<MatrixDebateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedScenarios, setExpandedScenarios] = useState<Set<number>>(new Set());

  const addScenario = useCallback(() => {
    setScenarios((prev) => [
      ...prev,
      { name: `Scenario ${prev.length + 1}`, parameters: {}, constraints: [], is_baseline: false },
    ]);
  }, []);

  const removeScenario = useCallback((index: number) => {
    setScenarios((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const updateScenario = useCallback((index: number, scenario: ScenarioInput) => {
    setScenarios((prev) => prev.map((s, i) => (i === index ? scenario : s)));
  }, []);

  const toggleExpanded = useCallback((index: number) => {
    setExpandedScenarios((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);

  const runMatrix = async () => {
    if (!task.trim() || scenarios.length === 0) return;

    try {
      setLoading(true);
      setError(null);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
      const response = await fetch(`${apiUrl}/api/debates/matrix`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task,
          agents: ['claude', 'gpt4'],
          scenarios: scenarios.map((s) => ({
            name: s.name,
            parameters: s.parameters,
            constraints: s.constraints,
            is_baseline: s.is_baseline,
          })),
          max_rounds: 3,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to run matrix debate');
      }

      const data = await response.json();
      setResult(data);
      setExpandedScenarios(new Set());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to run matrix debate');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-surface border border-acid-green/30 p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-mono text-acid-green">{'>'} SCENARIO MATRIX</h2>
          <span className="text-xs font-mono text-text-muted">
            Parallel scenario comparison
          </span>
        </div>

        {/* Task input */}
        <div className="mb-4">
          <label className="text-xs font-mono text-text-muted block mb-1">BASE TASK</label>
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Enter the debate topic that will be explored across all scenarios..."
            className="w-full px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green resize-none"
            rows={2}
          />
        </div>

        {/* Scenario builder */}
        <div className="mb-4">
          <label className="text-xs font-mono text-text-muted block mb-2">
            SCENARIOS ({scenarios.length})
          </label>
          <ScenarioBuilder
            scenarios={scenarios}
            onAdd={addScenario}
            onRemove={removeScenario}
            onUpdate={updateScenario}
          />
        </div>

        {/* Run button */}
        <button
          onClick={runMatrix}
          disabled={loading || !task.trim() || scenarios.length === 0}
          className="w-full py-3 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'RUNNING MATRIX...' : 'RUN SCENARIO MATRIX'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-surface border border-crimson/30 p-4">
          <div className="text-xs font-mono text-crimson">Error: {error}</div>
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Summary metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Scenarios"
              value={result.scenario_count}
              color="text-acid-cyan"
            />
            <MetricCard
              label="Consensus Rate"
              value={`${(result.comparison_matrix.consensus_rate * 100).toFixed(0)}%`}
              color={
                result.comparison_matrix.consensus_rate > 0.5
                  ? 'text-acid-green'
                  : 'text-yellow-400'
              }
            />
            <MetricCard
              label="Avg Confidence"
              value={`${(result.comparison_matrix.avg_confidence * 100).toFixed(0)}%`}
            />
            <MetricCard
              label="Avg Rounds"
              value={result.comparison_matrix.avg_rounds.toFixed(1)}
              color="text-gold"
            />
          </div>

          {/* Universal conclusions */}
          {result.universal_conclusions.length > 0 && (
            <div className="bg-surface border border-acid-green/30">
              <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
                <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
                  {'>'} UNIVERSAL CONCLUSIONS
                </span>
              </div>
              <div className="p-4 space-y-2">
                {result.universal_conclusions.map((conclusion, i) => (
                  <div
                    key={i}
                    className="px-3 py-2 bg-acid-green/10 border border-acid-green/30 text-sm font-mono text-text"
                  >
                    {conclusion}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scenario results */}
          <div className="bg-surface border border-acid-cyan/30">
            <div className="px-4 py-3 border-b border-acid-cyan/20 bg-bg/50">
              <span className="text-xs font-mono text-acid-cyan uppercase tracking-wider">
                {'>'} SCENARIO RESULTS
              </span>
            </div>
            <div className="p-4 space-y-3">
              {result.results.map((scenarioResult, i) => (
                <ScenarioCard
                  key={i}
                  result={scenarioResult}
                  isExpanded={expandedScenarios.has(i)}
                  onToggle={() => toggleExpanded(i)}
                />
              ))}
            </div>
          </div>

          {/* Conditional conclusions */}
          {result.conditional_conclusions.length > 0 && (
            <div className="bg-surface border border-gold/30">
              <div className="px-4 py-3 border-b border-gold/20 bg-bg/50">
                <span className="text-xs font-mono text-gold uppercase tracking-wider">
                  {'>'} CONDITIONAL CONCLUSIONS
                </span>
              </div>
              <div className="p-4 space-y-3">
                {result.conditional_conclusions.map((cc, i) => (
                  <div key={i} className="p-3 bg-bg/50 border border-gold/20">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs font-mono text-gold">{cc.condition}</span>
                      <span className="text-xs font-mono text-text-muted">
                        ({(cc.confidence * 100).toFixed(0)}% confidence)
                      </span>
                    </div>
                    {Object.keys(cc.parameters).length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-2">
                        {Object.entries(cc.parameters).map(([key, value]) => (
                          <span
                            key={key}
                            className="px-1 py-0.5 bg-acid-cyan/10 text-acid-cyan text-[10px] font-mono"
                          >
                            {key}={String(value)}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="text-sm font-mono text-text">
                      {cc.conclusion.length > 300
                        ? cc.conclusion.slice(0, 300) + '...'
                        : cc.conclusion}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Comparison grid */}
          <div className="bg-surface border border-purple/30">
            <div className="px-4 py-3 border-b border-purple/20 bg-bg/50">
              <span className="text-xs font-mono text-purple uppercase tracking-wider">
                {'>'} COMPARISON GRID
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="bg-bg/50">
                    <th className="px-4 py-2 text-left text-text-muted">Scenario</th>
                    <th className="px-4 py-2 text-center text-text-muted">Consensus</th>
                    <th className="px-4 py-2 text-center text-text-muted">Confidence</th>
                    <th className="px-4 py-2 text-center text-text-muted">Rounds</th>
                    <th className="px-4 py-2 text-center text-text-muted">Winner</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r, i) => {
                    const winnerColors = r.winner ? getAgentColors(r.winner) : null;
                    return (
                      <tr key={i} className="border-t border-border">
                        <td className="px-4 py-2">
                          <span className={r.is_baseline ? 'text-gold' : 'text-text'}>
                            {r.scenario_name}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-center">
                          <span
                            className={`px-2 py-0.5 ${
                              r.consensus_reached
                                ? 'bg-acid-green/20 text-acid-green'
                                : 'bg-crimson/20 text-crimson'
                            }`}
                          >
                            {r.consensus_reached ? 'YES' : 'NO'}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-center text-acid-cyan">
                          {(r.confidence * 100).toFixed(0)}%
                        </td>
                        <td className="px-4 py-2 text-center text-text-muted">
                          {r.rounds_used}
                        </td>
                        <td className="px-4 py-2 text-center">
                          {r.winner && winnerColors ? (
                            <span className={`px-2 py-0.5 ${winnerColors.bg} ${winnerColors.text}`}>
                              {r.winner}
                            </span>
                          ) : (
                            <span className="text-text-muted">-</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="bg-surface border border-acid-green/30 p-8 text-center">
          <div className="text-4xl font-mono text-acid-green/30 mb-4">[...]</div>
          <div className="text-sm font-mono text-text-muted">
            Configure scenarios above and run the matrix to see results
          </div>
        </div>
      )}
    </div>
  );
}
