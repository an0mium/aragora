'use client';

import { useState, useCallback, useMemo, useEffect } from 'react';
import { getAgentColors } from '@/utils/agentColors';
import type { StreamEvent } from '@/types/events';

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

// View mode types
type ViewMode = 'grid' | 'list' | 'compare';

// Side-by-side comparison component
function CompareView({
  left,
  right,
  onClose,
}: {
  left: ScenarioResult;
  right: ScenarioResult;
  onClose: () => void;
}) {
  const leftColors = left.winner ? getAgentColors(left.winner) : null;
  const rightColors = right.winner ? getAgentColors(right.winner) : null;

  const renderDiff = (label: string, leftVal: string | number, rightVal: string | number) => {
    const isDifferent = leftVal !== rightVal;
    return (
      <div className="grid grid-cols-3 gap-2 py-2 border-b border-border">
        <div className={`text-right ${isDifferent ? 'text-text' : 'text-text-muted'}`}>
          {typeof leftVal === 'number' ? leftVal.toFixed(0) : leftVal}
        </div>
        <div className="text-center text-xs font-mono text-text-muted">{label}</div>
        <div className={`text-left ${isDifferent ? 'text-text' : 'text-text-muted'}`}>
          {typeof rightVal === 'number' ? rightVal.toFixed(0) : rightVal}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-surface border border-purple/40">
      <div className="px-4 py-3 border-b border-purple/20 bg-bg/50 flex items-center justify-between">
        <span className="text-xs font-mono text-purple uppercase tracking-wider">
          SCENARIO COMPARISON
        </span>
        <button
          onClick={onClose}
          className="text-xs font-mono text-text-muted hover:text-purple"
        >
          [CLOSE]
        </button>
      </div>

      <div className="p-4">
        {/* Headers */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className={`p-3 text-center ${left.is_baseline ? 'bg-gold/10 border-gold/30' : 'bg-acid-cyan/10 border-acid-cyan/30'} border`}>
            <div className={`text-sm font-mono ${left.is_baseline ? 'text-gold' : 'text-acid-cyan'}`}>
              {left.scenario_name}
            </div>
            {left.is_baseline && <div className="text-xs text-gold/70">[BASELINE]</div>}
          </div>
          <div className={`p-3 text-center ${right.is_baseline ? 'bg-gold/10 border-gold/30' : 'bg-acid-cyan/10 border-acid-cyan/30'} border`}>
            <div className={`text-sm font-mono ${right.is_baseline ? 'text-gold' : 'text-acid-cyan'}`}>
              {right.scenario_name}
            </div>
            {right.is_baseline && <div className="text-xs text-gold/70">[BASELINE]</div>}
          </div>
        </div>

        {/* Comparison metrics */}
        <div className="bg-bg/50 border border-border p-4 text-xs font-mono">
          {renderDiff('Consensus', left.consensus_reached ? 'YES' : 'NO', right.consensus_reached ? 'YES' : 'NO')}
          {renderDiff('Confidence', `${(left.confidence * 100).toFixed(0)}%`, `${(right.confidence * 100).toFixed(0)}%`)}
          {renderDiff('Rounds', left.rounds_used, right.rounds_used)}
          {renderDiff('Winner', left.winner || '-', right.winner || '-')}
        </div>

        {/* Conclusions side by side */}
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div>
            <div className="text-xs font-mono text-text-muted mb-2">CONCLUSION</div>
            <div className="text-xs font-mono text-text bg-bg/50 p-3 border border-border max-h-40 overflow-y-auto">
              {left.final_answer || 'No conclusion'}
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-text-muted mb-2">CONCLUSION</div>
            <div className="text-xs font-mono text-text bg-bg/50 p-3 border border-border max-h-40 overflow-y-auto">
              {right.final_answer || 'No conclusion'}
            </div>
          </div>
        </div>

        {/* Parameters comparison */}
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div>
            <div className="text-xs font-mono text-text-muted mb-2">PARAMETERS</div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(left.parameters).map(([key, value]) => (
                <span key={key} className="px-1 py-0.5 bg-acid-cyan/10 text-acid-cyan text-[10px] font-mono">
                  {key}={String(value)}
                </span>
              ))}
              {Object.keys(left.parameters).length === 0 && (
                <span className="text-text-muted text-[10px]">None</span>
              )}
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-text-muted mb-2">PARAMETERS</div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(right.parameters).map(([key, value]) => (
                <span key={key} className="px-1 py-0.5 bg-acid-cyan/10 text-acid-cyan text-[10px] font-mono">
                  {key}={String(value)}
                </span>
              ))}
              {Object.keys(right.parameters).length === 0 && (
                <span className="text-text-muted text-[10px]">None</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Visual grid view showing parameters vs outcomes
function GridView({
  results,
  onSelectCompare,
}: {
  results: ScenarioResult[];
  onSelectCompare: (left: number, right: number) => void;
}) {
  // Extract all unique parameter keys
  const allParamKeys = useMemo(() => {
    const keys = new Set<string>();
    results.forEach(r => Object.keys(r.parameters).forEach(k => keys.add(k)));
    return Array.from(keys);
  }, [results]);

  const [selectedForCompare, setSelectedForCompare] = useState<number | null>(null);

  const handleCellClick = (index: number) => {
    if (selectedForCompare === null) {
      setSelectedForCompare(index);
    } else if (selectedForCompare !== index) {
      onSelectCompare(selectedForCompare, index);
      setSelectedForCompare(null);
    } else {
      setSelectedForCompare(null);
    }
  };

  return (
    <div className="overflow-x-auto">
      <div className="text-xs font-mono text-text-muted mb-2">
        Click two scenarios to compare them side-by-side
      </div>
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(results.length, 4)}, 1fr)` }}>
        {results.map((r, i) => (
          <div
            key={i}
            onClick={() => handleCellClick(i)}
            className={`p-3 border cursor-pointer transition-all duration-200 ${
              selectedForCompare === i
                ? 'border-purple bg-purple/20 scale-105'
                : r.is_baseline
                ? 'border-gold/40 hover:border-gold'
                : r.consensus_reached
                ? 'border-acid-green/40 hover:border-acid-green'
                : 'border-crimson/40 hover:border-crimson'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className={`text-xs font-mono ${r.is_baseline ? 'text-gold' : 'text-text'}`}>
                {r.scenario_name}
              </span>
              <span
                className={`w-2 h-2 rounded-full ${r.consensus_reached ? 'bg-acid-green' : 'bg-crimson'}`}
              />
            </div>

            {/* Mini parameter grid */}
            <div className="space-y-1 mb-2">
              {allParamKeys.slice(0, 3).map(key => (
                <div key={key} className="flex justify-between text-[10px] font-mono">
                  <span className="text-text-muted">{key}:</span>
                  <span className="text-acid-cyan">{String(r.parameters[key] || '-')}</span>
                </div>
              ))}
            </div>

            {/* Confidence bar */}
            <div className="h-1 bg-bg rounded-full overflow-hidden">
              <div
                className={`h-full ${r.consensus_reached ? 'bg-acid-green' : 'bg-crimson'}`}
                style={{ width: `${r.confidence * 100}%` }}
              />
            </div>
            <div className="text-[10px] font-mono text-text-muted mt-1 text-right">
              {(r.confidence * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Filter types
interface FilterState {
  consensusOnly: boolean;
  minConfidence: number;
  searchTerm: string;
}

interface ScenarioMatrixViewProps {
  events?: StreamEvent[];
}

export function ScenarioMatrixView({ events = [] }: ScenarioMatrixViewProps) {
  const [task, setTask] = useState('');
  const [scenarios, setScenarios] = useState<ScenarioInput[]>([
    { name: 'Baseline', parameters: {}, constraints: [], is_baseline: true },
  ]);
  const [result, setResult] = useState<MatrixDebateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedScenarios, setExpandedScenarios] = useState<Set<number>>(new Set());
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [compareIndexes, setCompareIndexes] = useState<[number, number] | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    consensusOnly: false,
    minConfidence: 0,
    searchTerm: '',
  });

  // Listen for matrix debate events
  const latestMatrixEvent = useMemo(() => {
    const relevant = events.filter(e =>
      e.type === 'scenario_complete' ||
      e.type === 'matrix_complete'
    );
    return relevant[relevant.length - 1];
  }, [events]);

  // Refresh on matrix events
  useEffect(() => {
    if (latestMatrixEvent && result) {
      // Re-fetch the matrix result
      const refreshResult = async () => {
        try {
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
          const response = await fetch(
            `${apiUrl}/api/debates/matrix/${result.matrix_id}`
          );
          if (response.ok) {
            const data = await response.json();
            setResult(data);
          }
        } catch (e) {
          // Ignore refresh errors
        }
      };
      refreshResult();
    }
  }, [latestMatrixEvent, result]);

  // Filter results
  const filteredResults = useMemo(() => {
    if (!result) return [];
    return result.results.filter(r => {
      if (filters.consensusOnly && !r.consensus_reached) return false;
      if (r.confidence < filters.minConfidence) return false;
      if (filters.searchTerm) {
        const term = filters.searchTerm.toLowerCase();
        if (!r.scenario_name.toLowerCase().includes(term) &&
            !r.final_answer.toLowerCase().includes(term)) {
          return false;
        }
      }
      return true;
    });
  }, [result, filters]);

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

          {/* Comparison view */}
          {compareIndexes && result.results[compareIndexes[0]] && result.results[compareIndexes[1]] && (
            <CompareView
              left={result.results[compareIndexes[0]]}
              right={result.results[compareIndexes[1]]}
              onClose={() => setCompareIndexes(null)}
            />
          )}

          {/* Scenario results */}
          <div className="bg-surface border border-acid-cyan/30">
            <div className="px-4 py-3 border-b border-acid-cyan/20 bg-bg/50 flex items-center justify-between flex-wrap gap-2">
              <span className="text-xs font-mono text-acid-cyan uppercase tracking-wider">
                {'>'} SCENARIO RESULTS ({filteredResults.length}/{result.results.length})
              </span>

              {/* View mode toggle */}
              <div className="flex items-center gap-1">
                {(['list', 'grid'] as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`px-2 py-1 text-[10px] font-mono transition-colors ${
                      viewMode === mode
                        ? 'bg-acid-cyan text-bg'
                        : 'text-text-muted hover:text-acid-cyan'
                    }`}
                  >
                    {mode.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Filters */}
            <div className="px-4 py-2 border-b border-acid-cyan/10 bg-bg/30 flex items-center gap-4 flex-wrap">
              <input
                type="text"
                value={filters.searchTerm}
                onChange={(e) => setFilters({ ...filters, searchTerm: e.target.value })}
                placeholder="Search scenarios..."
                className="px-2 py-1 bg-bg border border-border text-xs font-mono text-text focus:outline-none focus:border-acid-cyan"
              />
              <label className="flex items-center gap-1 text-[10px] font-mono text-text-muted">
                <input
                  type="checkbox"
                  checked={filters.consensusOnly}
                  onChange={(e) => setFilters({ ...filters, consensusOnly: e.target.checked })}
                  className="accent-acid-green"
                />
                Consensus only
              </label>
              <label className="flex items-center gap-1 text-[10px] font-mono text-text-muted">
                Min confidence:
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.minConfidence * 100}
                  onChange={(e) => setFilters({ ...filters, minConfidence: parseInt(e.target.value) / 100 })}
                  className="w-20 h-1 accent-acid-cyan"
                />
                <span className="text-acid-cyan">{Math.round(filters.minConfidence * 100)}%</span>
              </label>
            </div>

            <div className="p-4">
              {viewMode === 'grid' ? (
                <GridView
                  results={filteredResults}
                  onSelectCompare={(left, right) => {
                    // Find original indexes from filtered results
                    const leftIdx = result.results.indexOf(filteredResults[left]);
                    const rightIdx = result.results.indexOf(filteredResults[right]);
                    setCompareIndexes([leftIdx, rightIdx]);
                  }}
                />
              ) : (
                <div className="space-y-3">
                  {filteredResults.map((scenarioResult, i) => {
                    const originalIdx = result.results.indexOf(scenarioResult);
                    return (
                      <ScenarioCard
                        key={originalIdx}
                        result={scenarioResult}
                        isExpanded={expandedScenarios.has(originalIdx)}
                        onToggle={() => toggleExpanded(originalIdx)}
                      />
                    );
                  })}
                  {filteredResults.length === 0 && (
                    <div className="text-center text-text-muted text-xs font-mono py-4">
                      No scenarios match your filters
                    </div>
                  )}
                </div>
              )}
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
