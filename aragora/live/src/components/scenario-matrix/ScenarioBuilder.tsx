'use client';

import type { ScenarioInput } from './types';

export interface ScenarioBuilderProps {
  scenarios: ScenarioInput[];
  onAdd: () => void;
  onRemove: (index: number) => void;
  onUpdate: (index: number, scenario: ScenarioInput) => void;
}

export function ScenarioBuilder({
  scenarios,
  onAdd,
  onRemove,
  onUpdate,
}: ScenarioBuilderProps) {
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
              aria-label={`Remove scenario ${scenario.name}`}
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

export default ScenarioBuilder;
