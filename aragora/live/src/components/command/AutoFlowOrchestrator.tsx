'use client';

interface AutoFlowOrchestratorProps {
  currentPhase: string;
  phaseProgress: number;
  nodesCreated: number;
  onPause: () => void;
  onSkipToEnd: () => void;
  onCancel: () => void;
}

const PHASES = [
  { key: 'clustering', label: 'Clustering Ideas', icon: '\u{1F4A1}', color: 'indigo' },
  { key: 'goals', label: 'Extracting Goals', icon: '\u{1F3AF}', color: 'emerald' },
  { key: 'tasks', label: 'Decomposing into Tasks', icon: '\u2702', color: 'amber' },
  { key: 'agents', label: 'Assigning Agents', icon: '\u{1F916}', color: 'pink' },
  { key: 'validating', label: 'Validating', icon: '\u{1F52C}', color: 'violet' },
];

export function AutoFlowOrchestrator({ currentPhase, phaseProgress, nodesCreated, onPause, onSkipToEnd, onCancel }: AutoFlowOrchestratorProps) {
  const currentIndex = PHASES.findIndex(p => p.key === currentPhase);

  return (
    <div className="absolute inset-0 bg-bg/80 backdrop-blur-sm z-20 flex items-center justify-center">
      <div className="bg-surface border border-border rounded-xl p-6 w-[480px] shadow-2xl">
        {/* Progress Bar */}
        <div className="h-1.5 bg-bg rounded-full mb-6 overflow-hidden">
          <div
            className="h-full bg-acid-green rounded-full transition-all duration-500"
            style={{ width: `${((currentIndex + phaseProgress) / PHASES.length) * 100}%` }}
          />
        </div>

        {/* Phase List */}
        <div className="space-y-3 mb-6">
          {PHASES.map((phase, i) => {
            const isActive = i === currentIndex;
            const isDone = i < currentIndex;
            const colorMap: Record<string, string> = {
              indigo: 'text-indigo-400', emerald: 'text-emerald-400',
              amber: 'text-amber-400', pink: 'text-pink-400', violet: 'text-violet-400',
            };

            return (
              <div
                key={phase.key}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all ${
                  isActive ? 'bg-acid-green/10 border border-acid-green/30' :
                  isDone ? 'opacity-60' : 'opacity-30'
                }`}
              >
                {/* Status indicator */}
                <div className="w-6 h-6 flex items-center justify-center flex-shrink-0">
                  {isDone ? (
                    <span className="text-emerald-400 text-sm">{'\u2713'}</span>
                  ) : isActive ? (
                    <div className="w-4 h-4 border-2 border-acid-green/30 border-t-acid-green rounded-full animate-spin" />
                  ) : (
                    <span className="text-text-muted text-xs">{i + 1}</span>
                  )}
                </div>

                {/* Phase info */}
                <span className="text-lg">{phase.icon}</span>
                <div className="flex-1">
                  <span className={`text-sm font-mono ${isActive ? 'text-text font-bold' : 'text-text-muted'}`}>
                    {phase.label}
                  </span>
                  {isActive && (
                    <span className={`ml-2 text-xs font-mono ${colorMap[phase.color]}`}>
                      ({nodesCreated} nodes created)
                    </span>
                  )}
                </div>

                {/* Phase number */}
                <span className="text-xs font-mono text-text-muted">
                  {i + 1}/{PHASES.length}
                </span>
              </div>
            );
          })}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between">
          <button
            onClick={onCancel}
            className="px-3 py-1.5 text-xs font-mono text-red-400 hover:bg-red-500/10 rounded transition-colors"
          >
            Cancel
          </button>
          <div className="flex gap-2">
            <button
              onClick={onPause}
              className="px-3 py-1.5 text-xs font-mono text-text-muted border border-border rounded hover:bg-bg transition-colors"
            >
              Pause
            </button>
            <button
              onClick={onSkipToEnd}
              className="px-3 py-1.5 text-xs font-mono text-acid-green border border-acid-green/30 rounded hover:bg-acid-green/10 transition-colors"
            >
              Skip to End
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
