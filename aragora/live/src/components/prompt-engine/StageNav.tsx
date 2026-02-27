'use client';

import type { EngineStage } from '@/hooks/usePromptEngine';

const STAGES: { key: EngineStage; label: string }[] = [
  { key: 'intake', label: 'Intake' },
  { key: 'decompose', label: 'Decompose' },
  { key: 'interrogate', label: 'Interrogate' },
  { key: 'research', label: 'Research' },
  { key: 'spec', label: 'Spec' },
  { key: 'validate', label: 'Validate' },
  { key: 'handoff', label: 'Handoff' },
];

interface StageNavProps {
  currentStage: EngineStage | null;
}

export function StageNav({ currentStage }: StageNavProps) {
  const currentIdx = currentStage ? STAGES.findIndex(s => s.key === currentStage) : -1;

  return (
    <nav className="w-48 border-r border-[var(--border)] p-4 font-mono text-xs space-y-1">
      <h2 className="text-[var(--acid-green)] font-bold mb-4 text-sm">STAGES</h2>
      {STAGES.map((stage, i) => {
        const isActive = stage.key === currentStage;
        const isComplete = i < currentIdx;
        return (
          <div
            key={stage.key}
            className={`flex items-center gap-2 py-1.5 px-2 ${
              isActive ? 'text-[var(--acid-green)] bg-[var(--acid-green)]/10' :
              isComplete ? 'text-[var(--text-muted)]' :
              'text-[var(--text-muted)]/50'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${
              isActive ? 'bg-[var(--acid-green)] animate-pulse' :
              isComplete ? 'bg-[var(--text-muted)]' :
              'bg-[var(--border)]'
            }`} />
            <span>{stage.label}</span>
            {isComplete && <span className="ml-auto text-[var(--text-muted)]">&#10003;</span>}
          </div>
        );
      })}
    </nav>
  );
}
