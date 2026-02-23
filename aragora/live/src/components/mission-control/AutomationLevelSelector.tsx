'use client';

import { memo } from 'react';

export type AutomationLevel = 'full' | 'guided' | 'manual';

export interface AutomationLevelSelectorProps {
  value: AutomationLevel;
  onChange: (level: AutomationLevel) => void;
  disabled?: boolean;
}

const levels: { value: AutomationLevel; label: string; description: string; icon: string }[] = [
  {
    value: 'full',
    label: 'Full Auto',
    description: 'AI runs the entire pipeline automatically',
    icon: '⚡',
  },
  {
    value: 'guided',
    label: 'Guided',
    description: 'AI pauses at each stage for your approval',
    icon: '🎯',
  },
  {
    value: 'manual',
    label: 'Manual',
    description: 'AI extracts ideas, you control the rest',
    icon: '✋',
  },
];

export const AutomationLevelSelector = memo(function AutomationLevelSelector({
  value,
  onChange,
  disabled,
}: AutomationLevelSelectorProps) {
  return (
    <div className="flex gap-2" data-testid="automation-level-selector">
      {levels.map((level) => {
        const isSelected = value === level.value;
        return (
          <button
            key={level.value}
            onClick={() => onChange(level.value)}
            disabled={disabled}
            className={`
              flex-1 px-3 py-2 rounded-lg border text-left transition-all
              ${
                isSelected
                  ? 'border-[var(--acid-green)] bg-[var(--acid-green)]/10'
                  : 'border-[var(--border)] bg-[var(--surface)] hover:border-[var(--text-muted)]'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
            data-testid={`automation-level-${level.value}`}
          >
            <div className="flex items-center gap-1.5">
              <span className="text-sm">{level.icon}</span>
              <span
                className={`text-xs font-mono font-bold ${
                  isSelected ? 'text-[var(--acid-green)]' : 'text-[var(--text)]'
                }`}
              >
                {level.label}
              </span>
            </div>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">{level.description}</p>
          </button>
        );
      })}
    </div>
  );
});

export default AutomationLevelSelector;
