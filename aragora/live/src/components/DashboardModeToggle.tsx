'use client';

import { DashboardMode } from '@/hooks/useDashboardPreferences';

interface DashboardModeToggleProps {
  mode: DashboardMode;
  onModeChange: (mode: DashboardMode) => void;
  compact?: boolean;
}

export function DashboardModeToggle({ mode, onModeChange, compact = false }: DashboardModeToggleProps) {
  if (compact) {
    return (
      <button
        onClick={() => onModeChange(mode === 'focus' ? 'explorer' : 'focus')}
        className="px-2 py-1 text-xs font-mono border border-acid-green/30 text-text-muted hover:text-acid-green hover:border-acid-green transition-colors"
        title={mode === 'focus' ? 'Switch to Explorer Mode (show all panels)' : 'Switch to Focus Mode (minimal panels)'}
      >
        {mode === 'focus' ? '[FOCUS]' : '[EXPLORER]'}
      </button>
    );
  }

  return (
    <div className="flex items-center gap-1 bg-bg border border-acid-green/30 p-0.5 font-mono text-xs">
      <button
        onClick={() => onModeChange('focus')}
        className={`px-3 py-1.5 transition-colors ${
          mode === 'focus'
            ? 'bg-acid-green text-bg'
            : 'text-text-muted hover:text-acid-green'
        }`}
        title="Focus Mode: Show only essential panels for running debates"
      >
        [FOCUS]
      </button>
      <button
        onClick={() => onModeChange('explorer')}
        className={`px-3 py-1.5 transition-colors ${
          mode === 'explorer'
            ? 'bg-acid-green text-bg'
            : 'text-text-muted hover:text-acid-green'
        }`}
        title="Explorer Mode: Show all available panels and features"
      >
        [EXPLORER]
      </button>
    </div>
  );
}

export function FocusModeIndicator({ onClick }: { onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-1.5 text-xs font-mono bg-acid-cyan/10 border border-acid-cyan/30 text-acid-cyan hover:bg-acid-cyan/20 transition-colors"
      title="You're in Focus Mode. Click to explore more features."
    >
      <span className="w-2 h-2 bg-acid-cyan rounded-full animate-pulse" />
      FOCUS MODE
    </button>
  );
}
