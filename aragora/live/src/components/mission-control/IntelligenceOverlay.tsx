'use client';

import { memo } from 'react';

export interface IntelligenceOverlayProps {
  overlays: {
    confidence: boolean;
    cruxBadges: boolean;
    evidenceCounts: boolean;
    precedents: boolean;
  };
  onToggle: (key: keyof IntelligenceOverlayProps['overlays']) => void;
}

const OVERLAY_OPTIONS: {
  key: keyof IntelligenceOverlayProps['overlays'];
  label: string;
  icon: string;
  description: string;
  color: string;
}[] = [
  {
    key: 'confidence',
    label: 'Confidence',
    icon: '◐',
    description: 'Belief probability halos',
    color: 'text-blue-400',
  },
  {
    key: 'cruxBadges',
    label: 'Crux',
    icon: '⚡',
    description: 'Key decision points',
    color: 'text-amber-400',
  },
  {
    key: 'evidenceCounts',
    label: 'Evidence',
    icon: '📊',
    description: 'Supporting evidence count',
    color: 'text-emerald-400',
  },
  {
    key: 'precedents',
    label: 'Precedents',
    icon: '📚',
    description: 'KM precedent matches',
    color: 'text-violet-400',
  },
];

export const IntelligenceOverlay = memo(function IntelligenceOverlay({
  overlays,
  onToggle,
}: IntelligenceOverlayProps) {
  return (
    <div
      className="flex flex-col gap-1 p-2 bg-[var(--surface)] border border-[var(--border)] rounded-lg"
      data-testid="intelligence-overlay"
    >
      <div className="text-xs font-mono font-bold text-[var(--text-muted)] px-1 mb-1">Intelligence</div>
      {OVERLAY_OPTIONS.map((opt) => {
        const active = overlays[opt.key];
        return (
          <button
            key={opt.key}
            onClick={() => onToggle(opt.key)}
            className={`flex items-center gap-2 px-2 py-1.5 rounded text-left transition-all text-xs
              ${active ? 'bg-[var(--bg)] border border-[var(--border)]' : 'hover:bg-[var(--bg)]/50 border border-transparent'}
            `}
            data-testid={`intelligence-toggle-${opt.key}`}
          >
            <span className={`${opt.color} text-sm`}>{opt.icon}</span>
            <span className={`font-mono ${active ? 'text-[var(--text)]' : 'text-[var(--text-muted)]'}`}>
              {opt.label}
            </span>
            <span
              className={`ml-auto w-3 h-3 rounded-sm border ${
                active ? 'bg-[var(--acid-green)] border-[var(--acid-green)]' : 'border-[var(--text-muted)]'
              }`}
            />
          </button>
        );
      })}
    </div>
  );
});

export default IntelligenceOverlay;
