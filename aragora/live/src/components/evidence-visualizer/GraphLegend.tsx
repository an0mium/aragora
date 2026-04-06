'use client';

/**
 * Graph legend component for displaying node type explanations
 */

import React from 'react';

const NODE_TYPES = [
  { color: 'bg-[var(--accent)]', label: 'Argument' },
  { color: 'bg-acid-red', label: 'Rebuttal' },
  { color: 'bg-[var(--acid-cyan)]', label: 'Synthesis' },
  { color: 'bg-acid-yellow', label: 'Evidence' },
] as const;

export function GraphLegend() {
  return (
    <div className="p-3 bg-surface/50 rounded">
      <h4 className="font-theme-data text-xs text-[var(--acid-cyan)] mb-2">Node Types</h4>
      <div className="flex flex-wrap gap-4 text-xs font-theme-data">
        {NODE_TYPES.map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1">
            <span className={`w-3 h-3 rounded-full ${color}`} />
            <span className="text-text-muted">{label}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export default GraphLegend;
