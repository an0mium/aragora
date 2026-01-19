'use client';

/**
 * Graph legend component for displaying node type explanations
 */

import React from 'react';

const NODE_TYPES = [
  { color: 'bg-acid-green', label: 'Argument' },
  { color: 'bg-acid-red', label: 'Rebuttal' },
  { color: 'bg-acid-cyan', label: 'Synthesis' },
  { color: 'bg-acid-yellow', label: 'Evidence' },
] as const;

export function GraphLegend() {
  return (
    <div className="p-3 bg-surface/50 rounded">
      <h4 className="font-mono text-xs text-acid-cyan mb-2">Node Types</h4>
      <div className="flex flex-wrap gap-4 text-xs font-mono">
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
