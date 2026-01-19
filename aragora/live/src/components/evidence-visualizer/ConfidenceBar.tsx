'use client';

/**
 * Confidence bar component for displaying percentage values
 */

import React from 'react';

interface ConfidenceBarProps {
  value: number;
  label: string;
  color?: string;
}

export function ConfidenceBar({ value, label, color = 'acid-green' }: ConfidenceBarProps) {
  const percentage = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-xs text-text-muted w-16">{label}</span>
      <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
        <div
          className={`h-full bg-${color} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="font-mono text-xs text-text-muted w-10 text-right">{percentage}%</span>
    </div>
  );
}

export default ConfidenceBar;
