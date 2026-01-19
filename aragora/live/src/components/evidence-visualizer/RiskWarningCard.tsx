'use client';

/**
 * Risk warning card component for displaying individual risk warnings
 */

import React from 'react';
import { SEVERITY_COLORS } from './types';
import type { RiskWarning } from './types';

interface RiskWarningCardProps {
  warning: RiskWarning;
}

export function RiskWarningCard({ warning }: RiskWarningCardProps) {
  const severityStyle = SEVERITY_COLORS[warning.severity] || SEVERITY_COLORS.low;

  return (
    <div
      className={`p-3 rounded ${severityStyle.bg} border border-${warning.severity === 'critical' ? 'acid-red' : 'acid-green'}/30`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className={`font-mono text-xs uppercase ${severityStyle.text}`}>
          {warning.severity} - {warning.risk_type}
        </span>
        <span className="font-mono text-xs text-text-muted">
          {warning.domain}
        </span>
      </div>
      <p className="font-mono text-sm text-text">{warning.description}</p>
      {warning.mitigation && (
        <p className="font-mono text-xs text-acid-green mt-2">
          Mitigation: {warning.mitigation}
        </p>
      )}
    </div>
  );
}

export default RiskWarningCard;
