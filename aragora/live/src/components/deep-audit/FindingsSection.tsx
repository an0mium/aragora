'use client';

import type { AuditFinding } from '@/types/events';
import { CATEGORY_CONFIG } from './types';

interface FindingsSectionProps {
  findings: AuditFinding[];
  showFindings: boolean;
  onToggle: () => void;
}

export function FindingsSection({ findings, showFindings, onToggle }: FindingsSectionProps) {
  if (findings.length === 0) return null;

  return (
    <div className="px-4 pb-4">
      <button
        onClick={onToggle}
        className="w-full p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg text-left"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">📋</span>
            <span className="text-sm font-medium text-amber-400">
              {findings.length} Finding{findings.length !== 1 ? 's' : ''} Detected
            </span>
          </div>
          <span className="text-text-muted text-xs">{showFindings ? '▼' : '▶'}</span>
        </div>
      </button>

      {showFindings && (
        <div className="mt-2 space-y-2">
          {findings.map((finding, idx) => (
            <FindingCard key={idx} finding={finding} />
          ))}
        </div>
      )}
    </div>
  );
}

function FindingCard({ finding }: { finding: AuditFinding }) {
  const config = CATEGORY_CONFIG[finding.category] || CATEGORY_CONFIG.insight;

  return (
    <div className={`p-3 rounded ${config.bg} border ${config.border}`}>
      <div className="flex items-start gap-2">
        <span>{config.icon}</span>
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <span className={`text-sm font-medium ${config.color}`}>
              {finding.category.charAt(0).toUpperCase() + finding.category.slice(1)}
            </span>
            {finding.severity > 0 && (
              <span
                className={`text-xs font-theme-data ${
                  finding.severity >= 0.7
                    ? 'text-red-400'
                    : finding.severity >= 0.4
                      ? 'text-yellow-400'
                      : 'text-green-400'
                }`}
              >
                Severity: {Math.round(finding.severity * 100)}%
              </span>
            )}
          </div>
          <p className="text-sm text-text">{finding.summary}</p>
          {finding.details && (
            <p className="text-xs text-text-muted mt-1">{finding.details.slice(0, 200)}...</p>
          )}
          <div className="flex gap-4 mt-2 text-xs">
            {finding.agents_agree.length > 0 && (
              <span className="text-green-400">Agree: {finding.agents_agree.join(', ')}</span>
            )}
            {finding.agents_disagree.length > 0 && (
              <span className="text-red-400">Disagree: {finding.agents_disagree.join(', ')}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
