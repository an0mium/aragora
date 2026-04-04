'use client';

import type { AuditVerdict } from './types';

interface VerdictSectionProps {
  verdict: AuditVerdict;
}

export function VerdictSection({ verdict }: VerdictSectionProps) {
  return (
    <div className="px-4 pb-4">
      <div className="p-4 bg-gradient-to-br from-purple-500/20 to-indigo-500/20 border border-purple-500/40 rounded-lg">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-lg">🎯</span>
          <span className="text-sm font-semibold text-purple-400">Final Audit Verdict</span>
          <span
            className={`ml-auto text-xs font-theme-data px-2 py-0.5 rounded ${
              verdict.confidence >= 0.8
                ? 'bg-green-500/20 text-green-400'
                : verdict.confidence >= 0.6
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : 'bg-red-500/20 text-red-400'
            }`}
          >
            {Math.round(verdict.confidence * 100)}% confidence
          </span>
        </div>

        <p className="text-sm text-text mb-4">{verdict.recommendation}</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {verdict.unanimousIssues.length > 0 && (
            <VerdictColumn
              title={`Unanimous Issues (${verdict.unanimousIssues.length})`}
              items={verdict.unanimousIssues}
              colorClass="text-green-400"
              bgClass="bg-green-500/10"
              borderClass="border-green-500/20"
            />
          )}

          {verdict.splitOpinions.length > 0 && (
            <VerdictColumn
              title={`Split Opinions (${verdict.splitOpinions.length})`}
              items={verdict.splitOpinions}
              colorClass="text-yellow-400"
              bgClass="bg-yellow-500/10"
              borderClass="border-yellow-500/20"
            />
          )}

          {verdict.riskAreas.length > 0 && (
            <VerdictColumn
              title={`Risk Areas (${verdict.riskAreas.length})`}
              items={verdict.riskAreas}
              colorClass="text-red-400"
              bgClass="bg-red-500/10"
              borderClass="border-red-500/20"
            />
          )}
        </div>
      </div>
    </div>
  );
}

interface VerdictColumnProps {
  title: string;
  items: string[];
  colorClass: string;
  bgClass: string;
  borderClass: string;
}

function VerdictColumn({ title, items, colorClass, bgClass, borderClass }: VerdictColumnProps) {
  return (
    <div className={`p-2 ${bgClass} border ${borderClass} rounded`}>
      <div className={`${colorClass} font-medium mb-1`}>{title}</div>
      <ul className="text-text-muted space-y-0.5">
        {items.slice(0, 3).map((item, idx) => (
          <li key={idx}>• {item.slice(0, 50)}...</li>
        ))}
      </ul>
    </div>
  );
}
