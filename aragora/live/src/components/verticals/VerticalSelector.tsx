'use client';

/**
 * Vertical Selector Component
 *
 * Allows users to select and configure vertical specialists for
 * domain-specific AI tasks (software, legal, healthcare, accounting, research).
 */

import { useState, useCallback } from 'react';

export interface Vertical {
  id: string;
  displayName: string;
  description: string;
  expertiseAreas: string[];
  complianceFrameworks: string[];
  defaultModel: string;
  icon: string;
}

// Available verticals (would be fetched from API in production)
const VERTICALS: Vertical[] = [
  {
    id: 'software',
    displayName: 'Software Engineering',
    description: 'Code review, security analysis, and architecture design',
    expertiseAreas: ['Code Review', 'Security Analysis', 'Architecture Design', 'Performance'],
    complianceFrameworks: ['OWASP', 'CWE'],
    defaultModel: 'claude-sonnet-4',
    icon: '💻',
  },
  {
    id: 'legal',
    displayName: 'Legal',
    description: 'Contract analysis, compliance review, and regulatory matters',
    expertiseAreas: ['Contract Analysis', 'Regulatory Compliance', 'Risk Assessment', 'Privacy Law'],
    complianceFrameworks: ['GDPR', 'CCPA', 'HIPAA'],
    defaultModel: 'claude-sonnet-4',
    icon: '⚖️',
  },
  {
    id: 'healthcare',
    displayName: 'Healthcare',
    description: 'Clinical analysis, HIPAA compliance, and medical research',
    expertiseAreas: ['Clinical Documentation', 'Medical Research', 'HIPAA Compliance', 'PHI Protection'],
    complianceFrameworks: ['HIPAA', 'HITECH', 'FDA 21 CFR 11'],
    defaultModel: 'claude-sonnet-4',
    icon: '🏥',
  },
  {
    id: 'accounting',
    displayName: 'Accounting & Finance',
    description: 'Financial analysis, audit review, and SOX compliance',
    expertiseAreas: ['Financial Statement Analysis', 'Audit', 'SOX Compliance', 'Internal Controls'],
    complianceFrameworks: ['SOX', 'GAAP', 'PCAOB'],
    defaultModel: 'claude-sonnet-4',
    icon: '📊',
  },
  {
    id: 'research',
    displayName: 'Research',
    description: 'Methodology analysis, literature review, and statistical review',
    expertiseAreas: ['Research Methodology', 'Statistical Analysis', 'Literature Review', 'Ethics'],
    complianceFrameworks: ['IRB', 'CONSORT', 'PRISMA'],
    defaultModel: 'claude-sonnet-4',
    icon: '🔬',
  },
];

interface VerticalSelectorProps {
  selectedVertical?: string;
  onSelect: (vertical: Vertical) => void;
  showDetails?: boolean;
}

export function VerticalSelector({
  selectedVertical,
  onSelect,
  showDetails = true,
}: VerticalSelectorProps) {
  const [hoveredVertical, setHoveredVertical] = useState<string | null>(null);

  const handleSelect = useCallback(
    (vertical: Vertical) => {
      onSelect(vertical);
    },
    [onSelect]
  );

  const selectedData = VERTICALS.find((v) => v.id === selectedVertical);
  const displayVertical = hoveredVertical
    ? VERTICALS.find((v) => v.id === hoveredVertical)
    : selectedData;

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-bg">
        <h3 className="text-sm font-mono font-bold text-acid-green">
          SELECT VERTICAL SPECIALIST
        </h3>
        <p className="text-xs text-text-muted mt-1">
          Choose a domain specialist for your task
        </p>
      </div>

      {/* Vertical Grid */}
      <div className="p-4 grid grid-cols-5 gap-2">
        {VERTICALS.map((vertical) => (
          <button
            key={vertical.id}
            onClick={() => handleSelect(vertical)}
            onMouseEnter={() => setHoveredVertical(vertical.id)}
            onMouseLeave={() => setHoveredVertical(null)}
            className={`
              p-3 rounded-lg border-2 transition-all duration-200
              flex flex-col items-center gap-2
              ${
                selectedVertical === vertical.id
                  ? 'border-acid-green bg-acid-green/10'
                  : 'border-border hover:border-text-muted bg-bg'
              }
            `}
          >
            <span className="text-2xl">{vertical.icon}</span>
            <span className="text-xs font-mono text-center leading-tight">
              {vertical.displayName}
            </span>
          </button>
        ))}
      </div>

      {/* Details Panel */}
      {showDetails && displayVertical && (
        <div className="px-4 pb-4 border-t border-border pt-4">
          <div className="flex items-start gap-4">
            <span className="text-4xl">{displayVertical.icon}</span>
            <div className="flex-1">
              <h4 className="font-mono font-bold text-text">
                {displayVertical.displayName}
              </h4>
              <p className="text-sm text-text-muted mt-1">
                {displayVertical.description}
              </p>

              {/* Expertise Areas */}
              <div className="mt-3">
                <span className="text-xs font-mono text-text-muted">EXPERTISE:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {displayVertical.expertiseAreas.map((area) => (
                    <span
                      key={area}
                      className="px-2 py-0.5 text-xs bg-bg border border-border rounded font-mono"
                    >
                      {area}
                    </span>
                  ))}
                </div>
              </div>

              {/* Compliance Frameworks */}
              <div className="mt-3">
                <span className="text-xs font-mono text-text-muted">COMPLIANCE:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {displayVertical.complianceFrameworks.map((fw) => (
                    <span
                      key={fw}
                      className="px-2 py-0.5 text-xs bg-acid-green/20 text-acid-green rounded font-mono"
                    >
                      {fw}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default VerticalSelector;
