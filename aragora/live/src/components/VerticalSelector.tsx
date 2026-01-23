'use client';

import { useState, useEffect, useCallback } from 'react';

/**
 * Industry vertical configuration for specialized debate agents.
 */
export interface Vertical {
  id: string;
  displayName: string;
  description: string;
  icon: string;
  expertiseAreas: string[];
  complianceFrameworks: string[];
  suggestedAgents: string[];
  costTier: 'standard' | 'professional' | 'enterprise';
  keywords: string[];
}

// Industry verticals with specialized configurations
const INDUSTRY_VERTICALS: Vertical[] = [
  {
    id: 'general',
    displayName: 'General',
    description: 'Multi-purpose debate for any topic',
    icon: '\u2699\uFE0F', // Gear
    expertiseAreas: ['Reasoning', 'Analysis', 'Problem-solving'],
    complianceFrameworks: [],
    suggestedAgents: ['claude', 'gpt', 'deepseek'],
    costTier: 'standard',
    keywords: [],
  },
  {
    id: 'software',
    displayName: 'Software Engineering',
    description: 'Code review, architecture, security analysis',
    icon: '\uD83D\uDCBB', // Laptop
    expertiseAreas: ['Architecture', 'Code Review', 'Security', 'Performance', 'Testing'],
    complianceFrameworks: ['OWASP', 'CWE', 'SANS'],
    suggestedAgents: ['claude', 'deepseek', 'codestral'],
    costTier: 'professional',
    keywords: ['code', 'api', 'software', 'bug', 'function', 'programming', 'typescript', 'python'],
  },
  {
    id: 'legal',
    displayName: 'Legal & Compliance',
    description: 'Contract review, regulatory analysis, compliance',
    icon: '\u2696\uFE0F', // Balance scale
    expertiseAreas: ['Contract Analysis', 'Regulatory', 'Risk Assessment', 'IP'],
    complianceFrameworks: ['GDPR', 'SOX', 'HIPAA', 'PCI-DSS'],
    suggestedAgents: ['claude', 'gpt-4o', 'gemini'],
    costTier: 'enterprise',
    keywords: ['legal', 'contract', 'compliance', 'regulation', 'law', 'liability', 'terms'],
  },
  {
    id: 'healthcare',
    displayName: 'Healthcare',
    description: 'Clinical analysis, medical research, health policy',
    icon: '\uD83C\uDFE5', // Hospital
    expertiseAreas: ['Clinical', 'Research', 'Policy', 'Bioethics'],
    complianceFrameworks: ['HIPAA', 'FDA', 'HL7 FHIR'],
    suggestedAgents: ['claude', 'gpt-4o', 'gemini'],
    costTier: 'enterprise',
    keywords: ['health', 'medical', 'clinical', 'patient', 'treatment', 'diagnosis', 'healthcare'],
  },
  {
    id: 'accounting',
    displayName: 'Accounting & Finance',
    description: 'Financial analysis, audit, tax planning',
    icon: '\uD83D\uDCB0', // Money bag
    expertiseAreas: ['Audit', 'Tax', 'Financial Analysis', 'Reporting'],
    complianceFrameworks: ['SOX', 'GAAP', 'IFRS', 'AML/KYC'],
    suggestedAgents: ['claude', 'gpt-4o', 'gemini'],
    costTier: 'professional',
    keywords: ['finance', 'accounting', 'tax', 'audit', 'budget', 'revenue', 'cost'],
  },
  {
    id: 'academic',
    displayName: 'Academic Research',
    description: 'Scientific analysis, literature review, methodology',
    icon: '\uD83C\uDF93', // Graduation cap
    expertiseAreas: ['Research Methods', 'Literature Review', 'Data Analysis', 'Peer Review'],
    complianceFrameworks: ['IRB', 'NIH Guidelines'],
    suggestedAgents: ['claude', 'gpt-4o', 'gemini'],
    costTier: 'professional',
    keywords: ['research', 'study', 'analysis', 'data', 'hypothesis', 'methodology', 'academic'],
  },
];

// Cost tier styling
const COST_TIER_STYLES = {
  standard: {
    label: 'Standard',
    color: 'text-acid-green',
    bgColor: 'bg-acid-green/10',
    borderColor: 'border-acid-green/30',
  },
  professional: {
    label: 'Pro',
    color: 'text-acid-cyan',
    bgColor: 'bg-acid-cyan/10',
    borderColor: 'border-acid-cyan/30',
  },
  enterprise: {
    label: 'Enterprise',
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/30',
  },
};

interface VerticalSelectorProps {
  apiBase: string;
  selectedVertical: string;
  onVerticalChange: (verticalId: string) => void;
  onAgentsChange?: (agents: string) => void;
  questionText?: string;
  compact?: boolean;
}

export function VerticalSelector({
  apiBase,
  selectedVertical,
  onVerticalChange,
  onAgentsChange,
  questionText = '',
  compact = false,
}: VerticalSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [suggestedVertical, setSuggestedVertical] = useState<string | null>(null);
  const [loadingBackend, setLoadingBackend] = useState(false);

  // Get selected vertical config
  const currentVertical = INDUSTRY_VERTICALS.find(v => v.id === selectedVertical) || INDUSTRY_VERTICALS[0];

  // Auto-detect vertical from question text
  useEffect(() => {
    if (!questionText.trim()) {
      setSuggestedVertical(null);
      return;
    }

    const questionLower = questionText.toLowerCase();
    let bestMatch: string | null = null;
    let bestScore = 0;

    for (const vertical of INDUSTRY_VERTICALS) {
      if (vertical.id === 'general') continue;

      let score = 0;
      for (const keyword of vertical.keywords) {
        if (questionLower.includes(keyword)) {
          score += 1;
        }
      }

      if (score > bestScore) {
        bestScore = score;
        bestMatch = vertical.id;
      }
    }

    // Only suggest if we have at least 2 keyword matches
    setSuggestedVertical(bestScore >= 2 ? bestMatch : null);
  }, [questionText]);

  // Fetch backend verticals (if available)
  useEffect(() => {
    const fetchBackendVerticals = async () => {
      try {
        setLoadingBackend(true);
        const response = await fetch(`${apiBase}/api/v1/verticals`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (response.ok) {
          // Backend verticals could extend the list
          // For now, we just check if the endpoint exists
        }
      } catch {
        // Backend not available - use client-side verticals only
      } finally {
        setLoadingBackend(false);
      }
    };

    fetchBackendVerticals();
  }, [apiBase]);

  // Apply suggested agents when vertical changes
  const handleVerticalSelect = useCallback((verticalId: string) => {
    onVerticalChange(verticalId);
    setIsOpen(false);

    const vertical = INDUSTRY_VERTICALS.find(v => v.id === verticalId);
    if (vertical && onAgentsChange) {
      onAgentsChange(vertical.suggestedAgents.join(','));
    }
  }, [onVerticalChange, onAgentsChange]);

  // Compact mode - just a chip
  if (compact) {
    return (
      <div className="relative inline-block">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className={`px-2 py-1 text-xs font-mono border rounded flex items-center gap-1.5
                     ${COST_TIER_STYLES[currentVertical.costTier].borderColor}
                     hover:border-acid-green/60 transition-colors`}
        >
          <span>{currentVertical.icon}</span>
          <span className="text-text-muted">{currentVertical.displayName}</span>
          <span className={`text-[10px] ${COST_TIER_STYLES[currentVertical.costTier].color}`}>
            [{COST_TIER_STYLES[currentVertical.costTier].label}]
          </span>
        </button>

        {isOpen && (
          <VerticalDropdown
            verticals={INDUSTRY_VERTICALS}
            selectedId={selectedVertical}
            suggestedId={suggestedVertical}
            onSelect={handleVerticalSelect}
            onClose={() => setIsOpen(false)}
          />
        )}
      </div>
    );
  }

  // Full mode - expanded selector
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-xs font-mono text-text-muted">
          INDUSTRY VERTICAL
        </label>
        {suggestedVertical && suggestedVertical !== selectedVertical && (
          <button
            type="button"
            onClick={() => handleVerticalSelect(suggestedVertical)}
            className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
          >
            [Detected: {INDUSTRY_VERTICALS.find(v => v.id === suggestedVertical)?.displayName}]
          </button>
        )}
      </div>

      <div className="relative">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-4 py-3 bg-bg border border-acid-green/30 rounded
                     flex items-center justify-between gap-2
                     hover:border-acid-green/60 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className="text-xl">{currentVertical.icon}</span>
            <div className="text-left">
              <div className="font-mono text-sm text-text">{currentVertical.displayName}</div>
              <div className="text-[10px] text-text-muted">{currentVertical.description}</div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Cost tier badge */}
            <span className={`px-2 py-0.5 text-[10px] font-mono rounded
                            ${COST_TIER_STYLES[currentVertical.costTier].bgColor}
                            ${COST_TIER_STYLES[currentVertical.costTier].color}`}>
              {COST_TIER_STYLES[currentVertical.costTier].label}
            </span>

            {/* Dropdown arrow */}
            <span className="text-text-muted">{isOpen ? '\u25B2' : '\u25BC'}</span>
          </div>
        </button>

        {isOpen && (
          <VerticalDropdown
            verticals={INDUSTRY_VERTICALS}
            selectedId={selectedVertical}
            suggestedId={suggestedVertical}
            onSelect={handleVerticalSelect}
            onClose={() => setIsOpen(false)}
            showDetails
          />
        )}
      </div>

      {/* Expertise areas and compliance */}
      {currentVertical.id !== 'general' && (
        <div className="flex flex-wrap gap-2 mt-2">
          {/* Expertise tags */}
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-text-muted">Expertise:</span>
            {currentVertical.expertiseAreas.slice(0, 3).map((area) => (
              <span
                key={area}
                className="px-1.5 py-0.5 text-[10px] font-mono bg-surface border border-acid-green/20 rounded"
              >
                {area}
              </span>
            ))}
          </div>

          {/* Compliance frameworks */}
          {currentVertical.complianceFrameworks.length > 0 && (
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-text-muted">Compliance:</span>
              {currentVertical.complianceFrameworks.slice(0, 2).map((framework) => (
                <span
                  key={framework}
                  className="px-1.5 py-0.5 text-[10px] font-mono bg-warning/10 text-warning border border-warning/20 rounded"
                >
                  {framework}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface VerticalDropdownProps {
  verticals: Vertical[];
  selectedId: string;
  suggestedId: string | null;
  onSelect: (verticalId: string) => void;
  onClose: () => void;
  showDetails?: boolean;
}

function VerticalDropdown({
  verticals,
  selectedId,
  suggestedId,
  onSelect,
  onClose,
  showDetails = false,
}: VerticalDropdownProps) {
  // Close on escape or click outside
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.vertical-dropdown')) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [onClose]);

  return (
    <div
      className="vertical-dropdown absolute top-full left-0 right-0 mt-1 z-50
                 bg-surface border border-acid-green/30 rounded shadow-lg
                 max-h-[400px] overflow-y-auto"
    >
      {verticals.map((vertical) => {
        const isSelected = vertical.id === selectedId;
        const isSuggested = vertical.id === suggestedId;
        const tierStyle = COST_TIER_STYLES[vertical.costTier];

        return (
          <button
            key={vertical.id}
            type="button"
            onClick={() => onSelect(vertical.id)}
            className={`w-full px-4 py-3 flex items-start gap-3 text-left
                       border-b border-acid-green/10 last:border-b-0
                       hover:bg-bg transition-colors
                       ${isSelected ? 'bg-acid-green/10' : ''}`}
          >
            <span className="text-xl flex-shrink-0">{vertical.icon}</span>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-text">{vertical.displayName}</span>
                {isSuggested && (
                  <span className="px-1.5 py-0.5 text-[10px] font-mono bg-acid-cyan/20 text-acid-cyan rounded">
                    Suggested
                  </span>
                )}
                {isSelected && (
                  <span className="text-acid-green">\u2713</span>
                )}
              </div>

              <div className="text-[10px] text-text-muted mt-0.5">{vertical.description}</div>

              {showDetails && vertical.id !== 'general' && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {vertical.expertiseAreas.slice(0, 3).map((area) => (
                    <span
                      key={area}
                      className="px-1 py-0.5 text-[9px] font-mono bg-bg rounded text-text-muted"
                    >
                      {area}
                    </span>
                  ))}
                  {vertical.complianceFrameworks.length > 0 && (
                    <span
                      className="px-1 py-0.5 text-[9px] font-mono bg-warning/10 text-warning rounded"
                    >
                      +{vertical.complianceFrameworks.length} compliance
                    </span>
                  )}
                </div>
              )}
            </div>

            <span className={`px-2 py-0.5 text-[10px] font-mono rounded flex-shrink-0
                            ${tierStyle.bgColor} ${tierStyle.color}`}>
              {tierStyle.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// Export for use in other components
export { INDUSTRY_VERTICALS };
export type { VerticalSelectorProps };
