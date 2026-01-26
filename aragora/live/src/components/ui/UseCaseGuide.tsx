'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ProgressiveMode, useProgressiveMode } from '@/context/ProgressiveModeContext';

interface UseCase {
  id: string;
  title: string;
  description: string;
  icon: string;
  href: string;
  minMode: ProgressiveMode;
  tags: string[];
  features: string[];
}

const USE_CASES: UseCase[] = [
  {
    id: 'quick-debate',
    title: 'Quick Debate',
    description: 'Get AI perspectives on any topic in minutes',
    icon: '!',
    href: '/arena',
    minMode: 'simple',
    tags: ['beginner', 'fast'],
    features: ['AI discussion', 'Multiple viewpoints', 'Quick consensus'],
  },
  // SME Quickstart Templates
  {
    id: 'quick-decision',
    title: 'Quick Decision',
    description: 'Fast yes/no decisions with AI consensus',
    icon: 'Y',
    href: '/arena?template=quickstart/yes-no',
    minMode: 'simple',
    tags: ['sme', 'fast', 'decision'],
    features: ['Binary choice', '2-minute turnaround', 'Confidence score'],
  },
  {
    id: 'pros-cons',
    title: 'Pros & Cons',
    description: 'Structured analysis of advantages and disadvantages',
    icon: '+',
    href: '/arena?template=quickstart/pros-cons',
    minMode: 'simple',
    tags: ['sme', 'analysis'],
    features: ['Weighted factors', 'Clear recommendation', 'Trade-off matrix'],
  },
  {
    id: 'risk-assessment',
    title: 'Risk Assessment',
    description: 'Quick risk identification and scoring',
    icon: '~',
    href: '/arena?template=quickstart/risk-assessment',
    minMode: 'simple',
    tags: ['sme', 'risk'],
    features: ['Risk scoring', 'Mitigation strategies', 'Priority ranking'],
  },
  {
    id: 'brainstorm',
    title: 'Brainstorm',
    description: 'Multi-perspective idea generation',
    icon: '*',
    href: '/arena?template=quickstart/brainstorm',
    minMode: 'simple',
    tags: ['sme', 'creative'],
    features: ['Diverse perspectives', 'Idea ranking', 'Action items'],
  },
  // SME Business Templates
  {
    id: 'vendor-evaluation',
    title: 'Vendor Evaluation',
    description: 'Compare vendors across key criteria',
    icon: 'V',
    href: '/arena?template=sme/vendor-evaluation',
    minMode: 'standard',
    tags: ['sme', 'business', 'procurement'],
    features: ['Criteria scoring', 'Cost analysis', 'Recommendation report'],
  },
  {
    id: 'hiring-decision',
    title: 'Hiring Decision',
    description: 'Evaluate candidates with multi-agent analysis',
    icon: 'H',
    href: '/arena?template=sme/hiring-decision',
    minMode: 'standard',
    tags: ['sme', 'hiring', 'hr'],
    features: ['Skills assessment', 'Culture fit', 'Offer recommendation'],
  },
  {
    id: 'budget-allocation',
    title: 'Budget Allocation',
    description: 'Optimize budget distribution across categories',
    icon: '$',
    href: '/arena?template=sme/budget-allocation',
    minMode: 'standard',
    tags: ['sme', 'finance', 'planning'],
    features: ['Category analysis', 'ROI projection', 'Allocation strategy'],
  },
  {
    id: 'business-decision',
    title: 'Business Decision',
    description: 'Strategic decision analysis with impact assessment',
    icon: 'B',
    href: '/arena?template=sme/business-decision',
    minMode: 'standard',
    tags: ['sme', 'strategy'],
    features: ['Stakeholder analysis', 'Impact assessment', 'Go/No-go recommendation'],
  },
  {
    id: 'code-review',
    title: 'Code Review',
    description: 'Have AI agents review and critique code changes',
    icon: '<',
    href: '/reviews',
    minMode: 'standard',
    tags: ['developer', 'code'],
    features: ['PR analysis', 'Security checks', 'Best practices'],
  },
  {
    id: 'document-audit',
    title: 'Document Audit',
    description: 'Analyze documents with multiple AI perspectives',
    icon: '|',
    href: '/audit',
    minMode: 'standard',
    tags: ['legal', 'compliance'],
    features: ['Multi-doc analysis', 'Risk detection', 'Recommendations'],
  },
  {
    id: 'stress-test',
    title: 'Stress Test',
    description: 'Put ideas through a gauntlet of AI scrutiny',
    icon: '%',
    href: '/gauntlet',
    minMode: 'standard',
    tags: ['research', 'validation'],
    features: ['Adversarial testing', 'Edge cases', 'Robustness'],
  },
  {
    id: 'research',
    title: 'Deep Research',
    description: 'Multi-round debates for complex research questions',
    icon: '?',
    href: '/arena?mode=research',
    minMode: 'advanced',
    tags: ['research', 'academic'],
    features: ['Extended rounds', 'Citation tracking', 'Consensus proofs'],
  },
  {
    id: 'agent-compare',
    title: 'Agent Comparison',
    description: 'Compare different AI models on specific tasks',
    icon: '&',
    href: '/agents',
    minMode: 'advanced',
    tags: ['comparison', 'evaluation'],
    features: ['ELO rankings', 'Performance metrics', 'Calibration'],
  },
  {
    id: 'workflows',
    title: 'Custom Workflows',
    description: 'Build automated debate pipelines',
    icon: '>',
    href: '/workflows',
    minMode: 'advanced',
    tags: ['automation', 'enterprise'],
    features: ['Batch processing', 'Scheduling', 'Integrations'],
  },
  {
    id: 'genesis',
    title: 'Genesis Mode',
    description: 'Create and evolve new agent configurations',
    icon: '@',
    href: '/genesis',
    minMode: 'expert',
    tags: ['advanced', 'experimental'],
    features: ['Agent evolution', 'Prompt engineering', 'Fine-tuning'],
  },
];

interface UseCaseGuideProps {
  className?: string;
  showModeFilter?: boolean;
  maxItems?: number;
}

/**
 * Interactive use case guide component
 *
 * Helps users discover features based on their goals
 */
export function UseCaseGuide({
  className = '',
  showModeFilter = true,
  maxItems,
}: UseCaseGuideProps) {
  const { mode, isFeatureVisible, setMode } = useProgressiveMode();
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Get all unique tags
  const allTags = Array.from(
    new Set(USE_CASES.flatMap(uc => uc.tags))
  ).sort();

  // Filter use cases
  const filteredUseCases = USE_CASES.filter(uc => {
    // Filter by mode
    if (!isFeatureVisible(uc.minMode)) return false;

    // Filter by tag
    if (selectedTag && !uc.tags.includes(selectedTag)) return false;

    // Filter by search
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        uc.title.toLowerCase().includes(query) ||
        uc.description.toLowerCase().includes(query) ||
        uc.tags.some(t => t.toLowerCase().includes(query))
      );
    }

    return true;
  }).slice(0, maxItems);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="font-mono font-bold text-text">
          What would you like to do?
        </h2>
        {showModeFilter && (
          <ModeBadges currentMode={mode} onModeChange={setMode} />
        )}
      </div>

      {/* Search and filters */}
      <div className="flex flex-wrap gap-2">
        <input
          type="text"
          placeholder="Search use cases..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          className="
            flex-1 min-w-48 px-3 py-1.5 text-sm font-mono
            bg-surface border border-acid-green/30
            text-text placeholder-text-muted
            focus:border-acid-green focus:outline-none
          "
        />

        {/* Tag filters */}
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by category">
          <button
            onClick={() => setSelectedTag(null)}
            aria-pressed={!selectedTag}
            className={`
              px-2 py-1 text-xs font-mono
              ${!selectedTag
                ? 'bg-acid-green text-bg'
                : 'border border-acid-green/30 text-acid-green/70 hover:border-acid-green/50'
              }
              transition-colors
            `}
          >
            All
          </button>
          {allTags.map(tag => (
            <button
              key={tag}
              onClick={() => setSelectedTag(tag === selectedTag ? null : tag)}
              aria-pressed={selectedTag === tag}
              className={`
                px-2 py-1 text-xs font-mono
                ${selectedTag === tag
                  ? 'bg-acid-green text-bg'
                  : 'border border-acid-green/30 text-acid-green/70 hover:border-acid-green/50'
                }
                transition-colors
              `}
            >
              {tag}
            </button>
          ))}
        </div>
      </div>

      {/* Use case cards */}
      <div className="grid gap-3 md:grid-cols-2">
        {filteredUseCases.map(useCase => (
          <UseCaseCard key={useCase.id} useCase={useCase} />
        ))}
      </div>

      {filteredUseCases.length === 0 && (
        <div className="text-center py-8 text-text-muted">
          <div className="text-2xl mb-2">?</div>
          <div className="text-sm font-mono">
            No matching use cases found.
            {selectedTag && (
              <button
                onClick={() => setSelectedTag(null)}
                aria-label="Clear category filter"
                className="ml-2 text-acid-green underline"
              >
                Clear filter
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Individual use case card
 */
function UseCaseCard({ useCase }: { useCase: UseCase }) {
  return (
    <Link href={useCase.href}>
      <div className="
        p-4 bg-surface border border-acid-green/20
        hover:border-acid-green/50 hover:bg-acid-green/5
        transition-colors cursor-pointer
      ">
        <div className="flex items-start gap-3">
          <span className="text-2xl text-acid-green font-mono">
            {useCase.icon}
          </span>
          <div className="flex-1 min-w-0">
            <h3 className="font-mono font-bold text-text">
              {useCase.title}
            </h3>
            <p className="text-sm text-text-muted mt-1">
              {useCase.description}
            </p>

            {/* Features preview */}
            <div className="flex flex-wrap gap-1 mt-2">
              {useCase.features.slice(0, 3).map(feature => (
                <span
                  key={feature}
                  className="
                    px-1.5 py-0.5 text-xs font-mono
                    bg-acid-green/10 text-acid-green/80
                  "
                >
                  {feature}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Link>
  );
}

/**
 * Mode badges for filtering
 */
function ModeBadges({
  currentMode,
  onModeChange,
}: {
  currentMode: ProgressiveMode;
  onModeChange: (mode: ProgressiveMode) => void;
}) {
  const modes: { mode: ProgressiveMode; label: string }[] = [
    { mode: 'simple', label: 'S' },
    { mode: 'standard', label: 'ST' },
    { mode: 'advanced', label: 'A' },
    { mode: 'expert', label: 'E' },
  ];

  return (
    <div className="flex border border-acid-green/30 rounded overflow-hidden">
      {modes.map(({ mode, label }) => (
        <button
          key={mode}
          onClick={() => onModeChange(mode)}
          className={`
            px-2 py-1 text-xs font-mono
            ${currentMode === mode
              ? 'bg-acid-green text-bg'
              : 'text-acid-green/70 hover:bg-acid-green/10'
            }
            transition-colors
          `}
          title={mode}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

/**
 * Compact "I want to..." dropdown for quick navigation
 */
export function QuickActionDropdown({ className = '' }: { className?: string }) {
  const { isFeatureVisible } = useProgressiveMode();
  const [isOpen, setIsOpen] = useState(false);

  const visibleUseCases = USE_CASES.filter(uc => isFeatureVisible(uc.minMode));

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="
          flex items-center gap-2 px-3 py-1.5
          text-sm font-mono
          border border-acid-green/30
          text-acid-green hover:border-acid-green/50
          transition-colors
        "
      >
        <span>I want to...</span>
        <span>{isOpen ? '[-]' : '[+]'}</span>
      </button>

      {isOpen && (
        <div className="
          absolute top-full left-0 mt-1 w-64 z-50
          bg-surface border border-acid-green/30
          shadow-lg
        ">
          {visibleUseCases.map(uc => (
            <Link
              key={uc.id}
              href={uc.href}
              onClick={() => setIsOpen(false)}
              className="
                flex items-center gap-2 px-3 py-2
                text-sm font-mono text-text
                hover:bg-acid-green/10
                transition-colors
              "
            >
              <span className="w-5 text-center text-acid-green">{uc.icon}</span>
              <span>{uc.title}</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Tooltip helper for feature explanations
 */
export function FeatureTooltip({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div className="
          absolute bottom-full left-1/2 -translate-x-1/2 mb-2
          px-3 py-2 w-48 z-50
          bg-surface border border-acid-green/30
          text-xs font-mono
          shadow-lg
        ">
          <div className="text-acid-green font-bold mb-1">{title}</div>
          <div className="text-text-muted">{description}</div>
          {/* Arrow */}
          <div className="
            absolute top-full left-1/2 -translate-x-1/2
            border-4 border-transparent border-t-acid-green/30
          " />
        </div>
      )}
    </div>
  );
}
