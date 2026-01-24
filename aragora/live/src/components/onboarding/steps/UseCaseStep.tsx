'use client';

import { useState } from 'react';
import { useOnboardingStore } from '@/store/onboardingStore';

interface UseCase {
  id: string;
  label: string;
  description: string;
  icon: string;
}

const USE_CASES: UseCase[] = [
  {
    id: 'team_decisions',
    label: 'Team Decisions',
    description: 'Facilitate team decisions with structured AI debate',
    icon: '&#x1F465;',
  },
  {
    id: 'architecture_review',
    label: 'Architecture Review',
    description: 'Review system designs for best practices',
    icon: '&#x1F3D7;',
  },
  {
    id: 'security_audit',
    label: 'Security Audit',
    description: 'Identify vulnerabilities and security gaps',
    icon: '&#x1F512;',
  },
  {
    id: 'vendor_selection',
    label: 'Vendor Selection',
    description: 'Compare tools and vendors objectively',
    icon: '&#x2696;',
  },
  {
    id: 'policy_review',
    label: 'Policy Review',
    description: 'Review policies for gaps and improvements',
    icon: '&#x1F4DC;',
  },
  {
    id: 'general',
    label: 'General Use',
    description: 'Multi-perspective analysis for any topic',
    icon: '&#x1F4A1;',
  },
];

interface UseCaseStepProps {
  onNext: (useCase: string) => void;
  onBack: () => void;
}

export function UseCaseStep({ onNext, onBack }: UseCaseStepProps) {
  const { useCase: savedUseCase, setUseCase } = useOnboardingStore();
  const [selected, setSelected] = useState<string | null>(savedUseCase);

  const handleSelect = (id: string) => {
    setSelected(id);
    setUseCase(id as any);
  };

  const handleNext = () => {
    if (selected) {
      onNext(selected);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-mono text-acid-green mb-2">
          What brings you to Aragora?
        </h2>
        <p className="font-mono text-text-muted text-sm">
          We&apos;ll personalize your experience based on your primary use case
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {USE_CASES.map((useCase) => (
          <button
            key={useCase.id}
            onClick={() => handleSelect(useCase.id)}
            className={`p-4 text-left border transition-colors ${
              selected === useCase.id
                ? 'border-acid-green bg-acid-green/10'
                : 'border-acid-green/20 hover:border-acid-green/50 bg-surface'
            }`}
          >
            <div className="flex items-start gap-3">
              <span
                className="text-xl"
                dangerouslySetInnerHTML={{ __html: useCase.icon }}
              />
              <div>
                <div className="font-mono text-sm text-acid-green">
                  {useCase.label}
                </div>
                <div className="font-mono text-xs text-text-muted mt-1">
                  {useCase.description}
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>

      <div className="flex gap-3 pt-4">
        <button
          onClick={onBack}
          className="px-4 py-2 font-mono text-sm border border-acid-green/30 text-text-muted hover:border-acid-green hover:text-acid-green transition-colors"
        >
          Back
        </button>
        <div className="flex-1" />
        <button
          onClick={handleNext}
          disabled={!selected}
          className={`px-6 py-2 font-mono text-sm transition-colors ${
            selected
              ? 'bg-acid-green text-bg hover:bg-acid-green/80'
              : 'bg-surface text-text-muted border border-acid-green/20 cursor-not-allowed'
          }`}
        >
          Continue
        </button>
      </div>
    </div>
  );
}

export default UseCaseStep;
