'use client';

import { useOnboardingStore } from '@/store/onboardingStore';

const USE_CASES = [
  {
    id: 'team_decisions' as const,
    icon: '[T]',
    name: 'Team Decisions',
    example: '"Should we adopt microservices?"',
  },
  {
    id: 'project_planning' as const,
    icon: '[P]',
    name: 'Project Planning',
    example: '"Prioritize Q2 roadmap items"',
  },
  {
    id: 'vendor_selection' as const,
    icon: '[V]',
    name: 'Vendor Selection',
    example: '"Compare AWS vs GCP vs Azure"',
  },
  {
    id: 'policy_review' as const,
    icon: '[R]',
    name: 'Policy Review',
    example: '"Review our remote work policy"',
  },
  {
    id: 'technical_decisions' as const,
    icon: '[D]',
    name: 'Technical Decisions',
    example: '"Evaluate React vs Vue vs Svelte"',
  },
  {
    id: 'general' as const,
    icon: '[G]',
    name: 'General',
    example: '"Any decision that needs rigor"',
  },
];

export function UseCaseSelector() {
  const useCase = useOnboardingStore((s) => s.useCase);
  const setUseCase = useOnboardingStore((s) => s.setUseCase);

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] p-5">
      <div className="text-xs font-mono text-[var(--acid-green)] mb-3">
        {'>'} WHAT WILL YOU USE ARAGORA FOR?
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {USE_CASES.map((uc) => (
          <button
            key={uc.id}
            onClick={() => setUseCase(uc.id)}
            className={`p-3 text-left border transition-colors ${
              useCase === uc.id
                ? 'bg-[var(--acid-green)]/10 border-[var(--acid-green)]/40'
                : 'bg-[var(--bg)] border-[var(--border)] hover:border-[var(--acid-green)]/30'
            }`}
          >
            <div className="text-sm font-mono text-[var(--acid-green)] mb-1">
              {uc.icon} {uc.name}
            </div>
            <div className="text-[10px] font-mono text-[var(--text-muted)]">{uc.example}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
