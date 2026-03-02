'use client';

import { useTheme } from '@/context/ThemeContext';

interface ProblemCard {
  title: string;
  description: string;
}

const PROBLEMS: ProblemCard[] = [
  {
    title: 'Hallucination',
    description: 'Cross-model verification catches fabrications before they reach you.',
  },
  {
    title: 'Sycophancy',
    description: 'Agents are structurally incentivized to disagree and find flaws.',
  },
  {
    title: 'Inconsistency',
    description: 'Debate convergence produces stable, defensible positions.',
  },
];

export function ProblemSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <section
      className="px-4"
      style={{
        paddingTop: 'var(--section-padding)',
        paddingBottom: 'var(--section-padding)',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-3xl mx-auto">
        {/* Section label */}
        <p
          className="text-center mb-4 uppercase tracking-widest"
          style={{
            fontSize: isDark ? '11px' : '12px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
          }}
        >
          {isDark ? '> THE PROBLEM' : 'THE PROBLEM'}
        </p>

        {/* Statement */}
        <p
          className="text-center mb-12 max-w-xl mx-auto leading-relaxed"
          style={{
            fontSize: isDark ? '16px' : '18px',
            color: 'var(--text)',
            fontFamily: 'var(--font-landing)',
          }}
        >
          A single AI hallucinates, agrees with you, and contradicts itself.
          Adversarial debate fixes all three.
        </p>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PROBLEMS.map((problem) => (
            <div
              key={problem.title}
              className="p-6 transition-shadow"
              style={{
                backgroundColor: 'var(--surface)',
                borderRadius: 'var(--radius-card)',
                border: '1px solid var(--border)',
                borderTopColor: 'var(--accent)',
                borderTopWidth: isDark ? '1px' : '3px',
                boxShadow: 'var(--shadow-card)',
              }}
            >
              <h3
                className="mb-2 font-semibold"
                style={{
                  fontSize: '14px',
                  color: 'var(--accent)',
                  fontFamily: 'var(--font-landing)',
                }}
              >
                {problem.title}
              </h3>
              <p
                className="leading-relaxed"
                style={{
                  fontSize: isDark ? '13px' : '14px',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-landing)',
                }}
              >
                {problem.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
