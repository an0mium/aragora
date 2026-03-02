'use client';

import { useTheme } from '@/context/ThemeContext';

interface ProblemCard {
  icon: string;
  title: string;
  description: string;
}

const PROBLEMS: ProblemCard[] = [
  {
    icon: '\u26A0\uFE0F',
    title: 'Hallucination',
    description: 'Cross-model verification catches fabrications before they reach you.',
  },
  {
    icon: '\uD83E\uDD1D',
    title: 'Sycophancy',
    description: 'Agents are structurally incentivized to disagree and find flaws.',
  },
  {
    icon: '\uD83D\uDD00',
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
        paddingTop: '120px',
        paddingBottom: '120px',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-3xl mx-auto">
        {/* Section label */}
        <p
          className="text-center uppercase tracking-widest"
          style={{
            fontSize: isDark ? '11px' : '12px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
            marginBottom: '20px',
          }}
        >
          {isDark ? '> THE PROBLEM' : 'THE PROBLEM'}
        </p>

        {/* Statement */}
        <h2
          className="text-center max-w-lg mx-auto leading-snug"
          style={{
            fontSize: isDark ? '24px' : '28px',
            fontWeight: 600,
            color: 'var(--text)',
            fontFamily: 'var(--font-landing)',
            marginBottom: '64px',
          }}
        >
          A single AI hallucinates, flatters you, and contradicts itself.
        </h2>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {PROBLEMS.map((problem) => (
            <div
              key={problem.title}
              className="transition-all hover:translate-y-[-2px]"
              style={{
                backgroundColor: 'var(--surface)',
                borderRadius: 'var(--radius-card)',
                border: '1px solid var(--border)',
                borderTopColor: 'var(--accent)',
                borderTopWidth: '3px',
                boxShadow: 'var(--shadow-card)',
                padding: '32px 24px',
              }}
            >
              <div className="flex items-center gap-3" style={{ marginBottom: '16px' }}>
                <span style={{ color: 'var(--accent)' }}>{problem.icon}</span>
                <h3
                  className="font-semibold"
                  style={{
                    fontSize: '15px',
                    color: 'var(--text)',
                    fontFamily: 'var(--font-landing)',
                  }}
                >
                  {problem.title}
                </h3>
              </div>
              <p
                className="leading-relaxed"
                style={{
                  fontSize: isDark ? '13px' : '15px',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-landing)',
                  lineHeight: '1.7',
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
