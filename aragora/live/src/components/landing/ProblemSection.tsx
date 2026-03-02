'use client';

import type React from 'react';
import { useTheme } from '@/context/ThemeContext';

interface ProblemCard {
  title: string;
  description: string;
  icon: React.ReactNode;
}

const PROBLEMS: ProblemCard[] = [
  {
    title: 'Hallucination',
    description: 'Cross-model verification catches fabrications before they reach you.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
        <path d="M12 9v4" />
        <path d="M12 17h.01" />
      </svg>
    ),
  },
  {
    title: 'Sycophancy',
    description: 'Agents are structurally incentivized to disagree and find flaws.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 10v12" />
        <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.88Z" />
      </svg>
    ),
  },
  {
    title: 'Inconsistency',
    description: 'Debate convergence produces stable, defensible positions.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 3h5v5" />
        <path d="M8 3H3v5" />
        <path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3" />
        <path d="m15 9 6-6" />
      </svg>
    ),
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
          {isDark ? '> WHY THIS MATTERS' : 'WHY THIS MATTERS'}
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
          One AI is a liability. A panel is a process.
        </p>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PROBLEMS.map((problem) => (
            <div
              key={problem.title}
              className="p-6 transition-all hover:translate-y-[-2px]"
              style={{
                backgroundColor: 'var(--surface)',
                borderRadius: 'var(--radius-card)',
                border: '1px solid var(--border)',
                borderLeftColor: 'var(--accent)',
                borderLeftWidth: '3px',
                boxShadow: 'var(--shadow-card)',
              }}
            >
              <div className="flex items-center gap-2 mb-2">
                <span style={{ color: 'var(--accent)' }}>{problem.icon}</span>
                <h3
                  className="font-semibold"
                  style={{
                    fontSize: '14px',
                    color: 'var(--accent)',
                    fontFamily: 'var(--font-landing)',
                  }}
                >
                  {problem.title}
                </h3>
              </div>
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
