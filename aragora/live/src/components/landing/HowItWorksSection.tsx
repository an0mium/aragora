'use client';

import type React from 'react';
import { useTheme } from '@/context/ThemeContext';

interface Step {
  number: string;
  title: string;
  description: string;
  icon: React.ReactNode;
}

const STEPS: Step[] = [
  {
    number: '01',
    title: 'You ask a question',
    description: 'Any decision, strategy, or architecture question you need vetted.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.3-4.3" />
      </svg>
    ),
  },
  {
    number: '02',
    title: 'AI agents debate it',
    description: 'Claude, GPT, Gemini, Mistral, and others argue every angle. Different models catch different blind spots.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    number: '03',
    title: 'You get a decision receipt',
    description: 'An audit-ready verdict with evidence chains, confidence scores, and dissenting views preserved.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
        <rect width="8" height="4" x="8" y="2" rx="1" ry="1" />
        <path d="m9 14 2 2 4-4" />
      </svg>
    ),
  },
];

export function HowItWorksSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <section
      id="how-it-works"
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
          className="text-center mb-12 uppercase tracking-widest"
          style={{
            fontSize: isDark ? '11px' : '12px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
          }}
        >
          {isDark ? '> HOW IT WORKS' : 'HOW IT WORKS'}
        </p>

        {/* Steps */}
        <div className="space-y-0">
          {STEPS.map((step, idx) => (
            <div key={step.number} className="relative">
              {/* Vertical connector line between steps */}
              {idx < STEPS.length - 1 && (
                <div
                  className="absolute w-px"
                  style={{
                    left: '19px',
                    top: '40px',
                    bottom: '0',
                    backgroundColor: 'var(--border)',
                  }}
                />
              )}

              <div className="flex gap-6 items-start pb-12">
                {/* Circular badge */}
                <div
                  className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center"
                  style={{
                    backgroundColor: 'var(--accent-glow)',
                    color: 'var(--accent)',
                    fontSize: '14px',
                    fontWeight: 700,
                    fontFamily: isDark ? "'JetBrains Mono', monospace" : 'var(--font-landing)',
                  }}
                >
                  {step.number}
                </div>

                <div>
                  {/* Title with icon */}
                  <div className="flex items-center gap-2 mb-1">
                    <span style={{ color: 'var(--accent)' }}>{step.icon}</span>
                    <h3
                      style={{
                        fontSize: '16px',
                        fontWeight: 500,
                        color: 'var(--text)',
                        fontFamily: 'var(--font-landing)',
                      }}
                    >
                      {isDark ? `> ${step.title}` : step.title}
                    </h3>
                  </div>

                  {/* Description */}
                  <p
                    className="leading-relaxed"
                    style={{
                      fontSize: isDark ? '14px' : '16px',
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-landing)',
                    }}
                  >
                    {step.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
