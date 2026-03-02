'use client';

import { useTheme } from '@/context/ThemeContext';

interface Step {
  number: string;
  title: string;
  description: string;
}

const STEPS: Step[] = [
  {
    number: '01',
    title: 'You ask a question',
    description: 'Any decision, strategy, or architecture question you need vetted.',
  },
  {
    number: '02',
    title: 'AI agents debate it',
    description: 'Claude, GPT, Gemini, Mistral, and others argue every angle. Different models catch different blind spots.',
  },
  {
    number: '03',
    title: 'You get a decision receipt',
    description: 'An audit-ready verdict with evidence chains, confidence scores, and dissenting views preserved.',
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
        <div className="space-y-12">
          {STEPS.map((step) => (
            <div key={step.number} className="flex gap-6 items-start">
              {/* Step number */}
              <span
                className="flex-shrink-0 mt-0.5"
                style={{
                  color: 'var(--accent)',
                  fontSize: '14px',
                  fontFamily: isDark ? "'JetBrains Mono', monospace" : 'var(--font-landing)',
                  fontWeight: 600,
                  textShadow: isDark ? '0 0 10px var(--accent)' : 'none',
                }}
              >
                {isDark ? `[${step.number}]` : step.number}
              </span>

              <div>
                {/* Title */}
                <h3
                  className="mb-1"
                  style={{
                    fontSize: '16px',
                    fontWeight: 500,
                    color: 'var(--text)',
                    fontFamily: 'var(--font-landing)',
                  }}
                >
                  {isDark ? `> ${step.title}` : step.title}
                </h3>

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
          ))}
        </div>
      </div>
    </section>
  );
}
