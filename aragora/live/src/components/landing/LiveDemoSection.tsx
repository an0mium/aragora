'use client';

import { useTheme } from '@/context/ThemeContext';

const DEMO_AGENTS = [
  {
    name: 'Strategic Analyst',
    role: 'analyst',
    accent: '#059669',
    content: 'Microservices make sense at your scale (50+ engineers), but only if you invest in service mesh and observability first. The organizational cost of splitting prematurely exceeds the technical debt of a well-structured monolith.',
  },
  {
    name: "Devil's Advocate",
    role: 'critic',
    accent: '#dc2626',
    content: "The industry push toward microservices is survivorship bias. Most teams that succeed with them had strong platform engineering before the migration. Your team's current deployment cadence suggests the monolith isn't actually the bottleneck.",
  },
  {
    name: 'Implementation Expert',
    role: 'engineer',
    accent: '#2563eb',
    content: 'Start with the strangler fig pattern: extract the 2-3 domains with the highest change frequency first. Keep shared authentication and data access in the monolith until you have proven service boundaries.',
  },
];

export function LiveDemoSection() {
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
      <div className="max-w-4xl mx-auto">
        <p
          className="text-center mb-4 uppercase tracking-widest"
          style={{ fontSize: isDark ? '11px' : '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
        >
          {isDark ? '> SEE IT IN ACTION' : 'SEE IT IN ACTION'}
        </p>
        <p
          className="text-center mb-12"
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
        >
          Every debate produces a defensible, auditable result.
        </p>

        {/* Demo debate card */}
        <div
          style={{
            backgroundColor: 'var(--surface)',
            borderRadius: 'var(--radius-card)',
            border: '1px solid var(--border)',
            boxShadow: 'var(--shadow-card)',
            overflow: 'hidden',
          }}
        >
          {/* Topic header bar */}
          <div
            className="p-4 flex flex-wrap items-center gap-3"
            style={{ borderBottom: '1px solid var(--border)' }}
          >
            <span
              className="text-xs font-bold px-2 py-0.5 uppercase tracking-wider"
              style={{
                backgroundColor: 'var(--accent)',
                color: 'var(--bg)',
                borderRadius: 'var(--radius-button)',
              }}
            >
              Approved with conditions
            </span>
            <span
              className="text-sm font-medium"
              style={{ color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              Should we adopt microservices or keep our monolith?
            </span>
            <span
              className="text-xs ml-auto"
              style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              78% confidence · 6 agents · 3 rounds
            </span>
          </div>

          {/* Agent perspectives grid */}
          <div className="grid grid-cols-1 md:grid-cols-3">
            {DEMO_AGENTS.map((agent, i) => (
              <div
                key={agent.name}
                className="p-5"
                style={{
                  borderRight: i < DEMO_AGENTS.length - 1 ? '1px solid var(--border)' : 'none',
                  borderBottom: '1px solid var(--border)',
                }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: agent.accent }}
                  />
                  <span
                    className="text-xs font-bold uppercase tracking-wider"
                    style={{ color: agent.accent, fontFamily: 'var(--font-landing)' }}
                  >
                    {agent.name}
                  </span>
                </div>
                <p
                  className="text-sm leading-relaxed"
                  style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                >
                  {agent.content}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA button */}
        <div className="text-center mt-8">
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="text-sm font-semibold py-3 px-8 transition-all hover:scale-[1.02] cursor-pointer"
            style={{
              border: '1px solid var(--accent)',
              borderRadius: 'var(--radius-button)',
              color: 'var(--accent)',
              backgroundColor: 'transparent',
              fontFamily: 'var(--font-landing)',
            }}
          >
            Run your own debate
          </button>
        </div>
      </div>
    </section>
  );
}
