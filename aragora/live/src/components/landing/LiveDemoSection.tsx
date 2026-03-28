'use client';

import { useTheme } from '@/context/ThemeContext';

const LIVE_DEBATE_EVENTS = [
  {
    timestamp: '00:08',
    phase: 'Opening argument',
    name: 'Strategic Analyst',
    accent: '#059669',
    content: 'Microservices only pay off if the org can absorb platform overhead. With 50+ engineers, the debate is not architecture purity, it is whether coordination costs are already slowing releases.',
  },
  {
    timestamp: '00:16',
    phase: 'Cross-examination',
    name: "Devil's Advocate",
    accent: '#dc2626',
    content: "The monolith is not failing yet. Teams often mistake roadmap stress for architectural failure, then lock themselves into higher operational drag before proving the bottleneck is technical.",
  },
  {
    timestamp: '00:24',
    phase: 'Implementation plan',
    name: 'Implementation Expert',
    accent: '#2563eb',
    content: 'If we move, do it incrementally. Start with the highest-churn domains, keep auth and shared data centralised, and measure deploy frequency before expanding the split.',
  },
  {
    timestamp: '00:31',
    phase: 'Consensus update',
    name: 'Strategic Analyst',
    accent: '#059669',
    content: 'Consensus is converging on a staged migration: prove service boundaries under live traffic, keep rollback simple, and do not rewrite the platform before the first extraction succeeds.',
  },
];

export function LiveDemoSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <section
      aria-labelledby="landing-live-demo-heading"
      className="px-4"
      style={{
        paddingTop: '120px',
        paddingBottom: '120px',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-4xl mx-auto">
        <p
          className="text-center uppercase tracking-widest"
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)', marginBottom: '20px' }}
        >
          {isDark ? '> SEE IT IN ACTION' : 'SEE IT IN ACTION'}
        </p>
        <h2
          id="landing-live-demo-heading"
          className="text-center font-semibold"
          style={{ fontSize: isDark ? '34px' : '38px', color: 'var(--text)', fontFamily: 'var(--font-display)', marginBottom: '16px' }}
        >
          Watch a live debate unfold turn by turn.
        </h2>
        <p
          className="text-center mx-auto"
          style={{ fontSize: isDark ? '16px' : '18px', color: 'var(--text)', fontFamily: 'var(--font-landing)', marginBottom: '48px', maxWidth: '760px' }}
        >
          Visitors can watch agents argue back and forth in real time before the final verdict lands.
        </p>

        <div
          style={{
            backgroundColor: 'var(--surface)',
            borderRadius: 'var(--radius-card)',
            border: '1px solid var(--border)',
            borderTopColor: 'var(--accent)',
            borderTopWidth: '3px',
            boxShadow: 'var(--shadow-card)',
            overflow: 'hidden',
            margin: '0 24px',
          }}
        >
          <div
            className="flex flex-wrap items-center gap-3"
            style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}
          >
            <span
              className="inline-flex items-center gap-2 font-bold px-2 py-0.5 uppercase tracking-wider"
              style={{
                fontSize: '10px',
                backgroundColor: 'var(--accent)',
                color: 'var(--bg)',
                borderRadius: 'var(--radius-button)',
              }}
            >
              <span
                aria-hidden="true"
                className="w-2 h-2 rounded-full animate-pulse"
                style={{ backgroundColor: 'currentColor' }}
              />
              Live debate streaming
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              Should we adopt microservices or keep our monolith?
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              Round 2 of 3 · 3 agents live · verdict updating
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.7fr)_minmax(280px,0.9fr)]">
            <div
              aria-live="polite"
              aria-label="Live debate transcript"
              style={{ borderRight: '1px solid var(--border)' }}
            >
              {LIVE_DEBATE_EVENTS.map((agent, i) => (
                <div
                  key={`${agent.name}-${agent.timestamp}`}
                  data-testid="stream-message"
                  style={{
                    padding: '20px',
                    borderBottom: i < LIVE_DEBATE_EVENTS.length - 1 ? '1px solid var(--border)' : 'none',
                  }}
                >
                  <div
                    className="flex flex-wrap items-center gap-2"
                    style={{ marginBottom: '12px' }}
                  >
                    <span
                      className="font-bold uppercase tracking-wider"
                      style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                    >
                      {agent.timestamp}
                    </span>
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
                    <span
                      className="text-xs uppercase tracking-wider"
                      style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                    >
                      {agent.phase}
                    </span>
                  </div>
                  <p
                    className="leading-relaxed"
                    style={{ fontSize: '14px', color: 'var(--text)', fontFamily: 'var(--font-landing)', lineHeight: '1.7' }}
                  >
                    {agent.content}
                  </p>
                </div>
              ))}
            </div>

            <div style={{ padding: '24px' }}>
              <div
                style={{
                  padding: '20px',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-card)',
                  backgroundColor: 'color-mix(in srgb, var(--surface) 85%, var(--accent) 15%)',
                  marginBottom: '16px',
                }}
              >
                <p
                  className="font-bold uppercase tracking-wider"
                  style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)', marginBottom: '10px' }}
                >
                  Current verdict
                </p>
                <p
                  className="font-semibold"
                  style={{ fontSize: '22px', color: 'var(--text)', fontFamily: 'var(--font-display)', marginBottom: '10px' }}
                >
                  Approved with conditions
                </p>
                <p
                  style={{ fontSize: '14px', color: 'var(--text)', fontFamily: 'var(--font-landing)', lineHeight: '1.7' }}
                >
                  Extract only the highest-churn domains first and keep the rest of the platform in the monolith until observability proves the boundary.
                </p>
              </div>

              <div
                style={{
                  padding: '20px',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-card)',
                }}
              >
                <div className="flex items-center justify-between" style={{ marginBottom: '12px' }}>
                  <span
                    className="text-xs font-bold uppercase tracking-wider"
                    style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                  >
                    Consensus pressure
                  </span>
                  <span
                    className="text-xs font-bold uppercase tracking-wider"
                    style={{ color: 'var(--accent)', fontFamily: 'var(--font-landing)' }}
                  >
                    78% confidence
                  </span>
                </div>
                <p
                  style={{ fontSize: '13px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)', marginBottom: '12px', lineHeight: '1.7' }}
                >
                  The stream keeps updating as agents rebut each other, expose weak assumptions, and tighten the final recommendation.
                </p>
                <div
                  style={{
                    height: '10px',
                    borderRadius: '999px',
                    backgroundColor: 'var(--border)',
                    overflow: 'hidden',
                    marginBottom: '10px',
                  }}
                >
                  <div
                    aria-hidden="true"
                    style={{
                      width: '78%',
                      height: '100%',
                      background: 'linear-gradient(90deg, #2563eb 0%, var(--accent) 100%)',
                    }}
                  />
                </div>
                <p
                  style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                >
                  2 rebuttals remaining before the verdict locks.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="text-center mt-12">
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
            style={{
              border: '1px solid var(--accent)',
              borderRadius: 'var(--radius-button)',
              color: 'var(--accent)',
              backgroundColor: 'transparent',
              fontFamily: 'var(--font-landing)',
              padding: '18px 48px',
            }}
          >
            Run your own debate
          </button>
        </div>
      </div>
    </section>
  );
}
