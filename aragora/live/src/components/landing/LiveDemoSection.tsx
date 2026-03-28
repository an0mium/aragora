'use client';

import { useEffect, useState } from 'react';
import { useTheme } from '@/context/ThemeContext';

const LIVE_DEMO_TURNS = [
  {
    agent: 'Claude',
    role: 'Strategic Analyst',
    accent: '#22c55e',
    phase: 'Opening arguments',
    time: 'T+04s',
    content:
      'Split billing and notifications first. Your delivery risk comes from deployment coupling, not from the monolith by itself.',
  },
  {
    agent: 'GPT-4',
    role: 'Skeptical Operator',
    accent: '#38bdf8',
    phase: 'Counterargument',
    time: 'T+09s',
    content:
      'You do not have the platform team for a full migration yet. Fix release discipline and test isolation before adding new failure modes.',
  },
  {
    agent: 'Gemini',
    role: 'Systems Synthesizer',
    accent: '#f97316',
    phase: 'Cross-examination',
    time: 'T+14s',
    content:
      'Both claims hold. Extract only the order ingest path, keep auth centralized, and require observability SLAs before the second service.',
  },
  {
    agent: 'Claude',
    role: 'Strategic Analyst',
    accent: '#22c55e',
    phase: 'Consensus forming',
    time: 'T+20s',
    content:
      'Approved with conditions: phase the migration after launch hardening and use queue latency plus incident rate as the expansion gate.',
  },
] as const;

const LIVE_DEMO_TOPIC =
  'Should we split our monolith into services before the next product launch?';
const LIVE_DEMO_INTERVAL_MS = 2600;

export function LiveDemoSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const [activeTurnIndex, setActiveTurnIndex] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setActiveTurnIndex((currentIndex) => (currentIndex + 1) % LIVE_DEMO_TURNS.length);
    }, LIVE_DEMO_INTERVAL_MS);

    return () => window.clearInterval(interval);
  }, []);

  const activeTurn = LIVE_DEMO_TURNS[activeTurnIndex];
  const visibleTurns = LIVE_DEMO_TURNS.slice(0, activeTurnIndex + 1);

  return (
    <section
      className="px-4"
      style={{
        paddingTop: '96px',
        paddingBottom: '96px',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
      aria-labelledby="live-debate-heading"
    >
      <div className="max-w-4xl mx-auto">
        <p
          className="text-center uppercase tracking-widest"
          style={{
            fontSize: isDark ? '14px' : '16px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
            marginBottom: '16px',
          }}
        >
          {isDark ? '> WATCH A LIVE DEBATE' : 'WATCH A LIVE DEBATE'}
        </p>
        <div className="text-center" style={{ marginBottom: '48px' }}>
          <h2
            id="live-debate-heading"
            style={{
              fontSize: isDark ? '32px' : '36px',
              color: 'var(--text)',
              fontFamily: 'var(--font-display, var(--font-landing))',
              marginBottom: '12px',
            }}
          >
            Watch agents argue in real time
          </h2>
          <p
            style={{
              fontSize: isDark ? '15px' : '17px',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-landing)',
              maxWidth: '720px',
              margin: '0 auto',
              lineHeight: '1.7',
            }}
          >
            The landing page now previews a live exchange so visitors can see claims,
            pushback, and synthesis before the verdict lands.
          </p>
        </div>

        <div
          style={{
            backgroundColor: 'var(--surface)',
            borderRadius: 'var(--radius-card)',
            border: '1px solid var(--border)',
            borderTopColor: 'var(--accent)',
            borderTopWidth: '3px',
            boxShadow: 'var(--shadow-card)',
            overflow: 'hidden',
          }}
        >
          <div
            className="flex flex-wrap items-center gap-3"
            style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}
          >
            <span
              className="font-bold px-2 py-0.5 uppercase tracking-wider"
              style={{
                fontSize: '10px',
                backgroundColor: 'var(--accent)',
                color: 'var(--bg)',
                borderRadius: 'var(--radius-button)',
              }}
            >
              Live now
            </span>
            <span
              className="font-medium"
              style={{ fontSize: '12px', color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
            >
              {LIVE_DEMO_TOPIC}
            </span>
            <span
              className="ml-auto"
              style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              {activeTurn.phase} | Round 2 of 3 | Streaming
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[1.8fr_0.8fr]">
            <div
              role="log"
              aria-live="polite"
              style={{
                padding: '20px',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <div className="space-y-3">
                {visibleTurns.map((turn, index) => {
                  const isActive = index === activeTurnIndex;
                  return (
                    <article
                      key={`${turn.agent}-${turn.time}`}
                      style={{
                        padding: '16px',
                        borderRadius: 'var(--radius-card)',
                        border: `1px solid ${isActive ? turn.accent : 'var(--border)'}`,
                        backgroundColor: isActive
                          ? isDark
                            ? 'rgba(255,255,255,0.02)'
                            : 'rgba(255,255,255,0.8)'
                          : 'transparent',
                        boxShadow: isActive ? `0 0 0 1px ${turn.accent}20` : 'none',
                        opacity: isActive ? 1 : 0.72,
                        transition: 'opacity 200ms ease, transform 200ms ease',
                        transform: isActive ? 'translateY(0)' : 'translateY(2px)',
                      }}
                    >
                      <div className="flex items-center gap-3" style={{ marginBottom: '10px' }}>
                        <div
                          className={isActive ? 'animate-pulse' : undefined}
                          style={{
                            width: '10px',
                            height: '10px',
                            borderRadius: '9999px',
                            backgroundColor: turn.accent,
                            flexShrink: 0,
                          }}
                        />
                        <div>
                          <div
                            style={{
                              fontSize: '12px',
                              fontWeight: 700,
                              color: turn.accent,
                              fontFamily: 'var(--font-landing)',
                              letterSpacing: '0.08em',
                              textTransform: 'uppercase',
                            }}
                          >
                            {turn.agent} | {turn.role}
                          </div>
                          <div
                            style={{
                              fontSize: '11px',
                              color: 'var(--text-muted)',
                              fontFamily: 'var(--font-landing)',
                            }}
                          >
                            {turn.phase} | {turn.time}
                          </div>
                        </div>
                      </div>
                      <p
                        style={{
                          fontSize: '14px',
                          color: 'var(--text)',
                          fontFamily: 'var(--font-landing)',
                          lineHeight: '1.7',
                        }}
                      >
                        {turn.content}
                      </p>
                    </article>
                  );
                })}
              </div>
            </div>

            <aside
              style={{
                padding: '20px',
                borderLeft: '1px solid var(--border)',
              }}
            >
              <div style={{ marginBottom: '20px' }}>
                <p
                  style={{
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-landing)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    marginBottom: '8px',
                  }}
                >
                  Live status
                </p>
                <p
                  style={{
                    fontSize: '20px',
                    color: 'var(--text)',
                    fontFamily: 'var(--font-display, var(--font-landing))',
                    marginBottom: '8px',
                  }}
                >
                  {activeTurn.phase}
                </p>
                <p
                  style={{
                    fontSize: '13px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-landing)',
                    lineHeight: '1.7',
                  }}
                >
                  Agents are challenging assumptions, forcing tradeoffs into the open,
                  and converging on a decision with explicit conditions.
                </p>
              </div>

              <div className="space-y-3" style={{ marginBottom: '24px' }}>
                {LIVE_DEMO_TURNS.map((turn, index) => {
                  const isCurrent = index === activeTurnIndex;
                  return (
                    <div key={`${turn.agent}-${turn.phase}`} className="flex items-center gap-3">
                      <div
                        style={{
                          width: '8px',
                          height: '8px',
                          borderRadius: '9999px',
                          backgroundColor: isCurrent ? turn.accent : 'var(--border)',
                          boxShadow: isCurrent ? `0 0 12px ${turn.accent}` : 'none',
                          flexShrink: 0,
                        }}
                      />
                      <div>
                        <div
                          style={{
                            fontSize: '12px',
                            color: isCurrent ? 'var(--text)' : 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {turn.phase}
                        </div>
                        <div
                          style={{
                            fontSize: '10px',
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                            textTransform: 'uppercase',
                            letterSpacing: '0.06em',
                          }}
                        >
                          {turn.agent}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <button
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className="text-sm font-semibold transition-all hover:scale-[1.02] cursor-pointer"
                style={{
                  width: '100%',
                  border: '1px solid var(--accent)',
                  borderRadius: 'var(--radius-button)',
                  color: 'var(--accent)',
                  backgroundColor: 'transparent',
                  fontFamily: 'var(--font-landing)',
                  padding: '16px 24px',
                }}
              >
                Run your own debate
              </button>
            </aside>
          </div>
        </div>
      </div>
    </section>
  );
}
