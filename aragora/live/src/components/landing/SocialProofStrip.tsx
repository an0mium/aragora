'use client';

import { useTheme } from '@/context/ThemeContext';

interface Stat {
  value: string;
  label: string;
}

const STATS: Stat[] = [
  { value: '43', label: 'analyst roles' },
  { value: '6+', label: 'AI models' },
  { value: '<30s', label: 'to verdict' },
  { value: '100%', label: 'auditable' },
];

export function SocialProofStrip() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <section
      className="px-4 py-10 sm:py-12"
      style={{
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-3xl mx-auto">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 sm:gap-8 text-center">
          {STATS.map((stat) => (
            <div key={stat.label}>
              <p
                className="font-bold"
                style={{
                  fontSize: isDark ? '28px' : '32px',
                  color: 'var(--accent)',
                  fontFamily: isDark ? "'JetBrains Mono', monospace" : 'var(--font-landing)',
                  lineHeight: 1.2,
                  textShadow: isDark ? '0 0 10px var(--accent-glow)' : 'none',
                }}
              >
                {stat.value}
              </p>
              <p
                style={{
                  fontSize: '13px',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-landing)',
                  marginTop: '4px',
                }}
              >
                {stat.label}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
