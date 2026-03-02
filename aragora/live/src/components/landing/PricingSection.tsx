'use client';

import Link from 'next/link';
import { useTheme } from '@/context/ThemeContext';

interface Tier {
  name: string;
  price: string;
  period: string;
  highlight?: boolean;
  features: string[];
  cta: string;
  href: string;
}

const TIERS: Tier[] = [
  {
    name: 'Free',
    price: '$0',
    period: '/month',
    features: [
      '10 debates per month',
      '3 agents per debate',
      'Markdown receipts',
      'Demo mode',
    ],
    cta: 'Try it now',
    href: '/playground',
  },
  {
    name: 'Pro',
    price: '$49',
    period: '/seat/mo',
    highlight: true,
    features: [
      'Unlimited debates',
      '10 agents per debate',
      'All export formats',
      'CI/CD integration',
      'Slack, Teams, Email delivery',
      'Cross-debate memory',
    ],
    cta: 'Start free trial',
    href: '/signup?plan=pro',
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    features: [
      'SSO / RBAC / encryption',
      'SOC 2, HIPAA, EU AI Act',
      'Self-hosted option',
      'Dedicated support + SLA',
    ],
    cta: 'Contact sales',
    href: 'mailto:sales@aragora.ai?subject=Enterprise%20Inquiry',
  },
];

export function PricingSection() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <section
      id="pricing"
      className="px-4"
      style={{
        paddingTop: 'var(--section-padding)',
        paddingBottom: 'var(--section-padding)',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-4xl mx-auto">
        {/* Section label */}
        <p
          className="text-center mb-4 uppercase tracking-widest"
          style={{
            fontSize: isDark ? '11px' : '12px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
          }}
        >
          {isDark ? '> PRICING' : 'PRICING'}
        </p>

        <p
          className="text-center mb-12 max-w-md mx-auto"
          style={{
            fontSize: '14px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
          }}
        >
          Start free. Upgrade when you need more agents and audit-ready receipts.
        </p>

        {/* Tier cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {TIERS.map((tier) => {
            const isHighlighted = tier.highlight === true;
            return (
              <div
                key={tier.name}
                className="p-6 flex flex-col"
                style={{
                  backgroundColor: 'var(--surface)',
                  borderRadius: 'var(--radius-card)',
                  border: isHighlighted
                    ? '2px solid var(--accent)'
                    : '1px solid var(--border)',
                  boxShadow: isHighlighted ? 'var(--shadow-card-hover)' : 'var(--shadow-card)',
                }}
              >
                {/* Tier header */}
                <div className="mb-4">
                  <h3
                    className="text-sm font-semibold mb-1"
                    style={{
                      color: isHighlighted ? 'var(--accent)' : 'var(--text)',
                      fontFamily: 'var(--font-landing)',
                      textShadow: isDark && isHighlighted ? '0 0 10px var(--accent)' : 'none',
                    }}
                  >
                    {isDark ? `[${tier.name.toUpperCase()}]` : tier.name}
                  </h3>
                  <div className="flex items-baseline gap-1">
                    <span
                      className="text-3xl font-bold"
                      style={{ color: 'var(--text)', fontFamily: 'var(--font-landing)' }}
                    >
                      {tier.price}
                    </span>
                    {tier.period && (
                      <span
                        className="text-sm"
                        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                      >
                        {tier.period}
                      </span>
                    )}
                  </div>
                </div>

                {/* Features */}
                <ul className="space-y-2.5 mb-6 flex-1">
                  {tier.features.map((feature) => (
                    <li
                      key={feature}
                      className="flex items-start gap-2 text-sm"
                      style={{ fontFamily: 'var(--font-landing)' }}
                    >
                      <span style={{ color: 'var(--accent)', marginTop: '2px' }}>
                        {isDark ? '+' : '\u2713'}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>{feature}</span>
                    </li>
                  ))}
                </ul>

                {/* CTA */}
                <Link
                  href={tier.href}
                  className="block text-center text-sm font-semibold py-2.5 transition-opacity hover:opacity-80"
                  style={{
                    fontFamily: 'var(--font-landing)',
                    borderRadius: 'var(--radius-button)',
                    backgroundColor: isHighlighted ? 'var(--accent)' : 'transparent',
                    color: isHighlighted ? 'var(--bg)' : 'var(--accent)',
                    border: isHighlighted ? 'none' : '1px solid var(--accent)',
                    boxShadow: isDark && isHighlighted ? '0 0 20px var(--accent-glow)' : 'none',
                  }}
                >
                  {tier.cta}
                </Link>
              </div>
            );
          })}
        </div>

        {/* BYOK note */}
        <p
          className="text-center mt-8 text-xs"
          style={{
            color: 'var(--text-muted)',
            opacity: 0.6,
            fontFamily: 'var(--font-landing)',
          }}
        >
          Bring your own API keys for even lower costs.
        </p>
      </div>
    </section>
  );
}
