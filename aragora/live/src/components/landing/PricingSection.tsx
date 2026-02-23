'use client';

import Link from 'next/link';
import { SectionHeader } from './SectionHeader';

interface Tier {
  name: string;
  price: string;
  period: string;
  accent: string;
  highlight?: boolean;
  features: string[];
  cta: { label: string; href: string };
}

const TIERS: Tier[] = [
  {
    name: 'FREE',
    price: '$0',
    period: 'forever',
    accent: 'acid-cyan',
    features: [
      'Playground debates (mock agents)',
      'Instant results, no signup',
      'See proposals, critiques, votes',
      'Community templates',
    ],
    cta: { label: 'TRY NOW', href: '/playground' },
  },
  {
    name: 'PRO',
    price: '$49',
    period: '/month',
    accent: 'acid-green',
    highlight: true,
    features: [
      '15+ real AI models (Claude, GPT, Gemini, Mistral, DeepSeek)',
      'Decision receipts with SHA-256 audit trail',
      'Knowledge base integration',
      'ELO rankings and calibration tracking',
      'Slack, Teams, Email delivery',
      '1,000 debates/month',
    ],
    cta: { label: 'START FREE TRIAL', href: '/signup?plan=pro' },
  },
  {
    name: 'ENTERPRISE',
    price: 'Custom',
    period: '',
    accent: 'acid-cyan',
    features: [
      'Everything in Pro',
      'SSO (OIDC/SAML) + SCIM provisioning',
      'RBAC with 50+ granular permissions',
      'SOC 2 / HIPAA / EU AI Act compliance',
      'On-prem or dedicated cloud',
      'SLA + dedicated support',
    ],
    cta: { label: 'CONTACT SALES', href: '/contact?plan=enterprise' },
  },
];

export function PricingSection() {
  return (
    <section id="pricing" className="py-12 border-t border-acid-green/20">
      <div className="container mx-auto px-4">
        <SectionHeader title="PRICING" />
        <p className="text-text-muted font-mono text-xs text-center mb-8 max-w-xl mx-auto">
          Start free. Upgrade when you need real AI models and audit-ready receipts.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-5xl mx-auto">
          {TIERS.map((tier) => (
            <div
              key={tier.name}
              className={`border ${
                tier.highlight
                  ? 'border-acid-green bg-acid-green/5'
                  : `border-${tier.accent}/30 bg-surface/30`
              } p-6 flex flex-col`}
            >
              <h3 className={`text-${tier.accent} font-mono text-sm font-bold mb-1`}>
                {tier.name}
              </h3>
              <div className="mb-4">
                <span className="text-text font-mono text-2xl font-bold">{tier.price}</span>
                {tier.period && (
                  <span className="text-text-muted font-mono text-xs">{tier.period}</span>
                )}
              </div>

              <ul className="space-y-2 mb-6 flex-1">
                {tier.features.map((feature) => (
                  <li key={feature} className="text-text-muted text-xs font-mono flex items-start gap-2">
                    <span className={`text-${tier.accent} mt-0.5`}>+</span>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>

              <Link
                href={tier.cta.href}
                className={`block text-center font-mono text-sm py-2 transition-colors ${
                  tier.highlight
                    ? 'bg-acid-green text-bg font-bold hover:bg-acid-green/80'
                    : `border border-${tier.accent}/40 text-${tier.accent} hover:bg-${tier.accent}/10`
                }`}
              >
                [{tier.cta.label}]
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
