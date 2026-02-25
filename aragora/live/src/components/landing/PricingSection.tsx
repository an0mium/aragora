'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { SectionHeader } from './SectionHeader';
import { useAuth } from '@/context/AuthContext';
import { useToastContext } from '@/context/ToastContext';
import { apiPost } from '@/lib/api';

interface Tier {
  name: string;
  id: string;
  price: string;
  period: string;
  accent: string;
  highlight?: boolean;
  features: string[];
  cta: string;
  action: 'link' | 'checkout' | 'contact';
  href?: string;
}

const TIERS: Tier[] = [
  {
    name: 'FREE',
    id: 'free',
    price: '$0',
    period: '/month',
    accent: 'acid-cyan',
    features: [
      '10 debates/month',
      '3 agents per debate',
      'Basic analytics',
      'Community support',
    ],
    cta: 'TRY NOW',
    action: 'link',
    href: '/playground',
  },
  {
    name: 'STARTER',
    id: 'starter',
    price: '$29',
    period: '/month',
    accent: 'acid-cyan',
    features: [
      '100 debates/month',
      '10 agents per debate',
      'Full analytics + API access',
      'Email support',
    ],
    cta: 'START FREE TRIAL',
    action: 'checkout',
  },
  {
    name: 'PRO',
    id: 'professional',
    price: '$99',
    period: '/month',
    accent: 'acid-green',
    highlight: true,
    features: [
      '1,000 debates/month',
      'All 42+ agents',
      'Advanced analytics + priority support',
      'Slack, Teams, Email delivery',
      'Decision receipts with SHA-256 audit trail',
      'Knowledge base integration',
    ],
    cta: 'START FREE TRIAL',
    action: 'checkout',
  },
  {
    name: 'ENTERPRISE',
    id: 'enterprise',
    price: 'Custom',
    period: '',
    accent: 'acid-cyan',
    features: [
      'Everything in Pro + unlimited debates',
      'SSO (OIDC/SAML) + SCIM provisioning',
      'SOC 2 / HIPAA / EU AI Act compliance',
      'SLA + dedicated support',
    ],
    cta: 'CONTACT SALES',
    action: 'contact',
    href: 'mailto:sales@aragora.ai?subject=Enterprise%20Inquiry',
  },
];

/* ── Feature comparison table data ── */

interface ComparisonFeature {
  label: string;
  free: string;
  starter: string;
  pro: string;
  enterprise: string;
}

const COMPARISON_CATEGORIES: { category: string; features: ComparisonFeature[] }[] = [
  {
    category: 'USAGE',
    features: [
      { label: 'Debates per month', free: '10', starter: '100', pro: '1,000', enterprise: 'Unlimited' },
      { label: 'AI agents', free: '3 basic', starter: '10', pro: 'All 42+', enterprise: 'All 42+ + custom' },
    ],
  },
  {
    category: 'CORE FEATURES',
    features: [
      { label: 'Decision receipts', free: 'Basic', starter: 'Basic', pro: 'Full with provenance', enterprise: 'Full with provenance' },
      { label: 'Workflow templates', free: '\u2014', starter: '\u2713', pro: '\u2713', enterprise: '\u2713' },
      { label: 'Knowledge Mound access', free: '\u2014', starter: '\u2014', pro: '\u2713', enterprise: '\u2713' },
      { label: 'Custom agent training', free: '\u2014', starter: '\u2014', pro: '\u2014', enterprise: '\u2713' },
    ],
  },
  {
    category: 'SUPPORT & DELIVERY',
    features: [
      { label: 'Community support', free: '\u2713', starter: '\u2713', pro: '\u2713', enterprise: '\u2713' },
      { label: 'Email support', free: '\u2014', starter: '\u2713', pro: '\u2713', enterprise: '\u2713' },
      { label: 'Slack + Teams delivery', free: '\u2014', starter: '\u2014', pro: '\u2713', enterprise: '\u2713' },
      { label: 'Priority support', free: '\u2014', starter: '\u2014', pro: '\u2713', enterprise: '\u2713' },
      { label: 'Dedicated support + SLA', free: '\u2014', starter: '\u2014', pro: '\u2014', enterprise: '\u2713' },
    ],
  },
  {
    category: 'ENTERPRISE',
    features: [
      { label: 'SAML / SCIM SSO', free: '\u2014', starter: '\u2014', pro: '\u2014', enterprise: '\u2713' },
      { label: 'SOC 2 compliance artifacts', free: '\u2014', starter: '\u2014', pro: '\u2014', enterprise: '\u2713' },
      { label: 'On-premise deployment', free: '\u2014', starter: '\u2014', pro: '\u2014', enterprise: '\u2713' },
      { label: 'Decision Intelligence analytics', free: '\u2014', starter: '\u2014', pro: '\u2014', enterprise: '\u2713' },
    ],
  },
];

const TABLE_TIERS = [
  { key: 'free' as const, id: 'free', name: 'FREE', accent: 'acid-cyan', cta: 'TRY NOW', action: 'link' as const, href: '/playground', highlight: false },
  { key: 'starter' as const, id: 'starter', name: 'STARTER', accent: 'acid-cyan', cta: 'START FREE TRIAL', action: 'checkout' as const, highlight: false },
  { key: 'pro' as const, id: 'professional', name: 'PRO', accent: 'acid-green', cta: 'START FREE TRIAL', action: 'checkout' as const, highlight: true },
  { key: 'enterprise' as const, id: 'enterprise', name: 'ENTERPRISE', accent: 'acid-cyan', cta: 'CONTACT SALES', action: 'contact' as const, href: 'mailto:sales@aragora.ai?subject=Enterprise%20Inquiry', highlight: false },
];

function cellColor(value: string): string {
  if (value === '\u2713') return 'text-acid-green';
  if (value === '\u2014') return 'text-text-muted/40';
  return 'text-text-muted';
}

/* ── Mobile comparison cards (stacked) ── */

function ComparisonMobile({ loadingTier, onCheckout }: { loadingTier: string | null; onCheckout: (tierId: string) => void }) {
  return (
    <div className="md:hidden space-y-6">
      {TABLE_TIERS.map((tier) => (
        <div
          key={tier.key}
          className={`border ${
            tier.highlight
              ? 'border-acid-green bg-acid-green/5'
              : 'border-border bg-surface/30'
          } p-4`}
        >
          <div className="flex items-center gap-2 mb-1">
            <h4 className={`text-${tier.accent} font-mono text-sm font-bold`}>{tier.name}</h4>
            {tier.highlight && (
              <span className="font-mono text-[10px] px-1.5 py-0.5 bg-acid-green/20 text-acid-green border border-acid-green/40">
                MOST POPULAR
              </span>
            )}
          </div>

          {COMPARISON_CATEGORIES.map((cat) => (
            <div key={cat.category} className="mt-3">
              <p className="font-mono text-[10px] text-text-muted/60 mb-1">{cat.category}</p>
              {cat.features.map((f) => {
                const value = f[tier.key];
                return (
                  <div key={f.label} className="flex justify-between py-1 border-b border-border/30 last:border-b-0">
                    <span className="font-mono text-xs text-text-muted">{f.label}</span>
                    <span className={`font-mono text-xs ${cellColor(value)}`}>{value}</span>
                  </div>
                );
              })}
            </div>
          ))}

          {tier.action === 'checkout' ? (
            <button
              onClick={() => onCheckout(tier.id)}
              disabled={loadingTier === tier.id}
              className={`block w-full text-center font-mono text-sm py-2 mt-4 transition-colors disabled:opacity-50 disabled:cursor-wait ${
                tier.highlight
                  ? 'bg-acid-green text-bg font-bold hover:bg-acid-green/80'
                  : `border border-${tier.accent}/40 text-${tier.accent} hover:bg-${tier.accent}/10`
              }`}
            >
              {loadingTier === tier.id ? '[LOADING...]' : `[${tier.cta}]`}
            </button>
          ) : (
            <Link
              href={tier.href || '#'}
              className={`block text-center font-mono text-sm py-2 mt-4 transition-colors ${
                tier.highlight
                  ? 'bg-acid-green text-bg font-bold hover:bg-acid-green/80'
                  : `border border-${tier.accent}/40 text-${tier.accent} hover:bg-${tier.accent}/10`
              }`}
            >
              [{tier.cta}]
            </Link>
          )}
        </div>
      ))}
    </div>
  );
}

/* ── Desktop comparison table ── */

function ComparisonTable({ loadingTier, onCheckout }: { loadingTier: string | null; onCheckout: (tierId: string) => void }) {
  return (
    <div className="hidden md:block overflow-x-auto">
      <table className="w-full border-collapse font-mono text-xs">
        {/* Header */}
        <thead>
          <tr className="border-b border-acid-green/30">
            <th className="text-left py-3 px-3 text-text-muted font-normal w-[28%]">FEATURE</th>
            {TABLE_TIERS.map((tier) => (
              <th key={tier.key} className="text-center py-3 px-3 w-[18%]">
                <div className="flex flex-col items-center gap-1">
                  <span className={`text-${tier.accent} font-bold text-sm`}>{tier.name}</span>
                  {tier.highlight && (
                    <span className="font-mono text-[10px] px-1.5 py-0.5 bg-acid-green/20 text-acid-green border border-acid-green/40">
                      MOST POPULAR
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {COMPARISON_CATEGORIES.map((cat) => (
            <React.Fragment key={cat.category}>
              {/* Category header row */}
              <tr className="border-t border-border/40">
                <td colSpan={5} className="py-2 px-3 text-text-muted/60 text-[10px] tracking-wider">
                  {cat.category}
                </td>
              </tr>
              {/* Feature rows */}
              {cat.features.map((f) => (
                <tr key={f.label} className="border-b border-border/20 hover:bg-surface/40 transition-colors">
                  <td className="py-2 px-3 text-text-muted">{f.label}</td>
                  <td className={`py-2 px-3 text-center ${cellColor(f.free)}`}>{f.free}</td>
                  <td className={`py-2 px-3 text-center ${cellColor(f.starter)}`}>{f.starter}</td>
                  <td className={`py-2 px-3 text-center ${cellColor(f.pro)}`}>{f.pro}</td>
                  <td className={`py-2 px-3 text-center ${cellColor(f.enterprise)}`}>{f.enterprise}</td>
                </tr>
              ))}
            </React.Fragment>
          ))}
        </tbody>
        {/* CTA footer row */}
        <tfoot>
          <tr className="border-t border-acid-green/30">
            <td className="py-4 px-3" />
            {TABLE_TIERS.map((tier) => (
              <td key={tier.key} className="py-4 px-3 text-center">
                {tier.action === 'checkout' ? (
                  <button
                    onClick={() => onCheckout(tier.id)}
                    disabled={loadingTier === tier.id}
                    className={`inline-block font-mono text-xs py-2 px-4 transition-colors disabled:opacity-50 disabled:cursor-wait ${
                      tier.highlight
                        ? 'bg-acid-green text-bg font-bold hover:bg-acid-green/80'
                        : `border border-${tier.accent}/40 text-${tier.accent} hover:bg-${tier.accent}/10`
                    }`}
                  >
                    {loadingTier === tier.id ? '[LOADING...]' : `[${tier.cta}]`}
                  </button>
                ) : (
                  <Link
                    href={tier.href || '#'}
                    className={`inline-block font-mono text-xs py-2 px-4 transition-colors ${
                      tier.highlight
                        ? 'bg-acid-green text-bg font-bold hover:bg-acid-green/80'
                        : `border border-${tier.accent}/40 text-${tier.accent} hover:bg-${tier.accent}/10`
                    }`}
                  >
                    [{tier.cta}]
                  </Link>
                )}
              </td>
            ))}
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

interface CheckoutResponse {
  checkout?: { url?: string };
  checkout_url?: string;
}

export function PricingSection() {
  const { isAuthenticated } = useAuth();
  const { showError } = useToastContext();
  const router = useRouter();
  const [loadingTier, setLoadingTier] = useState<string | null>(null);

  const handleCheckout = async (tierId: string) => {
    if (!isAuthenticated) {
      router.push(`/auth/login?redirect=/pricing&plan=${tierId}`);
      return;
    }

    setLoadingTier(tierId);
    try {
      const origin = window.location.origin;
      const data = await apiPost<CheckoutResponse>('/api/billing/checkout', {
        tier: tierId,
        success_url: `${origin}/billing/success?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${origin}/#pricing`,
      });

      const checkoutUrl = data.checkout?.url || data.checkout_url;
      if (checkoutUrl) {
        window.location.href = checkoutUrl;
      } else {
        showError('No checkout URL returned. Please try again.');
      }
    } catch (err) {
      showError(
        err instanceof Error ? err.message : 'Failed to start checkout. Please try again.'
      );
    } finally {
      setLoadingTier(null);
    }
  };

  return (
    <section id="pricing" className="py-12 border-t border-acid-green/20">
      <div className="container mx-auto px-4">
        <SectionHeader title="PRICING" />
        <p className="text-text-muted font-mono text-xs text-center mb-8 max-w-xl mx-auto">
          Start free. Upgrade when you need real AI models and audit-ready receipts.
        </p>

        {/* ── Tier cards ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 max-w-6xl mx-auto">
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

              {tier.action === 'checkout' ? (
                <button
                  onClick={() => handleCheckout(tier.id)}
                  disabled={loadingTier === tier.id}
                  className={`block w-full text-center font-mono text-sm py-2 transition-colors disabled:opacity-50 disabled:cursor-wait ${
                    tier.highlight
                      ? 'bg-acid-green text-bg font-bold hover:bg-acid-green/80'
                      : `border border-${tier.accent}/40 text-${tier.accent} hover:bg-${tier.accent}/10`
                  }`}
                >
                  {loadingTier === tier.id ? '[LOADING...]' : `[${tier.cta}]`}
                </button>
              ) : (
                <Link
                  href={tier.href || '#'}
                  className={`block text-center font-mono text-sm py-2 transition-colors ${
                    tier.highlight
                      ? 'bg-acid-green text-bg font-bold hover:bg-acid-green/80'
                      : `border border-${tier.accent}/40 text-${tier.accent} hover:bg-${tier.accent}/10`
                  }`}
                >
                  [{tier.cta}]
                </Link>
              )}
            </div>
          ))}
        </div>

        {/* ── Feature comparison table ── */}
        <div className="max-w-5xl mx-auto mt-12">
          <h3 className="font-mono text-sm text-text-muted text-center mb-6">
            {'// DETAILED FEATURE COMPARISON'}
          </h3>
          <ComparisonTable loadingTier={loadingTier} onCheckout={handleCheckout} />
          <ComparisonMobile loadingTier={loadingTier} onCheckout={handleCheckout} />
        </div>
      </div>
    </section>
  );
}
