'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { useAuth } from '@/context/AuthContext';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

interface PlanFeature {
  text: string;
  included: boolean;
}

interface Plan {
  id: string;
  name: string;
  price: number;
  period: string;
  description: string;
  features: PlanFeature[];
  highlighted?: boolean;
  cta: string;
}

const plans: Plan[] = [
  {
    id: 'free',
    name: 'FREE',
    price: 0,
    period: 'forever',
    description: 'Get started with decision stress-tests',
    cta: 'Current Plan',
    features: [
      { text: '10 stress-tests / month', included: true },
      { text: '3 agents per stress-test', included: true },
      { text: 'Public stress-tests only', included: true },
      { text: 'Basic analytics', included: true },
      { text: 'API access', included: false },
      { text: 'Priority support', included: false },
    ],
  },
  {
    id: 'starter',
    name: 'STARTER',
    price: 29,
    period: 'month',
    description: 'For individuals and small teams',
    cta: 'Upgrade',
    features: [
      { text: '100 stress-tests / month', included: true },
      { text: '5 agents per stress-test', included: true },
      { text: 'Private stress-tests', included: true },
      { text: 'Full analytics', included: true },
      { text: 'API access', included: false },
      { text: 'Priority support', included: false },
    ],
  },
  {
    id: 'professional',
    name: 'PROFESSIONAL',
    price: 99,
    period: 'month',
    description: 'For power users and growing teams',
    highlighted: true,
    cta: 'Upgrade',
    features: [
      { text: '1,000 stress-tests / month', included: true },
      { text: 'Unlimited agents', included: true },
      { text: 'Private stress-tests', included: true },
      { text: 'Advanced analytics', included: true },
      { text: 'Full API access', included: true },
      { text: 'Email support', included: true },
    ],
  },
  {
    id: 'enterprise',
    name: 'ENTERPRISE',
    price: 299,
    period: 'month',
    description: 'For organizations with custom needs',
    cta: 'Contact Sales',
    features: [
      { text: 'Unlimited stress-tests', included: true },
      { text: 'Unlimited agents', included: true },
      { text: 'SSO / SAML', included: true },
      { text: 'Custom integrations', included: true },
      { text: 'Dedicated support', included: true },
      { text: 'SLA guarantee', included: true },
    ],
  },
];

export default function PricingPage() {
  const { user, isAuthenticated, tokens } = useAuth();
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubscribe = async (planId: string) => {
    if (!isAuthenticated) {
      window.location.href = '/auth/register';
      return;
    }

    if (planId === 'free') return;
    if (planId === 'enterprise') {
      window.location.href = 'mailto:sales@aragora.ai?subject=Enterprise%20Inquiry';
      return;
    }

    setLoading(planId);
    setError(null);

    try {
      const currentUrl = window.location.origin;
      const response = await fetch(`${API_BASE}/api/billing/checkout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${tokens?.access_token}`,
        },
        body: JSON.stringify({
          tier: planId,
          success_url: `${currentUrl}/billing/success?session_id={CHECKOUT_SESSION_ID}`,
          cancel_url: `${currentUrl}/pricing`,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create checkout session');
      }

      // Redirect to Stripe Checkout
      const checkoutUrl = data.checkout?.url || data.checkout_url;
      if (checkoutUrl) {
        window.location.href = checkoutUrl;
      } else {
        throw new Error('No checkout URL returned');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(null);
    }
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              {isAuthenticated ? (
                <Link
                  href="/"
                  className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
                >
                  [DASHBOARD]
                </Link>
              ) : (
                <Link
                  href="/auth/login"
                  className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
                >
                  [LOGIN]
                </Link>
              )}
            </div>
          </div>
        </header>

        {/* Hero */}
        <section className="py-16 px-4 text-center">
          <h1 className="text-3xl sm:text-4xl font-mono text-acid-green mb-4">
            PRICING
          </h1>
          <p className="text-text-muted font-mono max-w-xl mx-auto">
            Choose the plan that fits your debate needs.
            All plans include access to the live debate arena.
          </p>
        </section>

        {/* Error Message */}
        {error && (
          <div className="max-w-4xl mx-auto px-4 mb-8">
            <div className="p-4 border border-warning/50 bg-warning/10 text-warning text-sm font-mono">
              {error}
            </div>
          </div>
        )}

        {/* Pricing Grid */}
        <section className="pb-16 px-4">
          <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {plans.map((plan) => (
              <div
                key={plan.id}
                className={`border p-6 flex flex-col ${
                  plan.highlighted
                    ? 'border-acid-green bg-acid-green/5'
                    : 'border-acid-green/30 bg-surface/30'
                }`}
              >
                {/* Plan Header */}
                <div className="mb-6">
                  {plan.highlighted && (
                    <div className="text-xs font-mono text-acid-green mb-2">
                      MOST POPULAR
                    </div>
                  )}
                  <h2 className="text-xl font-mono text-acid-cyan mb-2">
                    {plan.name}
                  </h2>
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-mono text-text">
                      ${plan.price}
                    </span>
                    <span className="text-sm font-mono text-text-muted">
                      /{plan.period}
                    </span>
                  </div>
                  <p className="mt-2 text-sm font-mono text-text-muted">
                    {plan.description}
                  </p>
                </div>

                {/* Features */}
                <ul className="flex-1 space-y-3 mb-6">
                  {plan.features.map((feature, idx) => (
                    <li
                      key={idx}
                      className={`flex items-center gap-2 text-sm font-mono ${
                        feature.included ? 'text-text' : 'text-text-muted/50'
                      }`}
                    >
                      <span className={feature.included ? 'text-acid-green' : 'text-text-muted/30'}>
                        {feature.included ? '[+]' : '[-]'}
                      </span>
                      {feature.text}
                    </li>
                  ))}
                </ul>

                {/* CTA Button */}
                <button
                  onClick={() => handleSubscribe(plan.id)}
                  disabled={loading === plan.id || plan.id === 'free'}
                  className={`w-full py-3 font-mono font-bold transition-colors disabled:cursor-not-allowed ${
                    plan.highlighted
                      ? 'bg-acid-green text-bg hover:bg-acid-green/80 disabled:opacity-50'
                      : plan.id === 'free'
                      ? 'bg-surface border border-acid-green/30 text-text-muted cursor-default'
                      : 'bg-surface border border-acid-green/50 text-acid-green hover:bg-acid-green/10 disabled:opacity-50'
                  }`}
                >
                  {loading === plan.id ? 'LOADING...' : plan.cta}
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* FAQ */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">
              FREQUENTLY ASKED QUESTIONS
            </h2>
            <div className="space-y-6">
              <div className="border border-acid-green/20 p-4">
                <h3 className="font-mono text-acid-cyan mb-2">
                  Can I change plans later?
                </h3>
                <p className="text-sm font-mono text-text-muted">
                  Yes, you can upgrade or downgrade at any time. Changes take effect
                  immediately, and we&apos;ll prorate your billing.
                </p>
              </div>
              <div className="border border-acid-green/20 p-4">
                <h3 className="font-mono text-acid-cyan mb-2">
                  What happens if I exceed my debate limit?
                </h3>
                <p className="text-sm font-mono text-text-muted">
                  You&apos;ll see a prompt to upgrade. Your existing debates remain
                  accessible, but you can&apos;t start new ones until the next billing cycle
                  or you upgrade.
                </p>
              </div>
              <div className="border border-acid-green/20 p-4">
                <h3 className="font-mono text-acid-cyan mb-2">
                  Do you offer refunds?
                </h3>
                <p className="text-sm font-mono text-text-muted">
                  We offer a 7-day money-back guarantee. If you&apos;re not satisfied,
                  contact support for a full refund.
                </p>
              </div>
              <div className="border border-acid-green/20 p-4">
                <h3 className="font-mono text-acid-cyan mb-2">
                  What payment methods do you accept?
                </h3>
                <p className="text-sm font-mono text-text-muted">
                  We accept all major credit cards (Visa, Mastercard, Amex) via Stripe.
                  Enterprise customers can pay by invoice.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20">
          <div className="text-acid-green/50 mb-4">{'═'.repeat(40)}</div>
          <p className="text-text-muted">
            Questions? Contact{' '}
            <a href="mailto:support@aragora.ai" className="text-acid-cyan hover:text-acid-green">
              support@aragora.ai
            </a>
          </p>
        </footer>
      </main>
    </>
  );
}
