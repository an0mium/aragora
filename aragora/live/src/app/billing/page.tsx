'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { useAuth } from '@/context/AuthContext';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { API_BASE_URL } from '@/config';
import {
  getClient,
  BillingPlan,
  BillingUsage,
  BillingSubscription,
  BillingInvoice,
  UsageForecast as ForecastType,
} from '@/lib/aragora-client';

const API_BASE = API_BASE_URL;

// Use SDK types where available, keeping backward compat aliases
type UsageData = BillingUsage;
type SubscriptionData = BillingSubscription;
type Invoice = BillingInvoice;
type UsageForecast = ForecastType;

export default function BillingPage() {
  const { user, isAuthenticated, tokens, isLoading: authLoading } = useAuth();
  const [usage, setUsage] = useState<UsageData | null>(null);
  const [subscription, setSubscription] = useState<SubscriptionData | null>(null);
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [forecast, setForecast] = useState<UsageForecast | null>(null);
  const [plans, setPlans] = useState<BillingPlan[]>([]);
  const [loading, setLoading] = useState(true);
  const [portalLoading, setPortalLoading] = useState(false);
  const [upgradeLoading, setUpgradeLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'plans' | 'invoices'>('overview');
  const accessToken = tokens?.access_token;

  const fetchBillingData = useCallback(async () => {
    try {
      const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
      };

      const [usageRes, subRes, invoicesRes, forecastRes, plansRes] = await Promise.all([
        fetch(`${API_BASE}/api/billing/usage`, { headers }),
        fetch(`${API_BASE}/api/billing/subscription`, { headers }),
        fetch(`${API_BASE}/api/billing/invoices?limit=10`, { headers }),
        fetch(`${API_BASE}/api/billing/usage/forecast`, { headers }),
        fetch(`${API_BASE}/api/billing/plans`, { headers }),
      ]);

      if (usageRes.ok) {
        const usageData = await usageRes.json();
        setUsage(usageData.usage);
      }

      if (subRes.ok) {
        const subData = await subRes.json();
        setSubscription(subData.subscription);
      }

      if (invoicesRes.ok) {
        const invoicesData = await invoicesRes.json();
        setInvoices(invoicesData.invoices || []);
      }

      if (forecastRes.ok) {
        const forecastData = await forecastRes.json();
        setForecast(forecastData.forecast);
      }

      if (plansRes.ok) {
        const plansData = await plansRes.json();
        setPlans(plansData.plans || []);
      }
    } catch (err) {
      setError('Failed to load billing data');
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    if (!authLoading && isAuthenticated && accessToken) {
      fetchBillingData();
    } else if (!authLoading && !isAuthenticated) {
      setLoading(false);
    }
  }, [authLoading, isAuthenticated, accessToken, fetchBillingData]);

  const handleManageBilling = async () => {
    setPortalLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/billing/portal`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${tokens?.access_token}`,
        },
        body: JSON.stringify({
          return_url: window.location.href,
        }),
      });

      const data = await response.json();
      if (data.portal?.url) {
        window.location.href = data.portal.url;
      }
    } catch (err) {
      setError('Failed to open billing portal');
    } finally {
      setPortalLoading(false);
    }
  };

  const handleUpgrade = async (tier: string) => {
    setUpgradeLoading(tier);
    try {
      const response = await fetch(`${API_BASE}/api/billing/checkout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${tokens?.access_token}`,
        },
        body: JSON.stringify({
          tier,
          success_url: window.location.href,
          cancel_url: window.location.href,
        }),
      });

      const data = await response.json();
      if (data.checkout?.url) {
        window.location.href = data.checkout.url;
      } else {
        setError('Failed to start checkout');
      }
    } catch (err) {
      setError('Failed to start checkout');
    } finally {
      setUpgradeLoading(null);
    }
  };

  const usagePercent = usage
    ? Math.min(100, (usage.debates_used / usage.debates_limit) * 100)
    : 0;
  const forecastProjection = forecast?.projection ?? (forecast ? {
    debates_end_of_cycle: (forecast as { projected_debates?: number }).projected_debates ?? 0,
    cost_end_of_cycle_usd: (forecast as { cost_end_of_cycle_usd?: number }).cost_end_of_cycle_usd ?? 0,
  } : null);

  // Calculate trial days remaining
  const trialDaysRemaining = subscription?.trial_end
    ? Math.max(0, Math.ceil((new Date(subscription.trial_end).getTime() - Date.now()) / (1000 * 60 * 60 * 24)))
    : 0;
  const isInTrial = subscription?.is_trialing && trialDaysRemaining > 0;

  return (
    <ProtectedRoute>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/pricing"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [PRICING]
              </Link>
              <Link
                href="/settings"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [SETTINGS]
              </Link>
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="max-w-4xl mx-auto px-4 py-8">
          <h1 className="text-2xl font-mono text-acid-green mb-6">
            BILLING & SUBSCRIPTION
          </h1>

          {/* Tab Navigation */}
          <div className="flex gap-4 mb-6 border-b border-acid-green/30">
            <button
              onClick={() => setActiveTab('overview')}
              className={`pb-2 font-mono text-sm transition-colors ${
                activeTab === 'overview'
                  ? 'text-acid-green border-b-2 border-acid-green'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              OVERVIEW
            </button>
            <button
              onClick={() => setActiveTab('plans')}
              className={`pb-2 font-mono text-sm transition-colors ${
                activeTab === 'plans'
                  ? 'text-acid-green border-b-2 border-acid-green'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              PLANS
            </button>
            <button
              onClick={() => setActiveTab('invoices')}
              className={`pb-2 font-mono text-sm transition-colors ${
                activeTab === 'invoices'
                  ? 'text-acid-green border-b-2 border-acid-green'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              INVOICES ({invoices.length})
            </button>
          </div>

          {error && (
            <div className="mb-6 p-4 border border-warning/50 bg-warning/10 text-warning text-sm font-mono">
              {error}
            </div>
          )}

          {/* Trial Banner */}
          {isInTrial && (
            <div className="mb-6 p-4 border border-acid-cyan/50 bg-acid-cyan/10 text-acid-cyan text-sm font-mono">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-bold">TRIAL PERIOD ACTIVE</span>
                  <span className="ml-2">
                    {trialDaysRemaining} {trialDaysRemaining === 1 ? 'day' : 'days'} remaining
                  </span>
                </div>
                <Link
                  href="/pricing"
                  className="px-3 py-1 bg-acid-cyan/20 hover:bg-acid-cyan/30 border border-acid-cyan/50 transition-colors"
                >
                  UPGRADE NOW
                </Link>
              </div>
              {trialDaysRemaining <= 3 && (
                <div className="mt-2 text-xs text-warning">
                  Your trial is ending soon. Upgrade to keep your full access.
                </div>
              )}
            </div>
          )}

          {/* Payment Failed Banner */}
          {subscription?.payment_failed && (
            <div className="mb-6 p-4 border border-warning/50 bg-warning/10 text-warning text-sm font-mono">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-bold">PAYMENT FAILED</span>
                  <span className="ml-2">Please update your payment method to avoid service interruption.</span>
                </div>
                <button
                  onClick={handleManageBilling}
                  disabled={portalLoading}
                  className="px-3 py-1 bg-warning/20 hover:bg-warning/30 border border-warning/50 transition-colors disabled:opacity-50"
                >
                  UPDATE PAYMENT
                </button>
              </div>
            </div>
          )}

          {/* Past Due Status */}
          {subscription?.status === 'past_due' && !subscription?.payment_failed && (
            <div className="mb-6 p-4 border border-warning/50 bg-warning/10 text-warning text-sm font-mono">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-bold">PAYMENT PAST DUE</span>
                  <span className="ml-2">Your subscription payment is overdue.</span>
                </div>
                <button
                  onClick={handleManageBilling}
                  disabled={portalLoading}
                  className="px-3 py-1 bg-warning/20 hover:bg-warning/30 border border-warning/50 transition-colors disabled:opacity-50"
                >
                  RESOLVE NOW
                </button>
              </div>
            </div>
          )}

          {loading ? (
            <div className="text-center py-12 font-mono text-text-muted">
              Loading billing data...
            </div>
          ) : activeTab === 'overview' ? (
            <div className="grid gap-6 md:grid-cols-2">
              {/* Subscription Card */}
              <div className="border border-acid-green/30 bg-surface/30 p-6">
                <h2 className="text-lg font-mono text-acid-cyan mb-4">
                  CURRENT PLAN
                </h2>
                <div className="mb-4">
                  <div className="text-2xl font-mono text-acid-green uppercase">
                    {subscription?.tier || 'FREE'}
                    {isInTrial && (
                      <span className="ml-2 text-sm text-acid-cyan">(TRIAL)</span>
                    )}
                  </div>
                  <div className="text-sm font-mono text-text-muted">
                    Status: {isInTrial ? (
                      <span className="text-acid-cyan">Trialing</span>
                    ) : subscription?.status === 'past_due' ? (
                      <span className="text-warning">Past Due</span>
                    ) : subscription?.is_active ? (
                      <span className="text-acid-green">Active</span>
                    ) : (
                      <span className="text-warning">Inactive</span>
                    )}
                  </div>
                  {isInTrial && (
                    <div className="text-sm font-mono text-acid-cyan mt-1">
                      Trial ends: {new Date(subscription!.trial_end!).toLocaleDateString()}
                    </div>
                  )}
                  {subscription?.cancel_at_period_end && (
                    <div className="text-sm font-mono text-warning mt-1">
                      Cancels at period end
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  {subscription?.tier !== 'free' && (
                    <button
                      onClick={handleManageBilling}
                      disabled={portalLoading}
                      className="w-full py-2 font-mono text-sm border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors disabled:opacity-50"
                    >
                      {portalLoading ? 'LOADING...' : 'MANAGE SUBSCRIPTION'}
                    </button>
                  )}
                  <Link
                    href="/pricing"
                    className="block w-full py-2 font-mono text-sm text-center border border-acid-cyan/50 text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                  >
                    {subscription?.tier === 'free' ? 'UPGRADE PLAN' : 'CHANGE PLAN'}
                  </Link>
                </div>
              </div>

              {/* Usage Card */}
              <div className="border border-acid-green/30 bg-surface/30 p-6">
                <h2 className="text-lg font-mono text-acid-cyan mb-4">
                  USAGE THIS MONTH
                </h2>

                <div className="mb-4">
                  <div className="flex justify-between text-sm font-mono mb-1">
                    <span>Debates</span>
                    <span>
                      {usage?.debates_used || 0} / {usage?.debates_limit || 10}
                    </span>
                  </div>
                  <div className="h-2 bg-surface border border-acid-green/20">
                    <div
                      className={`h-full transition-all ${
                        usagePercent >= 90
                          ? 'bg-warning'
                          : usagePercent >= 75
                          ? 'bg-acid-cyan'
                          : 'bg-acid-green'
                      }`}
                      style={{ width: `${usagePercent}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono text-text-muted mt-1">
                    {usage?.debates_remaining || 0} remaining
                  </div>
                </div>

                {usage?.tokens_used ? (
                  <div className="text-sm font-mono text-text-muted">
                    <div>Tokens used: {usage.tokens_used.toLocaleString()}</div>
                    <div>Est. cost: ${usage.estimated_cost_usd.toFixed(2)}</div>
                  </div>
                ) : null}
              </div>

              {/* Usage Forecast Card */}
              {forecast && forecastProjection && (
                <div className="border border-acid-green/30 bg-surface/30 p-6">
                  <h2 className="text-lg font-mono text-acid-cyan mb-4">
                    USAGE FORECAST
                  </h2>
                  <div className="space-y-3 text-sm font-mono">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Projected debates:</span>
                      <span className={forecast.will_hit_limit ? 'text-warning' : 'text-acid-green'}>
                        {forecastProjection.debates_end_of_cycle}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Est. end-of-cycle cost:</span>
                      <span className="text-text">${forecastProjection.cost_end_of_cycle_usd.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Days remaining:</span>
                      <span className="text-text">{forecast.days_remaining}</span>
                    </div>
                    {forecast.will_hit_limit && (
                      <div className="mt-3 p-2 border border-warning/50 bg-warning/10 text-warning text-xs">
                        You may exceed your limit before the cycle ends.
                        {forecast.tier_recommendation && (
                          <span className="block mt-1">
                            Recommended: Upgrade to {forecast.tier_recommendation.recommended_tier}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Features Card */}
              {subscription?.limits && (
                <div className="border border-acid-green/30 bg-surface/30 p-6">
                  <h2 className="text-lg font-mono text-acid-cyan mb-4">
                    PLAN FEATURES
                  </h2>
                  <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                    <div>
                      <div className="text-text-muted">Debates/Month</div>
                      <div className="text-acid-green">
                        {subscription.limits.debates_per_month >= 999999
                          ? 'Unlimited'
                          : subscription.limits.debates_per_month}
                      </div>
                    </div>
                    <div>
                      <div className="text-text-muted">Team Members</div>
                      <div className="text-acid-green">
                        {subscription.limits.users_per_org >= 999999
                          ? 'Unlimited'
                          : subscription.limits.users_per_org}
                      </div>
                    </div>
                    <div>
                      <div className="text-text-muted">API Access</div>
                      <div className={subscription.limits.api_access ? 'text-acid-green' : 'text-text-muted'}>
                        {subscription.limits.api_access ? 'Enabled' : 'Disabled'}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : activeTab === 'plans' ? (
            /* Plans Tab */
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              {plans.map((plan) => {
                const isCurrentPlan = subscription?.tier === plan.id;
                const isLoading = upgradeLoading === plan.id;

                return (
                  <div
                    key={plan.id}
                    className={`border bg-surface/30 p-6 ${
                      isCurrentPlan
                        ? 'border-acid-green ring-1 ring-acid-green'
                        : 'border-acid-green/30'
                    }`}
                  >
                    <div className="text-center mb-4">
                      <h3 className="text-lg font-mono text-acid-cyan uppercase">
                        {plan.name}
                      </h3>
                      <div className="mt-2">
                        <span className="text-2xl font-mono text-acid-green">
                          {plan.price_monthly}
                        </span>
                        <span className="text-text-muted font-mono text-sm">/mo</span>
                      </div>
                    </div>

                    <ul className="space-y-2 mb-6 text-sm font-mono">
                      <li className="flex items-center gap-2">
                        <span className="text-acid-green">[+]</span>
                        <span className="text-text">
                          {plan.features.debates_per_month === -1
                            ? 'Unlimited'
                            : plan.features.debates_per_month}{' '}
                          debates/mo
                        </span>
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="text-acid-green">[+]</span>
                        <span className="text-text">
                          {plan.features.users_per_org} team members
                        </span>
                      </li>
                      {plan.features.api_access && (
                        <li className="flex items-center gap-2">
                          <span className="text-acid-green">[+]</span>
                          <span className="text-text">API access</span>
                        </li>
                      )}
                      {plan.features.all_agents && (
                        <li className="flex items-center gap-2">
                          <span className="text-acid-green">[+]</span>
                          <span className="text-text">All AI agents</span>
                        </li>
                      )}
                      {plan.features.sso_enabled && (
                        <li className="flex items-center gap-2">
                          <span className="text-acid-green">[+]</span>
                          <span className="text-text">SSO/SAML</span>
                        </li>
                      )}
                      {plan.features.audit_logs && (
                        <li className="flex items-center gap-2">
                          <span className="text-acid-green">[+]</span>
                          <span className="text-text">Audit logs</span>
                        </li>
                      )}
                      {plan.features.priority_support && (
                        <li className="flex items-center gap-2">
                          <span className="text-acid-green">[+]</span>
                          <span className="text-text">Priority support</span>
                        </li>
                      )}
                    </ul>

                    {isCurrentPlan ? (
                      <button
                        disabled
                        className="w-full py-2 font-mono text-sm border border-acid-green/30 text-text-muted cursor-not-allowed"
                      >
                        CURRENT PLAN
                      </button>
                    ) : plan.id === 'free' ? (
                      <button
                        disabled
                        className="w-full py-2 font-mono text-sm border border-acid-green/30 text-text-muted cursor-not-allowed"
                      >
                        FREE TIER
                      </button>
                    ) : (
                      <button
                        onClick={() => handleUpgrade(plan.id)}
                        disabled={isLoading}
                        className="w-full py-2 font-mono text-sm border border-acid-cyan/50 text-acid-cyan hover:bg-acid-cyan/10 transition-colors disabled:opacity-50"
                      >
                        {isLoading ? 'LOADING...' : 'UPGRADE'}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            /* Invoices Tab */
            <div className="border border-acid-green/30 bg-surface/30 p-6">
              <h2 className="text-lg font-mono text-acid-cyan mb-4">
                INVOICE HISTORY
              </h2>
              {invoices.length === 0 ? (
                <div className="text-center py-8 text-text-muted font-mono">
                  No invoices found
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm font-mono">
                    <thead>
                      <tr className="border-b border-acid-green/20">
                        <th className="text-left py-2 text-text-muted">Invoice</th>
                        <th className="text-left py-2 text-text-muted">Date</th>
                        <th className="text-left py-2 text-text-muted">Status</th>
                        <th className="text-right py-2 text-text-muted">Amount</th>
                        <th className="text-right py-2 text-text-muted">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {invoices.map((invoice) => (
                        <tr key={invoice.id} className="border-b border-acid-green/10">
                          <td className="py-3 text-text">{invoice.number || invoice.id.slice(0, 12)}</td>
                          <td className="py-3 text-text-muted">
                            {new Date(invoice.created).toLocaleDateString()}
                          </td>
                          <td className="py-3">
                            <span
                              className={`px-2 py-0.5 text-xs rounded ${
                                invoice.status === 'paid'
                                  ? 'bg-acid-green/20 text-acid-green'
                                  : invoice.status === 'open'
                                  ? 'bg-acid-cyan/20 text-acid-cyan'
                                  : 'bg-warning/20 text-warning'
                              }`}
                            >
                              {invoice.status.toUpperCase()}
                            </span>
                          </td>
                          <td className="py-3 text-right text-text">
                            ${invoice.amount_paid.toFixed(2)} {invoice.currency}
                          </td>
                          <td className="py-3 text-right">
                            <div className="flex gap-2 justify-end">
                              {invoice.hosted_invoice_url && (
                                <a
                                  href={invoice.hosted_invoice_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-acid-cyan hover:text-acid-green text-xs"
                                >
                                  [VIEW]
                                </a>
                              )}
                              {invoice.invoice_pdf && (
                                <a
                                  href={invoice.invoice_pdf}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-acid-cyan hover:text-acid-green text-xs"
                                >
                                  [PDF]
                                </a>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </ProtectedRoute>
  );
}
