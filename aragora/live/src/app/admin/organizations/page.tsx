'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { useAuth } from '@/context/AuthContext';

interface Organization {
  id: string;
  name: string;
  slug: string;
  tier: string;
  debates_used_this_month: number;
  owner_id: string;
  stripe_customer_id?: string;
  stripe_subscription_id?: string;
  billing_cycle_start: string;
  created_at: string;
  updated_at: string;
}

interface OrganizationsResponse {
  organizations: Organization[];
  total: number;
  limit: number;
  offset: number;
}

function TierBadge({ tier }: { tier: string }) {
  const colors: Record<string, string> = {
    free: 'bg-text-muted/20 text-text-muted border-text-muted/40',
    starter: 'bg-acid-cyan/20 text-acid-cyan border-acid-cyan/40',
    professional: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    enterprise: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
    enterprise_plus: 'bg-acid-magenta/20 text-acid-magenta border-acid-magenta/40',
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${colors[tier] || colors.free}`}>
      {tier.replace('_', ' ').toUpperCase()}
    </span>
  );
}

export default function OrganizationsAdminPage() {
  const { config: backendConfig } = useBackend();
  const { user, isAuthenticated, tokens } = useAuth();
  const token = tokens?.access_token;
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [tierFilter, setTierFilter] = useState<string>('');
  const limit = 20;

  const fetchOrganizations = useCallback(async () => {
    if (!token) return;

    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams({
        limit: String(limit),
        offset: String(page * limit),
      });
      if (tierFilter) {
        params.set('tier', tierFilter);
      }

      const res = await fetch(
        `${backendConfig.api}/api/admin/organizations?${params}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!res.ok) {
        if (res.status === 403) {
          throw new Error('Admin access required');
        }
        throw new Error(`Failed to fetch organizations: ${res.status}`);
      }

      const data: OrganizationsResponse = await res.json();
      setOrganizations(data.organizations);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch organizations');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api, token, page, tierFilter]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchOrganizations();
    }
  }, [fetchOrganizations, isAuthenticated]);

  const isAdmin = isAuthenticated && (user?.role === 'admin' || user?.role === 'owner');
  const totalPages = Math.ceil(total / limit);

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
              <Link
                href="/admin"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [ADMIN]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Sub Navigation */}
        <div className="border-b border-acid-green/20 bg-surface/40">
          <div className="container mx-auto px-4">
            <div className="flex gap-4 overflow-x-auto">
              <Link
                href="/admin"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                SYSTEM
              </Link>
              <Link
                href="/admin/organizations"
                className="px-4 py-2 font-mono text-sm text-acid-green border-b-2 border-acid-green"
              >
                ORGANIZATIONS
              </Link>
              <Link
                href="/admin/users"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                USERS
              </Link>
              <Link
                href="/admin/revenue"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                REVENUE
              </Link>
              <Link
                href="/admin/training"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                TRAINING
              </Link>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6 flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-2xl font-mono text-acid-green mb-2">
                Organizations
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Manage all organizations and their subscriptions.
              </p>
            </div>
            <div className="flex items-center gap-4">
              <select
                value={tierFilter}
                onChange={(e) => {
                  setTierFilter(e.target.value);
                  setPage(0);
                }}
                className="bg-surface border border-acid-green/40 text-text font-mono text-sm rounded px-3 py-2"
              >
                <option value="">All Tiers</option>
                <option value="free">Free</option>
                <option value="starter">Starter</option>
                <option value="professional">Professional</option>
                <option value="enterprise">Enterprise</option>
                <option value="enterprise_plus">Enterprise+</option>
              </select>
              <button
                onClick={fetchOrganizations}
                disabled={loading}
                className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
              >
                {loading ? 'Loading...' : 'Refresh'}
              </button>
            </div>
          </div>

          {!isAdmin && (
            <div className="card p-6 mb-6 border-acid-yellow/40">
              <div className="flex items-center gap-2 text-acid-yellow font-mono text-sm">
                <span>!</span>
                <span>Admin access required. Please sign in with an admin account.</span>
              </div>
            </div>
          )}

          {error && (
            <div className="card p-4 mb-6 border-acid-red/40 bg-acid-red/10">
              <p className="text-acid-red font-mono text-sm">{error}</p>
            </div>
          )}

          {/* Stats Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">Total Organizations</div>
              <div className="font-mono text-2xl text-acid-green">{total}</div>
            </div>
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">On This Page</div>
              <div className="font-mono text-2xl text-text">{organizations.length}</div>
            </div>
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">Current Page</div>
              <div className="font-mono text-2xl text-acid-cyan">{page + 1} / {totalPages || 1}</div>
            </div>
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">Filter</div>
              <div className="font-mono text-lg text-text">{tierFilter || 'All'}</div>
            </div>
          </div>

          {/* Organizations Table */}
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-surface border-b border-acid-green/20">
                  <tr>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">NAME</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">SLUG</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">TIER</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">DEBATES</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">BILLING</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">CREATED</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center">
                        <div className="font-mono text-text-muted animate-pulse">Loading...</div>
                      </td>
                    </tr>
                  )}
                  {!loading && organizations.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center">
                        <div className="font-mono text-text-muted">No organizations found</div>
                      </td>
                    </tr>
                  )}
                  {!loading && organizations.map((org) => (
                    <tr key={org.id} className="border-b border-acid-green/10 hover:bg-surface/50">
                      <td className="px-4 py-3">
                        <div className="font-mono text-sm text-text">{org.name}</div>
                        <div className="font-mono text-xs text-text-muted">{org.id.slice(0, 8)}...</div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-mono text-sm text-acid-cyan">{org.slug}</div>
                      </td>
                      <td className="px-4 py-3">
                        <TierBadge tier={org.tier} />
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-mono text-sm text-text">{org.debates_used_this_month}</div>
                      </td>
                      <td className="px-4 py-3">
                        {org.stripe_customer_id ? (
                          <span className="px-2 py-0.5 text-xs font-mono rounded border bg-acid-green/20 text-acid-green border-acid-green/40">
                            CONNECTED
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 text-xs font-mono rounded border bg-text-muted/20 text-text-muted border-text-muted/40">
                            NOT SET
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-mono text-xs text-text-muted">
                          {new Date(org.created_at).toLocaleDateString()}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-acid-green/20">
                <button
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-3 py-1 font-mono text-sm text-acid-cyan hover:text-acid-green disabled:text-text-muted disabled:cursor-not-allowed"
                >
                  &lt; PREV
                </button>
                <span className="font-mono text-sm text-text-muted">
                  Page {page + 1} of {totalPages}
                </span>
                <button
                  onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="px-3 py-1 font-mono text-sm text-acid-cyan hover:text-acid-green disabled:text-text-muted disabled:cursor-not-allowed"
                >
                  NEXT &gt;
                </button>
              </div>
            )}
          </div>
        </div>
      </main>
    </>
  );
}
