'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { useAuth } from '@/context/AuthContext';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

interface OrganizationDetails {
  id: string;
  name: string;
  slug: string;
  tier: string;
  owner_id: string;
  member_count: number;
  member_limit: number;
  created_at: string;
}

export default function OrganizationPage() {
  const { organization, tokens, user } = useAuth();
  const [orgDetails, setOrgDetails] = useState<OrganizationDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [editName, setEditName] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (organization?.id && tokens?.access_token) {
      fetchOrgDetails();
    }
  }, [organization, tokens]);

  const fetchOrgDetails = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/org/${organization?.id}`, {
        headers: {
          'Authorization': `Bearer ${tokens?.access_token}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch organization details');
      }

      const data = await response.json();
      setOrgDetails(data.organization);
      setEditName(data.organization.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load organization');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!editName.trim()) {
      setError('Organization name cannot be empty');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/org/${organization?.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${tokens?.access_token}`,
        },
        body: JSON.stringify({ name: editName.trim() }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to update organization');
      }

      await fetchOrgDetails();
      setEditMode(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const isOwner = user?.id === orgDetails?.owner_id;

  const getTierBadgeColor = (tier: string) => {
    switch (tier) {
      case 'free': return 'text-text-muted border-text-muted/30';
      case 'starter': return 'text-acid-cyan border-acid-cyan/30';
      case 'professional': return 'text-acid-green border-acid-green/30';
      case 'enterprise': return 'text-warning border-warning/30';
      default: return 'text-text-muted border-text-muted/30';
    }
  };

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
            <Link
              href="/"
              className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
            >
              [DASHBOARD]
            </Link>
          </div>
        </header>

        {/* Content */}
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-mono text-acid-green">
              ORGANIZATION SETTINGS
            </h1>
          </div>

          {/* Sub-navigation */}
          <div className="flex gap-4 mb-6 border-b border-acid-green/30">
            <Link
              href="/organization"
              className="pb-2 font-mono text-sm text-acid-green border-b-2 border-acid-green"
            >
              SETTINGS
            </Link>
            <Link
              href="/organization/members"
              className="pb-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
            >
              MEMBERS
            </Link>
          </div>

          {error && (
            <div className="mb-6 p-4 border border-warning/50 bg-warning/10 text-warning text-sm font-mono">
              {error}
              <button onClick={() => setError(null)} className="ml-4 text-xs underline">
                Dismiss
              </button>
            </div>
          )}

          {loading ? (
            <div className="text-center py-12 font-mono text-text-muted">
              Loading organization details...
            </div>
          ) : orgDetails ? (
            <div className="space-y-6">
              {/* Organization Info Card */}
              <div className="border border-acid-green/30 bg-surface/30 p-6">
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <div className="text-xs font-mono text-text-muted mb-1">ORGANIZATION NAME</div>
                    {editMode ? (
                      <div className="flex gap-3">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          className="bg-bg border border-acid-green/30 px-3 py-2 font-mono text-lg text-acid-green focus:border-acid-green focus:outline-none"
                        />
                        <button
                          onClick={handleSave}
                          disabled={saving}
                          className="px-4 py-2 font-mono text-sm border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors disabled:opacity-50"
                        >
                          {saving ? 'SAVING...' : 'SAVE'}
                        </button>
                        <button
                          onClick={() => {
                            setEditMode(false);
                            setEditName(orgDetails.name);
                          }}
                          className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
                        >
                          CANCEL
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-4">
                        <div className="text-2xl font-mono text-acid-green">{orgDetails.name}</div>
                        {isOwner && (
                          <button
                            onClick={() => setEditMode(true)}
                            className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
                          >
                            [EDIT]
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                  <div className={`px-3 py-1 border font-mono text-sm uppercase ${getTierBadgeColor(orgDetails.tier)}`}>
                    {orgDetails.tier}
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <div>
                    <div className="text-xs font-mono text-text-muted mb-1">SLUG</div>
                    <div className="text-sm font-mono text-text">{orgDetails.slug}</div>
                  </div>
                  <div>
                    <div className="text-xs font-mono text-text-muted mb-1">MEMBERS</div>
                    <div className="text-sm font-mono text-text">
                      {orgDetails.member_count} / {orgDetails.member_limit === 999999 ? 'Unlimited' : orgDetails.member_limit}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs font-mono text-text-muted mb-1">CREATED</div>
                    <div className="text-sm font-mono text-text">
                      {new Date(orgDetails.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs font-mono text-text-muted mb-1">YOUR ROLE</div>
                    <div className="text-sm font-mono text-acid-cyan uppercase">
                      {isOwner ? 'Owner' : 'Member'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="border border-acid-green/30 bg-surface/30 p-6">
                <h2 className="text-lg font-mono text-acid-cyan mb-4">QUICK ACTIONS</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Link
                    href="/organization/members"
                    className="block p-4 border border-acid-green/20 hover:border-acid-green/50 transition-colors"
                  >
                    <div className="text-sm font-mono text-acid-green mb-1">Manage Members</div>
                    <div className="text-xs font-mono text-text-muted">
                      Add, remove, or update member roles
                    </div>
                  </Link>
                  <Link
                    href="/billing"
                    className="block p-4 border border-acid-green/20 hover:border-acid-green/50 transition-colors"
                  >
                    <div className="text-sm font-mono text-acid-green mb-1">Billing & Subscription</div>
                    <div className="text-xs font-mono text-text-muted">
                      Manage your subscription and usage
                    </div>
                  </Link>
                </div>
              </div>

              {/* Danger Zone - Owner Only */}
              {isOwner && (
                <div className="border border-warning/30 bg-warning/5 p-6">
                  <h2 className="text-lg font-mono text-warning mb-4">DANGER ZONE</h2>
                  <div className="text-sm font-mono text-text-muted mb-4">
                    These actions are irreversible. Please be certain.
                  </div>
                  <button
                    className="px-4 py-2 font-mono text-sm border border-warning/50 text-warning hover:bg-warning/10 transition-colors opacity-50 cursor-not-allowed"
                    disabled
                    title="Contact support to delete organization"
                  >
                    DELETE ORGANIZATION
                  </button>
                  <div className="text-xs font-mono text-text-muted mt-2">
                    Contact support to delete your organization
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="font-mono text-text-muted mb-4">No organization found</div>
              <div className="text-sm font-mono text-text-muted">
                You are not part of any organization
              </div>
            </div>
          )}
        </div>
      </main>
    </ProtectedRoute>
  );
}
