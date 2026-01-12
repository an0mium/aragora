'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

interface SubscriptionData {
  tier: string;
  status: string;
  is_active: boolean;
  current_period_end?: string;
  cancel_at_period_end?: boolean;
}

interface SubscriptionCardProps {
  compact?: boolean;
  showActions?: boolean;
  className?: string;
}

export function SubscriptionCard({
  compact = false,
  showActions = true,
  className = '',
}: SubscriptionCardProps) {
  const { isAuthenticated, tokens } = useAuth();
  const [subscription, setSubscription] = useState<SubscriptionData | null>(null);
  const [loading, setLoading] = useState(true);
  const accessToken = tokens?.access_token;

  const fetchSubscription = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/billing/subscription`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (res.ok) {
        const data = await res.json();
        setSubscription(data.subscription);
      }
    } catch (err) {
      console.error('Failed to fetch subscription:', err);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    if (isAuthenticated && accessToken) {
      fetchSubscription();
    } else {
      setLoading(false);
    }
  }, [isAuthenticated, accessToken, fetchSubscription]);

  if (!isAuthenticated) return null;

  const tierColors: Record<string, string> = {
    free: 'text-text-muted',
    starter: 'text-acid-cyan',
    professional: 'text-acid-green',
    enterprise: 'text-acid-magenta',
  };

  if (compact) {
    return (
      <div className={`font-mono text-xs flex items-center gap-2 ${className}`}>
        <span className="text-text-muted">Plan:</span>
        <span className={tierColors[subscription?.tier || 'free'] || 'text-text'}>
          {(subscription?.tier || 'FREE').toUpperCase()}
        </span>
        {subscription?.cancel_at_period_end && (
          <span className="text-warning">(canceling)</span>
        )}
      </div>
    );
  }

  return (
    <div className={`border border-acid-green/30 bg-surface/30 p-4 ${className}`}>
      <h3 className="text-sm font-mono text-acid-cyan mb-3">SUBSCRIPTION</h3>

      {loading ? (
        <div className="text-xs font-mono text-text-muted">Loading...</div>
      ) : (
        <div className="space-y-3">
          {/* Tier display */}
          <div>
            <div className={`text-lg font-mono uppercase ${tierColors[subscription?.tier || 'free']}`}>
              {subscription?.tier || 'FREE'}
            </div>
            <div className="text-xs font-mono text-text-muted">
              Status:{' '}
              {subscription?.is_active ? (
                <span className="text-acid-green">Active</span>
              ) : (
                <span className="text-warning">Inactive</span>
              )}
            </div>
          </div>

          {/* Cancellation notice */}
          {subscription?.cancel_at_period_end && (
            <div className="text-xs font-mono text-warning bg-warning/10 p-2 border border-warning/30">
              Subscription will cancel at period end
            </div>
          )}

          {/* Actions */}
          {showActions && (
            <div className="pt-2 space-y-2">
              <Link
                href="/billing"
                className="block text-center py-2 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors"
              >
                MANAGE BILLING
              </Link>
              {subscription?.tier === 'free' && (
                <Link
                  href="/pricing"
                  className="block text-center py-2 text-xs font-mono bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
                >
                  UPGRADE
                </Link>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SubscriptionCard;
