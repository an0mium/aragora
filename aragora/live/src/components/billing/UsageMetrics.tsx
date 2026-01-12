'use client';

import { useEffect, useState, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

interface UsageData {
  debates_used: number;
  debates_limit: number;
  debates_remaining: number;
  tokens_used: number;
  estimated_cost_usd: number;
  period_start: string | null;
}

interface UsageMetricsProps {
  compact?: boolean;
  className?: string;
}

export function UsageMetrics({ compact = false, className = '' }: UsageMetricsProps) {
  const { isAuthenticated, tokens } = useAuth();
  const [usage, setUsage] = useState<UsageData | null>(null);
  const [loading, setLoading] = useState(true);
  const accessToken = tokens?.access_token;

  const fetchUsage = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/billing/usage`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (res.ok) {
        const data = await res.json();
        setUsage(data.usage);
      }
    } catch (err) {
      console.error('Failed to fetch usage:', err);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    if (isAuthenticated && accessToken) {
      fetchUsage();
    } else {
      setLoading(false);
    }
  }, [isAuthenticated, accessToken, fetchUsage]);

  if (!isAuthenticated) return null;

  const usagePercent = usage
    ? Math.min(100, (usage.debates_used / usage.debates_limit) * 100)
    : 0;

  const getBarColor = () => {
    if (usagePercent >= 90) return 'bg-warning';
    if (usagePercent >= 75) return 'bg-acid-cyan';
    return 'bg-acid-green';
  };

  if (compact) {
    return (
      <div className={`font-mono text-xs ${className}`}>
        <div className="flex items-center gap-2">
          <div className="w-16 h-1.5 bg-surface border border-acid-green/20">
            <div
              className={`h-full transition-all ${getBarColor()}`}
              style={{ width: `${usagePercent}%` }}
            />
          </div>
          <span className="text-text-muted">
            {usage?.debates_used ?? '-'}/{usage?.debates_limit ?? '-'}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className={`border border-acid-green/30 bg-surface/30 p-4 ${className}`}>
      <h3 className="text-sm font-mono text-acid-cyan mb-3">USAGE THIS MONTH</h3>

      {loading ? (
        <div className="text-xs font-mono text-text-muted">Loading...</div>
      ) : usage ? (
        <div className="space-y-3">
          {/* Debates usage bar */}
          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-text-muted">Debates</span>
              <span className="text-text">
                {usage.debates_used} / {usage.debates_limit}
              </span>
            </div>
            <div className="h-2 bg-surface border border-acid-green/20">
              <div
                className={`h-full transition-all ${getBarColor()}`}
                style={{ width: `${usagePercent}%` }}
              />
            </div>
            <div className="text-xs font-mono text-text-muted mt-1">
              {usage.debates_remaining} remaining
            </div>
          </div>

          {/* Token usage */}
          {usage.tokens_used > 0 && (
            <div className="pt-2 border-t border-acid-green/10">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-text-muted">Tokens</span>
                <span className="text-text">{usage.tokens_used.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-xs font-mono mt-1">
                <span className="text-text-muted">Est. Cost</span>
                <span className="text-acid-cyan">${usage.estimated_cost_usd.toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-xs font-mono text-text-muted">No usage data</div>
      )}
    </div>
  );
}

export default UsageMetrics;
