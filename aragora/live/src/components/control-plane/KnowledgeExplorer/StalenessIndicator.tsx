'use client';

import { useMemo } from 'react';

export interface StalenessBucket {
  /** Age range label (e.g., "< 1 week", "1-4 weeks") */
  label: string;
  /** Number of nodes in this bucket */
  count: number;
  /** Age threshold in days */
  maxDays: number;
  /** Health status */
  status: 'fresh' | 'aging' | 'stale' | 'critical';
}

export interface StalenessIndicatorProps {
  /** Distribution of nodes by age */
  buckets: StalenessBucket[];
  /** Total number of nodes */
  totalNodes: number;
  /** Average age in days */
  avgAgeDays: number;
  /** Nodes updated in last 24h */
  recentUpdates: number;
  /** Loading state */
  loading?: boolean;
  /** Callback when a bucket is clicked */
  onBucketClick?: (bucket: StalenessBucket) => void;
}

/**
 * Visual indicator showing knowledge freshness distribution.
 */
export function StalenessIndicator({
  buckets,
  totalNodes,
  avgAgeDays,
  recentUpdates,
  loading = false,
  onBucketClick,
}: StalenessIndicatorProps) {
  const maxCount = useMemo(() => Math.max(...buckets.map((b) => b.count), 1), [buckets]);

  const freshnessScore = useMemo(() => {
    if (totalNodes === 0) return 100;
    const freshCount = buckets
      .filter((b) => b.status === 'fresh' || b.status === 'aging')
      .reduce((sum, b) => sum + b.count, 0);
    return Math.round((freshCount / totalNodes) * 100);
  }, [buckets, totalNodes]);

  const statusColors: Record<string, string> = {
    fresh: 'bg-green-500',
    aging: 'bg-yellow-500',
    stale: 'bg-orange-500',
    critical: 'bg-red-500',
  };

  if (loading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="grid grid-cols-3 gap-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-16 bg-surface-lighter rounded-lg" />
          ))}
        </div>
        <div className="h-32 bg-surface-lighter rounded-lg" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 bg-surface rounded-lg border border-border">
          <div className="text-xs text-text-muted mb-1">Freshness Score</div>
          <div
            className={`text-2xl font-mono font-bold ${
              freshnessScore >= 70
                ? 'text-green-400'
                : freshnessScore >= 40
                  ? 'text-yellow-400'
                  : 'text-red-400'
            }`}
          >
            {freshnessScore}%
          </div>
        </div>
        <div className="p-3 bg-surface rounded-lg border border-border">
          <div className="text-xs text-text-muted mb-1">Avg Age</div>
          <div className="text-2xl font-mono font-bold text-acid-cyan">
            {avgAgeDays < 1 ? '<1' : avgAgeDays}
            <span className="text-sm text-text-muted ml-1">days</span>
          </div>
        </div>
        <div className="p-3 bg-surface rounded-lg border border-border">
          <div className="text-xs text-text-muted mb-1">Last 24h</div>
          <div className="text-2xl font-mono font-bold text-acid-green">
            +{recentUpdates}
            <span className="text-sm text-text-muted ml-1">updated</span>
          </div>
        </div>
      </div>

      {/* Age Distribution Chart */}
      <div className="p-4 bg-surface rounded-lg border border-border">
        <div className="text-sm font-medium mb-3">Age Distribution</div>
        <div className="space-y-2">
          {buckets.map((bucket) => (
            <button
              key={bucket.label}
              onClick={() => onBucketClick?.(bucket)}
              className="w-full flex items-center gap-3 p-2 rounded hover:bg-surface-lighter
                         transition-colors group"
            >
              <div className="w-24 text-xs text-text-muted text-left">{bucket.label}</div>
              <div className="flex-1 h-6 bg-surface-lighter rounded overflow-hidden">
                <div
                  className={`h-full ${statusColors[bucket.status]} transition-all duration-300
                             group-hover:opacity-80`}
                  style={{ width: `${(bucket.count / maxCount) * 100}%` }}
                />
              </div>
              <div className="w-16 text-right">
                <span className="text-sm font-mono">{bucket.count.toLocaleString()}</span>
                <span className="text-xs text-text-muted ml-1">
                  ({Math.round((bucket.count / totalNodes) * 100)}%)
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Freshness Timeline */}
      <div className="p-4 bg-surface rounded-lg border border-border">
        <div className="text-sm font-medium mb-3">Freshness Overview</div>
        <div className="flex gap-0.5 h-4 rounded overflow-hidden">
          {buckets.map((bucket) => (
            <div
              key={bucket.label}
              className={`${statusColors[bucket.status]} transition-all duration-300`}
              style={{ width: `${(bucket.count / totalNodes) * 100}%` }}
              title={`${bucket.label}: ${bucket.count} nodes`}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-text-muted">
          <span>Fresh</span>
          <span>Critical</span>
        </div>
      </div>
    </div>
  );
}

export default StalenessIndicator;
