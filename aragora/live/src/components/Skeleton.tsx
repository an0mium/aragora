'use client';

import { memo } from 'react';

interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
  animate?: boolean;
}

function SkeletonComponent({
  className = '',
  width,
  height,
  rounded = 'md',
  animate = true,
}: SkeletonProps) {
  const roundedClass = {
    none: '',
    sm: 'rounded-sm',
    md: 'rounded',
    lg: 'rounded-lg',
    full: 'rounded-full',
  }[rounded];

  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  return (
    <div
      className={`bg-surface/50 ${roundedClass} ${animate ? 'animate-pulse' : ''} ${className}`}
      style={style}
    />
  );
}

export const Skeleton = memo(SkeletonComponent);

// Pre-built skeleton components for common use cases
export function AgentRankingSkeleton() {
  return (
    <div className="flex items-center gap-3 p-2 bg-bg border border-border rounded-lg">
      <Skeleton width={28} height={28} rounded="full" />
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <Skeleton width={100} height={14} />
          <Skeleton width={40} height={14} />
          <Skeleton width={30} height={16} rounded="sm" />
        </div>
        <Skeleton width={120} height={10} />
      </div>
      <Skeleton width={50} height={12} />
    </div>
  );
}

export function MatchSkeleton() {
  return (
    <div className="p-2 bg-bg border border-border rounded-lg space-y-2">
      <div className="flex items-center justify-between">
        <Skeleton width={120} height={14} />
        <Skeleton width={60} height={18} rounded="sm" />
      </div>
      <div className="flex gap-2">
        <Skeleton width={80} height={12} />
        <Skeleton width={80} height={12} />
      </div>
      <Skeleton width={140} height={10} />
    </div>
  );
}

export function StatCardSkeleton() {
  return (
    <div className="p-3 bg-bg border border-border rounded-lg">
      <Skeleton width={60} height={10} className="mb-2" />
      <Skeleton width={50} height={24} />
    </div>
  );
}

export function IntrospectionSkeleton() {
  return (
    <div className="p-3 bg-bg border border-border rounded-lg space-y-3">
      <div className="flex items-center justify-between">
        <Skeleton width={100} height={16} />
        <Skeleton width={80} height={18} rounded="sm" />
      </div>
      <Skeleton width="100%" height={32} />
      <div className="grid grid-cols-3 gap-2">
        <div className="space-y-1">
          <Skeleton width={60} height={10} />
          <Skeleton width="80%" height={10} />
        </div>
        <div className="space-y-1">
          <Skeleton width={70} height={10} />
          <Skeleton width="70%" height={10} />
        </div>
        <div className="space-y-1">
          <Skeleton width={65} height={10} />
          <Skeleton width="75%" height={10} />
        </div>
      </div>
    </div>
  );
}

export function LeaderboardSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <AgentRankingSkeleton key={i} />
      ))}
    </div>
  );
}

export function MatchesSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <MatchSkeleton key={i} />
      ))}
    </div>
  );
}

export function StatsSkeleton() {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <StatCardSkeleton />
        <StatCardSkeleton />
        <StatCardSkeleton />
        <StatCardSkeleton />
      </div>
      <div className="p-3 bg-bg border border-border rounded-lg">
        <Skeleton width={100} height={10} className="mb-3" />
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Skeleton width={40} height={10} />
            <Skeleton width="60%" height={8} />
            <Skeleton width={20} height={10} />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton width={40} height={10} />
            <Skeleton width="45%" height={8} />
            <Skeleton width={20} height={10} />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton width={40} height={10} />
            <Skeleton width="30%" height={8} />
            <Skeleton width={20} height={10} />
          </div>
        </div>
      </div>
    </div>
  );
}

export function IntrospectionListSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, i) => (
        <IntrospectionSkeleton key={i} />
      ))}
    </div>
  );
}
