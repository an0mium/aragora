'use client';

import { memo } from 'react';

interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
  animate?: boolean;
  /** Accessible label for the loading element */
  label?: string;
}

function SkeletonComponent({
  className = '',
  width,
  height,
  rounded = 'md',
  animate = true,
  label = 'Loading content',
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
      role="status"
      aria-busy="true"
      aria-label={label}
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

// Additional common skeletons

export function TextLineSkeleton({ width = '100%' }: { width?: string | number }) {
  return <Skeleton width={width} height={14} />;
}

export function ParagraphSkeleton({ lines = 3 }: { lines?: number }) {
  const widths = ['100%', '95%', '90%', '80%', '85%', '70%'];
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} width={widths[i % widths.length]} height={14} />
      ))}
    </div>
  );
}

export function CardSkeleton() {
  return (
    <div className="p-4 bg-surface border border-border rounded-lg space-y-3">
      <div className="flex items-center gap-3">
        <Skeleton width={40} height={40} rounded="full" />
        <div className="flex-1 space-y-2">
          <Skeleton width="60%" height={16} />
          <Skeleton width="40%" height={12} />
        </div>
      </div>
      <ParagraphSkeleton lines={2} />
      <div className="flex gap-2">
        <Skeleton width={80} height={28} rounded="md" />
        <Skeleton width={80} height={28} rounded="md" />
      </div>
    </div>
  );
}

export function TableRowSkeleton({ columns = 4 }: { columns?: number }) {
  return (
    <div className="flex items-center gap-4 p-2 border-b border-border">
      {Array.from({ length: columns }).map((_, i) => (
        <Skeleton key={i} width={`${100 / columns}%`} height={14} />
      ))}
    </div>
  );
}

export function TableSkeleton({ rows = 5, columns = 4 }: { rows?: number; columns?: number }) {
  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <div className="bg-surface/30 p-2 flex items-center gap-4">
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton key={i} width={`${100 / columns}%`} height={12} />
        ))}
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <TableRowSkeleton key={i} columns={columns} />
      ))}
    </div>
  );
}

export function AvatarSkeleton({ size = 40 }: { size?: number }) {
  return <Skeleton width={size} height={size} rounded="full" />;
}

export function ButtonSkeleton({ width = 100 }: { width?: number }) {
  return <Skeleton width={width} height={36} rounded="md" />;
}

export function BadgeSkeleton({ width = 60 }: { width?: number }) {
  return <Skeleton width={width} height={20} rounded="full" />;
}

export function DebateSkeleton() {
  return (
    <div className="p-4 bg-surface border border-border rounded-lg space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton width={80} height={20} rounded="sm" />
          <Skeleton width={60} height={20} rounded="full" />
        </div>
        <Skeleton width={100} height={12} />
      </div>
      <Skeleton width="80%" height={18} />
      <div className="flex flex-wrap gap-2">
        <BadgeSkeleton width={70} />
        <BadgeSkeleton width={50} />
        <BadgeSkeleton width={60} />
      </div>
      <div className="flex items-center gap-4 text-xs">
        <Skeleton width={100} height={10} />
        <Skeleton width={80} height={10} />
      </div>
    </div>
  );
}

export function DebateListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, i) => (
        <DebateSkeleton key={i} />
      ))}
    </div>
  );
}

export function InsightSkeleton() {
  return (
    <div className="p-3 bg-bg border border-border rounded-lg space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton width={20} height={20} rounded="sm" />
          <Skeleton width={100} height={14} />
        </div>
        <Skeleton width={40} height={16} rounded="sm" />
      </div>
      <Skeleton width="100%" height={12} />
      <Skeleton width="70%" height={12} />
    </div>
  );
}

export function InsightListSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <InsightSkeleton key={i} />
      ))}
    </div>
  );
}
