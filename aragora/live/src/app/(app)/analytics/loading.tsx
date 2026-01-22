'use client';

import { Skeleton, StatsSkeleton } from '@/components/Skeleton';

export default function AnalyticsLoading() {
  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton width={140} height={24} label="Loading page title" />
          <Skeleton width={300} height={14} label="Loading page description" />
        </div>
        <div className="flex gap-2">
          <Skeleton width={120} height={36} rounded="md" />
          <Skeleton width={80} height={36} rounded="md" />
        </div>
      </div>

      {/* Date Range Selector */}
      <div className="flex gap-4 items-center">
        <Skeleton width={200} height={40} rounded="md" />
        <div className="flex gap-2">
          <Skeleton width={60} height={32} rounded="md" />
          <Skeleton width={60} height={32} rounded="md" />
          <Skeleton width={60} height={32} rounded="md" />
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="p-4 bg-surface border border-border rounded-lg">
            <Skeleton width={80} height={12} className="mb-2" />
            <Skeleton width={70} height={32} />
            <div className="flex items-center gap-2 mt-2">
              <Skeleton width={40} height={14} rounded="sm" />
              <Skeleton width={60} height={10} />
            </div>
          </div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Main Chart */}
        <div className="p-4 bg-surface border border-border rounded-lg">
          <div className="flex justify-between items-center mb-4">
            <Skeleton width={120} height={16} />
            <div className="flex gap-2">
              <Skeleton width={60} height={24} rounded="sm" />
              <Skeleton width={60} height={24} rounded="sm" />
            </div>
          </div>
          <Skeleton width="100%" height={200} rounded="md" label="Loading chart" />
        </div>

        {/* Secondary Chart */}
        <div className="p-4 bg-surface border border-border rounded-lg">
          <div className="flex justify-between items-center mb-4">
            <Skeleton width={140} height={16} />
            <Skeleton width={80} height={24} rounded="sm" />
          </div>
          <Skeleton width="100%" height={200} rounded="md" label="Loading chart" />
        </div>
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2">
          <div className="p-4 bg-surface border border-border rounded-lg">
            <Skeleton width={160} height={16} className="mb-4" />
            <Skeleton width="100%" height={150} rounded="md" />
          </div>
        </div>
        <StatsSkeleton />
      </div>
    </div>
  );
}
