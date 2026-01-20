'use client';

import { Skeleton, InsightListSkeleton, CardSkeleton, BadgeSkeleton } from '@/components/Skeleton';

export default function InsightsLoading() {
  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton width={140} height={24} label="Loading page title" />
          <Skeleton width={320} height={14} label="Loading page description" />
        </div>
        <div className="flex gap-2">
          <Skeleton width={100} height={36} rounded="md" />
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-4 border-b border-border pb-2">
        <Skeleton width={60} height={28} rounded="md" />
        <Skeleton width={80} height={28} rounded="md" />
        <Skeleton width={70} height={28} rounded="md" />
        <Skeleton width={90} height={28} rounded="md" />
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="p-4 bg-surface border border-border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <Skeleton width={100} height={12} />
              <BadgeSkeleton width={50} />
            </div>
            <Skeleton width={60} height={28} />
          </div>
        ))}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-3 gap-6">
        {/* Insights List */}
        <div className="col-span-2 space-y-4">
          <div className="flex justify-between items-center">
            <Skeleton width={120} height={16} />
            <div className="flex gap-2">
              <Skeleton width={80} height={32} rounded="md" />
              <Skeleton width={80} height={32} rounded="md" />
            </div>
          </div>
          <InsightListSkeleton count={5} />
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <div className="p-4 bg-surface border border-border rounded-lg">
            <Skeleton width={100} height={14} className="mb-3" />
            <div className="space-y-2">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="flex items-center gap-2">
                  <Skeleton width={12} height={12} rounded="full" />
                  <Skeleton width="80%" height={10} />
                </div>
              ))}
            </div>
          </div>
          <CardSkeleton />
        </div>
      </div>
    </div>
  );
}
