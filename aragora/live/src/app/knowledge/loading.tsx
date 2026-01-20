'use client';

import { Skeleton, CardSkeleton, TableSkeleton, BadgeSkeleton } from '@/components/Skeleton';

export default function KnowledgeLoading() {
  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton width={200} height={24} label="Loading page title" />
          <Skeleton width={300} height={14} label="Loading page description" />
        </div>
        <div className="flex gap-2">
          <Skeleton width={100} height={36} rounded="md" label="Loading button" />
          <Skeleton width={100} height={36} rounded="md" label="Loading button" />
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="p-4 bg-surface border border-border rounded-lg">
            <Skeleton width={80} height={12} className="mb-2" />
            <Skeleton width={60} height={28} />
          </div>
        ))}
      </div>

      {/* Search & Filters */}
      <div className="flex gap-4 items-center">
        <Skeleton width={300} height={40} rounded="md" label="Loading search" />
        <div className="flex gap-2">
          <BadgeSkeleton width={80} />
          <BadgeSkeleton width={60} />
          <BadgeSkeleton width={70} />
        </div>
      </div>

      {/* Knowledge Graph / Table */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <TableSkeleton rows={8} columns={4} />
        </div>
        <div className="space-y-4">
          <CardSkeleton />
          <CardSkeleton />
        </div>
      </div>
    </div>
  );
}
