'use client';

import { Skeleton, TableSkeleton, CardSkeleton, BadgeSkeleton } from '@/components/Skeleton';

export default function MLLoading() {
  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton width={220} height={24} label="Loading page title" />
          <Skeleton width={350} height={14} label="Loading page description" />
        </div>
        <div className="flex gap-2">
          <Skeleton width={120} height={36} rounded="md" label="Loading button" />
          <Skeleton width={120} height={36} rounded="md" label="Loading button" />
        </div>
      </div>

      {/* Model Stats */}
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="p-4 bg-surface border border-border rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Skeleton width={20} height={20} rounded="sm" />
              <Skeleton width={80} height={12} />
            </div>
            <Skeleton width={60} height={28} />
            <Skeleton width={100} height={10} className="mt-2" />
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-4 border-b border-border pb-2">
        <Skeleton width={80} height={28} rounded="md" />
        <Skeleton width={100} height={28} rounded="md" />
        <Skeleton width={90} height={28} rounded="md" />
        <Skeleton width={70} height={28} rounded="md" />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Models List */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <Skeleton width={120} height={16} />
            <BadgeSkeleton width={60} />
          </div>
          {[1, 2, 3].map((i) => (
            <div key={i} className="p-4 bg-surface border border-border rounded-lg">
              <div className="flex justify-between items-start mb-3">
                <div className="space-y-1">
                  <Skeleton width={150} height={16} />
                  <Skeleton width={100} height={12} />
                </div>
                <Skeleton width={60} height={24} rounded="full" />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Skeleton width={60} height={10} className="mb-1" />
                  <Skeleton width={40} height={14} />
                </div>
                <div>
                  <Skeleton width={50} height={10} className="mb-1" />
                  <Skeleton width={45} height={14} />
                </div>
                <div>
                  <Skeleton width={55} height={10} className="mb-1" />
                  <Skeleton width={35} height={14} />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Training Jobs */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <Skeleton width={100} height={16} />
            <Skeleton width={100} height={32} rounded="md" />
          </div>
          <TableSkeleton rows={5} columns={4} />
        </div>
      </div>
    </div>
  );
}
