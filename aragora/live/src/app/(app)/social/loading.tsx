'use client';

import { Skeleton, CardSkeleton, AvatarSkeleton, ParagraphSkeleton } from '@/components/Skeleton';

export default function SocialLoading() {
  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton width={180} height={24} label="Loading page title" />
          <Skeleton width={250} height={14} label="Loading page description" />
        </div>
        <Skeleton width={120} height={36} rounded="md" label="Loading button" />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="p-4 bg-surface border border-border rounded-lg">
            <Skeleton width={100} height={12} className="mb-2" />
            <Skeleton width={50} height={24} />
          </div>
        ))}
      </div>

      {/* Activity Feed & Sidebar */}
      <div className="grid grid-cols-3 gap-6">
        {/* Feed */}
        <div className="col-span-2 space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="p-4 bg-surface border border-border rounded-lg">
              <div className="flex items-start gap-3">
                <AvatarSkeleton size={40} />
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2">
                    <Skeleton width={120} height={14} />
                    <Skeleton width={60} height={10} />
                  </div>
                  <ParagraphSkeleton lines={2} />
                  <div className="flex gap-4 pt-2">
                    <Skeleton width={40} height={20} />
                    <Skeleton width={40} height={20} />
                    <Skeleton width={40} height={20} />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <div className="p-4 bg-surface border border-border rounded-lg">
            <Skeleton width={100} height={14} className="mb-3" />
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-2 py-2">
                <AvatarSkeleton size={32} />
                <div className="flex-1">
                  <Skeleton width={100} height={12} />
                </div>
                <Skeleton width={50} height={24} rounded="md" />
              </div>
            ))}
          </div>
          <CardSkeleton />
        </div>
      </div>
    </div>
  );
}
