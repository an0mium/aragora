'use client';

import { Suspense } from 'react';
import { QBODashboard } from '@/components/accounting/QBODashboard';

function LoadingFallback() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 bg-[var(--surface)] rounded w-1/3" />
      <div className="grid grid-cols-4 gap-4">
        <div className="h-24 bg-[var(--surface)] rounded" />
        <div className="h-24 bg-[var(--surface)] rounded" />
        <div className="h-24 bg-[var(--surface)] rounded" />
        <div className="h-24 bg-[var(--surface)] rounded" />
      </div>
      <div className="h-64 bg-[var(--surface)] rounded" />
    </div>
  );
}

export default function AccountingPage() {
  return (
    <div className="container mx-auto px-4 py-6 max-w-6xl">
      <Suspense fallback={<LoadingFallback />}>
        <QBODashboard />
      </Suspense>
    </div>
  );
}
