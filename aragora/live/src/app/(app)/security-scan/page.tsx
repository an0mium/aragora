'use client';

import { Suspense } from 'react';
import { SecurityScanWizard } from '@/components/codebase/SecurityScanWizard';

function LoadingFallback() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 bg-[var(--surface)] rounded w-1/3" />
      <div className="h-64 bg-[var(--surface)] rounded" />
    </div>
  );
}

export default function SecurityScanPage() {
  return (
    <div className="container mx-auto px-4 py-6 max-w-5xl">
      <Suspense fallback={<LoadingFallback />}>
        <SecurityScanWizard />
      </Suspense>
    </div>
  );
}
