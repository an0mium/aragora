'use client';

import { Suspense } from 'react';
import AuditSessionDetail from './AuditSessionDetail';

function LoadingState() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-muted font-mono animate-pulse">LOADING SESSION...</div>
    </div>
  );
}

export default function AuditViewPage() {
  return (
    <Suspense fallback={<LoadingState />}>
      <AuditSessionDetail />
    </Suspense>
  );
}
