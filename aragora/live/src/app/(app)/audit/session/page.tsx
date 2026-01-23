'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';

function AuditSessionContent() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get('id');

  if (!sessionId) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-xl font-mono text-acid-cyan mb-4">No Session ID Provided</h1>
          <p className="text-text-muted mb-6">Please provide a session ID to view audit details.</p>
          <Link href="/audit" className="text-acid-green hover:underline">
            &larr; Back to Audit Sessions
          </Link>
        </div>
      </div>
    );
  }

  // Load the session detail dynamically
  // This is a placeholder - the actual implementation would fetch session data
  return (
    <div className="min-h-screen bg-bg p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Link href="/audit" className="text-acid-cyan hover:underline text-sm">
            &larr; Back to Audit Sessions
          </Link>
        </div>
        <h1 className="text-2xl font-mono text-acid-green mb-4">
          Audit Session: {sessionId}
        </h1>
        <p className="text-text-muted">
          Loading session details...
        </p>
        {/* Session details would be loaded here */}
      </div>
    </div>
  );
}

export default function AuditSessionPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="animate-spin text-4xl text-acid-cyan">&#x21BB;</div>
      </div>
    }>
      <AuditSessionContent />
    </Suspense>
  );
}
