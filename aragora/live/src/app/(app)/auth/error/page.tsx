'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';

function OAuthErrorContent() {
  const searchParams = useSearchParams();
  const error = searchParams.get('error') || 'An unknown error occurred';

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10 flex flex-col items-center justify-center">
        <div className="w-full max-w-md p-8">
          <div className="border border-warning/30 bg-surface/50 p-8 text-center">
            {/* Error Icon */}
            <div className="mb-6">
              <div className="text-4xl text-warning">&#x26A0;</div>
            </div>

            {/* Title */}
            <h1 className="text-xl font-mono text-warning mb-4">
              AUTHENTICATION ERROR
            </h1>

            {/* Error Message */}
            <div className="mb-6 p-4 border border-warning/30 bg-warning/5">
              <p className="text-text-muted text-sm font-mono break-words">
                {decodeURIComponent(error)}
              </p>
            </div>

            {/* Actions */}
            <div className="space-y-3">
              <Link
                href="/auth/login"
                className="block w-full py-3 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors text-center"
              >
                TRY AGAIN
              </Link>
              <Link
                href="/"
                className="block w-full py-3 border border-acid-green/30 text-acid-cyan font-mono hover:border-acid-green transition-colors text-center"
              >
                RETURN HOME
              </Link>
            </div>

            {/* Help Text */}
            <div className="mt-8 pt-6 border-t border-acid-green/20">
              <p className="text-xs font-mono text-text-muted">
                If this error persists, please try:
              </p>
              <ul className="text-xs font-mono text-text-muted/70 mt-2 space-y-1">
                <li>- Clearing your browser cookies</li>
                <li>- Using a different browser</li>
                <li>- Checking your OAuth provider settings</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

function LoadingFallback() {
  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />
      <main className="min-h-screen bg-bg text-text relative z-10 flex flex-col items-center justify-center">
        <div className="w-full max-w-md p-8">
          <div className="border border-warning/30 bg-surface/50 p-8 text-center">
            <div className="mb-6">
              <div className="text-4xl text-warning">&#x26A0;</div>
            </div>
            <h1 className="text-xl font-mono text-warning mb-4">AUTHENTICATION ERROR</h1>
            <div className="mb-6 p-4 border border-warning/30 bg-warning/5">
              <p className="text-text-muted text-sm font-mono">Loading...</p>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

export default function OAuthErrorPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <OAuthErrorContent />
    </Suspense>
  );
}
