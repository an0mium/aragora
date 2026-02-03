'use client';

import { Suspense, useState, useEffect } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';

/** Map raw backend error strings to user-friendly messages. */
function friendlyMessage(raw: string): { message: string; isTransient: boolean } {
  const decoded = decodeURIComponent(raw);
  const lower = decoded.toLowerCase();

  // Transient database/connection errors - auto-retry is appropriate
  if (lower.includes('interfaceerror') || lower.includes('connectiondoesnotexisterror')) {
    return {
      message: 'A temporary connection issue occurred. Retrying automatically\u2026',
      isTransient: true,
    };
  }
  if (lower.includes('timeouterror') || lower.includes('connectionrefusederror')) {
    return {
      message: 'The server is temporarily unreachable. Retrying automatically\u2026',
      isTransient: true,
    };
  }

  // Permanent errors - show a clear explanation
  if (lower.includes('invalid or expired state')) {
    return {
      message: 'Your login session expired. Please try signing in again.',
      isTransient: false,
    };
  }
  if (lower.includes('failed to exchange authorization code')) {
    return {
      message: 'The login provider returned an invalid response. Please try again.',
      isTransient: false,
    };
  }
  if (lower.includes('jwt') || lower.includes('secret not configured')) {
    return {
      message: 'Server configuration error. Please contact the administrator.',
      isTransient: false,
    };
  }
  if (lower.includes('user service unavailable')) {
    return {
      message: 'The authentication service is currently unavailable. Please try again shortly.',
      isTransient: true,
    };
  }

  // Fallback - show the decoded message as-is
  return { message: decoded, isTransient: false };
}

function OAuthErrorContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const rawError = searchParams.get('error') || 'An unknown error occurred';

  const { message, isTransient } = friendlyMessage(rawError);
  const [countdown, setCountdown] = useState(isTransient ? 3 : 0);

  useEffect(() => {
    if (!isTransient || countdown <= 0) return;

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          // Retry by navigating back to login
          router.push('/auth/login');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isTransient, countdown, router]);

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
              {isTransient ? 'CONNECTION ISSUE' : 'AUTHENTICATION ERROR'}
            </h1>

            {/* User-friendly Error Message */}
            <div className="mb-6 p-4 border border-warning/30 bg-warning/5">
              <p className="text-text-muted text-sm font-mono break-words">
                {message}
              </p>
            </div>

            {/* Auto-retry countdown for transient errors */}
            {isTransient && countdown > 0 && (
              <p className="text-acid-cyan text-xs font-mono mb-4 animate-pulse">
                Retrying in {countdown}s...
              </p>
            )}

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
            {!isTransient && (
              <div className="mt-8 pt-6 border-t border-acid-green/20">
                <p className="text-xs font-mono text-text-muted">
                  If this error persists, please try:
                </p>
                <ul className="text-xs font-mono text-text-muted/70 mt-2 space-y-1">
                  <li>- Clearing your browser cookies</li>
                  <li>- Using a different browser</li>
                  <li>- Waiting a few minutes and trying again</li>
                </ul>
              </div>
            )}
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
