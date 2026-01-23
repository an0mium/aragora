'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useAuth } from '@/context/AuthContext';

type Status = 'processing' | 'success' | 'error';

function OAuthCallbackContent() {
  const router = useRouter();
  const { setTokens } = useAuth();
  const [status, setStatus] = useState<Status>('processing');
  const [message, setMessage] = useState('Processing authentication...');

  useEffect(() => {
    const processCallback = async () => {
      // Debug: Log the full URL to help diagnose OAuth callback issues
      console.log('[OAuth Callback] Full URL:', window.location.href);
      console.log('[OAuth Callback] Hash:', window.location.hash);
      console.log('[OAuth Callback] Search:', window.location.search);

      // Parse query params directly from window.location (more reliable in static export)
      const urlParams = new URLSearchParams(window.location.search);

      // Check for account linking success
      const linked = urlParams.get('linked');
      if (linked) {
        setStatus('success');
        setMessage(`Successfully linked ${linked.charAt(0).toUpperCase() + linked.slice(1)} account`);
        setTimeout(() => router.push('/settings'), 1500);
        return;
      }

      // Parse tokens from URL query params (primary) or fragment (legacy fallback)
      let tokenString = window.location.search.substring(1); // Remove leading '?'
      console.log('[OAuth Callback] Query params:', tokenString ? `${tokenString.substring(0, 50)}...` : '(empty)');

      // Legacy fallback: Check hash fragment if query params are empty
      if (!tokenString) {
        tokenString = window.location.hash.substring(1);
        console.log('[OAuth Callback] Hash fragment:', tokenString ? `${tokenString.substring(0, 50)}...` : '(empty)');
      }

      if (!tokenString) {
        setStatus('error');
        setMessage('No authentication data received');
        console.error('[OAuth Callback] No query params or hash fragment with tokens found');
        return;
      }

      const params = new URLSearchParams(tokenString);
      const accessToken = params.get('access_token');
      const refreshToken = params.get('refresh_token');

      if (accessToken && refreshToken) {
        try {
          // Store tokens and wait for user profile to be fetched
          // This is async - we must await it before redirecting
          await setTokens(accessToken, refreshToken);
          setStatus('success');
          setMessage('Authentication successful');
          // Clear the hash from URL for security
          window.history.replaceState(null, '', window.location.pathname);
          // Small delay for user to see success message, then redirect
          setTimeout(() => router.push('/'), 500);
        } catch (err) {
          console.error('[OAuth Callback] Failed to set tokens:', err);
          setStatus('error');
          setMessage('Failed to complete authentication');
        }
      } else {
        setStatus('error');
        setMessage('Missing authentication tokens');
      }
    };

    processCallback();
  }, [router, setTokens]);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10 flex flex-col items-center justify-center">
        <div className="w-full max-w-md p-8">
          <div className="border border-acid-green/30 bg-surface/50 p-8 text-center">
            {/* Status Icon */}
            <div className="mb-6">
              {status === 'processing' && (
                <div className="inline-block animate-spin text-4xl text-acid-cyan">
                  &#x21BB;
                </div>
              )}
              {status === 'success' && (
                <div className="text-4xl text-acid-green">&#x2713;</div>
              )}
              {status === 'error' && (
                <div className="text-4xl text-warning">&#x2717;</div>
              )}
            </div>

            {/* Title */}
            <h1 className="text-xl font-mono text-acid-green mb-4">
              {status === 'processing' && 'AUTHENTICATING...'}
              {status === 'success' && 'ACCESS GRANTED'}
              {status === 'error' && 'AUTHENTICATION FAILED'}
            </h1>

            {/* Message */}
            <p className="text-text-muted text-sm font-mono mb-6">{message}</p>

            {/* Actions */}
            {status === 'error' && (
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
            )}

            {status === 'success' && (
              <p className="text-acid-cyan text-xs font-mono animate-pulse">
                Redirecting...
              </p>
            )}

            {status === 'processing' && (
              <div className="text-acid-green/50 text-xs font-mono">
                <p>{'═'.repeat(25)}</p>
                <p className="mt-2">Please wait...</p>
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
          <div className="border border-acid-green/30 bg-surface/50 p-8 text-center">
            <div className="mb-6">
              <div className="inline-block animate-spin text-4xl text-acid-cyan">
                &#x21BB;
              </div>
            </div>
            <h1 className="text-xl font-mono text-acid-green mb-4">AUTHENTICATING...</h1>
            <p className="text-text-muted text-sm font-mono mb-6">Processing authentication...</p>
            <div className="text-acid-green/50 text-xs font-mono">
              <p>{'═'.repeat(25)}</p>
              <p className="mt-2">Please wait...</p>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

export default function OAuthCallbackPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <OAuthCallbackContent />
    </Suspense>
  );
}
