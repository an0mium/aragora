'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useAuth } from '@/context/AuthContext';
import { logger } from '@/utils/logger';

type Status = 'processing' | 'success' | 'error';

function OAuthCallbackContent() {
  const router = useRouter();
  const { setTokens } = useAuth();
  const [status, setStatus] = useState<Status>('processing');
  const [message, setMessage] = useState('Processing authentication...');

  useEffect(() => {
    const processCallback = async () => {
      // Debug: Log the full URL to help diagnose OAuth callback issues
      logger.debug('[OAuth Callback] Full URL:', window.location.href);
      logger.debug('[OAuth Callback] Hash:', window.location.hash);
      logger.debug('[OAuth Callback] Search:', window.location.search);

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
      logger.debug('[OAuth Callback] Query params:', tokenString ? `${tokenString.substring(0, 50)}...` : '(empty)');

      // Legacy fallback: Check hash fragment if query params are empty
      if (!tokenString) {
        tokenString = window.location.hash.substring(1);
        logger.debug('[OAuth Callback] Hash fragment:', tokenString ? `${tokenString.substring(0, 50)}...` : '(empty)');
      }

      if (!tokenString) {
        setStatus('error');
        setMessage('No authentication data received');
        logger.error('[OAuth Callback] No query params or hash fragment with tokens found');
        return;
      }

      const params = new URLSearchParams(tokenString);
      const accessToken = params.get('access_token');
      const refreshToken = params.get('refresh_token');

      if (accessToken && refreshToken) {
        try {
          logger.debug('[OAuth Callback] Calling setTokens with access_token:', accessToken.substring(0, 20) + '...');
          // Store tokens and wait for user profile to be fetched
          // This is async - we must await it before redirecting
          await setTokens(accessToken, refreshToken);
          logger.debug('[OAuth Callback] setTokens completed successfully');
          setStatus('success');
          setMessage('Authentication successful');
          // Clear the hash from URL for security
          window.history.replaceState(null, '', window.location.pathname);

          // Verify tokens are stored before redirect
          const storedTokens = localStorage.getItem('aragora_tokens');
          const storedUser = localStorage.getItem('aragora_user');
          logger.debug('[OAuth Callback] Pre-redirect check - tokens:', !!storedTokens, 'user:', !!storedUser);

          // Slightly longer delay to ensure state is settled before navigation
          setTimeout(() => {
            logger.debug('[OAuth Callback] Redirecting to home...');
            router.push('/');
          }, 750);
        } catch (err) {
          logger.error('[OAuth Callback] Failed to set tokens:', err);
          setStatus('error');
          // Provide more descriptive error messages
          if (err instanceof Error) {
            if (err.message.includes('Invalid tokens')) {
              setMessage('OAuth tokens were rejected by the server. Please try logging in again.');
            } else if (err.message.includes('401')) {
              setMessage('Authentication failed. Please try logging in again.');
            } else if (err.message.includes('Network error') || err.message.includes('Server error')) {
              setMessage(err.message + ' Your tokens have been saved.');
            } else {
              setMessage(err.message || 'Failed to complete authentication');
            }
          } else {
            setMessage('Failed to complete authentication');
          }
        }
      } else {
        setStatus('error');
        setMessage('Missing authentication tokens');
        logger.error('[OAuth Callback] Tokens missing from URL params. access_token:', !!accessToken, 'refresh_token:', !!refreshToken);
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
