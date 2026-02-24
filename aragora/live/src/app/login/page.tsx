'use client';

import { Suspense, useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useAuth } from '@/context/AuthContext';
import { SocialLoginButtons } from '@/components/auth/SocialLoginButtons';
import { TopBar } from '@/components/layout/TopBar';
import { normalizeReturnUrl, RETURN_URL_STORAGE_KEY } from '@/utils/returnUrl';

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedReturnUrl =
    searchParams.get('returnUrl') ||
    searchParams.get('redirect') ||
    (typeof window !== 'undefined' ? sessionStorage.getItem(RETURN_URL_STORAGE_KEY) : null);
  const redirectTo = normalizeReturnUrl(requestedReturnUrl);
  const { login, isLoading: authLoading } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Save return URL to sessionStorage so the OAuth callback can use it too
  useEffect(() => {
    if (redirectTo && redirectTo !== '/') {
      sessionStorage.setItem(RETURN_URL_STORAGE_KEY, redirectTo);
    }
  }, [redirectTo]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    const result = await login(email, password);

    if (result.success) {
      sessionStorage.removeItem(RETURN_URL_STORAGE_KEY);
      router.push(redirectTo);
    } else {
      setError(result.error || 'Login failed');
    }

    setIsLoading(false);
  };

  return (
    <div className="w-full max-w-md">
      <div className="border border-acid-green/30 bg-surface/50 p-8">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-mono text-acid-green mb-2">SYSTEM ACCESS</h1>
          <p className="text-text-muted text-sm font-mono">
            Enter credentials to authenticate
          </p>
        </div>

        {error && (
          <div
            role="alert"
            className="mb-6 p-3 border border-warning/50 bg-warning/10 text-warning text-sm font-mono"
          >
            <p>{error}</p>
            {(error.toLowerCase().includes('invalid') ||
              error.toLowerCase().includes('failed')) && (
              <p className="mt-2 text-text-muted text-xs">
                Tip: Try signing in with Google or GitHub below
              </p>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-xs font-mono text-acid-cyan mb-2">
              EMAIL ADDRESS
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={isLoading || authLoading}
              autoComplete="email"
              aria-describedby={error ? 'login-error' : undefined}
              className="w-full px-4 py-3 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50 disabled:opacity-50 disabled:cursor-not-allowed"
              placeholder="user@example.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-xs font-mono text-acid-cyan mb-2">
              PASSWORD
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={isLoading || authLoading}
              autoComplete="current-password"
              aria-describedby={error ? 'login-error' : undefined}
              className="w-full px-4 py-3 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50 disabled:opacity-50 disabled:cursor-not-allowed"
              placeholder="********"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading || authLoading}
            className="w-full py-3 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'AUTHENTICATING...' : 'LOGIN'}
          </button>
        </form>

        <SocialLoginButtons mode="login" />

        <div className="mt-6 text-center">
          <Link
            href="/signup"
            className="text-sm font-mono text-acid-cyan hover:text-acid-green transition-colors"
          >
            No account? Sign up free
          </Link>
        </div>

        <div className="mt-8 pt-6 border-t border-acid-green/20">
          <div className="text-xs font-mono text-text-muted text-center">
            <p className="mb-2">CONNECTION STATUS: SECURE</p>
            <p className="text-acid-green/50">{'═'.repeat(30)}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Login page at /login (canonical URL).
 * Wrapped in Suspense for static export compatibility with useSearchParams.
 */
export default function LoginPage() {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <TopBar />
      <main className="pt-12">
        <Scanlines opacity={0.02} />
        <CRTVignette />

        <div className="min-h-screen bg-bg text-text relative z-10 flex flex-col">
          <div className="flex-1 flex items-center justify-center px-4 py-16">
            <Suspense fallback={
              <div className="text-[var(--acid-green)] font-mono animate-pulse">
                {'>'} LOADING...
              </div>
            }>
              <LoginForm />
            </Suspense>
          </div>
        </div>
      </main>
    </div>
  );
}
