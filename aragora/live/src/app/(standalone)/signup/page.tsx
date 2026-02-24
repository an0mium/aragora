'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { SocialLoginButtons } from '@/components/auth/SocialLoginButtons';
import { normalizeReturnUrl, RETURN_URL_STORAGE_KEY } from '@/utils/returnUrl';

export default function SignupPage() {
  const router = useRouter();
  const { register, isAuthenticated, isLoading: authLoading } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Determine post-signup destination:
  // 1. Stored return URL from a pre-login redirect (e.g. user was on a protected page)
  // 2. Onboarding wizard if there's a pending question
  // 3. Default: onboarding for new users
  const getPostSignupRoute = () => {
    if (typeof window !== 'undefined') {
      // Check for a stored redirect URL (from ProtectedRoute or login flow)
      const storedReturnUrl = sessionStorage.getItem(RETURN_URL_STORAGE_KEY);
      if (storedReturnUrl) {
        const normalized = normalizeReturnUrl(storedReturnUrl);
        // If there is a meaningful stored redirect, use it and clear it
        if (normalized !== '/') {
          sessionStorage.removeItem(RETURN_URL_STORAGE_KEY);
          return normalized;
        }
      }
      // If there is a pending onboarding question, go to onboarding
      if (sessionStorage.getItem('aragora_onboarding_question')) {
        return '/onboarding';
      }
    }
    return '/onboarding';
  };

  // If already authenticated, redirect to stored destination or onboarding
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      router.replace(getPostSignupRoute());
    }
  }, [authLoading, isAuthenticated, router]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setIsLoading(true);

    const result = await register(email, password, name || undefined);

    if (result.success) {
      router.push(getPostSignupRoute());
    } else {
      setError(result.error || 'Registration failed');
    }

    setIsLoading(false);
  };

  return (
    <main className="min-h-screen bg-bg text-text">
      {/* Minimal nav */}
      <nav className="border-b border-border bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link href="/" className="font-mono text-acid-green font-bold text-sm tracking-wider">
            ARAGORA
          </Link>
          <Link
            href="/login"
            className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
          >
            Already have an account? LOG IN
          </Link>
        </div>
      </nav>

      <div className="flex-1 flex items-center justify-center px-4 py-16">
        <div className="w-full max-w-md">
          <div className="border border-acid-green/30 bg-surface/50 p-8">
            {/* Header */}
            <div className="text-center mb-8">
              <h1 className="text-2xl font-mono text-acid-green mb-2">
                CREATE YOUR ACCOUNT
              </h1>
              <p className="text-text-muted text-sm font-mono">
                Get your first AI-powered decision in under 2 minutes
              </p>
            </div>

            {/* Error */}
            {error && (
              <div className="mb-6 p-3 border border-warning/50 bg-warning/10 text-warning text-sm font-mono">
                {error}
              </div>
            )}

            {/* Social login first -- less friction */}
            <SocialLoginButtons mode="register" />

            {/* Divider */}
            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-acid-green/20" />
              </div>
              <div className="relative flex justify-center">
                <span className="px-4 bg-surface/50 text-xs font-mono text-text-muted uppercase">
                  Or sign up with email
                </span>
              </div>
            </div>

            {/* Email/password form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="signup-name" className="block text-xs font-mono text-acid-cyan mb-1.5">
                  YOUR NAME
                </label>
                <input
                  id="signup-name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  autoComplete="name"
                  className="w-full px-4 py-2.5 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50"
                  placeholder="Jane Smith"
                />
              </div>

              <div>
                <label htmlFor="signup-email" className="block text-xs font-mono text-acid-cyan mb-1.5">
                  EMAIL ADDRESS *
                </label>
                <input
                  id="signup-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                  className="w-full px-4 py-2.5 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50"
                  placeholder="you@company.com"
                />
              </div>

              <div>
                <label htmlFor="signup-password" className="block text-xs font-mono text-acid-cyan mb-1.5">
                  PASSWORD *
                </label>
                <input
                  id="signup-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  autoComplete="new-password"
                  className="w-full px-4 py-2.5 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50"
                  placeholder="Min 8 characters"
                />
              </div>

              <div>
                <label htmlFor="signup-confirm" className="block text-xs font-mono text-acid-cyan mb-1.5">
                  CONFIRM PASSWORD *
                </label>
                <input
                  id="signup-confirm"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  autoComplete="new-password"
                  className="w-full px-4 py-2.5 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50"
                  placeholder="Repeat password"
                />
              </div>

              <button
                type="submit"
                disabled={isLoading || authLoading}
                className="w-full py-3 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed mt-2"
              >
                {isLoading ? 'CREATING ACCOUNT...' : 'CREATE ACCOUNT'}
              </button>
            </form>

            {/* Perks reminder */}
            <div className="mt-6 pt-5 border-t border-acid-green/20 space-y-2">
              {[
                'Free tier: 10 debates/month',
                'Real AI models (Claude, GPT, Gemini, Mistral)',
                'Audit-ready decision receipts',
              ].map((perk) => (
                <div key={perk} className="flex items-center gap-2 text-xs font-mono text-text-muted">
                  <span className="text-acid-green">+</span>
                  <span>{perk}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
