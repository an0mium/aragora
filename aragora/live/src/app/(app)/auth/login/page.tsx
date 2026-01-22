'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { useAuth } from '@/context/AuthContext';
import { SocialLoginButtons } from '@/components/auth/SocialLoginButtons';

export default function LoginPage() {
  const router = useRouter();
  const { login, isLoading: authLoading } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    const result = await login(email, password);

    if (result.success) {
      router.push('/');
    } else {
      setError(result.error || 'Login failed');
    }

    setIsLoading(false);
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10 flex flex-col">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="/auth/register"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [CREATE ACCOUNT]
              </Link>
            </div>
          </div>
        </header>

        {/* Login Form */}
        <div className="flex-1 flex items-center justify-center px-4 py-16">
          <div className="w-full max-w-md">
            <div className="border border-acid-green/30 bg-surface/50 p-8">
              <div className="text-center mb-8">
                <h1 className="text-2xl font-mono text-acid-green mb-2">SYSTEM ACCESS</h1>
                <p className="text-text-muted text-sm font-mono">
                  Enter credentials to authenticate
                </p>
              </div>

              {error && (
                <div className="mb-6 p-3 border border-warning/50 bg-warning/10 text-warning text-sm font-mono">
                  {error}
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
                    autoComplete="email"
                    className="w-full px-4 py-3 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50"
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
                    autoComplete="current-password"
                    className="w-full px-4 py-3 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green placeholder-text-muted/50"
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

              {/* Social Login Options */}
              <SocialLoginButtons mode="login" />

              <div className="mt-6 text-center">
                <Link
                  href="/auth/register"
                  className="text-sm font-mono text-acid-cyan hover:text-acid-green transition-colors"
                >
                  No account? Register here
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
        </div>
      </main>
    </>
  );
}
