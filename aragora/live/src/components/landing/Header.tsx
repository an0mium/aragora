'use client';

import Link from 'next/link';
import { useTheme } from '@/context/ThemeContext';
import { ThemeSelector } from './ThemeSelector';

export function Header() {
  const { theme } = useTheme();

  return (
    <header
      className="sticky top-0 z-50 backdrop-blur-sm"
      style={{
        backgroundColor: theme === 'dark' ? 'rgba(10,10,10,0.85)' : theme === 'professional' ? 'rgba(255,255,255,0.85)' : 'rgba(250,249,247,0.85)',
        borderBottom: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Wordmark */}
        <Link href="/landing" className="flex items-center">
          <span
            className="font-bold tracking-wider"
            style={{
              color: 'var(--accent)',
              fontSize: theme === 'dark' ? '14px' : '16px',
              fontFamily: theme === 'dark' ? "'JetBrains Mono', monospace" : "'Inter', system-ui, sans-serif",
              letterSpacing: theme === 'dark' ? '0.15em' : '0.08em',
            }}
          >
            {theme === 'dark' ? '> ARAGORA' : 'Aragora'}
          </span>
        </Link>

        {/* Nav links + Theme selector */}
        <div className="flex items-center gap-6">
          <nav className="hidden sm:flex items-center gap-5">
            <a
              href="#how-it-works"
              className="text-sm transition-colors hover:opacity-80"
              style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              How it works
            </a>
            <a
              href="#pricing"
              className="text-sm transition-colors hover:opacity-80"
              style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              Pricing
            </a>
            <Link
              href="/login"
              className="text-sm transition-colors hover:opacity-80"
              style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
            >
              Log in
            </Link>
          </nav>
          <ThemeSelector />
        </div>
      </div>
    </header>
  );
}
