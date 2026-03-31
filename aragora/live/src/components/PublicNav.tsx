'use client';

import Link from 'next/link';
import { Logo } from '@/components/Logo';
import { ThemeSelector } from '@/components/landing/ThemeSelector';

interface PublicNavProps {
  /** Page-specific controls rendered between logo and right section */
  children?: React.ReactNode;
  /** Max width of the nav content (default: 72rem / 1152px) */
  maxWidth?: string;
}

export function PublicNav({ children, maxWidth = '72rem' }: PublicNavProps) {
  return (
    <nav className="border-b border-[var(--border)] bg-[var(--surface)]/80 backdrop-blur-sm sticky top-0 z-50">
      <div
        className="mx-auto px-4 py-3 flex items-center justify-between gap-4"
        style={{ maxWidth }}
      >
        {/* Branded logo — matches landing Header */}
        <div className="flex items-center gap-3 shrink-0">
          <Logo size="lg" pixelSize={28} />
          <Link href="/landing" className="flex items-center">
            <span
              className="font-bold"
              style={{
                color: 'var(--accent)',
                fontSize: '14px',
                fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: '0.15em',
              }}
            >
              {'>'} ARAGORA
            </span>
          </Link>
        </div>

        {/* Page-specific controls slot */}
        {children}

        {/* Standard right section */}
        <div className="flex items-center gap-3 shrink-0">
          <ThemeSelector />
          <Link
            href="/signup"
            className="hidden sm:inline-flex rounded-full bg-[var(--acid-green)] text-xs font-semibold transition-opacity hover:opacity-90"
            style={{ color: 'var(--bg)', padding: '8px 14px' }}
          >
            Get started
          </Link>
        </div>
      </div>
    </nav>
  );
}
