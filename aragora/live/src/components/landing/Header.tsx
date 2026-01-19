'use client';

import Link from 'next/link';
import { AsciiBannerCompact } from '../AsciiBanner';
import { ThemeToggle } from '../ThemeToggle';
import { BackendSelector } from '../BackendSelector';
import { useSidebar } from '@/context/SidebarContext';

// Core navigation items - consistent with Sidebar.tsx
const coreNavItems = [
  { label: 'DEBATE', href: '/debate' },
  { label: 'DEBATES', href: '/debates' },
  { label: 'GAUNTLET', href: '/gauntlet' },
  { label: 'LEADERBOARD', href: '/leaderboard' },
  { label: 'AGENTS', href: '/agents' },
];

// Secondary items shown in expanded nav
const secondaryNavItems = [
  { label: 'MEMORY', href: '/memory' },
  { label: 'ANALYTICS', href: '/analytics' },
  { label: 'DOCS', href: '/developer' },
];

export function Header() {
  const { toggle } = useSidebar();

  return (
    <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Menu button + Logo */}
          <div className="flex items-center gap-3">
            <button
              onClick={toggle}
              className="p-2 text-acid-green hover:text-acid-cyan transition-colors"
              aria-label="Toggle navigation menu"
            >
              <span className="font-mono text-lg">☰</span>
            </button>
            <AsciiBannerCompact connected={true} />
          </div>

          {/* Desktop Navigation - Core links */}
          <nav className="hidden md:flex items-center gap-3 overflow-x-auto" aria-label="Main navigation">
            {coreNavItems.map(item => (
              <Link
                key={item.href}
                href={item.href}
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors whitespace-nowrap"
              >
                [{item.label}]
              </Link>
            ))}
            <span className="text-acid-green/30">|</span>
            {secondaryNavItems.map(item => (
              <Link
                key={item.href}
                href={item.href}
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors whitespace-nowrap"
              >
                [{item.label}]
              </Link>
            ))}
            <BackendSelector compact />
            <ThemeToggle />
          </nav>

          {/* Mobile - Theme toggle only (menu via hamburger) */}
          <div className="flex md:hidden items-center gap-2">
            <BackendSelector compact />
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}
