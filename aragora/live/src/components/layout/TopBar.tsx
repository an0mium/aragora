'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { useLayout } from '@/context/LayoutContext';
import { useCommandPalette } from '@/context/CommandPaletteContext';
import { useAuth } from '@/context/AuthContext';
import { ThemeToggle } from '@/components/ThemeToggle';

export function TopBar() {
  const { isMobile, toggleLeftSidebar, toggleRightSidebar, rightSidebarOpen } = useLayout();
  const { open: openCommandPalette } = useCommandPalette();
  const { isAuthenticated, user, logout } = useAuth();

  return (
    <header className="fixed top-0 left-0 right-0 h-12 bg-[var(--surface)] border-b border-[var(--border)] z-50 flex items-center px-3 gap-3">
      {/* Left section: Logo (clickable to toggle sidebar) + Title */}
      <div className="flex items-center gap-2">
        {/* Aragora logo - toggles sidebar on click */}
        <button
          onClick={toggleLeftSidebar}
          className="p-1.5 hover:bg-[var(--surface-elevated)] rounded transition-colors"
          aria-label="Toggle navigation menu"
        >
          <Image
            src="/aragora-logo.png"
            alt="Aragora"
            width={28}
            height={28}
            className="rounded"
          />
        </button>

        {/* Title link */}
        <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
          <span className="text-[var(--acid-green)] font-mono font-bold text-lg tracking-tight">
            ARAGORA
          </span>
        </Link>
      </div>

      {/* Center section: Search */}
      <div className="flex-1 flex justify-center">
        <button
          onClick={openCommandPalette}
          className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg)] border border-[var(--border)] rounded-md hover:border-[var(--acid-green)]/30 transition-colors max-w-md w-full"
        >
          <span className="text-[var(--text-muted)] font-mono text-sm">⌘</span>
          <span className="text-[var(--text-muted)] text-sm flex-1 text-left">
            Search or command...
          </span>
          <kbd className="hidden sm:inline-block px-1.5 py-0.5 bg-[var(--surface-elevated)] border border-[var(--border)] rounded text-xs text-[var(--text-muted)] font-mono">
            ⌘K
          </kbd>
        </button>
      </div>

      {/* Right section: Actions */}
      <div className="flex items-center gap-2">
        {/* Right sidebar toggle (desktop only) */}
        {!isMobile && (
          <button
            onClick={toggleRightSidebar}
            className={`p-2 hover:bg-[var(--surface-elevated)] rounded transition-colors ${
              rightSidebarOpen ? 'text-[var(--acid-green)]' : 'text-[var(--text-muted)]'
            }`}
            aria-label="Toggle context panel"
            title={rightSidebarOpen ? 'Hide context panel' : 'Show context panel'}
          >
            <span className="font-mono">⊞</span>
          </button>
        )}

        {/* Theme toggle */}
        <ThemeToggle />

        {/* Login/User menu */}
        {isAuthenticated ? (
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline text-xs text-[var(--text-muted)] truncate max-w-[120px]">
              {user?.email || user?.name}
            </span>
            <button
              onClick={() => logout?.()}
              className="p-2 hover:bg-[var(--surface-elevated)] rounded transition-colors"
              aria-label="Logout"
              title="Logout"
            >
              <span className="text-[var(--acid-green)] font-mono">●</span>
            </button>
          </div>
        ) : (
          <Link
            href="/auth/login"
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[var(--accent)]/10 hover:bg-[var(--accent)]/20 border border-[var(--accent)]/30 rounded-md transition-colors"
            title="Login"
          >
            <span className="text-[var(--accent)] font-mono text-sm">→</span>
            <span className="text-[var(--accent)] text-xs font-medium hidden sm:inline">Login</span>
          </Link>
        )}
      </div>
    </header>
  );
}
