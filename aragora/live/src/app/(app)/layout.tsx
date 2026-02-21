'use client';

import { usePathname } from 'next/navigation';
import { AppShell } from '@/components/layout';
import { TopBar } from '@/components/layout/TopBar';
import { useAuth } from '@/context/AuthContext';

const NO_SHELL_PREFIXES = ['/auth'];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || '';
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const hideShell = NO_SHELL_PREFIXES.some((prefix) => pathname.startsWith(prefix));

  // Onboarding is accessible at /onboarding but we don't force-redirect to it.
  // Forced redirects were causing crash loops for OAuth users whose
  // Zustand store defaults needsOnboarding=true before they can interact.

  // Unauthenticated users at root see LandingPage (which has its own nav) — skip AppShell
  if (!hideShell && pathname === '/' && !authLoading && !isAuthenticated) {
    return <>{children}</>;
  }

  // Show minimal loading indicator while auth resolves on root
  if (!hideShell && pathname === '/' && authLoading) {
    return (
      <div className="min-h-screen bg-[var(--bg)] flex items-center justify-center">
        <span className="font-mono text-sm text-[var(--text-muted)] animate-pulse">Loading...</span>
      </div>
    );
  }

  if (hideShell) {
    return (
      <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
        <TopBar />
        <main className="pt-12">{children}</main>
      </div>
    );
  }

  return <AppShell>{children}</AppShell>;
}
