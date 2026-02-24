'use client';

import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { AppShell } from '@/components/layout';
import { TopBar } from '@/components/layout/TopBar';
import { useAuth } from '@/context/AuthContext';
import { useBackend } from '@/components/BackendSelector';

const NO_SHELL_PREFIXES = ['/auth'];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || '';
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const { config: backendConfig } = useBackend();
  const hideShell = NO_SHELL_PREFIXES.some((prefix) => pathname.startsWith(prefix));

  // Prevent hydration mismatch: server renders loading spinner, client must
  // render the same tree until mounted to avoid React Error #185.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Demo mode detection from backend health endpoint
  const [isDemoMode, setIsDemoMode] = useState(false);
  useEffect(() => {
    if (isAuthenticated) return;
    const controller = new AbortController();
    fetch(`${backendConfig.api}/api/health`, { signal: controller.signal })
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.demo_mode || data?.mode === 'demo' || data?.offline) {
          setIsDemoMode(true);
        }
      })
      .catch(() => { /* backend not available */ });
    return () => controller.abort();
  }, [backendConfig.api, isAuthenticated]);

  // Onboarding is accessible at /onboarding but we don't force-redirect to it.
  // Forced redirects were causing crash loops for OAuth users whose
  // Zustand store defaults needsOnboarding=true before they can interact.

  // Until client has mounted, render the same loading spinner the server rendered
  // to avoid hydration mismatch (React Error #185) from auth-dependent branching.
  if (!mounted) {
    return (
      <div className="min-h-screen bg-[var(--bg)] flex items-center justify-center">
        <span className="font-mono text-sm text-[var(--text-muted)] animate-pulse">Loading...</span>
      </div>
    );
  }

  // Unauthenticated users at root see LandingPage (which has its own nav) — skip AppShell
  // In demo mode, show AppShell so sidebar navigation works
  if (!hideShell && pathname === '/' && !authLoading && !isAuthenticated && !isDemoMode) {
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
