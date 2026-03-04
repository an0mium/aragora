'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

/**
 * Root page redirect.
 *
 * In runtime mode, next.config.js redirects `/` → `/landing/` server-side,
 * so this page never renders. In static-export mode (Docker, Cloudflare Pages),
 * this client component catches the fallback and redirects based on auth state.
 */
export default function RootPage() {
  const router = useRouter();

  useEffect(() => {
    try {
      const raw = localStorage.getItem('aragora_tokens');
      if (raw) {
        const tokens = JSON.parse(raw);
        if (tokens?.access_token) {
          router.replace('/demo');
          return;
        }
      }
    } catch {
      // Invalid JSON in localStorage — treat as unauthenticated
    }
    router.replace('/landing');
  }, [router]);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'var(--bg)',
        color: 'var(--text-muted)',
        fontFamily: 'var(--font-landing)',
        fontSize: '14px',
      }}
    >
      Loading...
    </div>
  );
}
