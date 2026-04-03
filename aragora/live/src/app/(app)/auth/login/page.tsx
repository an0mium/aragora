'use client';

import { Suspense, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

function LoginRedirectContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const query = searchParams.toString();
    router.replace(query ? `/login?${query}` : '/login');
  }, [router, searchParams]);

  return (
    <main className="min-h-screen bg-bg text-text flex items-center justify-center">
      <div className="font-mono text-acid-green animate-pulse text-sm">
        Redirecting to login...
      </div>
    </main>
  );
}

/**
 * /auth/login is a compatibility shim for legacy links.
 * /login is the canonical sign-in page.
 */
export default function LoginRedirectPage() {
  return (
    <Suspense
      fallback={
        <main className="min-h-screen bg-bg text-text flex items-center justify-center">
          <div className="font-mono text-acid-green animate-pulse text-sm">
            Redirecting to login...
          </div>
        </main>
      }
    >
      <LoginRedirectContent />
    </Suspense>
  );
}
