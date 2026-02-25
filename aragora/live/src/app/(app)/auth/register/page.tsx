'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

/**
 * /auth/register redirects to /signup (the canonical registration page).
 * Keeps old links working without maintaining two registration flows.
 */
export default function RegisterRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/signup');
  }, [router]);

  return (
    <main className="min-h-screen bg-bg text-text flex items-center justify-center">
      <div className="font-mono text-acid-green animate-pulse text-sm">
        Redirecting to sign up...
      </div>
    </main>
  );
}
