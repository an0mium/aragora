'use client';

import { useEffect } from 'react';
import { getCrashReporter } from '@/lib/crash-reporter';

/**
 * Global error page for root layout errors
 *
 * This catches errors that occur in the root layout itself.
 * It must provide its own <html> and <body> tags since the root layout may have failed.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Global app error:', error);
    const reporter = getCrashReporter();
    const accepted = reporter.capture(error, {
      componentName: 'next-global-error-boundary',
    });
    if (accepted) {
      reporter.flush();
    }
  }, [error]);

  return (
    <html lang="en" className="dark">
      <body className="bg-[#0a0a0a] text-[#e0e0e0]" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
        <div className="min-h-screen flex items-center justify-center p-4">
          <div className="max-w-2xl w-full border border-[#ff0040] bg-[#1a1a2e] p-6">
            <div className="text-[#ff0040] text-center mb-6">
              <div className="text-4xl font-bold mb-2">
                ARAGORA // CRITICAL ERROR
              </div>
              <div className="text-[#ffff00] text-sm">
                System encountered a fatal error
              </div>
            </div>

            <div className="bg-[#0a0a0a] border border-[#333] p-4 mb-4 text-[#888] text-sm overflow-x-auto">
              <div className="mb-2 text-[#ff0040] font-bold">
                {'>'} {error.message || 'Fatal system error'}
              </div>
              {error.digest && (
                <div className="text-[#666] text-xs mt-2">
                  Error digest: {error.digest}
                </div>
              )}
            </div>

            <button
              onClick={reset}
              className="w-full border border-[#00ff88] text-[#00ff88] py-3 px-4 hover:bg-[#00ff88] hover:text-[#0a0a0a] transition-colors font-bold text-lg mb-4"
            >
              {'>'} REBOOT_SYSTEM
            </button>

            <div className="text-[#666] text-xs text-center">
              If problem persists, please report this issue
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
