'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  useEffect(() => {
    console.error('App error:', error);
  }, [error]);

  // Detect hydration mismatch errors (React Error #418, #423, #425)
  const isHydrationError =
    error.message?.includes('Hydration') ||
    error.message?.includes('hydrat') ||
    error.message?.includes('server-rendered') ||
    error.message?.includes('Text content does not match') ||
    error.digest?.includes('NEXT_');

  // Hard refresh: bypass cache and force full page reload
  const handleHardRefresh = () => {
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  return (
    <div className="min-h-screen bg-bg flex items-center justify-center p-4">
      <div className="max-w-2xl w-full border border-crimson bg-surface p-6 font-mono">
        <div className="flex items-start gap-3 mb-4">
          <div className="text-crimson text-2xl glow-text-subtle">{'>'}</div>
          <div>
            <div className="text-crimson font-bold mb-2 text-xl">
              APPLICATION ERROR
            </div>
            <div className="text-warning text-sm mb-2">
              {isHydrationError
                ? 'A rendering mismatch occurred. This is usually harmless — try refreshing.'
                : 'Something went wrong in the Aragora Live interface'}
            </div>
            {error.digest && (
              <div className="text-text-muted text-xs">
                Error ID: {error.digest}
              </div>
            )}
          </div>
        </div>

        <div className="bg-bg border border-border p-4 mb-4 text-text-muted text-sm overflow-x-auto">
          <div className="mb-2 text-text font-bold">
            {'>'} {error.name || 'Error'}
          </div>
          <div className="pl-4 text-crimson">
            {error.message || 'An unexpected error occurred'}
          </div>
        </div>

        <div className="flex gap-3 mb-4">
          <button
            onClick={reset}
            className="flex-1 border border-accent text-accent py-2 px-4 hover:bg-accent hover:text-bg transition-colors font-bold"
          >
            {'>'} RETRY
          </button>
          <button
            onClick={handleHardRefresh}
            className="flex-1 border border-acid-green text-acid-green py-2 px-4 hover:bg-acid-green hover:text-bg transition-colors font-bold"
            title="Force full page reload (bypasses cache)"
          >
            {'>'} HARD REFRESH
          </button>
          <Link
            href="/"
            className="flex-1 border border-text-muted text-text-muted py-2 px-4 hover:bg-text-muted hover:text-bg transition-colors text-center"
          >
            {'>'} HOME
          </Link>
        </div>

        {/* Diagnostics panel (expandable) */}
        <button
          onClick={() => setShowDiagnostics(!showDiagnostics)}
          className="w-full text-left text-xs font-mono text-text-muted hover:text-acid-green transition-colors mb-2"
        >
          {showDiagnostics ? '[-]' : '[+]'} DIAGNOSTICS
        </button>

        {showDiagnostics && (
          <div className="bg-bg border border-border p-3 text-xs space-y-2">
            <div>
              <span className="text-text-muted">Error Type: </span>
              <span className="text-text">{error.name || 'Unknown'}</span>
            </div>
            <div>
              <span className="text-text-muted">Digest: </span>
              <span className="text-text">{error.digest || 'N/A'}</span>
            </div>
            <div>
              <span className="text-text-muted">Hydration Issue: </span>
              <span className={isHydrationError ? 'text-acid-yellow' : 'text-acid-green'}>
                {isHydrationError ? 'YES' : 'NO'}
              </span>
            </div>
            <div>
              <span className="text-text-muted">URL: </span>
              <span className="text-text">{typeof window !== 'undefined' ? window.location.href : 'SSR'}</span>
            </div>
            <div>
              <span className="text-text-muted">User Agent: </span>
              <span className="text-text truncate block">
                {typeof navigator !== 'undefined' ? navigator.userAgent.slice(0, 80) : 'N/A'}
              </span>
            </div>
            {error.stack && (
              <div>
                <span className="text-text-muted block mb-1">Stack Trace:</span>
                <pre className="text-[10px] text-text-muted overflow-x-auto whitespace-pre-wrap max-h-40 overflow-y-auto">
                  {error.stack}
                </pre>
              </div>
            )}
          </div>
        )}

        <div className="mt-4 p-3 bg-warning/10 border border-warning/30 text-warning text-xs">
          <div className="font-bold mb-1">{'>'} TROUBLESHOOTING</div>
          <ul className="pl-4 space-y-1">
            {isHydrationError && (
              <li>* Hydration errors are often caused by browser extensions or cached data</li>
            )}
            <li>* Try the Hard Refresh button to clear cached state</li>
            <li>* Check browser console for details</li>
            <li>* Verify backend connection at /admin/system-health</li>
            <li>* Clear cache and reload if issue persists</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
