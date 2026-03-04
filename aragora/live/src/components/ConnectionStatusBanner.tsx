'use client';

import { useState, useEffect, useCallback } from 'react';
import { API_BASE_URL } from '@/config';

const POLL_INTERVAL = 30_000;
const TIMEOUT_MS = 5_000;

/**
 * Global API connectivity banner.
 *
 * Pings the health endpoint on mount and every 30 seconds.
 * Shows a fixed-bottom banner when the API is unreachable.
 * Dismissible per-session (resets on page refresh).
 */
export function ConnectionStatusBanner() {
  const [mounted, setMounted] = useState(false);
  const [reachable, setReachable] = useState(true);
  const [dismissed, setDismissed] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`, {
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      setReachable(response.ok);
    } catch {
      setReachable(false);
    }
  }, []);

  useEffect(() => {
    setMounted(true);
    checkHealth();
    const interval = setInterval(checkHealth, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [checkHealth]);

  if (!mounted || reachable || dismissed) return null;

  return (
    <div
      className="fixed bottom-0 left-0 right-0 z-[100] px-4 py-2 text-xs font-mono"
      style={{
        backgroundColor: 'var(--surface)',
        borderTop: '1px solid var(--crimson, #dc2626)',
        color: 'var(--text-muted)',
      }}
      role="alert"
    >
      <div className="container mx-auto flex items-center justify-between gap-4">
        <div className="flex-1">
          <span className="font-bold mr-2" style={{ color: 'var(--crimson, #dc2626)' }}>
            API UNREACHABLE
          </span>
          Backend at {API_BASE_URL} is not responding.
        </div>
        <button
          onClick={() => setDismissed(true)}
          className="px-2 py-0.5 border border-current hover:opacity-70 transition-opacity"
          aria-label="Dismiss connectivity warning"
        >
          [DISMISS]
        </button>
      </div>
    </div>
  );
}

export default ConnectionStatusBanner;
