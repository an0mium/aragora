'use client';

import { useState, useEffect } from 'react';
import { getEnvWarnings, IS_DEV_MODE, type EnvWarning } from '@/config';

const DISMISS_KEY = 'aragora-config-warnings-dismissed';

/**
 * Runtime configuration health banner.
 *
 * Displays warnings when environment variables are missing or misconfigured.
 * Only shows in development mode or when there are critical warnings.
 * Dismissible with persistence in localStorage.
 */
export function ConfigHealthBanner() {
  const [warnings, setWarnings] = useState<EnvWarning[]>([]);
  const [dismissed, setDismissed] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);

    // Check for warnings
    const envWarnings = getEnvWarnings();
    setWarnings(envWarnings);

    // Check if previously dismissed (only for warnings, not errors)
    const hasErrors = envWarnings.some((w) => w.severity === 'error');
    if (hasErrors) {
      setDismissed(false);
    } else {
      const wasDismissed = localStorage.getItem(DISMISS_KEY) === 'true';
      setDismissed(wasDismissed);
    }
  }, []);

  const handleDismiss = () => {
    setDismissed(true);
    localStorage.setItem(DISMISS_KEY, 'true');
  };

  // Don't render anything until mounted (avoid hydration mismatch)
  if (!mounted) return null;

  // Don't show if no warnings or dismissed
  if (warnings.length === 0 || dismissed) return null;

  // Don't show in production unless there are errors
  const hasErrors = warnings.some((w) => w.severity === 'error');
  if (!IS_DEV_MODE && !hasErrors) return null;

  return (
    <div
      className={`fixed top-0 left-0 right-0 z-[100] px-4 py-2 text-xs font-mono ${
        hasErrors
          ? 'bg-warning/90 text-black'
          : 'bg-acid-cyan/20 text-acid-cyan border-b border-acid-cyan/30'
      }`}
      role="alert"
    >
      <div className="container mx-auto flex items-center justify-between gap-4">
        <div className="flex-1">
          <span className="font-bold mr-2">
            {hasErrors ? '[CONFIG ERROR]' : '[CONFIG WARNING]'}
          </span>
          {warnings.map((w, i) => (
            <span key={w.key} className="mr-3">
              {w.key}: {w.message}
              {i < warnings.length - 1 ? ' |' : ''}
            </span>
          ))}
        </div>
        {!hasErrors && (
          <button
            onClick={handleDismiss}
            className="px-2 py-0.5 border border-current hover:bg-acid-cyan/20 transition-colors"
            aria-label="Dismiss configuration warning"
          >
            [DISMISS]
          </button>
        )}
      </div>
    </div>
  );
}

export default ConfigHealthBanner;
