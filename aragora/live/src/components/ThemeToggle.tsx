'use client';

import { useTheme } from '@/context/ThemeContext';

export function ThemeToggle() {
  const { effectiveTheme, toggleTheme, isInitialized } = useTheme();

  // Show placeholder during SSR/hydration to prevent mismatch
  if (!isInitialized) {
    return (
      <button
        className="p-2 text-text-muted hover:text-text transition-colors"
        aria-label="Toggle theme"
      >
        <span className="w-4 h-4 block" />
      </button>
    );
  }

  return (
    <button
      onClick={toggleTheme}
      className="p-2 text-text-muted hover:text-text transition-colors rounded hover:bg-surface"
      aria-label={`Switch to ${effectiveTheme === 'dark' ? 'light' : 'dark'} mode`}
      title={`Switch to ${effectiveTheme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {effectiveTheme === 'dark' ? (
        // Sun icon for switching to light mode
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="w-4 h-4"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z"
          />
        </svg>
      ) : (
        // Moon icon for switching to dark mode
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="w-4 h-4"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z"
          />
        </svg>
      )}
    </button>
  );
}

/**
 * Extended theme toggle with system preference option.
 * Shows three-state: dark, light, system (auto).
 */
export function ThemeSelector() {
  const { preference, setTheme, isInitialized } = useTheme();

  if (!isInitialized) {
    return (
      <div className="flex gap-1 p-1 bg-surface rounded">
        <span className="w-20 h-8 bg-surface-elevated rounded animate-pulse" />
      </div>
    );
  }

  return (
    <div className="flex gap-1 p-1 bg-surface rounded" role="radiogroup" aria-label="Theme selection">
      <button
        role="radio"
        aria-checked={preference === 'dark'}
        onClick={() => setTheme('dark')}
        className={`px-3 py-1.5 text-xs font-mono rounded transition-colors ${
          preference === 'dark'
            ? 'bg-acid-green/20 text-acid-green'
            : 'text-text-muted hover:text-text'
        }`}
      >
        Dark
      </button>
      <button
        role="radio"
        aria-checked={preference === 'light'}
        onClick={() => setTheme('light')}
        className={`px-3 py-1.5 text-xs font-mono rounded transition-colors ${
          preference === 'light'
            ? 'bg-acid-green/20 text-acid-green'
            : 'text-text-muted hover:text-text'
        }`}
      >
        Light
      </button>
      <button
        role="radio"
        aria-checked={preference === 'system'}
        onClick={() => setTheme('system')}
        className={`px-3 py-1.5 text-xs font-mono rounded transition-colors ${
          preference === 'system'
            ? 'bg-acid-green/20 text-acid-green'
            : 'text-text-muted hover:text-text'
        }`}
      >
        Auto
      </button>
    </div>
  );
}
