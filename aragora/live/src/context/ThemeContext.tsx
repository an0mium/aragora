'use client';

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react';

// ============================================================================
// Types
// ============================================================================

export type ThemePreference = 'dark' | 'light' | 'system';
export type EffectiveTheme = 'dark' | 'light';

export interface ThemeContextValue {
  /** User's theme preference (dark, light, or system) */
  preference: ThemePreference;
  /** The actual theme being rendered (resolved from system if preference is 'system') */
  effectiveTheme: EffectiveTheme;
  /** Set the theme preference */
  setTheme: (theme: ThemePreference) => void;
  /** Toggle between dark and light (ignores system, sets explicit preference) */
  toggleTheme: () => void;
  /** Whether the theme context has initialized */
  isInitialized: boolean;
}

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = 'aragora-theme';
const STORAGE_KEY_PREFERENCES = 'aragora_preferences';
const DATA_ATTRIBUTE = 'data-theme';

// ============================================================================
// Context
// ============================================================================

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

// ============================================================================
// Utilities
// ============================================================================

function getSystemTheme(): EffectiveTheme {
  if (typeof window === 'undefined') return 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function resolveEffectiveTheme(preference: ThemePreference): EffectiveTheme {
  if (preference === 'system') {
    return getSystemTheme();
  }
  return preference;
}

function applyTheme(theme: EffectiveTheme): void {
  if (typeof document === 'undefined') return;

  // Set or remove the data-theme attribute
  if (theme === 'light') {
    document.documentElement.setAttribute(DATA_ATTRIBUTE, 'light');
    document.body.setAttribute(DATA_ATTRIBUTE, 'light');
  } else {
    document.documentElement.removeAttribute(DATA_ATTRIBUTE);
    document.body.removeAttribute(DATA_ATTRIBUTE);
  }

  // Update meta theme-color for mobile browsers
  const metaThemeColor = document.querySelector('meta[name="theme-color"]');
  if (metaThemeColor) {
    metaThemeColor.setAttribute('content', theme === 'light' ? '#faf9f7' : '#00ff41');
  }
}

function getStoredPreference(): ThemePreference | null {
  if (typeof localStorage === 'undefined') return null;

  // Check legacy key first (simple dark/light)
  const legacyTheme = localStorage.getItem(STORAGE_KEY);
  if (legacyTheme === 'dark' || legacyTheme === 'light') {
    return legacyTheme;
  }

  // Check preferences object
  try {
    const prefs = localStorage.getItem(STORAGE_KEY_PREFERENCES);
    if (prefs) {
      const parsed = JSON.parse(prefs);
      if (parsed.theme === 'dark' || parsed.theme === 'light' || parsed.theme === 'system') {
        return parsed.theme;
      }
    }
  } catch {
    // Ignore parse errors
  }

  return null;
}

function storePreference(preference: ThemePreference): void {
  if (typeof localStorage === 'undefined') return;

  // Store in legacy key for backward compatibility
  if (preference === 'system') {
    localStorage.removeItem(STORAGE_KEY);
  } else {
    localStorage.setItem(STORAGE_KEY, preference);
  }

  // Also update preferences object if it exists
  try {
    const prefs = localStorage.getItem(STORAGE_KEY_PREFERENCES);
    if (prefs) {
      const parsed = JSON.parse(prefs);
      parsed.theme = preference;
      localStorage.setItem(STORAGE_KEY_PREFERENCES, JSON.stringify(parsed));
    }
  } catch {
    // Ignore errors
  }
}

// ============================================================================
// Provider
// ============================================================================

interface ThemeProviderProps {
  children: ReactNode;
  /** Default preference if none stored. Defaults to 'dark' */
  defaultPreference?: ThemePreference;
}

export function ThemeProvider({
  children,
  defaultPreference = 'dark',
}: ThemeProviderProps) {
  const [preference, setPreference] = useState<ThemePreference>(defaultPreference);
  const [effectiveTheme, setEffectiveTheme] = useState<EffectiveTheme>('dark');
  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize from storage on mount
  useEffect(() => {
    const stored = getStoredPreference();
    const initialPreference = stored ?? defaultPreference;
    setPreference(initialPreference);

    const resolved = resolveEffectiveTheme(initialPreference);
    setEffectiveTheme(resolved);
    applyTheme(resolved);

    setIsInitialized(true);
  }, [defaultPreference]);

  // Listen for system preference changes (when preference is 'system')
  useEffect(() => {
    if (preference !== 'system') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = (e: MediaQueryListEvent) => {
      const newTheme: EffectiveTheme = e.matches ? 'dark' : 'light';
      setEffectiveTheme(newTheme);
      applyTheme(newTheme);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [preference]);

  // Set theme handler
  const setTheme = useCallback((newPreference: ThemePreference) => {
    setPreference(newPreference);
    storePreference(newPreference);

    const resolved = resolveEffectiveTheme(newPreference);
    setEffectiveTheme(resolved);
    applyTheme(resolved);
  }, []);

  // Toggle theme handler
  const toggleTheme = useCallback(() => {
    const newTheme: EffectiveTheme = effectiveTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
  }, [effectiveTheme, setTheme]);

  const value = useMemo<ThemeContextValue>(() => ({
    preference,
    effectiveTheme,
    setTheme,
    toggleTheme,
    isInitialized,
  }), [preference, effectiveTheme, setTheme, toggleTheme, isInitialized]);

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

// ============================================================================
// Hook
// ============================================================================

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// ============================================================================
// SSR-Safe Initialization Script
// ============================================================================

/**
 * Inline script to prevent flash of wrong theme.
 * Should be placed in the <head> before CSS loads.
 */
export const themeInitScript = `
(function() {
  try {
    var theme = localStorage.getItem('aragora-theme');
    if (!theme) {
      var prefs = localStorage.getItem('aragora_preferences');
      if (prefs) {
        var parsed = JSON.parse(prefs);
        theme = parsed.theme;
      }
    }
    if (theme === 'system' || !theme) {
      theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    if (theme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
    }
  } catch (e) {}
})();
`;
