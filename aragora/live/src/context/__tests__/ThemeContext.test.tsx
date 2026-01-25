/**
 * Tests for ThemeContext
 *
 * Tests cover:
 * - Default theme (dark)
 * - Theme initialization from localStorage
 * - setTheme changes preference and effective theme
 * - toggleTheme cycles between dark and light
 * - System preference handling
 * - useTheme hook throws outside provider
 * - localStorage persistence
 */

import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../ThemeContext';

// Mock localStorage
const localStorageMock: Record<string, string> = {};
const mockLocalStorage = {
  getItem: jest.fn((key: string) => localStorageMock[key] || null),
  setItem: jest.fn((key: string, value: string) => {
    localStorageMock[key] = value;
  }),
  removeItem: jest.fn((key: string) => {
    delete localStorageMock[key];
  }),
  clear: jest.fn(() => {
    Object.keys(localStorageMock).forEach(key => delete localStorageMock[key]);
  }),
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

// Mock matchMedia
const mockMatchMedia = jest.fn();
Object.defineProperty(window, 'matchMedia', {
  value: mockMatchMedia,
});

// Mock document methods (prefixed with _ as currently unused, kept for future test expansion)
const _mockSetAttribute = jest.fn();
const _mockRemoveAttribute = jest.fn();
const _originalDocumentElement = document.documentElement;

describe('ThemeContext', () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider>{children}</ThemeProvider>
  );

  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();

    // Default to dark mode system preference
    mockMatchMedia.mockImplementation((query: string) => ({
      matches: query === '(prefers-color-scheme: dark)',
      media: query,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    }));
  });

  describe('Initialization', () => {
    it('starts with dark theme by default', async () => {
      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.preference).toBe('dark');
      expect(result.current.effectiveTheme).toBe('dark');
    });

    it('loads theme from legacy storage key', async () => {
      localStorageMock['aragora-theme'] = 'light';

      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.preference).toBe('light');
      expect(result.current.effectiveTheme).toBe('light');
    });

    it('loads theme from preferences object', async () => {
      localStorageMock['aragora_preferences'] = JSON.stringify({ theme: 'light' });

      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.preference).toBe('light');
    });

    it('uses defaultPreference prop when no storage', async () => {
      const customWrapper = ({ children }: { children: React.ReactNode }) => (
        <ThemeProvider defaultPreference="light">{children}</ThemeProvider>
      );

      const { result } = renderHook(() => useTheme(), { wrapper: customWrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.preference).toBe('light');
    });
  });

  describe('setTheme', () => {
    it('changes preference to light', async () => {
      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      act(() => {
        result.current.setTheme('light');
      });

      expect(result.current.preference).toBe('light');
      expect(result.current.effectiveTheme).toBe('light');
    });

    it('changes preference to system', async () => {
      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      act(() => {
        result.current.setTheme('system');
      });

      expect(result.current.preference).toBe('system');
      // Effective theme should be resolved from system (dark in our mock)
      expect(result.current.effectiveTheme).toBe('dark');
    });

    it('persists theme to localStorage', async () => {
      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      act(() => {
        result.current.setTheme('light');
      });

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('aragora-theme', 'light');
    });

    it('removes legacy key when set to system', async () => {
      localStorageMock['aragora-theme'] = 'light';

      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      act(() => {
        result.current.setTheme('system');
      });

      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('aragora-theme');
    });
  });

  describe('toggleTheme', () => {
    it('toggles from dark to light', async () => {
      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.effectiveTheme).toBe('dark');

      act(() => {
        result.current.toggleTheme();
      });

      expect(result.current.preference).toBe('light');
      expect(result.current.effectiveTheme).toBe('light');
    });

    it('toggles from light to dark', async () => {
      localStorageMock['aragora-theme'] = 'light';

      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.effectiveTheme).toBe('light');

      act(() => {
        result.current.toggleTheme();
      });

      expect(result.current.preference).toBe('dark');
      expect(result.current.effectiveTheme).toBe('dark');
    });

    it('toggle overrides system preference', async () => {
      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      act(() => {
        result.current.setTheme('system');
      });

      expect(result.current.preference).toBe('system');
      expect(result.current.effectiveTheme).toBe('dark'); // System is dark

      act(() => {
        result.current.toggleTheme();
      });

      // Should now be explicit light (not system anymore)
      expect(result.current.preference).toBe('light');
    });
  });

  describe('System Preference', () => {
    it('resolves system to dark when system prefers dark', async () => {
      mockMatchMedia.mockImplementation((query: string) => ({
        matches: query === '(prefers-color-scheme: dark)',
        media: query,
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
      }));

      const customWrapper = ({ children }: { children: React.ReactNode }) => (
        <ThemeProvider defaultPreference="system">{children}</ThemeProvider>
      );

      const { result } = renderHook(() => useTheme(), { wrapper: customWrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.effectiveTheme).toBe('dark');
    });

    it('resolves system to light when system prefers light', async () => {
      mockMatchMedia.mockImplementation((query: string) => ({
        matches: query !== '(prefers-color-scheme: dark)', // Prefers light
        media: query,
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
      }));

      const customWrapper = ({ children }: { children: React.ReactNode }) => (
        <ThemeProvider defaultPreference="system">{children}</ThemeProvider>
      );

      const { result } = renderHook(() => useTheme(), { wrapper: customWrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      expect(result.current.effectiveTheme).toBe('light');
    });
  });

  describe('useTheme hook', () => {
    it('throws error when used outside ThemeProvider', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      expect(() => {
        renderHook(() => useTheme());
      }).toThrow('useTheme must be used within a ThemeProvider');

      consoleSpy.mockRestore();
    });
  });

  describe('Edge Cases', () => {
    it('handles invalid localStorage JSON gracefully', async () => {
      localStorageMock['aragora_preferences'] = 'invalid json{';

      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      // Should fallback to default
      expect(result.current.preference).toBe('dark');
    });

    it('handles invalid theme value in localStorage', async () => {
      localStorageMock['aragora-theme'] = 'invalid-theme';

      const { result } = renderHook(() => useTheme(), { wrapper });

      await waitFor(() => {
        expect(result.current.isInitialized).toBe(true);
      });

      // Should fallback to default
      expect(result.current.preference).toBe('dark');
    });
  });
});
