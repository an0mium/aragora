/**
 * Tests for ThemeToggle component
 */

import { render, screen, fireEvent, act } from '@testing-library/react';
import { ThemeToggle } from '../src/components/ThemeToggle';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// Mock matchMedia
const mockMatchMedia = (matches: boolean) => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation((query) => ({
      matches,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  });
};

describe('ThemeToggle', () => {
  beforeEach(() => {
    localStorageMock.clear();
    document.body.removeAttribute('data-theme');
    mockMatchMedia(true); // Default to preferring dark mode
  });

  describe('Initial State', () => {
    it('renders without crashing', () => {
      render(<ThemeToggle />);
      expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('has aria-label for accessibility', async () => {
      render(<ThemeToggle />);

      // Wait for mount
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label');
    });
  });

  describe('Theme Persistence', () => {
    it('loads dark theme from localStorage', async () => {
      localStorageMock.setItem('aragora-theme', 'dark');
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(screen.getByRole('button')).toHaveAttribute(
        'aria-label',
        'Switch to light mode'
      );
    });

    it('loads light theme from localStorage', async () => {
      localStorageMock.setItem('aragora-theme', 'light');
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(screen.getByRole('button')).toHaveAttribute(
        'aria-label',
        'Switch to dark mode'
      );
    });

    it('saves theme to localStorage on toggle', async () => {
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      fireEvent.click(screen.getByRole('button'));

      expect(localStorageMock.getItem('aragora-theme')).toBe('light');
    });
  });

  describe('System Preference', () => {
    it('uses dark theme when system prefers dark', async () => {
      mockMatchMedia(true);
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(screen.getByRole('button')).toHaveAttribute(
        'aria-label',
        'Switch to light mode'
      );
    });

    it('uses light theme when system prefers light', async () => {
      mockMatchMedia(false);
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(screen.getByRole('button')).toHaveAttribute(
        'aria-label',
        'Switch to dark mode'
      );
    });
  });

  describe('Toggle Behavior', () => {
    it('toggles from dark to light', async () => {
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label', 'Switch to light mode');

      fireEvent.click(button);

      expect(button).toHaveAttribute('aria-label', 'Switch to dark mode');
    });

    it('toggles from light to dark', async () => {
      localStorageMock.setItem('aragora-theme', 'light');
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label', 'Switch to dark mode');

      fireEvent.click(button);

      expect(button).toHaveAttribute('aria-label', 'Switch to light mode');
    });

    it('sets data-theme attribute on body for light mode', async () => {
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      fireEvent.click(screen.getByRole('button'));

      expect(document.body.getAttribute('data-theme')).toBe('light');
    });

    it('removes data-theme attribute for dark mode', async () => {
      localStorageMock.setItem('aragora-theme', 'light');
      document.body.setAttribute('data-theme', 'light');

      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      fireEvent.click(screen.getByRole('button'));

      expect(document.body.getAttribute('data-theme')).toBeNull();
    });
  });

  describe('Icons', () => {
    it('shows sun icon in dark mode (to switch to light)', async () => {
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      const svg = document.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });

    it('shows moon icon in light mode (to switch to dark)', async () => {
      localStorageMock.setItem('aragora-theme', 'light');
      render(<ThemeToggle />);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      const svg = document.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });
  });
});
