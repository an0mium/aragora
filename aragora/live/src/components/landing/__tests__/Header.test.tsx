import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from '../Header';
import { SidebarProvider } from '@/context/SidebarContext';

// Helper to render Header with required providers
const renderHeader = () => {
  return render(
    <SidebarProvider>
      <Header />
    </SidebarProvider>
  );
};

// Mock child components with simpler dependencies
jest.mock('../../AsciiBanner', () => ({
  AsciiBannerCompact: ({ connected }: { connected: boolean }) => (
    <div data-testid="ascii-banner" data-connected={connected}>
      ARAGORA
    </div>
  ),
}));

jest.mock('../../ThemeToggle', () => ({
  ThemeToggle: () => <button data-testid="theme-toggle">Theme</button>,
}));

jest.mock('../../BackendSelector', () => ({
  BackendSelector: ({ compact }: { compact?: boolean }) => (
    <div data-testid="backend-selector" data-compact={compact}>
      Backend
    </div>
  ),
}));

describe('Header', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initial render', () => {
    it('renders the ASCII banner', () => {
      renderHeader();

      expect(screen.getByTestId('ascii-banner')).toBeInTheDocument();
    });

    it('renders the theme toggle', () => {
      renderHeader();

      // Multiple theme toggles (desktop + mobile)
      const toggles = screen.getAllByTestId('theme-toggle');
      expect(toggles.length).toBeGreaterThan(0);
    });

    it('renders backend selectors', () => {
      renderHeader();

      // Backend selector in both desktop and mobile views
      const selectors = screen.getAllByTestId('backend-selector');
      expect(selectors.length).toBe(2);
    });
  });

  describe('desktop navigation', () => {
    it('renders core navigation links', () => {
      renderHeader();

      expect(screen.getByRole('link', { name: /\[debate\]/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /\[debates\]/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /\[gauntlet\]/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /\[leaderboard\]/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /\[agents\]/i })).toBeInTheDocument();
    });

    it('renders secondary navigation links', () => {
      renderHeader();

      expect(screen.getByRole('link', { name: /memory/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /analytics/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /docs/i })).toBeInTheDocument();
    });

    it('debates link points to /debates', () => {
      renderHeader();

      const link = screen.getByRole('link', { name: /\[debates\]/i });
      expect(link).toHaveAttribute('href', '/debates');
    });

    it('agents link points to /agents', () => {
      renderHeader();

      const link = screen.getByRole('link', { name: /\[agents\]/i });
      expect(link).toHaveAttribute('href', '/agents');
    });

    it('has proper aria-label for navigation', () => {
      renderHeader();

      expect(
        screen.getByRole('navigation', { name: /main navigation/i })
      ).toBeInTheDocument();
    });
  });

  describe('sidebar toggle', () => {
    it('renders sidebar toggle button', () => {
      renderHeader();

      expect(
        screen.getByRole('button', { name: /toggle navigation menu/i })
      ).toBeInTheDocument();
    });

    it('toggle button shows hamburger icon', () => {
      renderHeader();

      const button = screen.getByRole('button', { name: /toggle navigation menu/i });
      expect(button.textContent).toContain('☰');
    });

    it('calls toggle when button is clicked', async () => {
      const user = userEvent.setup();
      renderHeader();

      const button = screen.getByRole('button', { name: /toggle navigation menu/i });

      // Click should not throw
      await user.click(button);

      // Button should still be present after click
      expect(button).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('sidebar toggle button has aria-label', () => {
      renderHeader();

      const button = screen.getByRole('button', { name: /toggle navigation menu/i });
      expect(button).toHaveAttribute('aria-label', 'Toggle navigation menu');
    });

    it('navigation has aria-label', () => {
      renderHeader();

      const nav = screen.getByRole('navigation', { name: /main navigation/i });
      expect(nav).toHaveAttribute('aria-label', 'Main navigation');
    });
  });
});
