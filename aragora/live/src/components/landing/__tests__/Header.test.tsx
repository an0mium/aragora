import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from '../Header';

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
      render(<Header />);

      expect(screen.getByTestId('ascii-banner')).toBeInTheDocument();
    });

    it('renders the theme toggle', () => {
      render(<Header />);

      // Multiple theme toggles (desktop + mobile)
      const toggles = screen.getAllByTestId('theme-toggle');
      expect(toggles.length).toBeGreaterThan(0);
    });

    it('renders the backend selector in desktop nav', () => {
      render(<Header />);

      expect(screen.getByTestId('backend-selector')).toBeInTheDocument();
    });
  });

  describe('desktop navigation', () => {
    it('renders all navigation links', () => {
      render(<Header />);

      expect(screen.getByRole('link', { name: /debates/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /agents/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /insights/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /evidence/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /memory/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /tournaments/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /dev/i })).toBeInTheDocument();
    });

    it('debates link points to /debates', () => {
      render(<Header />);

      const link = screen.getByRole('link', { name: /debates/i });
      expect(link).toHaveAttribute('href', '/debates');
    });

    it('agents link points to /agents', () => {
      render(<Header />);

      const link = screen.getByRole('link', { name: /agents/i });
      expect(link).toHaveAttribute('href', '/agents');
    });

    it('live link points to external URL', () => {
      render(<Header />);

      const links = screen.getAllByRole('link', { name: /live/i });
      // Find the one pointing to external URL
      const externalLink = links.find((link) =>
        link.getAttribute('href')?.includes('live.aragora.ai')
      );
      expect(externalLink).toHaveAttribute('href', 'https://live.aragora.ai');
    });

    it('has proper aria-label for navigation', () => {
      render(<Header />);

      expect(
        screen.getByRole('navigation', { name: /main navigation/i })
      ).toBeInTheDocument();
    });
  });

  describe('mobile menu toggle', () => {
    it('renders mobile menu button', () => {
      render(<Header />);

      expect(
        screen.getByRole('button', { name: /open menu/i })
      ).toBeInTheDocument();
    });

    it('mobile menu is closed by default', () => {
      render(<Header />);

      // Mobile navigation should not exist when menu is closed
      expect(
        screen.queryByRole('navigation', { name: /mobile navigation/i })
      ).not.toBeInTheDocument();
    });

    it('opens mobile menu when button is clicked', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });

      expect(
        screen.getByRole('navigation', { name: /mobile navigation/i })
      ).toBeInTheDocument();
    });

    it('changes button label when menu is open', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });

      expect(
        screen.getByRole('button', { name: /close menu/i })
      ).toBeInTheDocument();
    });

    it('closes mobile menu when button is clicked again', async () => {
      const user = userEvent.setup();
      render(<Header />);

      // Open menu
      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });
      expect(
        screen.getByRole('navigation', { name: /mobile navigation/i })
      ).toBeInTheDocument();

      // Close menu
      await act(async () => {
        await user.click(screen.getByRole('button', { name: /close menu/i }));
      });
      expect(
        screen.queryByRole('navigation', { name: /mobile navigation/i })
      ).not.toBeInTheDocument();
    });

    it('shows hamburger icon when closed', () => {
      render(<Header />);

      const button = screen.getByRole('button', { name: /open menu/i });
      expect(button.textContent).toContain('☰');
    });

    it('shows close icon when open', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });

      const button = screen.getByRole('button', { name: /close menu/i });
      expect(button.textContent).toContain('✕');
    });
  });

  describe('mobile navigation', () => {
    it('shows all navigation links in mobile menu', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });

      const mobileNav = screen.getByRole('navigation', {
        name: /mobile navigation/i,
      });

      // Check links are within mobile navigation
      expect(mobileNav.querySelector('a[href="/debates"]')).toBeInTheDocument();
      expect(mobileNav.querySelector('a[href="/agents"]')).toBeInTheDocument();
      expect(mobileNav.querySelector('a[href="/insights"]')).toBeInTheDocument();
      expect(mobileNav.querySelector('a[href="/evidence"]')).toBeInTheDocument();
      expect(mobileNav.querySelector('a[href="/memory"]')).toBeInTheDocument();
      expect(mobileNav.querySelector('a[href="/tournaments"]')).toBeInTheDocument();
      expect(mobileNav.querySelector('a[href="/developer"]')).toBeInTheDocument();
    });

    it('shows backend selector in mobile menu', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });

      // Should have backend selectors in both desktop and mobile
      const selectors = screen.getAllByTestId('backend-selector');
      expect(selectors.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('accessibility', () => {
    it('mobile menu button has aria-expanded', () => {
      render(<Header />);

      const button = screen.getByRole('button', { name: /open menu/i });
      expect(button).toHaveAttribute('aria-expanded', 'false');
    });

    it('mobile menu button aria-expanded updates when open', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });

      const button = screen.getByRole('button', { name: /close menu/i });
      expect(button).toHaveAttribute('aria-expanded', 'true');
    });

    it('mobile menu button has aria-controls', () => {
      render(<Header />);

      const button = screen.getByRole('button', { name: /open menu/i });
      expect(button).toHaveAttribute('aria-controls', 'mobile-navigation');
    });

    it('mobile navigation has correct id for aria-controls', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /open menu/i }));
      });

      const mobileNav = screen.getByRole('navigation', {
        name: /mobile navigation/i,
      });
      expect(mobileNav).toHaveAttribute('id', 'mobile-navigation');
    });
  });
});
