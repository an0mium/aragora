import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from '../Header';

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

jest.mock('next/navigation', () => ({
  usePathname: () => '/landing',
}));

jest.mock('../ThemeSelector', () => ({
  ThemeSelector: () => <div data-testid="theme-selector">ThemeSelector</div>,
}));

jest.mock('@/components/Logo', () => ({
  Logo: () => <div data-testid="logo">Logo</div>,
}));

describe('Header', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initial render', () => {
    it('renders the logo', () => {
      render(<Header />);
      expect(screen.getByTestId('logo')).toBeInTheDocument();
    });

    it('renders the ARAGORA wordmark', () => {
      render(<Header />);
      expect(screen.getByText('> ARAGORA')).toBeInTheDocument();
    });

    it('renders the theme selector', () => {
      render(<Header />);
      expect(screen.getByTestId('theme-selector')).toBeInTheDocument();
    });
  });

  describe('desktop navigation', () => {
    it('renders core navigation links', () => {
      render(<Header />);

      // Links appear in both desktop and mobile nav
      expect(screen.getAllByText('How it works').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Pricing').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Compliance').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Log in').length).toBeGreaterThan(0);
    });

    it('pricing link points to /pricing', () => {
      render(<Header />);
      // Pricing link in both desktop and mobile nav — find the first
      const links = screen.getAllByText('Pricing');
      const desktopLink = links[0].closest('a');
      expect(desktopLink).toHaveAttribute('href', '/pricing');
    });
  });

  describe('mobile menu', () => {
    it('renders hamburger button for mobile', () => {
      render(<Header />);
      expect(screen.getByLabelText('Open menu')).toBeInTheDocument();
    });

    it('toggles menu on hamburger click', async () => {
      const user = userEvent.setup();
      render(<Header />);

      const button = screen.getByLabelText('Open menu');
      await user.click(button);

      expect(screen.getByLabelText('Close menu')).toBeInTheDocument();
    });

    it('renders sign up CTA in mobile menu', async () => {
      const user = userEvent.setup();
      render(<Header />);

      await user.click(screen.getByLabelText('Open menu'));
      expect(screen.getByText('Sign up free')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('hamburger has aria-expanded attribute', () => {
      render(<Header />);
      const button = screen.getByLabelText('Open menu');
      expect(button).toHaveAttribute('aria-expanded', 'false');
    });

    it('header is a sticky banner', () => {
      render(<Header />);
      const header = document.querySelector('header');
      expect(header).toBeInTheDocument();
      expect(header).toHaveClass('sticky');
    });
  });
});
