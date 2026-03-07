import { render, screen } from '@testing-library/react';
import { LandingPage } from '../LandingPage';

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

// Mock all child components to isolate LandingPage logic
jest.mock('../Header', () => ({
  Header: () => <header data-testid="header">Header</header>,
}));

jest.mock('../HeroSection', () => ({
  HeroSection: () => (
    <div data-testid="hero-section">Hero</div>
  ),
}));

jest.mock('../HowItWorksSection', () => ({
  HowItWorksSection: () => <section data-testid="how-it-works">How It Works</section>,
}));

jest.mock('../ProblemSection', () => ({
  ProblemSection: () => <section data-testid="problem">Problem</section>,
}));

jest.mock('../PricingSection', () => ({
  PricingSection: () => <section data-testid="pricing-section">Pricing</section>,
}));

jest.mock('../Footer', () => ({
  Footer: () => <footer data-testid="footer">Footer</footer>,
}));

describe('LandingPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initial render', () => {
    it('renders all page sections', () => {
      render(<LandingPage />);

      expect(screen.getByTestId('header')).toBeInTheDocument();
      expect(screen.getByTestId('hero-section')).toBeInTheDocument();
      expect(screen.getByTestId('how-it-works')).toBeInTheDocument();
      expect(screen.getByTestId('problem')).toBeInTheDocument();
      expect(screen.getByTestId('pricing-section')).toBeInTheDocument();
      expect(screen.getByTestId('footer')).toBeInTheDocument();
    });

    it('renders the themed container with min-h-screen', () => {
      const { container } = render(<LandingPage />);

      const wrapper = container.firstElementChild;
      expect(wrapper).toHaveClass('min-h-screen');
      expect(wrapper).toHaveAttribute('data-landing-theme', 'dark');
    });
  });
});
