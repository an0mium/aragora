import { render, screen } from '@testing-library/react';
import { LandingPage } from '../LandingPage';

// Mock all child components to isolate LandingPage logic
jest.mock('../Header', () => ({
  Header: () => <header data-testid="header">Header</header>,
}));

jest.mock('../HeroSection', () => ({
  HeroSection: () => (
    <div data-testid="hero-section">
      <a href="/playground">TRY A FREE DEBATE</a>
      <a href="/login">SIGN IN FOR REAL AI MODELS</a>
    </div>
  ),
}));

jest.mock('../VerticalCards', () => ({
  VerticalCards: () => <section data-testid="vertical-cards">Vertical Cards</section>,
}));

jest.mock('../WhyAragoraSection', () => ({
  WhyAragoraSection: () => <section data-testid="why-aragora">Why Aragora</section>,
}));

jest.mock('../DebateProtocolSection', () => ({
  DebateProtocolSection: () => (
    <section data-testid="debate-protocol">Debate Protocol</section>
  ),
}));

jest.mock('../CapabilitiesSection', () => ({
  CapabilitiesSection: () => (
    <section data-testid="capabilities">Capabilities</section>
  ),
}));

jest.mock('../TrustSection', () => ({
  TrustSection: () => <section data-testid="trust">Trust</section>,
}));

jest.mock('../Footer', () => ({
  Footer: () => <footer data-testid="footer">Footer</footer>,
}));

describe('LandingPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initial render', () => {
    it('renders all page sections in correct order', () => {
      render(<LandingPage />);

      expect(screen.getByTestId('header')).toBeInTheDocument();
      expect(screen.getByTestId('hero-section')).toBeInTheDocument();
      expect(screen.getByTestId('vertical-cards')).toBeInTheDocument();
      expect(screen.getByTestId('why-aragora')).toBeInTheDocument();
      expect(screen.getByTestId('debate-protocol')).toBeInTheDocument();
      expect(screen.getByTestId('capabilities')).toBeInTheDocument();
      expect(screen.getByTestId('trust')).toBeInTheDocument();
      expect(screen.getByTestId('footer')).toBeInTheDocument();
    });

    it('renders main element with proper classes', () => {
      render(<LandingPage />);

      const main = screen.getByRole('main');
      expect(main).toHaveClass('min-h-screen');
    });

    it('renders dual CTA buttons in hero section', () => {
      render(<LandingPage />);

      expect(screen.getByText('TRY A FREE DEBATE')).toBeInTheDocument();
      expect(screen.getByText('SIGN IN FOR REAL AI MODELS')).toBeInTheDocument();
    });

    it('links playground CTA to /playground', () => {
      render(<LandingPage />);

      const playgroundLink = screen.getByText('TRY A FREE DEBATE').closest('a');
      expect(playgroundLink).toHaveAttribute('href', '/playground');
    });

    it('links sign-in CTA to /login', () => {
      render(<LandingPage />);

      const loginLink = screen.getByText('SIGN IN FOR REAL AI MODELS').closest('a');
      expect(loginLink).toHaveAttribute('href', '/login');
    });
  });
});
