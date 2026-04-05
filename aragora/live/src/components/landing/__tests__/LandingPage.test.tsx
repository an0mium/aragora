import { render, screen } from '@testing-library/react';
import { LandingPage } from '../LandingPage';

const mockUseBackend = jest.fn(() => ({
  config: { api: 'http://localhost:8080', ws: 'ws://localhost:8765/ws' },
}));
const mockUseSpectate = jest.fn();

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

jest.mock('@/hooks/useSpectate', () => ({
  useSpectate: (...args: unknown[]) => mockUseSpectate(...args),
}));

jest.mock('../../BackendSelector', () => ({
  BACKENDS: {
    production: { api: 'https://api.example.com', ws: 'wss://api.example.com/ws' },
  },
  useBackend: () => mockUseBackend(),
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

const mockLiveDebatePanel = jest.fn(() => (
  <section data-testid="live-debate-panel">Live Debate</section>
));
const mockLiveDemoSection = jest.fn(() => (
  <section data-testid="live-demo-section">Live Demo</section>
));

jest.mock('../LiveDebatePanel', () => ({
  LiveDebatePanel: (props: Record<string, unknown>) => {
    mockLiveDebatePanel(props);
    return <section data-testid="live-debate-panel">Live Debate</section>;
  },
}));

jest.mock('../LiveDemoSection', () => ({
  LiveDemoSection: (props: Record<string, unknown>) => {
    mockLiveDemoSection(props);
    return <section data-testid="live-demo-section">Live Demo</section>;
  },
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
  const bridgeState = {
    status: {
      active: true,
      subscribers: 2,
      buffer_size: 8,
      bridge_state: 'live_debates_available' as const,
      last_event_at: '2026-04-05T16:00:00Z',
      activity_age_seconds: 12,
      recent_activity_window_seconds: 120,
      recent_event_count: 4,
      live_debate_count: 1,
      live_debate_ids: ['debate-1'],
      live_debates: [],
      unattributed_recent_event_count: 0,
    },
    loaded: true,
    connected: true,
    events: [],
    refresh: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseBackend.mockReturnValue({
      config: { api: 'http://localhost:8080', ws: 'ws://localhost:8765/ws' },
    });
    mockUseSpectate.mockReturnValue(bridgeState);
  });

  describe('initial render', () => {
    it('renders all page sections', () => {
      render(<LandingPage />);

      expect(screen.getByTestId('header')).toBeInTheDocument();
      expect(screen.getByTestId('hero-section')).toBeInTheDocument();
      expect(screen.getByTestId('live-debate-panel')).toBeInTheDocument();
      expect(screen.getByTestId('live-demo-section')).toBeInTheDocument();
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

    it('passes resolved backend settings to the live debate panel', () => {
      render(<LandingPage />);

      expect(mockUseSpectate).toHaveBeenCalledWith(undefined, undefined, {
        apiBaseUrl: 'http://localhost:8080',
        pollInterval: 4000,
        maxEvents: 40,
      });
      expect(mockLiveDebatePanel).toHaveBeenCalledWith(
        expect.objectContaining({
          apiBase: 'http://localhost:8080',
          wsUrl: 'ws://localhost:8765/ws',
          bridgeState,
        }),
      );
      expect(mockLiveDemoSection).toHaveBeenCalledWith(
        expect.objectContaining({
          bridgeState,
        }),
      );
    });
  });
});
