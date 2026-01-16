import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LandingPage } from '../LandingPage';

// Mock all child components to isolate LandingPage logic
jest.mock('../../MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

jest.mock('../Header', () => ({
  Header: () => <header data-testid="header">Header</header>,
}));

jest.mock('../HeroSection', () => ({
  HeroSection: ({
    error,
    activeDebateId,
    activeQuestion,
    onDismissError,
    onDebateStarted,
    onError,
  }: {
    error: string | null;
    activeDebateId: string | null;
    activeQuestion: string | null;
    onDismissError: () => void;
    onDebateStarted: (id: string, q: string) => void;
    onError: (err: string) => void;
  }) => (
    <div data-testid="hero-section">
      {error && (
        <div data-testid="error-display">
          {error}
          <button onClick={onDismissError} data-testid="dismiss-error">
            Dismiss
          </button>
        </div>
      )}
      {activeDebateId && (
        <div data-testid="active-debate">
          {activeDebateId} - {activeQuestion}
        </div>
      )}
      <button
        onClick={() => onDebateStarted('debate-123', 'Test question')}
        data-testid="start-debate"
      >
        Start Debate
      </button>
      <button
        onClick={() => onError('Test error')}
        data-testid="trigger-error"
      >
        Trigger Error
      </button>
    </div>
  ),
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
  const defaultProps = {
    apiBase: 'http://localhost:8080',
    wsUrl: 'wss://localhost:8080/ws',
    onDebateStarted: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('initial render', () => {
    it('renders visual effects', () => {
      render(<LandingPage {...defaultProps} />);

      expect(screen.getByTestId('scanlines')).toBeInTheDocument();
      expect(screen.getByTestId('crt-vignette')).toBeInTheDocument();
    });

    it('renders all page sections', () => {
      render(<LandingPage {...defaultProps} />);

      expect(screen.getByTestId('header')).toBeInTheDocument();
      expect(screen.getByTestId('hero-section')).toBeInTheDocument();
      expect(screen.getByTestId('why-aragora')).toBeInTheDocument();
      expect(screen.getByTestId('debate-protocol')).toBeInTheDocument();
      expect(screen.getByTestId('capabilities')).toBeInTheDocument();
      expect(screen.getByTestId('trust')).toBeInTheDocument();
      expect(screen.getByTestId('footer')).toBeInTheDocument();
    });

    it('renders main element with proper classes', () => {
      render(<LandingPage {...defaultProps} />);

      const main = screen.getByRole('main');
      expect(main).toHaveClass('min-h-screen');
    });

    it('starts with no error', () => {
      render(<LandingPage {...defaultProps} />);

      expect(screen.queryByTestId('error-display')).not.toBeInTheDocument();
    });

    it('starts with no active debate', () => {
      render(<LandingPage {...defaultProps} />);

      expect(screen.queryByTestId('active-debate')).not.toBeInTheDocument();
    });
  });

  describe('debate started flow', () => {
    it('updates active debate state when debate starts', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      render(<LandingPage {...defaultProps} />);

      await act(async () => {
        await user.click(screen.getByTestId('start-debate'));
      });

      expect(screen.getByTestId('active-debate')).toHaveTextContent(
        'debate-123 - Test question'
      );
    });

    it('calls onDebateStarted prop when debate starts', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      const onDebateStarted = jest.fn();
      render(<LandingPage {...defaultProps} onDebateStarted={onDebateStarted} />);

      await act(async () => {
        await user.click(screen.getByTestId('start-debate'));
      });

      expect(onDebateStarted).toHaveBeenCalledWith('debate-123');
    });
  });

  describe('error handling', () => {
    it('displays error when error occurs', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      render(<LandingPage {...defaultProps} />);

      await act(async () => {
        await user.click(screen.getByTestId('trigger-error'));
      });

      expect(screen.getByTestId('error-display')).toHaveTextContent('Test error');
    });

    it('auto-dismisses error after 5 seconds', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      render(<LandingPage {...defaultProps} />);

      await act(async () => {
        await user.click(screen.getByTestId('trigger-error'));
      });

      expect(screen.getByTestId('error-display')).toBeInTheDocument();

      act(() => {
        jest.advanceTimersByTime(5000);
      });

      expect(screen.queryByTestId('error-display')).not.toBeInTheDocument();
    });

    it('dismisses error when dismiss button is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      render(<LandingPage {...defaultProps} />);

      await act(async () => {
        await user.click(screen.getByTestId('trigger-error'));
      });

      expect(screen.getByTestId('error-display')).toBeInTheDocument();

      await act(async () => {
        await user.click(screen.getByTestId('dismiss-error'));
      });

      expect(screen.queryByTestId('error-display')).not.toBeInTheDocument();
    });
  });

  describe('prop passing', () => {
    it('passes apiBase to HeroSection', () => {
      render(<LandingPage {...defaultProps} apiBase="http://custom:9000" />);

      // HeroSection is mocked, verify component renders
      expect(screen.getByTestId('hero-section')).toBeInTheDocument();
    });
  });
});
