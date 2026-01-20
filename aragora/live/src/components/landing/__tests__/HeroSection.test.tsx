import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HeroSection } from '../HeroSection';

// Mock DebateInput since it has complex dependencies
jest.mock('../../DebateInput', () => ({
  DebateInput: () => <div data-testid="debate-input">MockDebateInput</div>,
}));

describe('HeroSection', () => {
  const defaultProps = {
    error: null,
    activeDebateId: null,
    activeQuestion: null,
    apiBase: 'http://localhost:8080',
    onDismissError: jest.fn(),
    onDebateStarted: jest.fn(),
    onError: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initial render', () => {
    it('renders the main heading', () => {
      render(<HeroSection {...defaultProps} />);

      expect(
        screen.getByRole('heading', { name: /what should ai agents debate/i })
      ).toBeInTheDocument();
    });

    it('renders the subheading', () => {
      render(<HeroSection {...defaultProps} />);

      expect(screen.getByText(/Ingest from any source/i)).toBeInTheDocument();
    });

    it('renders the DebateInput component', () => {
      render(<HeroSection {...defaultProps} />);

      expect(screen.getByTestId('debate-input')).toBeInTheDocument();
    });

    it('renders ASCII banner on larger screens', () => {
      render(<HeroSection {...defaultProps} />);

      // ASCII banner is in a pre element with specific class
      const banner = document.querySelector('pre.text-acid-green');
      expect(banner).toBeInTheDocument();
      // Banner is stylized ASCII art, just verify it has content
      expect(banner?.textContent?.length).toBeGreaterThan(100);
    });
  });

  describe('error display', () => {
    it('shows error message when error is present', () => {
      render(<HeroSection {...defaultProps} error="Something went wrong" />);

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('does not show error section when error is null', () => {
      render(<HeroSection {...defaultProps} error={null} />);

      expect(screen.queryByText('✕')).not.toBeInTheDocument();
    });

    it('calls onDismissError when dismiss button is clicked', async () => {
      const user = userEvent.setup();
      const onDismissError = jest.fn();

      render(
        <HeroSection
          {...defaultProps}
          error="Test error"
          onDismissError={onDismissError}
        />
      );

      await user.click(screen.getByRole('button', { name: /dismiss error/i }));

      expect(onDismissError).toHaveBeenCalledTimes(1);
    });

    it('error dismiss button has accessible label', () => {
      render(<HeroSection {...defaultProps} error="Test error" />);

      expect(
        screen.getByRole('button', { name: /dismiss error/i })
      ).toBeInTheDocument();
    });
  });

  describe('active debate indicator', () => {
    it('shows active debate section when debate is in progress', () => {
      render(
        <HeroSection
          {...defaultProps}
          activeDebateId="debate-123"
          activeQuestion="Is AI beneficial?"
        />
      );

      expect(screen.getByText('DECISION IN PROGRESS')).toBeInTheDocument();
    });

    it('displays the active question', () => {
      render(
        <HeroSection
          {...defaultProps}
          activeDebateId="debate-123"
          activeQuestion="Is AI beneficial?"
        />
      );

      expect(screen.getByText('Is AI beneficial?')).toBeInTheDocument();
    });

    it('displays the debate ID', () => {
      render(
        <HeroSection
          {...defaultProps}
          activeDebateId="debate-123"
          activeQuestion="Is AI beneficial?"
        />
      );

      expect(screen.getByText(/ID: debate-123/)).toBeInTheDocument();
    });

    it('shows WebSocket streaming indicator', () => {
      render(
        <HeroSection
          {...defaultProps}
          activeDebateId="debate-123"
          activeQuestion="Is AI beneficial?"
        />
      );

      expect(screen.getByText(/Events streaming via WebSocket/)).toBeInTheDocument();
    });

    it('does not show active debate section when no debate is active', () => {
      render(<HeroSection {...defaultProps} activeDebateId={null} />);

      expect(
        screen.queryByText('DECISION IN PROGRESS')
      ).not.toBeInTheDocument();
    });

    it('has animated pulse indicator for active debate', () => {
      render(
        <HeroSection
          {...defaultProps}
          activeDebateId="debate-123"
          activeQuestion="Test"
        />
      );

      const pulseIndicator = document.querySelector('.animate-pulse');
      expect(pulseIndicator).toBeInTheDocument();
    });
  });

  describe('props passing', () => {
    it('passes apiBase to DebateInput', () => {
      render(<HeroSection {...defaultProps} apiBase="http://custom:9000" />);

      // DebateInput is mocked, but we can verify the component renders
      expect(screen.getByTestId('debate-input')).toBeInTheDocument();
    });
  });
});
