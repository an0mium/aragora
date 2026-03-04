import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HeroSection } from '../HeroSection';

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

jest.mock('../../BackendSelector', () => ({
  useBackend: () => ({
    config: { api: 'http://localhost:8080', ws: 'ws://localhost:8765' },
  }),
  BACKENDS: { production: { api: 'http://localhost:8080', ws: 'ws://localhost:8765' } },
  BackendSelector: () => <div data-testid="backend-selector">Backend</div>,
}));

jest.mock('react-markdown', () => ({
  __esModule: true,
  default: ({ children }: { children: string }) => <div>{children}</div>,
}));

jest.mock('../../DebateResultPreview', () => ({
  DebateResultPreview: () => <div data-testid="debate-result">Result</div>,
  PENDING_DEBATE_KEY: 'aragora_pending_debate',
  RETURN_URL_KEY: 'aragora_return_url',
}));

jest.mock('../../DebateInput', () => ({
  DebateInput: () => <div data-testid="debate-input">DebateInput</div>,
}));

jest.mock('@/utils/returnUrl', () => ({
  getCurrentReturnUrl: () => '/landing',
  normalizeReturnUrl: (url: string) => url,
  RETURN_URL_STORAGE_KEY: 'aragora_return_url',
}));

// Suppress unhandled fetch in tests
beforeAll(() => {
  global.fetch = jest.fn().mockRejectedValue(new Error('fetch not available'));
});

describe('HeroSection (landing mode)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initial render', () => {
    it('renders the main heading', () => {
      render(<HeroSection />);

      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
        /Don't trust one AI.*Make them compete/
      );
    });

    it('renders the subtitle', () => {
      render(<HeroSection />);

      expect(
        screen.getByText(/Pit Claude, GPT, Gemini, and Mistral/i)
      ).toBeInTheDocument();
    });

    it('renders the debate textarea', () => {
      render(<HeroSection />);

      expect(
        screen.getByPlaceholderText('What decision are you facing?')
      ).toBeInTheDocument();
    });

    it('renders the Start Debate button', () => {
      render(<HeroSection />);

      expect(
        screen.getByRole('button', { name: /start debate/i })
      ).toBeInTheDocument();
    });

    it('Start Debate button is disabled when input is empty', () => {
      render(<HeroSection />);

      const button = screen.getByRole('button', { name: /start debate/i });
      expect(button).toBeDisabled();
    });
  });

  describe('debate form interaction', () => {
    it('enables Start Debate button when text is entered', async () => {
      const user = userEvent.setup();
      render(<HeroSection />);

      const textarea = screen.getByPlaceholderText('What decision are you facing?');
      await user.type(textarea, 'Should we use TypeScript?');

      const button = screen.getByRole('button', { name: /start debate/i });
      expect(button).not.toBeDisabled();
    });
  });

  describe('dashboard mode', () => {
    const dashboardProps = {
      apiBase: 'http://localhost:8080',
      error: null,
      activeDebateId: null,
      activeQuestion: null,
      onDismissError: jest.fn(),
      onDebateStarted: jest.fn(),
      onError: jest.fn(),
    };

    it('shows error message when error prop is present', () => {
      render(<HeroSection {...dashboardProps} error="Something went wrong" />);

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('shows active debate indicator when debate is in progress', () => {
      render(
        <HeroSection
          {...dashboardProps}
          activeDebateId="debate-123"
          activeQuestion="Is AI beneficial?"
        />
      );

      expect(screen.getByText('DECISION IN PROGRESS')).toBeInTheDocument();
      expect(screen.getByText('Is AI beneficial?')).toBeInTheDocument();
    });
  });
});
