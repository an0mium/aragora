import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DebateInput } from '../DebateInput';

// Mock next/navigation
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock config
jest.mock('@/config', () => ({
  DEFAULT_AGENTS: 'grok,anthropic-api,openai-api',
  DEFAULT_ROUNDS: 8,  // 9-round format default
  DEFAULT_CONSENSUS: 'judge',
  AGENT_DISPLAY_NAMES: {
    grok: 'Grok 4',
    'anthropic-api': 'Claude Opus 4.5',
    'openai-api': 'GPT 5.2',
  },
}));

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('DebateInput', () => {
  const defaultProps = {
    apiBase: 'http://localhost:8080',
    onDebateStarted: jest.fn(),
    onError: jest.fn(),
  };
  const advanceTimers = async (ms: number) => {
    await act(async () => {
      await jest.advanceTimersByTimeAsync(ms);
    });
  };
  const actUser = async (action: () => Promise<void>) => {
    await act(async () => {
      await action();
    });
  };
  const queueHealthCheckSuccess = () => {
    mockFetch.mockImplementationOnce(() =>
      new Promise((resolve) => {
        setTimeout(() => {
          resolve({ ok: true, json: async () => ({ status: 'ok' }) });
        }, 0);
      })
    );
  };
  const queueHealthCheckFailure = (error: Error) => {
    mockFetch.mockImplementationOnce(() =>
      new Promise((_, reject) => {
        setTimeout(() => {
          reject(error);
        }, 0);
      })
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers({ advanceTimers: true });

    // Default: API is online
    mockFetch.mockImplementation(() =>
      new Promise((resolve) => {
        setTimeout(() => {
          resolve({ ok: true, json: async () => ({ status: 'ok' }) });
        }, 0);
      })
    );
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('initial render', () => {
    it('renders the debate input textarea', async () => {
      render(<DebateInput {...defaultProps} />);

      // Advance timers for health check
      await advanceTimers(100);

      expect(screen.getByLabelText(/enter your debate question/i)).toBeInTheDocument();
    });

    it('shows connecting state initially', () => {
      render(<DebateInput {...defaultProps} />);

      expect(screen.getByText('CONNECTING...')).toBeInTheDocument();
    });

    it('shows START DEBATE button when online', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });
    });

    it('has a placeholder question', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      expect(textarea).toHaveAttribute('placeholder');
      expect(textarea.getAttribute('placeholder')).not.toBe('');
    });
  });

  describe('API status', () => {
    it('shows online status when API is healthy', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });
    });

    it('shows offline banner when API is unhealthy', async () => {
      queueHealthCheckFailure(new Error('Network error'));

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('API server offline')).toBeInTheDocument();
      });
    });

    it('disables submit when offline', async () => {
      queueHealthCheckFailure(new Error('Network error'));

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('OFFLINE')).toBeInTheDocument();
      });
    });
  });

  describe('form submission', () => {
    it('submits debate with question', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      queueHealthCheckSuccess();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, debate_id: 'debate-123' }),
      }); // submit

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });

      const textarea = screen.getByLabelText(/enter your debate question/i);
      await actUser(() => user.type(textarea, 'What is the meaning of life?'));
      await actUser(() => user.click(screen.getByLabelText('Start debate')));
      await waitFor(() => {
        expect(defaultProps.onDebateStarted).toHaveBeenCalledWith(
          'debate-123',
          'What is the meaning of life?'
        );
      });
    });

    it('uses placeholder when question is empty', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      queueHealthCheckSuccess();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, debate_id: 'debate-456' }),
      }); // submit

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });

      await actUser(() => user.click(screen.getByLabelText('Start debate')));
      await waitFor(() => {
        expect(defaultProps.onDebateStarted).toHaveBeenCalled();
      });

      // Should have used placeholder text
      const callArgs = (mockFetch.mock.calls[1][1] as RequestInit).body;
      const parsed = JSON.parse(callArgs as string);
      expect(parsed.question).toBeTruthy();
    });

    it('calls onError when submission fails', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      queueHealthCheckSuccess();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: false, error: 'Server error' }),
      }); // submit

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });

      const textarea = screen.getByLabelText(/enter your debate question/i);
      await actUser(() => user.type(textarea, 'Test question'));
      await actUser(() => user.click(screen.getByLabelText('Start debate')));
      await waitFor(() => {
        expect(defaultProps.onError).toHaveBeenCalledWith('Server error');
      });
    });

    it('shows STARTING state during submission', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      queueHealthCheckSuccess();
      mockFetch.mockImplementationOnce(() => new Promise(() => {})); // Never resolves

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });

      const textarea = screen.getByLabelText(/enter your debate question/i);
      await actUser(() => user.type(textarea, 'Test question'));
      await actUser(() => user.click(screen.getByLabelText('Start debate')));
      expect(screen.getByText('STARTING...')).toBeInTheDocument();
    });
  });

  describe('advanced options', () => {
    it('hides advanced options by default', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      expect(screen.queryByLabelText(/debate mode/i)).not.toBeInTheDocument();
    });

    it('shows advanced options when toggled', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      expect(screen.getByText('DEBATE MODE')).toBeInTheDocument();
      expect(screen.getByLabelText(/agents/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/debate rounds/i)).toBeInTheDocument();
    });

    it('toggles advanced options text', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      expect(screen.getByText('[-] Hide options')).toBeInTheDocument();

      await actUser(() => user.click(screen.getByText('[-] Hide options')));
      expect(screen.getByText('[+] Show options')).toBeInTheDocument();
    });
  });

  describe('debate modes', () => {
    it('has three mode options', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      expect(screen.getByRole('tab', { name: /standard/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /graph/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /matrix/i })).toBeInTheDocument();
    });

    it('standard mode is selected by default', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      const standardTab = screen.getByRole('tab', { name: /standard/i });
      expect(standardTab).toHaveAttribute('aria-selected', 'true');
    });

    it('can switch to graph mode', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      await actUser(() => user.click(screen.getByRole('tab', { name: /graph/i })));
      const graphTab = screen.getByRole('tab', { name: /graph/i });
      expect(graphTab).toHaveAttribute('aria-selected', 'true');
    });

    it('navigates to graph view on graph debate submission', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      queueHealthCheckSuccess();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, debate_id: 'graph-123' }),
      }); // submit

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      await actUser(() => user.click(screen.getByRole('tab', { name: /graph/i })));
      const textarea = screen.getByLabelText(/enter your debate question/i);
      await actUser(() => user.type(textarea, 'Test question'));
      await actUser(() => user.click(screen.getByLabelText('Start debate')));
      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith('/debates/graph?id=graph-123');
      });
    });
  });

  describe('agent configuration', () => {
    it('allows changing agents', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      const agentsInput = screen.getByLabelText(/agents/i);
      await actUser(() => user.clear(agentsInput));
      await actUser(() => user.type(agentsInput, 'claude,gemini'));
      expect(agentsInput).toHaveValue('claude,gemini');
    });

    it('allows changing round count', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      const roundsSelect = screen.getByLabelText(/debate rounds/i);
      await actUser(() => user.selectOptions(roundsSelect, '5'));
      expect(roundsSelect).toHaveValue('5');
    });
  });

  describe('domain detection', () => {
    it('shows domain hint for technical questions', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      queueHealthCheckSuccess();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          recommendations: [
            { agent: 'codex', suitability: 0.9, domain_match: true },
          ],
        }),
      }); // recommendations

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      await actUser(() => user.type(textarea, 'How should I implement this API?'));
      // Wait for debounce
      await advanceTimers(600);

      await waitFor(() => {
        expect(screen.getByText('[TECHNICAL]')).toBeInTheDocument();
      });
    });
  });

  describe('keyboard shortcuts', () => {
    it('submits on Ctrl+Enter', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      queueHealthCheckSuccess();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, debate_id: 'debate-789' }),
      }); // submit

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });

      const textarea = screen.getByLabelText(/enter your debate question/i);
      await actUser(() => user.type(textarea, 'Test question'));
      await actUser(() => user.type(textarea, '{Control>}{Enter}{/Control}'));
      await waitFor(() => {
        expect(defaultProps.onDebateStarted).toHaveBeenCalled();
      });
    });
  });

  describe('character count', () => {
    it('shows character count when typing', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      await actUser(() => user.type(textarea, 'Hello world'));
      expect(screen.getByText('11 chars')).toBeInTheDocument();
    });

    it('shows keyboard hint when empty', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      expect(screen.getByText('Cmd+Enter to debate this question')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has accessible textarea with label', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      expect(textarea).toBeInTheDocument();
      expect(textarea).toHaveAttribute('aria-describedby', 'debate-input-hint');
    });

    it('has accessible options toggle', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      const toggle = screen.getByLabelText(/show advanced options/i);
      expect(toggle).toHaveAttribute('aria-expanded', 'false');
      expect(toggle).toHaveAttribute('aria-controls', 'advanced-options');
    });

    it('has accessible submit button', async () => {
      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await waitFor(() => {
        expect(screen.getByLabelText('Start debate')).toBeInTheDocument();
      });
    });

    it('mode tabs have proper ARIA attributes', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

      render(<DebateInput {...defaultProps} />);

      await advanceTimers(100);

      await actUser(() => user.click(screen.getByText('[+] Show options')));
      const tablist = screen.getByRole('tablist');
      expect(tablist).toHaveAttribute('aria-labelledby', 'debate-mode-label');
    });
  });
});
