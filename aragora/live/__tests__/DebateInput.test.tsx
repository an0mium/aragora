/**
 * Tests for DebateInput component
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DebateInput } from '../src/components/DebateInput';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock config
jest.mock('../src/config', () => ({
  DEFAULT_AGENTS: 'claude,gemini,gpt4',
  DEFAULT_ROUNDS: 3,
  AGENT_DISPLAY_NAMES: {
    claude: 'Claude',
    gemini: 'Gemini',
    gpt4: 'GPT-4',
  },
}));

describe('DebateInput', () => {
  const apiBase = 'http://localhost:8080';

  beforeEach(() => {
    mockFetch.mockClear();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('API Status', () => {
    it('shows checking state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves
      render(<DebateInput apiBase={apiBase} />);

      expect(screen.getByText('CONNECTING...')).toBeInTheDocument();
    });

    it('shows online state when API is healthy', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        return Promise.reject(new Error('Not found'));
      });

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      await waitFor(() => {
        expect(screen.getByText('START DEBATE')).toBeInTheDocument();
      });
    });

    it('shows offline state when API is unavailable', async () => {
      mockFetch.mockImplementation(() => Promise.reject(new Error('Network error')));

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(6000); // Timeout
      });

      await waitFor(() => {
        expect(screen.getByText('OFFLINE')).toBeInTheDocument();
      });
    });

    it('displays offline banner with helpful message', async () => {
      mockFetch.mockImplementation(() => Promise.reject(new Error('Network error')));

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(6000);
      });

      await waitFor(() => {
        expect(screen.getByText('API server offline')).toBeInTheDocument();
      });
    });
  });

  describe('Input Field', () => {
    beforeEach(() => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        return Promise.resolve({ ok: false });
      });
    });

    it('renders textarea for question input', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    it('accepts user input', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'What is AI?' } });

      expect(textarea).toHaveValue('What is AI?');
    });

    it('shows character count when typing', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Hello World' } });

      expect(screen.getByText('11 chars')).toBeInTheDocument();
    });

    it('shows keyboard hint when empty', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      expect(screen.getByText('Cmd+Enter to debate this question')).toBeInTheDocument();
    });
  });

  describe('Submit Behavior', () => {
    beforeEach(() => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debate')) {
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                success: true,
                debate_id: 'test-debate-123',
              }),
          });
        }
        return Promise.resolve({ ok: false });
      });
    });

    it('calls onDebateStarted when debate starts successfully', async () => {
      const onDebateStarted = jest.fn();
      render(<DebateInput apiBase={apiBase} onDebateStarted={onDebateStarted} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test question' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(onDebateStarted).toHaveBeenCalledWith('test-debate-123', 'Test question');
      });
    });

    it('uses placeholder when input is empty', async () => {
      const onDebateStarted = jest.fn();
      render(<DebateInput apiBase={apiBase} onDebateStarted={onDebateStarted} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Don't type anything, just submit
      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(onDebateStarted).toHaveBeenCalled();
        // The second argument should be the placeholder (one of the random questions)
        const callArgs = onDebateStarted.mock.calls[0];
        expect(callArgs[1]).toBeTruthy();
        expect(callArgs[1].length).toBeGreaterThan(10);
      });
    });

    it('clears input after successful submit', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test question' } });
      expect(textarea).toHaveValue('Test question');

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(textarea).toHaveValue('');
      });
    });

    it('disables submit button while submitting', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debate')) {
          return new Promise(() => {}); // Never resolves
        }
        return Promise.resolve({ ok: false });
      });

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('STARTING...')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('calls onError when debate fails', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debate')) {
          return Promise.resolve({
            ok: false,
            json: () => Promise.resolve({ error: 'Server error' }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const onError = jest.fn();
      render(<DebateInput apiBase={apiBase} onError={onError} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith('Server error');
      });
    });

    it('handles network errors gracefully', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debate')) {
          const error = new TypeError('Failed to fetch');
          return Promise.reject(error);
        }
        return Promise.resolve({ ok: false });
      });

      const onError = jest.fn();
      render(<DebateInput apiBase={apiBase} onError={onError} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith(
          expect.stringContaining('unavailable')
        );
      });
    });
  });

  describe('Advanced Options', () => {
    beforeEach(() => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        return Promise.resolve({ ok: false });
      });
    });

    it('hides advanced options by default', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      expect(screen.queryByText('AGENTS (comma-separated)')).not.toBeInTheDocument();
    });

    it('shows advanced options when toggled', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      fireEvent.click(screen.getByText('[+] Show options'));

      expect(screen.getByText('AGENTS (comma-separated)')).toBeInTheDocument();
      expect(screen.getByText('DEBATE ROUNDS')).toBeInTheDocument();
    });

    it('allows changing agents', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      fireEvent.click(screen.getByText('[+] Show options'));

      const agentInput = screen.getByPlaceholderText(/grok/);
      fireEvent.change(agentInput, { target: { value: 'custom-agent' } });

      expect(agentInput).toHaveValue('custom-agent');
    });

    it('allows changing rounds', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      fireEvent.click(screen.getByText('[+] Show options'));

      const roundsSelect = screen.getByRole('combobox');
      fireEvent.change(roundsSelect, { target: { value: '5' } });

      expect(roundsSelect).toHaveValue('5');
    });
  });

  describe('Domain Detection', () => {
    beforeEach(() => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/routing/recommendations')) {
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                recommendations: [
                  { agent: 'codex', suitability: 0.9, domain_match: true },
                  { agent: 'claude', suitability: 0.8, domain_match: true },
                ],
              }),
          });
        }
        return Promise.resolve({ ok: false });
      });
    });

    it('detects technical domain', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'How do I fix this JavaScript bug?' } });

      await act(async () => {
        jest.advanceTimersByTime(600); // Debounce
      });

      await waitFor(() => {
        expect(screen.getByText('[TECHNICAL]')).toBeInTheDocument();
      });
    });

    it('shows agent recommendations for detected domain', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'What is the best Python framework?' } });

      await act(async () => {
        jest.advanceTimersByTime(600);
      });

      await waitFor(() => {
        expect(screen.getByText('codex')).toBeInTheDocument();
      });
    });

    it('allows applying recommended agents', async () => {
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Type in question first (before showing advanced options)
      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'How to write TypeScript code?' } });

      await act(async () => {
        jest.advanceTimersByTime(600);
      });

      // Now show advanced options to see agents field
      fireEvent.click(screen.getByText('[+] Show options'));

      await waitFor(() => {
        const useButton = screen.getByText('[USE]');
        fireEvent.click(useButton);
      });

      await waitFor(() => {
        const agentInput = screen.getByPlaceholderText(/grok/);
        expect(agentInput).toHaveValue('codex,claude');
      });
    });
  });
});
