/**
 * Tests for DebateInput mode switching functionality
 *
 * Tests the STANDARD, GRAPH, and MATRIX mode selection and
 * navigation to the appropriate results pages.
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock next/navigation
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: jest.fn(),
    prefetch: jest.fn(),
  }),
  useSearchParams: () => ({
    get: jest.fn(),
  }),
}));

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

// Import after mocks
import { DebateInput } from '../src/components/DebateInput';

describe('DebateInput Mode Switching', () => {
  const apiBase = 'http://localhost:8080';

  beforeEach(() => {
    mockFetch.mockClear();
    mockPush.mockClear();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  const setupHealthyApi = () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/health')) {
        return Promise.resolve({ ok: true });
      }
      return Promise.resolve({ ok: false });
    });
  };

  describe('Mode Selection UI', () => {
    it('renders all three mode buttons', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      expect(screen.getByRole('button', { name: /standard/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /graph/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /matrix/i })).toBeInTheDocument();
    });

    it('defaults to STANDARD mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const standardButton = screen.getByRole('button', { name: /standard/i });
      expect(standardButton).toHaveClass('active');
    });

    it('switches to GRAPH mode when clicked', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.click(graphButton);

      expect(graphButton).toHaveClass('active');
    });

    it('switches to MATRIX mode when clicked', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const matrixButton = screen.getByRole('button', { name: /matrix/i });
      fireEvent.click(matrixButton);

      expect(matrixButton).toHaveClass('active');
    });

    it('shows mode description when hovering', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.mouseEnter(graphButton);

      await waitFor(() => {
        expect(screen.getByText(/branching debate/i)).toBeInTheDocument();
      });
    });
  });

  describe('STANDARD Mode Submission', () => {
    it('calls /api/debate for STANDARD mode', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debate') && !url.includes('/graph') && !url.includes('/matrix')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              debate_id: 'standard-debate-123',
            }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const onDebateStarted = jest.fn();
      render(<DebateInput apiBase={apiBase} onDebateStarted={onDebateStarted} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test standard debate' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/debate'),
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });

    it('does not navigate away for STANDARD mode', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debate')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              debate_id: 'standard-debate-123',
            }),
          });
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
        expect(mockPush).not.toHaveBeenCalled();
      });
    });
  });

  describe('GRAPH Mode Submission', () => {
    it('calls /api/debates/graph for GRAPH mode', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debates/graph')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              debate_id: 'graph-debate-123',
            }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to GRAPH mode
      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.click(graphButton);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test graph debate' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/debates/graph'),
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });

    it('navigates to /debates/graph after GRAPH debate starts', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debates/graph')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              debate_id: 'graph-debate-123',
            }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to GRAPH mode
      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.click(graphButton);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test graph debate' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith('/debates/graph?id=graph-debate-123');
      });
    });
  });

  describe('MATRIX Mode Submission', () => {
    it('calls /api/debates/matrix for MATRIX mode', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debates/matrix')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              matrix_id: 'matrix-123',
            }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to MATRIX mode
      const matrixButton = screen.getByRole('button', { name: /matrix/i });
      fireEvent.click(matrixButton);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test matrix debate' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/debates/matrix'),
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });

    it('navigates to /debates/matrix after MATRIX debate starts', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debates/matrix')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              matrix_id: 'matrix-123',
            }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to MATRIX mode
      const matrixButton = screen.getByRole('button', { name: /matrix/i });
      fireEvent.click(matrixButton);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test matrix debate' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith('/debates/matrix?id=matrix-123');
      });
    });

    it('shows matrix variables configuration in MATRIX mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to MATRIX mode
      const matrixButton = screen.getByRole('button', { name: /matrix/i });
      fireEvent.click(matrixButton);

      // Show options
      fireEvent.click(screen.getByText('[+] Show options'));

      await waitFor(() => {
        expect(screen.getByText(/variables/i)).toBeInTheDocument();
      });
    });
  });

  describe('Mode-specific UI Changes', () => {
    it('shows branching hint in GRAPH mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.click(graphButton);

      await waitFor(() => {
        expect(screen.getByText(/explore multiple branches/i)).toBeInTheDocument();
      });
    });

    it('shows scenario hint in MATRIX mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      const matrixButton = screen.getByRole('button', { name: /matrix/i });
      fireEvent.click(matrixButton);

      await waitFor(() => {
        expect(screen.getByText(/compare scenarios/i)).toBeInTheDocument();
      });
    });

    it('updates submit button text based on mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Default STANDARD mode
      expect(screen.getByRole('button', { name: /start debate/i })).toBeInTheDocument();

      // Switch to GRAPH
      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.click(graphButton);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /start graph debate/i })).toBeInTheDocument();
      });

      // Switch to MATRIX
      const matrixButton = screen.getByRole('button', { name: /matrix/i });
      fireEvent.click(matrixButton);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /start matrix/i })).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling by Mode', () => {
    it('shows appropriate error for failed GRAPH debate', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debates/graph')) {
          return Promise.resolve({
            ok: false,
            json: () => Promise.resolve({ error: 'Graph debates require at least 2 agents' }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const onError = jest.fn();
      render(<DebateInput apiBase={apiBase} onError={onError} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to GRAPH mode
      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.click(graphButton);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start graph debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith('Graph debates require at least 2 agents');
      });
    });

    it('shows appropriate error for failed MATRIX debate', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debates/matrix')) {
          return Promise.resolve({
            ok: false,
            json: () => Promise.resolve({ error: 'Matrix debates require variables' }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const onError = jest.fn();
      render(<DebateInput apiBase={apiBase} onError={onError} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to MATRIX mode
      const matrixButton = screen.getByRole('button', { name: /matrix/i });
      fireEvent.click(matrixButton);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start matrix/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith('Matrix debates require variables');
      });
    });
  });

  describe('Mode Persistence', () => {
    it('remembers mode selection after submission', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/health')) {
          return Promise.resolve({ ok: true });
        }
        if (url.includes('/api/debates/graph')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              success: true,
              debate_id: 'graph-debate-123',
            }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      render(<DebateInput apiBase={apiBase} />);

      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      // Switch to GRAPH mode
      const graphButton = screen.getByRole('button', { name: /graph/i });
      fireEvent.click(graphButton);

      const textarea = screen.getByRole('textbox');
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start graph debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        // After submission, mode should still be GRAPH
        expect(graphButton).toHaveClass('active');
      });
    });
  });
});
