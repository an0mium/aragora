/**
 * Tests for DebateInput mode switching functionality
 *
 * Tests the STANDARD, GRAPH, and MATRIX mode selection and
 * navigation to the appropriate results pages.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { mockRouter } from 'next/navigation';

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
    mockRouter.push.mockClear();
  });

  const setupHealthyApi = () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/health')) {
        return Promise.resolve({ ok: true });
      }
      return Promise.resolve({ ok: false });
    });
  };

  const waitForApiOnline = async () => {
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /start debate/i })).toBeEnabled();
    });
  };

  const openAdvancedOptions = () => {
    fireEvent.click(screen.getByRole('button', { name: /show advanced options/i }));
  };

  describe('Mode Selection UI', () => {
    it('renders all three mode tabs', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      expect(screen.getByRole('tab', { name: /standard/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /graph/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /matrix/i })).toBeInTheDocument();
    });

    it('defaults to STANDARD mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      const standardTab = screen.getByRole('tab', { name: /standard/i });
      expect(standardTab).toHaveAttribute('aria-selected', 'true');
    });

    it('switches to GRAPH mode when clicked', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      expect(graphTab).toHaveAttribute('aria-selected', 'true');
    });

    it('switches to MATRIX mode when clicked', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      const matrixTab = screen.getByRole('tab', { name: /matrix/i });
      fireEvent.click(matrixTab);

      expect(matrixTab).toHaveAttribute('aria-selected', 'true');
    });

    it('shows the description for the selected mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      expect(screen.getByText(/linear debate with critique rounds/i)).toBeInTheDocument();

      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      expect(screen.getByText(/branching debate exploring multiple paths/i)).toBeInTheDocument();
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

      await waitForApiOnline();

      const textarea = screen.getByLabelText(/enter your debate question/i);
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

      await waitForApiOnline();

      const textarea = screen.getByLabelText(/enter your debate question/i);
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockRouter.push).not.toHaveBeenCalled();
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

      await waitForApiOnline();
      openAdvancedOptions();

      // Switch to GRAPH mode
      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      const textarea = screen.getByLabelText(/enter your debate question/i);
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

      await waitForApiOnline();
      openAdvancedOptions();

      // Switch to GRAPH mode
      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      fireEvent.change(textarea, { target: { value: 'Test graph debate' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockRouter.push).toHaveBeenCalledWith('/debates/graph?id=graph-debate-123');
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

      await waitForApiOnline();
      openAdvancedOptions();

      // Switch to MATRIX mode
      const matrixTab = screen.getByRole('tab', { name: /matrix/i });
      fireEvent.click(matrixTab);

      const textarea = screen.getByLabelText(/enter your debate question/i);
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

      await waitForApiOnline();
      openAdvancedOptions();

      // Switch to MATRIX mode
      const matrixTab = screen.getByRole('tab', { name: /matrix/i });
      fireEvent.click(matrixTab);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      fireEvent.change(textarea, { target: { value: 'Test matrix debate' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(mockRouter.push).toHaveBeenCalledWith('/debates/matrix?id=matrix-123');
      });
    });

    it('shows matrix mode description in advanced options', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();

      openAdvancedOptions();
      const matrixTab = screen.getByRole('tab', { name: /matrix/i });
      fireEvent.click(matrixTab);

      await waitFor(() => {
        expect(screen.getByText(/parallel scenarios for comparison/i)).toBeInTheDocument();
      });
    });
  });

  describe('Mode-specific UI Changes', () => {
    it('updates description in GRAPH mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      expect(screen.getByText(/branching debate exploring multiple paths/i)).toBeInTheDocument();
    });

    it('updates description in MATRIX mode', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      const matrixTab = screen.getByRole('tab', { name: /matrix/i });
      fireEvent.click(matrixTab);

      expect(screen.getByText(/parallel scenarios for comparison/i)).toBeInTheDocument();
    });

    it('keeps the submit button label consistent across modes', async () => {
      setupHealthyApi();
      render(<DebateInput apiBase={apiBase} />);

      await waitForApiOnline();
      openAdvancedOptions();

      // Default STANDARD mode
      expect(screen.getByRole('button', { name: /start debate/i })).toBeInTheDocument();

      // Switch to GRAPH
      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      expect(screen.getByRole('button', { name: /start debate/i })).toBeInTheDocument();

      // Switch to MATRIX
      const matrixTab = screen.getByRole('tab', { name: /matrix/i });
      fireEvent.click(matrixTab);

      expect(screen.getByRole('button', { name: /start debate/i })).toBeInTheDocument();
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

      await waitForApiOnline();
      openAdvancedOptions();

      // Switch to GRAPH mode
      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
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

      await waitForApiOnline();
      openAdvancedOptions();

      // Switch to MATRIX mode
      const matrixTab = screen.getByRole('tab', { name: /matrix/i });
      fireEvent.click(matrixTab);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
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

      await waitForApiOnline();
      openAdvancedOptions();

      // Switch to GRAPH mode
      const graphTab = screen.getByRole('tab', { name: /graph/i });
      fireEvent.click(graphTab);

      const textarea = screen.getByLabelText(/enter your debate question/i);
      fireEvent.change(textarea, { target: { value: 'Test' } });

      const submitButton = screen.getByRole('button', { name: /start debate/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        // After submission, mode should still be GRAPH
        expect(graphTab).toHaveAttribute('aria-selected', 'true');
      });
    });
  });
});
