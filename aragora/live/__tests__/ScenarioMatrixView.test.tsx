/**
 * Tests for ScenarioMatrixView component
 *
 * Tests the matrix debate visualization with scenario grid,
 * comparison features, and filtering capabilities.
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Import after mocks are set up
import { ScenarioMatrixView } from '../src/components/ScenarioMatrixView';

const mockMatrixData = {
  matrix_id: 'matrix-1',
  base_topic: 'Best programming language for web development',
  created_at: '2024-01-15T10:00:00Z',
  status: 'completed',
  variables: [
    { name: 'framework', values: ['React', 'Vue', 'Angular'] },
    { name: 'scale', values: ['Small', 'Large'] },
  ],
  scenarios: [
    {
      id: 'scenario-1',
      variables: { framework: 'React', scale: 'Small' },
      debate_id: 'debate-react-small',
      result: {
        consensus_reached: true,
        confidence: 0.85,
        winner: 'claude',
        final_answer: 'React is excellent for small projects due to its simplicity.',
      },
    },
    {
      id: 'scenario-2',
      variables: { framework: 'React', scale: 'Large' },
      debate_id: 'debate-react-large',
      result: {
        consensus_reached: true,
        confidence: 0.78,
        winner: 'gpt4',
        final_answer: 'React scales well with proper architecture.',
      },
    },
    {
      id: 'scenario-3',
      variables: { framework: 'Vue', scale: 'Small' },
      debate_id: 'debate-vue-small',
      result: {
        consensus_reached: false,
        confidence: 0.45,
        winner: null,
        final_answer: null,
      },
    },
    {
      id: 'scenario-4',
      variables: { framework: 'Vue', scale: 'Large' },
      debate_id: 'debate-vue-large',
      result: null, // Pending
    },
    {
      id: 'scenario-5',
      variables: { framework: 'Angular', scale: 'Small' },
      debate_id: 'debate-angular-small',
      result: {
        consensus_reached: true,
        confidence: 0.72,
        winner: 'gemini',
        final_answer: 'Angular may be overkill for small projects.',
      },
    },
    {
      id: 'scenario-6',
      variables: { framework: 'Angular', scale: 'Large' },
      debate_id: 'debate-angular-large',
      result: {
        consensus_reached: true,
        confidence: 0.91,
        winner: 'claude',
        final_answer: 'Angular excels in large enterprise applications.',
      },
    },
  ],
};

const mockMatrixList = {
  matrices: [
    {
      matrix_id: 'matrix-1',
      base_topic: 'Best programming language for web development',
      created_at: '2024-01-15T10:00:00Z',
      status: 'completed',
      scenario_count: 6,
    },
    {
      matrix_id: 'matrix-2',
      base_topic: 'Cloud provider comparison',
      created_at: '2024-01-14T09:00:00Z',
      status: 'in_progress',
      scenario_count: 9,
    },
  ],
};

describe('ScenarioMatrixView', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('Rendering', () => {
    it('renders the component title', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMatrixList),
      });

      render(<ScenarioMatrixView />);

      expect(screen.getByText('Matrix Debates')).toBeInTheDocument();
    });

    it('shows loading state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<ScenarioMatrixView />);

      expect(screen.getByText('Loading matrices...')).toBeInTheDocument();
    });

    it('renders matrix list after loading', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMatrixList),
      });

      render(<ScenarioMatrixView />);

      await waitFor(() => {
        expect(screen.getByText('Best programming language for web development')).toBeInTheDocument();
        expect(screen.getByText('Cloud provider comparison')).toBeInTheDocument();
      });
    });

    it('shows empty state when no matrices', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ matrices: [] }),
      });

      render(<ScenarioMatrixView />);

      await waitFor(() => {
        expect(screen.getByText(/no matrix debates/i)).toBeInTheDocument();
      });
    });

    it('shows error state on fetch failure', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<ScenarioMatrixView />);

      await waitFor(() => {
        expect(screen.getByText(/failed to load/i)).toBeInTheDocument();
      });
    });
  });

  describe('Matrix Selection', () => {
    it('fetches matrix details when selected', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixList),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixData),
        });

      render(<ScenarioMatrixView />);

      await waitFor(() => {
        expect(screen.getByText('Best programming language for web development')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('Best programming language for web development'));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(2);
        expect(mockFetch).toHaveBeenLastCalledWith(
          expect.stringContaining('/matrix-1'),
          expect.any(Object)
        );
      });
    });

    it('auto-selects matrix when initialMatrixId is provided', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixList),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixData),
        });

      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/matrix-1'),
          expect.any(Object)
        );
      });
    });
  });

  describe('Matrix Grid Display', () => {
    beforeEach(() => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixList),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixData),
        });
    });

    it('displays scenario grid after selection', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        // Should show variable headers
        expect(screen.getByText('React')).toBeInTheDocument();
        expect(screen.getByText('Vue')).toBeInTheDocument();
        expect(screen.getByText('Angular')).toBeInTheDocument();
      });
    });

    it('shows consensus status for completed scenarios', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        // Should show checkmarks or X marks for consensus status
        const consensusIndicators = screen.getAllByTestId(/consensus-indicator/);
        expect(consensusIndicators.length).toBeGreaterThan(0);
      });
    });

    it('shows pending indicator for incomplete scenarios', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText(/pending/i)).toBeInTheDocument();
      });
    });

    it('displays confidence scores', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        // Should show confidence percentages
        expect(screen.getByText(/85%/)).toBeInTheDocument();
        expect(screen.getByText(/91%/)).toBeInTheDocument();
      });
    });
  });

  describe('Scenario Details', () => {
    beforeEach(() => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixList),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixData),
        });
    });

    it('shows scenario details when cell clicked', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText('React')).toBeInTheDocument();
      });

      // Click on a scenario cell
      const scenarioCells = screen.getAllByTestId(/scenario-cell/);
      fireEvent.click(scenarioCells[0]);

      await waitFor(() => {
        expect(screen.getByText(/scenario details/i)).toBeInTheDocument();
      });
    });

    it('displays winner in scenario details', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText('React')).toBeInTheDocument();
      });

      const scenarioCells = screen.getAllByTestId(/scenario-cell/);
      fireEvent.click(scenarioCells[0]);

      await waitFor(() => {
        expect(screen.getByText(/winner/i)).toBeInTheDocument();
      });
    });

    it('displays final answer preview', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText('React')).toBeInTheDocument();
      });

      const scenarioCells = screen.getAllByTestId(/scenario-cell/);
      fireEvent.click(scenarioCells[0]);

      await waitFor(() => {
        expect(screen.getByText(/excellent for small projects/i)).toBeInTheDocument();
      });
    });
  });

  describe('Filtering', () => {
    beforeEach(() => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixList),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixData),
        });
    });

    it('renders filter controls', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText(/filter/i)).toBeInTheDocument();
      });
    });

    it('filters by consensus status', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText('React')).toBeInTheDocument();
      });

      // Find and click consensus filter
      const consensusFilter = screen.getByRole('checkbox', { name: /consensus only/i });
      fireEvent.click(consensusFilter);

      await waitFor(() => {
        // Should hide non-consensus scenarios
        const scenarioCells = screen.getAllByTestId(/scenario-cell/);
        // Vue+Small had no consensus, should be filtered
        expect(scenarioCells.length).toBeLessThan(6);
      });
    });

    it('filters by minimum confidence', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText('React')).toBeInTheDocument();
      });

      // Find and adjust confidence slider
      const confidenceSlider = screen.getByRole('slider', { name: /confidence/i });
      fireEvent.change(confidenceSlider, { target: { value: '80' } });

      await waitFor(() => {
        // Should only show high-confidence scenarios
        expect(screen.getByText(/85%/)).toBeInTheDocument();
        expect(screen.getByText(/91%/)).toBeInTheDocument();
      });
    });
  });

  describe('Comparison Mode', () => {
    beforeEach(() => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixList),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixData),
        });
    });

    it('shows compare button when multiple scenarios selected', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText('React')).toBeInTheDocument();
      });

      // Select multiple scenarios (ctrl+click or shift+click simulation)
      const scenarioCells = screen.getAllByTestId(/scenario-cell/);
      fireEvent.click(scenarioCells[0], { ctrlKey: true });
      fireEvent.click(scenarioCells[1], { ctrlKey: true });

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /compare/i })).toBeInTheDocument();
      });
    });

    it('opens comparison view when compare clicked', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText('React')).toBeInTheDocument();
      });

      const scenarioCells = screen.getAllByTestId(/scenario-cell/);
      fireEvent.click(scenarioCells[0], { ctrlKey: true });
      fireEvent.click(scenarioCells[1], { ctrlKey: true });

      await waitFor(() => {
        const compareButton = screen.getByRole('button', { name: /compare/i });
        fireEvent.click(compareButton);
      });

      await waitFor(() => {
        expect(screen.getByText(/comparison/i)).toBeInTheDocument();
      });
    });
  });

  describe('Statistics Summary', () => {
    beforeEach(() => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixList),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockMatrixData),
        });
    });

    it('shows total scenario count', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText(/6 scenarios/i)).toBeInTheDocument();
      });
    });

    it('shows consensus rate', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        // 4 out of 5 completed scenarios reached consensus
        expect(screen.getByText(/80%/)).toBeInTheDocument();
      });
    });

    it('shows average confidence', async () => {
      render(<ScenarioMatrixView initialMatrixId="matrix-1" />);

      await waitFor(() => {
        expect(screen.getByText(/avg confidence/i)).toBeInTheDocument();
      });
    });
  });

  describe('Refresh', () => {
    it('renders refresh button', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMatrixList),
      });

      render(<ScenarioMatrixView />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
      });
    });

    it('refetches matrices when refresh clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMatrixList),
      });

      render(<ScenarioMatrixView />);

      await waitFor(() => {
        expect(screen.getByText('Best programming language for web development')).toBeInTheDocument();
      });

      mockFetch.mockClear();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockMatrixList),
      });

      fireEvent.click(screen.getByRole('button', { name: /refresh/i }));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });
    });
  });
});
