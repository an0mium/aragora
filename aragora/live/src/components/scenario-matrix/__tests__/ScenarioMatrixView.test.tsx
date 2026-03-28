import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ScenarioMatrixView } from '..';

const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;

function createMatrixResult(overrides?: {
  results?: Array<{
    scenario_name: string;
    parameters: Record<string, unknown>;
    constraints: string[];
    is_baseline: boolean;
    winner: string | null;
    final_answer: string | null;
    confidence: number;
    consensus_reached: boolean;
    rounds_used: number;
  }>;
}) {
  const results = overrides?.results ?? [
    {
      scenario_name: 'Baseline',
      parameters: { budget: 'low' },
      constraints: ['Keep costs fixed'],
      is_baseline: true,
      winner: 'claude',
      final_answer: 'Stay with the baseline staffing plan.',
      confidence: 0.62,
      consensus_reached: true,
      rounds_used: 3,
    },
    {
      scenario_name: 'Add Critics',
      parameters: { critics: 2 },
      constraints: [],
      is_baseline: false,
      winner: 'codex',
      final_answer: 'Use two critics for stronger adversarial coverage.',
      confidence: 0.84,
      consensus_reached: true,
      rounds_used: 4,
    },
    {
      scenario_name: 'Judge Finish',
      parameters: { judge: true },
      constraints: [],
      is_baseline: false,
      winner: null,
      final_answer: 'Add a judge-only final round.',
      confidence: 0.41,
      consensus_reached: false,
      rounds_used: 5,
    },
  ];

  return {
    matrix_id: 'matrix-123',
    task: 'Compare debate configurations',
    scenario_count: results.length,
    results,
    universal_conclusions: ['Structured critique improves decision quality.'],
    conditional_conclusions: [],
    comparison_matrix: {
      scenarios: results.map((result) => result.scenario_name),
      consensus_rate: 2 / 3,
      avg_confidence: 0.6233,
      avg_rounds: 4,
    },
  };
}

function mockMatrixFetch(result = createMatrixResult()) {
  mockFetch.mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => result,
  } as Response);
}

describe('ScenarioMatrixView', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('opens and closes the side-by-side comparison panel from grid selections', async () => {
    const user = userEvent.setup();

    mockMatrixFetch();

    render(<ScenarioMatrixView initialMatrixId="matrix-123" />);

    expect(await screen.findByText(/SCENARIO RESULTS \(3\/3\)/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'GRID' }));

    const grid = screen.getByRole('grid', { name: 'Scenario comparison grid' });
    const baselineCell = within(grid).getByRole('gridcell', { name: /Baseline \(baseline\)/i });
    const criticsCell = within(grid).getByRole('gridcell', { name: /Add Critics/i });

    await user.click(baselineCell);
    await user.click(criticsCell);

    expect(await screen.findByText('SCENARIO COMPARISON')).toBeInTheDocument();
    expect(screen.getByText('Baseline')).toBeInTheDocument();
    expect(screen.getByText('Add Critics')).toBeInTheDocument();
    expect(screen.getAllByText('[BASELINE]')).toHaveLength(1);

    await user.click(screen.getByRole('button', { name: '[CLOSE]' }));

    await waitFor(() => {
      expect(screen.queryByText('SCENARIO COMPARISON')).not.toBeInTheDocument();
    });
  });

  it('filters results safely when a scenario is missing a final answer', async () => {
    const user = userEvent.setup();

    mockMatrixFetch(
      createMatrixResult({
        results: [
          {
            scenario_name: 'Baseline',
            parameters: { budget: 'low' },
            constraints: [],
            is_baseline: true,
            winner: null,
            final_answer: null,
            confidence: 0.55,
            consensus_reached: false,
            rounds_used: 2,
          },
          {
            scenario_name: 'Alternative',
            parameters: { budget: 'high' },
            constraints: [],
            is_baseline: false,
            winner: 'claude',
            final_answer: 'Escalate to the stronger agent mix.',
            confidence: 0.88,
            consensus_reached: true,
            rounds_used: 4,
          },
        ],
      }),
    );

    render(<ScenarioMatrixView initialMatrixId="matrix-123" />);

    expect(await screen.findByText(/SCENARIO RESULTS \(2\/2\)/i)).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText('Search scenarios...'), 'alternative');

    await waitFor(() => {
      expect(screen.getByText(/SCENARIO RESULTS \(1\/2\)/i)).toBeInTheDocument();
    });

    expect(screen.getByText('Alternative')).toBeInTheDocument();
    expect(screen.queryByText('Baseline')).not.toBeInTheDocument();
    expect(screen.queryByText(/^Error:/)).not.toBeInTheDocument();
  });
});
