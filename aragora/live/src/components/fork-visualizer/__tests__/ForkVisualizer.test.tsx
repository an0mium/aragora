import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ForkVisualizer } from '../ForkVisualizer';

type MockResponse = {
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
};

const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;

describe('ForkVisualizer', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('switches to the side-by-side comparison view after selecting two forks', async () => {
    const user = userEvent.setup();
    const tree = {
      id: 'debate-123',
      type: 'root' as const,
      branch_point: 0,
      children: [
        {
          id: 'fork-alpha',
          type: 'fork' as const,
          branch_point: 3,
          pivot_claim: 'Use two critics',
          status: 'completed',
          modified_context: 'Increase adversarial pressure',
          messages_inherited: 3,
          children: [],
        },
        {
          id: 'fork-beta',
          type: 'fork' as const,
          branch_point: 5,
          pivot_claim: 'Use a judge',
          status: 'created',
          modified_context: 'Add a judge role',
          messages_inherited: 5,
          children: [],
        },
      ],
      total_nodes: 3,
      max_depth: 1,
    };

    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        forks: tree.children.map((child) => ({
          branch_id: child.id,
          parent_debate_id: tree.id,
          branch_point: child.branch_point,
          pivot_claim: child.pivot_claim,
          modified_context: child.modified_context,
          status: child.status,
          messages_inherited: child.messages_inherited,
        })),
        tree,
      }),
    } as MockResponse as Response);

    render(<ForkVisualizer debateId="debate-123" />);

    expect(await screen.findByText('Fork @ R3')).toBeInTheDocument();
    expect(screen.getByText('Fork @ R5')).toBeInTheDocument();

    const compareButtons = screen.getAllByRole('button', { name: 'COMPARE' });
    expect(compareButtons).toHaveLength(1);
    expect(compareButtons[0]).toBeDisabled();

    const leftButtons = screen.getAllByRole('button', { name: 'L' });
    const rightButtons = screen.getAllByRole('button', { name: 'R' });
    expect(leftButtons).toHaveLength(3);
    expect(rightButtons).toHaveLength(3);

    await user.click(leftButtons[1]);
    await user.click(rightButtons[2]);

    await waitFor(() => {
      expect(screen.getByText(/COMPARING FORK-fork-a vs FORK-fork-b/i)).toBeInTheDocument();
    });

    expect(screen.getByText('OUTCOME DIFFERENCES')).toBeInTheDocument();
    expect(screen.getByText('Round 3')).toBeInTheDocument();
    expect(screen.getByText('Messages: 3')).toBeInTheDocument();
    expect(screen.getByText('Messages: 5')).toBeInTheDocument();

    const compareTab = screen.getByRole('button', { name: 'COMPARE' });
    expect(compareTab).toBeEnabled();

    const differences = screen.getAllByText(/status|pivot claim|messages inherited/i);
    expect(differences).toHaveLength(3);

    const comparePanel = screen.getByText(/COMPARING FORK-fork-a vs FORK-fork-b/i).closest('div');
    expect(comparePanel).not.toBeNull();
    expect(within(comparePanel as HTMLElement).getByRole('button', { name: '[CLEAR]' })).toBeInTheDocument();
  });
});
