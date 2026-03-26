import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AgentNetworkPanel } from '../AgentNetworkPanel';

const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('AgentNetworkPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ leaderboard: [{ name: 'claude' }, { name: 'gpt-4' }] }),
    });
  });

  it('loads selectable agents from leaderboard payloads', async () => {
    const user = userEvent.setup();
    render(<AgentNetworkPanel />);

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /expand agent network panel/i }));
    });

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toHaveValue('claude');
    });

    expect(screen.getByRole('option', { name: 'claude' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'gpt-4' })).toBeInTheDocument();
  });
});
