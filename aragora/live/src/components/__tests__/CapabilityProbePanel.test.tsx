import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CapabilityProbePanel } from '../CapabilityProbePanel';

const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('CapabilityProbePanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ leaderboard: [{ name: 'claude' }, { name: 'gpt-4' }] }),
    });
  });

  it('loads target agents from leaderboard payloads', async () => {
    const user = userEvent.setup();
    render(<CapabilityProbePanel />);

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /expand capability probes panel/i }));
    });

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toHaveValue('claude');
    });

    expect(screen.getByRole('option', { name: 'claude' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'gpt-4' })).toBeInTheDocument();
  });
});
