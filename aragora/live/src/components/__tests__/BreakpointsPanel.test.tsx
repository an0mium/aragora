import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { BreakpointsPanel } from '../BreakpointsPanel';
import { useAuth } from '@/context/AuthContext';

jest.mock('@/context/AuthContext', () => ({
  useAuth: jest.fn(),
}));

const mockUseAuth = useAuth as jest.Mock;
const mockFetch = jest.fn();

describe('BreakpointsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseAuth.mockReturnValue({
      tokens: { access_token: 'test-token' },
    });
    global.fetch = mockFetch as typeof fetch;
  });

  it('loads pending breakpoints from the canonical v1 endpoint and normalizes the payload', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        breakpoints: [
          {
            breakpoint_id: 'bp-123',
            trigger: 'low_confidence',
            message: 'Confidence dropped below threshold',
            created_at: '2026-03-31T12:00:00Z',
            timeout_minutes: 30,
            snapshot: {
              debate_id: 'debate-456',
              round_num: 3,
              task: 'Analyze market trends',
              confidence: 0.45,
              agents: ['claude', 'gpt-4'],
            },
          },
        ],
      }),
    });

    render(<BreakpointsPanel apiBase="https://api.aragora.ai" />);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.aragora.ai/api/v1/breakpoints/pending',
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-token',
          },
        }
      );
    });

    expect(await screen.findByText('Confidence dropped below threshold')).toBeInTheDocument();
    expect(screen.getByText('debate-456')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'CONTINUE' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'REDIRECT' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'ABORT' })).toBeInTheDocument();
  });

  it('sends breakpoint resolutions with the canonical v1 route and message field', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          breakpoints: [
            {
              breakpoint_id: 'bp-123',
              trigger: 'low_confidence',
              message: 'Confidence dropped below threshold',
              created_at: '2026-03-31T12:00:00Z',
              snapshot: {
                debate_id: 'debate-456',
              },
            },
          ],
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'resolved' }),
      });

    render(<BreakpointsPanel apiBase="https://api.aragora.ai" />);

    expect(await screen.findByText('Confidence dropped below threshold')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'CONTINUE' }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenLastCalledWith(
        'https://api.aragora.ai/api/v1/breakpoints/bp-123/resolve',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-token',
          },
          body: JSON.stringify({
            action: 'continue',
            message: 'User selected: continue',
          }),
        }
      );
    });
  });
});
