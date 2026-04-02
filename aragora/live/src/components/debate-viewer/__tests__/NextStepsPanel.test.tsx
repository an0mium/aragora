import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { NextStepsPanel } from '../NextStepsPanel';

const mockApiFetch = jest.fn();

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children, className }: { href: string; children: React.ReactNode; className?: string }) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}));

jest.mock('@/lib/api', () => ({
  apiFetch: (...args: unknown[]) => mockApiFetch(...args),
}));

describe('NextStepsPanel', () => {
  beforeEach(() => {
    mockApiFetch.mockReset();
  });

  it('routes receipt navigation through the debate detail receipt tab', () => {
    render(<NextStepsPanel debateId="debate-123" />);

    const link = screen.getByRole('link', { name: /view receipt/i });
    expect(link).toHaveAttribute('href', '/debates/debate-123?tab=receipt');
  });

  it('extracts debate knowledge through the real knowledge mound endpoint', async () => {
    const user = userEvent.setup();
    mockApiFetch
      .mockResolvedValueOnce({
        question: 'Should we ship the debate save flow?',
        final_answer: 'Yes, the extraction path is ready.',
        arguments: [
          {
            agent: 'claude',
            round: 1,
            content: 'I think the knowledge extraction route should be wired end to end.',
          },
        ],
      })
      .mockResolvedValueOnce({
        debate_id: 'debate-123',
        claims: [],
        relationships: [],
      });

    render(<NextStepsPanel debateId="debate-123" />);

    await user.click(screen.getByRole('button', { name: /save to knowledge/i }));

    await waitFor(() => {
      expect(mockApiFetch).toHaveBeenNthCalledWith(1, '/api/v1/debates/debate-123/package');
      expect(mockApiFetch).toHaveBeenNthCalledWith(
        2,
        '/api/v1/knowledge/mound/extraction/debate',
        {
          method: 'POST',
          body: JSON.stringify({
            debate_id: 'debate-123',
            messages: [
              {
                agent: 'claude',
                round: 1,
                content: 'I think the knowledge extraction route should be wired end to end.',
              },
            ],
            consensus_text: 'Yes, the extraction path is ready.',
            topic: 'Should we ship the debate save flow?',
          }),
        },
      );
    });

    expect(await screen.findByText(/debate outcome persisted to the knowledge mound/i)).toBeInTheDocument();
  });

  it('shows a helpful error when the debate package has no transcript to extract', async () => {
    const user = userEvent.setup();
    mockApiFetch.mockResolvedValueOnce({
      question: 'Should we save this?',
      final_answer: 'Not enough context.',
      arguments: [],
    });

    render(<NextStepsPanel debateId="debate-123" />);

    await user.click(screen.getByRole('button', { name: /save to knowledge/i }));

    expect(await screen.findByText(/no debate transcript available to save/i)).toBeInTheDocument();
    expect(mockApiFetch).toHaveBeenCalledTimes(1);
  });
});
