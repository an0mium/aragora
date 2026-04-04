import type { ReactNode } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LandingReviewPage from '../page';
import { useSWRFetch } from '@/hooks/useSWRFetch';

jest.mock('@/components/admin/AdminLayout', () => ({
  AdminLayout: ({
    title,
    description,
    actions,
    children,
  }: {
    title: string;
    description?: string;
    actions?: ReactNode;
    children: ReactNode;
  }) => (
    <div>
      <h1>{title}</h1>
      {description && <p>{description}</p>}
      {actions}
      <div>{children}</div>
    </div>
  ),
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({ config: { api: 'http://localhost:8080' } }),
}));

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: jest.fn(),
}));

const mockUseSWRFetch = useSWRFetch as jest.Mock;
const mockFetch = jest.fn();

global.fetch = mockFetch as unknown as typeof fetch;

function buildSummary() {
  return {
    generated_at: '2026-04-04T01:00:00Z',
    window_seconds: 86400,
    total_events: 12,
    unique_client_count: 3,
    last_event_at: '2026-04-04T00:59:00Z',
    event_counts: {
      preflight_shown: 3,
      preflight_selected: 2,
      preview_rendered: 2,
      preview_timeout: 0,
      preview_clarification_requested: 0,
      retry_clicked: 0,
      wrong_answer_clicked: 1,
      open_full_debate_clicked: 0,
      share_clicked: 0,
    },
    rates: {
      preflight_selection_rate: 0.6667,
      preview_render_rate: 1,
      preview_timeout_rate: 0,
      preview_clarification_rate: 0,
      wrong_answer_rate: 0.5,
      open_full_debate_rate: 0,
      share_rate: 0,
      retry_rate: null,
    },
    question_length: {
      samples: 2,
      avg: 88,
      max: 120,
    },
    preview: {
      rendered_count: 2,
      avg_participant_count: 3,
    },
    timeouts: {
      count: 0,
      avg_timeout_seconds: null,
    },
    top_options: [],
  };
}

function buildFeedback() {
  return {
    generated_at: '2026-04-04T01:00:00Z',
    window_seconds: 86400,
    total_reports: 1,
    returned_reports: 1,
    unique_client_count: 1,
    last_report_at: '2026-04-04T00:58:00Z',
    stats: {
      rewritten_count: 1,
      rewritten_rate: 1,
      preview_mode_count: 1,
      preview_mode_rate: 1,
      review_status_counts: {
        pending: 1,
        reviewed: 0,
        resolved: 0,
        dismissed: 0,
      },
    },
    reports: [
      {
        id: 'lfb_1',
        timestamp: '2026-04-04T00:58:00Z',
        client_tag: 'ip:abc123',
        question: 'Should I microwave chicken nuggets for my child?',
        interpreted_question: 'Is it safe to reheat pre-cooked chicken nuggets?',
        final_answer_preview: 'Yes, reheat until hot all the way through.',
        result_warning: null,
        result_mode: 'preview',
        debate_id: 'debate-123',
        verdict: 'needs_review',
        participant_count: 3,
        rewritten: true,
        review_status: 'pending',
        reviewed_at: null,
        reviewed_by: null,
      },
    ],
  };
}

describe('LandingReviewPage', () => {
  const mutateSummary = jest.fn();
  const mutateFeedback = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    window.localStorage.clear();
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      headers: {
        get: () => 'application/json',
      },
      json: async () => ({ ok: true }),
    });

    mockUseSWRFetch.mockImplementation((endpoint: string) => {
      if (endpoint.startsWith('/api/v1/playground/landing/events/summary')) {
        return {
          data: buildSummary(),
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: mutateSummary,
        };
      }

      if (endpoint.startsWith('/api/v1/playground/landing/feedback')) {
        return {
          data: buildFeedback(),
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: mutateFeedback,
        };
      }

      return {
        data: null,
        error: null,
        isLoading: false,
        isValidating: false,
        mutate: jest.fn(),
      };
    });
  });

  it('renders review-state counts for the queue', async () => {
    render(<LandingReviewPage />);

    expect(await screen.findByText('Landing Review')).toBeInTheDocument();
    expect(screen.getByText('Pending 1')).toBeInTheDocument();
    expect(screen.getByText('Reviewed 0')).toBeInTheDocument();
    expect(screen.getByText('Resolved 0')).toBeInTheDocument();
    expect(screen.getByText('Dismissed 0')).toBeInTheDocument();
    expect(screen.getByText('Should I microwave chicken nuggets for my child?')).toBeInTheDocument();
  });

  it('posts review updates and refreshes the feedback queue', async () => {
    const user = userEvent.setup();
    window.localStorage.setItem(
      'aragora_tokens',
      JSON.stringify({ access_token: 'admin-token' }),
    );

    render(<LandingReviewPage />);

    await user.click(await screen.findByRole('button', { name: 'Resolve' }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/v1/playground/landing/feedback/review',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: 'Bearer admin-token',
          },
          body: JSON.stringify({ id: 'lfb_1', review_status: 'resolved' }),
        }),
      );
    });
    await waitFor(() => {
      expect(mutateFeedback).toHaveBeenCalled();
    });
  });
});
