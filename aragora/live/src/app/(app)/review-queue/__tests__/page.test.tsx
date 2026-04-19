import React from 'react';

import { fireEvent, renderWithProviders, screen, waitFor } from '@/test-utils';

import ReviewQueuePage from '../page';

const mockFetch = jest.fn();

global.fetch = mockFetch as typeof fetch;

beforeAll(() => {
  Object.defineProperty(window, 'open', {
    writable: true,
    value: jest.fn(),
  });
  Object.defineProperty(window, 'confirm', {
    writable: true,
    value: jest.fn(() => true),
  });
  Object.defineProperty(window, 'prompt', {
    writable: true,
    value: jest.fn(() => ''),
  });
});

function jsonResponse(data: unknown): Response {
  return {
    ok: true,
    status: 200,
    headers: {
      get: () => 'application/json',
    },
    json: async () => data,
  } as Response;
}

const LIST_RESPONSE = {
  prs: [
    {
      number: 6308,
      title: 'fix(ci): cancel test-fast only on synchronize events',
      url: 'https://github.com/synaptent/aragora/pull/6308',
      diff_url: 'https://github.com/synaptent/aragora/pull/6308/files',
      head_sha: 'abc123def456',
      author: 'an0mium',
      is_draft: false,
      mergeable: 'MERGEABLE',
      review_decision: 'REVIEW_REQUIRED',
      labels: ['codex'],
      additions: 1,
      deletions: 1,
      changed_files: 1,
      checks_summary: '5/5 green',
      lane: 'ready_now',
      lane_reason: 'all green',
      created_at: '2026-04-19T12:00:00Z',
      updated_at: '2026-04-19T12:05:00Z',
      status_counts: { success: 5, failure: 0, pending: 0, cancelled: 0, total: 5 },
      touched_subsystems: ['.github'],
      high_risk_paths_touched: ['.github/workflows/test.yml'],
      machine_recommendation: 'approve_candidate',
      machine_recommendation_reason: 'all green, bounded diff, no high-risk paths',
      brief_available: true,
      brief: {
        pr_number: 6308,
        verdict: 'approve_candidate',
        raw_verdict: '✓ APPROVE',
        confidence: 5,
        logic: 'Correct YAML.',
        security: 'None.',
        maintainability: 'Small follow-up likely.',
        skeptic: 'If cancellations persist, root cause is elsewhere.',
        recommended_action: 'Approve first.',
      },
    },
  ],
  count: 1,
  generated_at: '2026-04-19T12:10:00Z',
  source: 'local-review-queue',
};

const STATS_RESPONSE = {
  decisions_today: 3,
  approvals_today: 2,
  median_decision_seconds: 18,
  streak: 3,
  source: 'local-review-queue',
};

const DETAIL_RESPONSE = {
  pr: LIST_RESPONSE.prs[0],
  packet: {
    pr_number: 6308,
    title: 'fix(ci): cancel test-fast only on synchronize events',
    url: 'https://github.com/synaptent/aragora/pull/6308',
    head_sha: 'abc123def456',
    base_sha: 'base123',
    author: 'an0mium',
    is_draft: false,
    additions: 1,
    deletions: 1,
    changed_files: 1,
    queue_bucket: 'ready_now',
    touched_subsystems: ['.github'],
    high_risk_paths_touched: ['.github/workflows/test.yml'],
    validation: ['pytest tests/server/handlers/test_review_queue_handler.py'],
    checks_summary: '5/5 green',
    risk_flags: ['touches high-risk paths: .github/workflows/test.yml'],
    machine_recommendation: 'approve_candidate',
    machine_recommendation_reason: 'all green, bounded diff, no high-risk paths',
    packet_sha: 'sha256:test',
    generated_at: '2026-04-19T12:10:00Z',
    advisory_only: true,
    settlement_note: 'Human settlement required.',
  },
  brief: LIST_RESPONSE.prs[0].brief,
  checks: [
    {
      name: 'lint',
      status: 'COMPLETED',
      conclusion: 'SUCCESS',
      details_url: 'https://github.com/example/lint',
    },
  ],
  files: [
    {
      path: '.github/workflows/test.yml',
      additions: 1,
      deletions: 1,
    },
  ],
  diff_url: 'https://github.com/synaptent/aragora/pull/6308/files',
};

function installFetchMocks() {
  mockFetch.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method || 'GET').toUpperCase();

    if (url.includes('/api/v1/review-queue/prs/6308/approve') && method === 'POST') {
      return Promise.resolve(jsonResponse({ receipt: { action: 'approve', pr_number: 6308 } }));
    }

    if (url.includes('/api/v1/review-queue/prs/6308') && method === 'GET') {
      return Promise.resolve(jsonResponse(DETAIL_RESPONSE));
    }

    if (url.includes('/api/v1/review-queue/stats')) {
      return Promise.resolve(jsonResponse(STATS_RESPONSE));
    }

    if (url.includes('/api/v1/review-queue/prs')) {
      return Promise.resolve(jsonResponse(LIST_RESPONSE));
    }

    return Promise.reject(new Error(`Unhandled fetch: ${method} ${url}`));
  });
}

describe('ReviewQueuePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    installFetchMocks();
  });

  it('renders the queue and summary metrics', async () => {
    renderWithProviders(<ReviewQueuePage />, {
      authOverrides: {
        isAuthenticated: true,
        tokens: {
          access_token: 'token',
          refresh_token: 'refresh',
          expires_at: '2099-01-01T00:00:00Z',
        },
      },
    });

    await waitFor(() => {
      expect(screen.getByText('fix(ci): cancel test-fast only on synchronize events')).toBeInTheDocument();
    });

    expect(screen.getByText('Review Queue')).toBeInTheDocument();
    expect(screen.getByText('Pending')).toBeInTheDocument();
    expect(screen.getByText('18s')).toBeInTheDocument();
  });

  it('loads detail when a card is expanded', async () => {
    renderWithProviders(<ReviewQueuePage />, {
      authOverrides: {
        isAuthenticated: true,
        tokens: {
          access_token: 'token',
          refresh_token: 'refresh',
          expires_at: '2099-01-01T00:00:00Z',
        },
      },
    });

    await waitFor(() => {
      expect(screen.getByText('Expand')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Expand'));

    await waitFor(() => {
      expect(screen.getByText('Correct YAML.')).toBeInTheDocument();
    });

    expect(screen.getByText('Risk Flags')).toBeInTheDocument();
    expect(screen.getByText('.github/workflows/test.yml')).toBeInTheDocument();
  });

  it('settles a PR from the browser surface', async () => {
    renderWithProviders(<ReviewQueuePage />, {
      authOverrides: {
        isAuthenticated: true,
        tokens: {
          access_token: 'token',
          refresh_token: 'refresh',
          expires_at: '2099-01-01T00:00:00Z',
        },
      },
    });

    await waitFor(() => {
      expect(screen.getByText('Approve')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Approve'));

    await waitFor(() => {
      expect(screen.getByText('Approved #6308')).toBeInTheDocument();
    });

    expect(screen.queryByText('fix(ci): cancel test-fast only on synchronize events')).not.toBeInTheDocument();
  });
});
