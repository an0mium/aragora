/**
 * Integration test for the ReviewQueuePage: inbox-zero celebration, error display,
 * loading state.
 */
import { render, screen } from '@testing-library/react';

// Use real timers — most tests here do not need fake timers, and inbox-zero
// test explicitly swaps to them.

jest.mock('../src/config', () => ({
  API_BASE_URL: 'http://localhost:8080',
  WS_URL: 'ws://localhost:8765/ws',
}));

let mockQueue: {
  prs: unknown[];
  total: number;
  visible: number;
  deferredCount: number;
  degraded: boolean;
  reason?: string;
  isLoading: boolean;
  isValidating: boolean;
  error: Error | null;
  mutate: jest.Mock;
};

let mockStats: {
  stats: {
    approved: number;
    request_changes: number;
    deferred: number;
    streak: number;
    decision_count: number;
    median_decision_seconds: number | null;
    date: string | null;
  } | null;
  isLoading: boolean;
  mutate: jest.Mock;
};

jest.mock('../src/hooks/useReviewQueue', () => ({
  __esModule: true,
  useReviewQueue: () => mockQueue,
  useReviewQueueStats: () => mockStats,
  useSettlePR: () => jest.fn().mockResolvedValue({ status: 'ok' }),
  settlePR: jest.fn(),
  fetchBrief: jest.fn().mockResolvedValue(null),
  generateBrief: jest.fn().mockResolvedValue({ state: 'queued' }),
  getBriefState: jest.fn().mockResolvedValue({ state: 'absent' }),
  cancelBriefGeneration: jest.fn().mockResolvedValue(undefined),
  getBriefGenerationFlag: jest.fn(() => null),
  __resetBriefGenerationFlagForTests: jest.fn(),
  useBriefState: jest.fn(() => ({
    snapshot: null,
    isLoading: false,
    error: null,
    featureDisabled: false,
    refresh: jest.fn().mockResolvedValue(null),
    setSnapshot: jest.fn(),
  })),
}));

import ReviewQueuePage from '../src/app/(app)/review-queue/page';

beforeEach(() => {
  mockQueue = {
    prs: [],
    total: 0,
    visible: 0,
    deferredCount: 0,
    degraded: false,
    reason: undefined,
    isLoading: false,
    isValidating: false,
    error: null,
    mutate: jest.fn(),
  };
  mockStats = {
    stats: {
      date: '2026-04-19',
      approved: 0,
      request_changes: 0,
      deferred: 0,
      streak: 0,
      decision_count: 0,
      median_decision_seconds: null,
    },
    isLoading: false,
    mutate: jest.fn(),
  };
});

describe('ReviewQueuePage', () => {
  it('renders loading state', () => {
    mockQueue.isLoading = true;
    render(<ReviewQueuePage />);
    expect(screen.getByTestId('review-queue-page-loading')).toBeInTheDocument();
  });

  it('renders queue error', () => {
    mockQueue.error = new Error('connection refused');
    render(<ReviewQueuePage />);
    expect(screen.getByTestId('review-queue-page-error')).toHaveTextContent(/connection refused/);
  });

  it('shows inbox-zero celebration when queue empties after having items', () => {
    // Simulate that we had items and just cleared them.
    mockQueue.total = 3;
    mockQueue.visible = 0;
    render(<ReviewQueuePage />);
    expect(screen.getByTestId('review-queue-inbox-zero')).toBeInTheDocument();
  });

  it('does not celebrate when queue was already empty', () => {
    mockQueue.total = 0;
    mockQueue.visible = 0;
    render(<ReviewQueuePage />);
    expect(screen.queryByTestId('review-queue-inbox-zero')).toBeNull();
  });

  it('renders stats header with visible count', () => {
    mockQueue.visible = 2;
    mockQueue.total = 2;
    mockQueue.prs = [
      {
        number: 1,
        title: 'a',
        url: '',
        head_sha: '',
        is_draft: false,
        author: 'armand',
        labels: [],
        additions: 0,
        deletions: 0,
        changed_files: 0,
        created_at: '',
        updated_at: '',
        age_seconds: 0,
        touched_subsystems: [],
        ci: { success: 0, failure: 0, pending: 0, total: 0 },
        brief_present: false,
        verdict: null,
        confidence: null,
        deferred: false,
      },
      {
        number: 2,
        title: 'b',
        url: '',
        head_sha: '',
        is_draft: false,
        author: 'armand',
        labels: [],
        additions: 0,
        deletions: 0,
        changed_files: 0,
        created_at: '',
        updated_at: '',
        age_seconds: 0,
        touched_subsystems: [],
        ci: { success: 0, failure: 0, pending: 0, total: 0 },
        brief_present: false,
        verdict: null,
        confidence: null,
        deferred: false,
      },
    ];
    render(<ReviewQueuePage />);
    expect(screen.getByTestId('review-queue-visible')).toHaveTextContent('2');
  });

  it('surfaces degraded banner', () => {
    mockQueue.degraded = true;
    mockQueue.reason = 'gh CLI not found';
    render(<ReviewQueuePage />);
    expect(screen.getByTestId('review-queue-degraded')).toHaveTextContent(/gh CLI not found/);
  });
});
