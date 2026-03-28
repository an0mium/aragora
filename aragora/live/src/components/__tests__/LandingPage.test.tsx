import type { ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import { LandingPage } from '../LandingPage';
import { useSpectate, type SpectateEvent, type SpectateStatus } from '@/hooks/useSpectate';

jest.mock('next/link', () => {
  return function MockLink({
    children,
    href,
    className,
  }: {
    children: ReactNode;
    href: string;
    className?: string;
  }) {
    return (
      <a href={href} className={className}>
        {children}
      </a>
    );
  };
});

jest.mock('../DebateResultPreview', () => ({
  DebateResultPreview: () => <div data-testid="debate-result-preview" />,
  RETURN_URL_KEY: 'aragora-return-url',
  PENDING_DEBATE_KEY: 'aragora-pending-debate',
}));

jest.mock('@/hooks/useSpectate', () => ({
  useSpectate: jest.fn(),
}));

const mockUseSpectate = useSpectate as jest.MockedFunction<typeof useSpectate>;

function buildStatus(overrides: Partial<SpectateStatus> = {}): SpectateStatus {
  return {
    active: true,
    subscribers: 2,
    buffer_size: 12,
    bridge_state: 'live_debates_available',
    last_event_at: '2026-03-28T12:00:05.000Z',
    activity_age_seconds: 5,
    recent_activity_window_seconds: 120,
    recent_event_count: 4,
    live_debate_count: 1,
    live_debate_ids: ['debate-live-1'],
    live_debates: [
      {
        debate_id: 'debate-live-1',
        recent_event_count: 4,
        last_event_at: '2026-03-28T12:00:05.000Z',
        event_types: ['consensus', 'critique', 'proposal'],
      },
    ],
    unattributed_recent_event_count: 0,
    ...overrides,
  };
}

function buildEvent(overrides: Partial<SpectateEvent> = {}): SpectateEvent {
  return {
    event_type: 'proposal',
    timestamp: '2026-03-28T12:00:00.000Z',
    data: {
      details: 'Agent Alpha proposed a staged rollout with circuit breakers.',
      question: 'Should we roll out model gating this quarter?',
    },
    debate_id: 'debate-live-1',
    pipeline_id: null,
    agent_name: 'Agent Alpha',
    round_number: 1,
    ...overrides,
  };
}

describe('LandingPage live debate preview', () => {
  beforeEach(() => {
    jest.useFakeTimers().setSystemTime(new Date('2026-03-28T12:00:10.000Z'));
    mockUseSpectate.mockReturnValue({
      events: [],
      connected: false,
      loaded: true,
      status: buildStatus({
        bridge_state: 'idle',
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        recent_event_count: 0,
      }),
      refresh: jest.fn(),
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it('shows a live debate watcher when the bridge confirms an active debate', () => {
    mockUseSpectate.mockReturnValue({
      events: [
        buildEvent(),
        buildEvent({
          event_type: 'critique',
          timestamp: '2026-03-28T12:00:04.000Z',
          data: { details: 'Agent Beta attacked the rollback assumptions.' },
          agent_name: 'Agent Beta',
          round_number: 1,
        }),
      ],
      connected: true,
      loaded: true,
      status: buildStatus(),
      refresh: jest.fn(),
    });

    render(<LandingPage apiBase="https://api.example.com" />);

    expect(screen.getByTestId('landing-live-debate-card')).toBeInTheDocument();
    expect(
      screen.getByText('Should we roll out model gating this quarter?'),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Watch live debate' }),
    ).toHaveAttribute('href', '/spectate/debate-live-1');
    expect(screen.getByTestId('landing-live-debate-feed')).toHaveTextContent(
      'Agent Beta attacked the rollback assumptions.',
    );
    expect(screen.getByTestId('landing-live-debate-feed')).toHaveTextContent(
      'Agent Alpha proposed a staged rollout with circuit breakers.',
    );
  });

  it('falls back to recent attributed events when the bridge status endpoint is unavailable', () => {
    mockUseSpectate.mockReturnValue({
      events: [
        buildEvent({
          debate_id: 'debate-fallback-7',
          timestamp: '2026-03-28T12:00:07.000Z',
          data: {
            details: 'Agent Sigma opened with a hybrid staffing plan.',
            topic: 'Should we hire two platform engineers this half?',
          },
          agent_name: 'Agent Sigma',
        }),
        buildEvent({
          debate_id: 'debate-fallback-7',
          event_type: 'vote',
          timestamp: '2026-03-28T12:00:08.000Z',
          data: { details: 'Agent Delta cast a provisional yes vote.' },
          agent_name: 'Agent Delta',
          round_number: 2,
        }),
      ],
      connected: true,
      loaded: true,
      status: null,
      refresh: jest.fn(),
    });

    render(<LandingPage apiBase="https://api.example.com" />);

    expect(screen.getByText('RECENT FEED')).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Watch live debate' }),
    ).toHaveAttribute('href', '/spectate/debate-fallback-7');
    expect(
      screen.getByText('Should we hire two platform engineers this half?'),
    ).toBeInTheDocument();
  });

  it('does not invent a live watcher when recent activity lacks a debate id', () => {
    mockUseSpectate.mockReturnValue({
      events: [
        buildEvent({
          debate_id: null,
          timestamp: '2026-03-28T12:00:07.000Z',
          data: { details: 'An unattributed debate event arrived from the bridge.' },
          agent_name: 'Agent Ghost',
        }),
      ],
      connected: true,
      loaded: true,
      status: buildStatus({
        bridge_state: 'activity_unattributed',
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        recent_event_count: 1,
        unattributed_recent_event_count: 1,
      }),
      refresh: jest.fn(),
    });

    render(<LandingPage apiBase="https://api.example.com" />);

    expect(
      screen.queryByRole('link', { name: 'Watch live debate' }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(
        'Waiting for the bridge to tag the current debate before we show a live watcher.',
      ),
    ).toBeInTheDocument();
  });
});
