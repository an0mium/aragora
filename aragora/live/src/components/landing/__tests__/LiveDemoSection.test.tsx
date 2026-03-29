import { render, screen } from '@testing-library/react';
import { LiveDemoSection } from '../LiveDemoSection';

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

const mockUseSpectate = jest.fn();

jest.mock('@/hooks/useSpectate', () => ({
  useSpectate: (...args: unknown[]) => mockUseSpectate(...args),
}));

describe('LiveDemoSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers().setSystemTime(new Date('2026-03-29T12:00:00Z'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  function createSpectateState(overrides: Record<string, unknown> = {}) {
    return {
      status: null,
      loaded: true,
      connected: true,
      events: [],
      refresh: jest.fn(),
      ...overrides,
    };
  }

  it('shows loading bridge copy before spectate status arrives', () => {
    mockUseSpectate.mockReturnValue(
      createSpectateState({
        status: null,
        loaded: false,
        connected: false,
      }),
    );

    render(<LiveDemoSection />);

    expect(screen.getByTestId('live-demo-section')).toBeInTheDocument();
    expect(screen.getByText('Checking public bridge')).toBeInTheDocument();
    expect(
      screen.getByText('Checking public live bridge before showing recent activity.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Waiting on a live debate')).toBeInTheDocument();
  });

  it('renders a real live debate stream when the bridge exposes a featured debate', () => {
    const liveEvents = [
      {
        event_type: 'proposal',
        timestamp: '2026-03-29T11:59:20Z',
        data: { details: 'Split billing and identity into separate services first.' },
        debate_id: 'debate-live-7',
        pipeline_id: null,
        agent_name: 'Strategic Analyst',
        round_number: 1,
      },
      {
        event_type: 'critique',
        timestamp: '2026-03-29T11:59:37Z',
        data: { details: 'That adds cross-service latency before the team has platform support.' },
        debate_id: 'debate-live-7',
        pipeline_id: null,
        agent_name: "Devil's Advocate",
        round_number: 2,
      },
      {
        event_type: 'consensus',
        timestamp: '2026-03-29T11:59:50Z',
        data: { details: 'Keep the monolith, but peel off the highest-churn workflow behind a queue.' },
        debate_id: 'debate-live-7',
        pipeline_id: null,
        agent_name: 'Synthesizer',
        round_number: 3,
      },
    ];

    mockUseSpectate.mockImplementation((debateId?: string) => {
      if (debateId === 'debate-live-7') {
        return createSpectateState({
          status: null,
          events: liveEvents,
        });
      }

      return createSpectateState({
        status: {
          active: true,
          subscribers: 3,
          buffer_size: 12,
          bridge_state: 'live_debates_available',
          last_event_at: '2026-03-29T11:59:50Z',
          activity_age_seconds: 10,
          recent_activity_window_seconds: 120,
          recent_event_count: 3,
          live_debate_count: 1,
          live_debate_ids: ['debate-live-7'],
          live_debates: [
            {
              debate_id: 'debate-live-7',
              recent_event_count: 3,
              last_event_at: '2026-03-29T11:59:50Z',
              event_types: ['proposal', 'critique', 'consensus'],
            },
          ],
          unattributed_recent_event_count: 0,
        },
        events: liveEvents,
      });
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Live debate detected')).toBeInTheDocument();
    expect(
      screen.getByText('Streaming 3 recent events from debate debate-live-7.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Debate debate-live-7')).toBeInTheDocument();
    expect(screen.getByText('Strategic Analyst')).toBeInTheDocument();
    expect(screen.getByText("Devil's Advocate")).toBeInTheDocument();
    expect(
      screen.getByText('Split billing and identity into separate services first.'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('That adds cross-service latency before the team has platform support.'),
    ).toBeInTheDocument();
    expect(screen.getAllByTestId('live-debate-event')).toHaveLength(3);
    expect(
      screen.getByRole('link', { name: 'Open live arena' }),
    ).toHaveAttribute('href', '/spectate/debate-live-7');
  });

  it('falls back to debate ids discovered from recent events when status is redacted', () => {
    const fallbackEvents = [
      {
        event_type: 'proposal',
        timestamp: '2026-03-29T11:59:14Z',
        data: { details: 'Move ingestion to an event queue before splitting services.' },
        debate_id: 'debate-fallback-2',
        pipeline_id: null,
        agent_name: 'Implementation Expert',
        round_number: 1,
      },
      {
        event_type: 'critique',
        timestamp: '2026-03-29T11:59:41Z',
        data: { details: 'Queueing first contains blast radius while the monolith still owns writes.' },
        debate_id: 'debate-fallback-2',
        pipeline_id: null,
        agent_name: 'Risk Analyst',
        round_number: 2,
      },
    ];

    mockUseSpectate.mockImplementation((debateId?: string) => {
      if (debateId === 'debate-fallback-2') {
        return createSpectateState({
          status: null,
          events: fallbackEvents,
        });
      }

      return createSpectateState({
        status: {
          active: true,
          subscribers: 2,
          buffer_size: 6,
          bridge_state: 'activity_unattributed',
          last_event_at: '2026-03-29T11:59:41Z',
          activity_age_seconds: 19,
          recent_activity_window_seconds: 120,
          recent_event_count: 2,
          live_debate_count: 0,
          live_debate_ids: [],
          live_debates: [],
          unattributed_recent_event_count: 2,
        },
        events: fallbackEvents,
      });
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Debate debate-fallback-2')).toBeInTheDocument();
    expect(
      screen.getByText('Move ingestion to an event queue before splitting services.'),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Open live arena' }),
    ).toHaveAttribute('href', '/spectate/debate-fallback-2');
  });
});
