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
  });

  it('shows loading bridge copy before spectate status arrives', () => {
    mockUseSpectate.mockImplementation(() => ({
      status: null,
      loaded: false,
      connected: false,
      events: [],
      refresh: jest.fn(),
    }));

    render(<LiveDemoSection />);

    expect(screen.getByTestId('live-demo-section')).toBeInTheDocument();
    expect(screen.getByText('Checking public bridge')).toBeInTheDocument();
    expect(
      screen.getByText('Checking public live bridge before showing recent activity.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Sample decision trace')).toBeInTheDocument();
  });

  it('shows truthful public bridge activity when recent events are available', () => {
    mockUseSpectate.mockImplementation(() => ({
      status: {
        active: true,
        subscribers: 3,
        buffer_size: 12,
        bridge_state: 'activity_unattributed',
        last_event_at: '2026-03-28T20:00:00Z',
        activity_age_seconds: 34,
        recent_activity_window_seconds: 120,
        recent_event_count: 9,
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        unattributed_recent_event_count: 9,
      },
      loaded: true,
      connected: true,
      events: [],
      refresh: jest.fn(),
    }));

    render(<LiveDemoSection />);

    expect(screen.getByText('Bridge active')).toBeInTheDocument();
    expect(screen.getByText('9 recent events in the last 2 minutes.')).toBeInTheDocument();
    expect(screen.getByText('Last activity 34s ago')).toBeInTheDocument();
  });

  it('shows a live debate feed when recent public events expose a debate ID', () => {
    const now = Date.now();
    const proposalTs = new Date(now - 20_000).toISOString();
    const critiqueTs = new Date(now - 8_000).toISOString();

    mockUseSpectate.mockImplementation((debateId?: string) => {
      if (debateId === 'debate-live-1234567890') {
        return {
          status: null,
          loaded: true,
          connected: true,
          events: [
            {
              event_type: 'proposal',
              timestamp: proposalTs,
              data: { details: 'Keep the monolith until team boundaries stabilize.' },
              debate_id: 'debate-live-1234567890',
              pipeline_id: null,
              agent_name: 'Strategic Analyst',
              round_number: 1,
            },
            {
              event_type: 'critique',
              timestamp: critiqueTs,
              data: { details: 'That ignores deployment drag across three product teams.' },
              debate_id: 'debate-live-1234567890',
              pipeline_id: null,
              agent_name: "Devil's Advocate",
              round_number: 1,
            },
          ],
          refresh: jest.fn(),
        };
      }

      return {
        status: {
          active: true,
          subscribers: 4,
          buffer_size: 24,
          bridge_state: 'activity_unattributed',
          last_event_at: critiqueTs,
          activity_age_seconds: 8,
          recent_activity_window_seconds: 120,
          recent_event_count: 6,
          live_debate_count: 0,
          live_debate_ids: [],
          live_debates: [],
          unattributed_recent_event_count: 6,
        },
        loaded: true,
        connected: true,
        events: [
          {
            event_type: 'proposal',
            timestamp: proposalTs,
            data: { details: 'Keep the monolith until team boundaries stabilize.' },
            debate_id: 'debate-live-1234567890',
            pipeline_id: null,
            agent_name: 'Strategic Analyst',
            round_number: 1,
          },
          {
            event_type: 'critique',
            timestamp: critiqueTs,
            data: { details: 'That ignores deployment drag across three product teams.' },
            debate_id: 'debate-live-1234567890',
            pipeline_id: null,
            agent_name: "Devil's Advocate",
            round_number: 1,
          },
        ],
        refresh: jest.fn(),
      };
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Live debate on air')).toBeInTheDocument();
    expect(screen.getByText('Live public debate')).toBeInTheDocument();
    expect(screen.getByTestId('live-debate-feed')).toBeInTheDocument();
    expect(screen.getByText('Strategic Analyst')).toBeInTheDocument();
    expect(screen.getByText("Devil's Advocate")).toBeInTheDocument();
    expect(screen.getByText('Keep the monolith until team boundaries stabilize.')).toBeInTheDocument();
    expect(screen.getByText('That ignores deployment drag across three product teams.')).toBeInTheDocument();
    expect(screen.getByText('Open live feed')).toBeInTheDocument();
    expect(screen.getByText('Watch this debate live')).toBeInTheDocument();
  });
});
