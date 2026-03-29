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
    mockUseSpectate.mockReturnValue({
      status: null,
      loaded: false,
      connected: false,
      events: [],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getByTestId('live-demo-section')).toBeInTheDocument();
    expect(screen.getByText('Checking public bridge')).toBeInTheDocument();
    expect(
      screen.getByText('Checking public live bridge before showing recent activity.'),
    ).toBeInTheDocument();
    expect(screen.getByTestId('sample-transcript')).toBeInTheDocument();
  });

  it('shows a live public debate feed when recent events expose a debate id', () => {
    const now = Date.now();

    mockUseSpectate.mockReturnValue({
      status: {
        active: true,
        subscribers: 3,
        buffer_size: 12,
        bridge_state: 'activity_unattributed',
        last_event_at: new Date(now - 4000).toISOString(),
        activity_age_seconds: 4,
        recent_activity_window_seconds: 120,
        recent_event_count: 3,
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        unattributed_recent_event_count: 3,
      },
      loaded: true,
      connected: true,
      events: [
        {
          event_type: 'debate_start',
          timestamp: new Date(now - 5000).toISOString(),
          data: {
            task: 'Should we centralize feature flags across product lines?',
            agents: ['Strategic Analyst', 'Risk Manager'],
          },
          debate_id: 'adhoc_live-123',
          pipeline_id: null,
          agent_name: null,
          round_number: 0,
        },
        {
          event_type: 'proposal',
          timestamp: new Date(now - 3000).toISOString(),
          data: {
            details:
              'Centralize evaluation so policy changes propagate immediately across all products.',
          },
          debate_id: 'adhoc_live-123',
          pipeline_id: null,
          agent_name: 'Strategic Analyst',
          round_number: 1,
        },
        {
          event_type: 'critique',
          timestamp: new Date(now - 1000).toISOString(),
          data: {
            details:
              'A single control plane becomes a failure domain unless teams keep local fail-open rules.',
          },
          debate_id: 'adhoc_live-123',
          pipeline_id: null,
          agent_name: 'Risk Manager',
          round_number: 1,
        },
      ],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Streaming live')).toBeInTheDocument();
    expect(screen.getByTestId('live-debate-stream')).toBeInTheDocument();
    expect(
      screen.getByText('Should we centralize feature flags across product lines?'),
    ).toBeInTheDocument();
    expect(screen.getByText('Strategic Analyst')).toBeInTheDocument();
    expect(screen.getByText('Risk Manager')).toBeInTheDocument();
    expect(
      screen.getByText(
        'Centralize evaluation so policy changes propagate immediately across all products.',
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'A single control plane becomes a failure domain unless teams keep local fail-open rules.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Watch full live debate' })).toHaveAttribute(
      'href',
      '/debate/adhoc_live-123',
    );
  });

  it('shows truthful public bridge activity when recent events are available without a discoverable debate', () => {
    mockUseSpectate.mockReturnValue({
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
      events: [
        {
          event_type: 'system',
          timestamp: new Date(Date.now() - 5000).toISOString(),
          data: { details: 'Heartbeat from the public bridge.' },
          debate_id: null,
          pipeline_id: null,
          agent_name: null,
          round_number: null,
        },
      ],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Bridge active')).toBeInTheDocument();
    expect(screen.getByText('9 recent events in the last 2 minutes.')).toBeInTheDocument();
    expect(screen.getByText('Last activity 34s ago')).toBeInTheDocument();
    expect(screen.getByTestId('sample-transcript')).toBeInTheDocument();
  });
});
