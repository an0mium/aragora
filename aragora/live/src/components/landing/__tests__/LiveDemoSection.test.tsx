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
    expect(screen.getByText('Sample decision trace')).toBeInTheDocument();
  });

  it('shows truthful public bridge activity when recent events are available', () => {
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
      events: [],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Bridge active')).toBeInTheDocument();
    expect(screen.getByText('9 recent events in the last 2 minutes.')).toBeInTheDocument();
    expect(screen.getByText('Last activity 34s ago')).toBeInTheDocument();
  });

  it('renders a live debate feed when recent events resolve to a public debate', () => {
    mockUseSpectate.mockReturnValue({
      status: {
        active: true,
        subscribers: 4,
        buffer_size: 24,
        bridge_state: 'activity_unattributed',
        last_event_at: '2026-03-28T20:00:00Z',
        activity_age_seconds: 9,
        recent_activity_window_seconds: 120,
        recent_event_count: 12,
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        unattributed_recent_event_count: 12,
      },
      loaded: true,
      connected: true,
      events: [
        {
          event_type: 'proposal',
          timestamp: '2026-03-28T20:00:00Z',
          data: {
            details: 'We should keep the monolith until team boundaries are stable.',
            task: 'Should we split the platform into microservices this quarter?',
          },
          debate_id: 'debate-live-1',
          pipeline_id: null,
          agent_name: 'Strategic Analyst',
          round_number: 1,
        },
        {
          event_type: 'critique',
          timestamp: '2026-03-28T20:00:04Z',
          data: {
            details: 'Counterpoint: the billing domain is already isolated enough to extract safely.',
          },
          debate_id: 'debate-live-1',
          pipeline_id: null,
          agent_name: "Devil's Advocate",
          round_number: 1,
        },
      ],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getAllByText('Live public debate')).toHaveLength(2);
    expect(screen.getByTestId('live-debate-feed')).toBeInTheDocument();
    expect(
      screen.getByText('Should we split the platform into microservices this quarter?'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('We should keep the monolith until team boundaries are stable.'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Counterpoint: the billing domain is already isolated enough to extract safely.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('Open live feed')).toBeInTheDocument();
    expect(screen.queryByText('Sample decision trace')).not.toBeInTheDocument();
  });
});
