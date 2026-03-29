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
  beforeAll(() => {
    jest.spyOn(Date, 'now').mockReturnValue(new Date('2026-03-29T12:00:00Z').getTime());
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterAll(() => {
    jest.restoreAllMocks();
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
    expect(screen.getByTestId('live-debate-fallback')).toBeInTheDocument();
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

  it('renders a live debate transcript when recent public events include a debate id', () => {
    mockUseSpectate.mockImplementation((debateId?: string) => {
      if (debateId === 'debate-live-7') {
        return {
          status: null,
          loaded: true,
          connected: true,
          events: [
            {
              debate_id: 'debate-live-7',
              event_type: 'proposal',
              timestamp: '2026-03-29T11:59:52Z',
              agent_name: 'Strategic Analyst',
              round_number: 1,
              pipeline_id: null,
              data: {
                task: 'Should we split the monolith this quarter?',
                details: 'Service extraction should start with the payments boundary.',
              },
            },
            {
              debate_id: 'debate-live-7',
              event_type: 'critique',
              timestamp: '2026-03-29T11:59:57Z',
              agent_name: "Devil's Advocate",
              round_number: 1,
              pipeline_id: null,
              data: {
                details: 'The migration cost is still higher than the current deployment pain.',
              },
            },
          ],
          refresh: jest.fn(),
        };
      }

      return {
        status: {
          active: true,
          subscribers: 4,
          buffer_size: 18,
          bridge_state: 'activity_unattributed',
          last_event_at: '2026-03-29T11:59:57Z',
          activity_age_seconds: 3,
          recent_activity_window_seconds: 120,
          recent_event_count: 11,
          live_debate_count: 0,
          live_debate_ids: [],
          live_debates: [],
          unattributed_recent_event_count: 11,
        },
        loaded: true,
        connected: true,
        events: [
          {
            debate_id: 'debate-live-7',
            event_type: 'debate_start',
            timestamp: '2026-03-29T11:59:48Z',
            agent_name: null,
            round_number: null,
            pipeline_id: null,
            data: {
              task: 'Should we split the monolith this quarter?',
            },
          },
          {
            debate_id: 'debate-live-7',
            event_type: 'proposal',
            timestamp: '2026-03-29T11:59:52Z',
            agent_name: 'Strategic Analyst',
            round_number: 1,
            pipeline_id: null,
            data: {
              task: 'Should we split the monolith this quarter?',
              details: 'Service extraction should start with the payments boundary.',
            },
          },
          {
            debate_id: 'debate-live-7',
            event_type: 'critique',
            timestamp: '2026-03-29T11:59:57Z',
            agent_name: "Devil's Advocate",
            round_number: 1,
            pipeline_id: null,
            data: {
              details: 'The migration cost is still higher than the current deployment pain.',
            },
          },
        ],
        refresh: jest.fn(),
      };
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Live debate')).toBeInTheDocument();
    expect(screen.getByText('Live public debate')).toBeInTheDocument();
    expect(screen.getByText('Should we split the monolith this quarter?')).toBeInTheDocument();
    expect(screen.getByTestId('live-debate-stream')).toBeInTheDocument();
    expect(screen.getByText('Service extraction should start with the payments boundary.')).toBeInTheDocument();
    expect(screen.getByText('The migration cost is still higher than the current deployment pain.')).toBeInTheDocument();
    expect(screen.getByText('Strategic Analyst')).toBeInTheDocument();
    expect(screen.getByText("Devil's Advocate")).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Watch full debate' })).toHaveAttribute(
      'href',
      '/spectate/debate-live-7',
    );
  });
});
