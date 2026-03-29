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
    jest.spyOn(Date, 'now').mockReturnValue(new Date('2026-03-29T12:00:40Z').valueOf());
  });

  afterAll(() => {
    jest.restoreAllMocks();
  });

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
    expect(screen.getByTestId('sample-debate-trace')).toBeInTheDocument();
    expect(screen.getByText('Sample decision trace')).toBeInTheDocument();
  });

  it('shows unattributed live bridge activity without inventing a debate id', () => {
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
          event_type: 'agent_message',
          timestamp: '2026-03-29T12:00:05Z',
          data: { details: 'The team can absorb one more synchronous dependency.' },
          debate_id: null,
          pipeline_id: null,
          agent_name: 'Systems Analyst',
          round_number: 1,
        },
        {
          event_type: 'critique',
          timestamp: '2026-03-29T12:00:20Z',
          data: { details: 'That assumption ignores the current deployment bottleneck.' },
          debate_id: null,
          pipeline_id: null,
          agent_name: 'Contrarian Reviewer',
          round_number: 1,
        },
      ],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Bridge active')).toBeInTheDocument();
    expect(screen.getByText('2 recent live updates are visible while the bridge waits for a debate ID attribution.')).toBeInTheDocument();
    expect(screen.getByText('Last activity 34s ago')).toBeInTheDocument();
    expect(screen.getByTestId('live-bridge-feed')).toBeInTheDocument();
    expect(screen.getByText('Live bridge activity')).toBeInTheDocument();
    expect(screen.getByText('Systems Analyst')).toBeInTheDocument();
    expect(screen.getByText('The team can absorb one more synchronous dependency.')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open the live feed' })).toHaveAttribute('href', '/spectate');
  });

  it('streams the freshest attributed live debate on the landing page', () => {
    mockUseSpectate.mockReturnValue({
      status: {
        active: true,
        subscribers: 5,
        buffer_size: 18,
        bridge_state: 'live_debates_available',
        last_event_at: '2026-03-29T12:00:35Z',
        activity_age_seconds: 5,
        recent_activity_window_seconds: 120,
        recent_event_count: 3,
        live_debate_count: 1,
        live_debate_ids: ['debate-live-1'],
        live_debates: [
          {
            debate_id: 'debate-live-1',
            recent_event_count: 3,
            last_event_at: '2026-03-29T12:00:35Z',
            event_types: ['proposal', 'critique', 'consensus'],
          },
        ],
        unattributed_recent_event_count: 0,
      },
      loaded: true,
      connected: true,
      events: [
        {
          event_type: 'proposal',
          timestamp: '2026-03-29T12:00:10Z',
          data: {
            details: 'Open in Germany first and delay France until the support queue is staffed.',
            task: 'Should we launch the EU expansion this quarter?',
            agents: ['Market Analyst', 'Risk Officer', 'Operator'],
          },
          debate_id: 'debate-live-1',
          pipeline_id: null,
          agent_name: 'Market Analyst',
          round_number: 1,
        },
        {
          event_type: 'critique',
          timestamp: '2026-03-29T12:00:24Z',
          data: {
            details: 'Germany-first still overloads compliance unless onboarding is narrowed to one region.',
          },
          debate_id: 'debate-live-1',
          pipeline_id: null,
          agent_name: 'Risk Officer',
          round_number: 1,
        },
        {
          event_type: 'consensus',
          timestamp: '2026-03-29T12:00:35Z',
          data: {
            details: 'Consensus: launch Germany now, hold France until local support coverage clears the threshold.',
          },
          debate_id: 'debate-live-1',
          pipeline_id: null,
          agent_name: 'Operator',
          round_number: 2,
        },
      ],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Debate live')).toBeInTheDocument();
    expect(screen.getByTestId('live-debate-stream')).toBeInTheDocument();
    expect(screen.getByText('Live public debate')).toBeInTheDocument();
    expect(screen.getByText('Should we launch the EU expansion this quarter?')).toBeInTheDocument();
    expect(screen.getByText('Market Analyst')).toBeInTheDocument();
    expect(screen.getByText('Risk Officer')).toBeInTheDocument();
    expect(screen.getByText('Operator')).toBeInTheDocument();
    expect(screen.getByText('Consensus: launch Germany now, hold France until local support coverage clears the threshold.')).toBeInTheDocument();
    expect(screen.queryByTestId('sample-debate-trace')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Watch this debate live' })).toHaveAttribute('href', '/spectate/debate-live-1');
  });
});
