import { act, render, screen } from '@testing-library/react';
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

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  close = jest.fn();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  emitOpen() {
    this.onopen?.();
  }

  emitMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }
}

describe('LiveDemoSection', () => {
  const originalWebSocket = global.WebSocket;

  beforeAll(() => {
    Object.defineProperty(global, 'WebSocket', {
      writable: true,
      value: MockWebSocket,
    });
  });

  afterAll(() => {
    Object.defineProperty(global, 'WebSocket', {
      writable: true,
      value: originalWebSocket,
    });
  });

  beforeEach(() => {
    jest.clearAllMocks();
    MockWebSocket.instances = [];
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
    expect(screen.getByText('Waiting for live debate')).toBeInTheDocument();
  });

  it('streams the latest public debate instead of showing a fabricated sample transcript', async () => {
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
          event_type: 'proposal',
          timestamp: '2026-03-28T20:00:00Z',
          data: {
            details: 'Start with a modular monolith while the team is still consolidating ownership.',
            task: 'Should we break the platform into services this quarter?',
            agents: ['Strategic Analyst', "Devil's Advocate"],
          },
          debate_id: 'debate-live-42',
          pipeline_id: null,
          agent_name: 'Strategic Analyst',
          round_number: 1,
        },
      ],
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    const socket = MockWebSocket.instances[0];
    expect(socket.url).toContain('/spectate/debate-live-42');

    act(() => {
      socket.emitOpen();
      socket.emitMessage({
        type: 'metadata',
        debate_id: 'debate-live-42',
        task: 'Should we break the platform into services this quarter?',
        agents: ['Strategic Analyst', "Devil's Advocate"],
      });
      socket.emitMessage({
        type: 'critique',
        debate_id: 'debate-live-42',
        timestamp: 1743192060,
        agent: "Devil's Advocate",
        round: 1,
        details: 'Splitting now would add operational drag before the architecture boundaries are stable.',
      });
    });

    expect(screen.getByText('Bridge active')).toBeInTheDocument();
    expect(screen.getByText('9 recent events in the last 2 minutes.')).toBeInTheDocument();
    expect(screen.getByText('Last activity 34s ago')).toBeInTheDocument();
    expect(screen.getByText('Live public debate')).toBeInTheDocument();
    expect(
      screen.getByText('Should we break the platform into services this quarter?'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Start with a modular monolith while the team is still consolidating ownership.',
      ),
    ).toBeInTheDocument();
    expect(
      await screen.findByText(
        'Splitting now would add operational drag before the architecture boundaries are stable.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('Open full spectate')).toHaveAttribute(
      'href',
      '/spectate/debate-live-42',
    );
  });
});
