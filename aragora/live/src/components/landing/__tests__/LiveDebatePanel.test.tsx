import { render, screen } from '@testing-library/react';
import { LiveDebatePanel } from '../LiveDebatePanel';

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

const mockUseSpectate = jest.fn();

jest.mock('@/hooks/useSpectate', () => ({
  useSpectate: (...args: unknown[]) => mockUseSpectate(...args),
}));

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;
  onopen: (() => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
  }
}

describe('LiveDebatePanel', () => {
  const originalWebSocket = global.WebSocket;

  beforeAll(() => {
    (global as unknown as { WebSocket: typeof MockWebSocket }).WebSocket = MockWebSocket;
  });

  afterAll(() => {
    global.WebSocket = originalWebSocket;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    MockWebSocket.instances = [];
    mockUseSpectate.mockReturnValue({
      status: null,
      loaded: false,
      connected: false,
      events: [],
      refresh: jest.fn(),
    });
  });

  it('trusts the shared bridge status and stays offline without opening a socket', () => {
    render(
      <LiveDebatePanel
        apiBase="https://api.example.com"
        wsUrl="ws://spectate.example.com/ws"
        bridgeState={{
          status: {
            active: false,
            subscribers: 0,
            buffer_size: 0,
            bridge_state: 'inactive',
            last_event_at: null,
            activity_age_seconds: null,
            recent_activity_window_seconds: 120,
            recent_event_count: 0,
            live_debate_count: 0,
            live_debate_ids: [],
            live_debates: [],
            unattributed_recent_event_count: 0,
          },
          loaded: true,
          events: [
            {
              event_type: 'proposal',
              timestamp: '2026-04-05T16:00:00Z',
              data: { task: 'Should stale buffered events mark the bridge live?' },
              debate_id: 'debate-stale-1',
              pipeline_id: null,
              agent_name: 'Strategist',
              round_number: 1,
            },
          ],
        }}
      />,
    );

    expect(mockUseSpectate).toHaveBeenCalledWith(undefined, undefined, {
      apiBaseUrl: 'https://api.example.com',
      pollInterval: 4000,
      maxEvents: 40,
      enabled: false,
    });
    expect(screen.getByText('BRIDGE OFFLINE')).toBeInTheDocument();
    expect(
      screen.getByText(
        'The public spectate bridge is unreachable right now, so no live debate is shown.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open spectator view' })).toHaveAttribute(
      'href',
      '/spectate',
    );
    expect(MockWebSocket.instances).toHaveLength(0);
  });
});
