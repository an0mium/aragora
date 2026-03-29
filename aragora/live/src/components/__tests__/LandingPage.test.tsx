import type { ReactNode } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import { LandingPage } from '../LandingPage';

jest.mock('next/link', () => {
  const MockLink = ({ children, href }: { children: ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  );
  MockLink.displayName = 'MockLink';
  return MockLink;
});

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

  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

function createJsonResponse(payload: unknown, ok = true) {
  return Promise.resolve({
    ok,
    json: () => Promise.resolve(payload),
  });
}

describe('LandingPage live debate preview', () => {
  const originalWebSocket = global.WebSocket;
  const mockFetch = global.fetch as jest.Mock;

  beforeAll(() => {
    (global as unknown as { WebSocket: typeof MockWebSocket }).WebSocket = MockWebSocket;
  });

  afterAll(() => {
    global.WebSocket = originalWebSocket;
  });

  beforeEach(() => {
    MockWebSocket.instances = [];
    mockFetch.mockReset();
  });

  it('shows a live public debate from recent spectate activity', async () => {
    mockFetch.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/v1/spectate/recent?count=40')) {
        return createJsonResponse({
          events: [
            {
              event_type: 'proposal',
              timestamp: '2026-03-29T18:00:01Z',
              data: {
                task: 'Should Aragora open the live debate feed on the homepage?',
                details: 'Expose the strongest public debate so visitors can evaluate agent disagreement before signing up.',
                agents: ['Strategist', 'Critic'],
              },
              debate_id: 'debate-live-1',
              pipeline_id: null,
              agent_name: 'Strategist',
              round_number: 1,
            },
            {
              event_type: 'round_start',
              timestamp: '2026-03-29T18:00:00Z',
              data: {},
              debate_id: 'debate-live-1',
              pipeline_id: null,
              agent_name: null,
              round_number: 1,
            },
          ],
        });
      }

      if (url.endsWith('/api/v1/spectate/status')) {
        return createJsonResponse({
          active: true,
          recent_activity_window_seconds: 120,
          recent_event_count: 2,
          last_event_at: '2026-03-29T18:00:01Z',
        });
      }

      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    render(
      <LandingPage
        apiBase="https://api.example.com"
        wsUrl="ws://spectate.example.com/ws"
      />,
    );

    expect(await screen.findByText('LIVE DEBATE')).toBeInTheDocument();
    expect(screen.getByText('Watch agents argue in real time.')).toBeInTheDocument();
    expect(screen.getByText('2 recent events discovered for this debate.')).toBeInTheDocument();
    expect(
      screen.getByText('Should Aragora open the live debate feed on the homepage?'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Expose the strongest public debate so visitors can evaluate agent disagreement before signing up.',
      ),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1);
    });

    expect(MockWebSocket.instances[0].url).toBe(
      'ws://spectate.example.com/spectate/debate-live-1',
    );
    expect(screen.getByText('2 agents visible')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open spectator view' })).toHaveAttribute(
      'href',
      '/spectate/debate-live-1',
    );
  });

  it('streams new critique events into the public transcript in real time', async () => {
    mockFetch.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/v1/spectate/recent?count=40')) {
        return createJsonResponse({
          events: [
            {
              event_type: 'proposal',
              timestamp: '2026-03-29T18:10:00Z',
              data: {
                task: 'Should we expose live debates to new visitors?',
                details: 'Yes. A public feed proves the product is more than a static marketing promise.',
                agents: ['Planner', 'Skeptic'],
              },
              debate_id: 'debate-live-2',
              pipeline_id: null,
              agent_name: 'Planner',
              round_number: 1,
            },
          ],
        });
      }

      if (url.endsWith('/api/v1/spectate/status')) {
        return createJsonResponse({
          active: true,
          recent_activity_window_seconds: 120,
          recent_event_count: 1,
          last_event_at: '2026-03-29T18:10:00Z',
        });
      }

      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    render(
      <LandingPage
        apiBase="https://api.example.com"
        wsUrl="ws://spectate.example.com/ws"
      />,
    );

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1);
    });

    const socket = MockWebSocket.instances[0];

    act(() => {
      socket.simulateOpen();
      socket.simulateMessage({
        type: 'metadata',
        task: 'Should we expose live debates to new visitors?',
        agents: ['Planner', 'Skeptic', 'Judge'],
      });
      socket.simulateMessage({
        type: 'critique',
        timestamp: 1774807805,
        agent: 'Skeptic',
        round: 1,
        details: 'Counterpoint: do not fake liveness. Only stream it when the bridge has a real debate attached.',
      });
    });

    expect(await screen.findByText('STREAMING NOW')).toBeInTheDocument();
    expect(screen.getByText('3 agents visible')).toBeInTheDocument();
    expect(screen.getByText('Skeptic')).toBeInTheDocument();
    expect(
      screen.getByText(
        'Counterpoint: do not fake liveness. Only stream it when the bridge has a real debate attached.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('CRITIQUE')).toBeInTheDocument();
  });
});
