import { act, renderHook, waitFor } from '@testing-library/react';

import { useSpectate, type SpectateEvent } from '@/hooks/useSpectate';

jest.mock('@/config', () => ({
  API_BASE_URL: 'http://localhost:8080',
}));

type MockListener = (event: MessageEvent<string>) => void;

class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close = jest.fn(() => undefined);

  private listeners = new Map<string, Set<MockListener>>();

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListener): void {
    const typedListener = listener as unknown as MockListener;
    const existing = this.listeners.get(type) ?? new Set<MockListener>();
    existing.add(typedListener);
    this.listeners.set(type, existing);
  }

  removeEventListener(type: string, listener: EventListener): void {
    this.listeners.get(type)?.delete(listener as unknown as MockListener);
  }

  emit(type: string, payload: unknown): void {
    const event = { data: JSON.stringify(payload) } as MessageEvent<string>;
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }

  open(): void {
    this.onopen?.(new Event('open'));
  }

  fail(): void {
    this.onerror?.(new Event('error'));
  }

  static reset(): void {
    MockEventSource.instances = [];
  }
}

const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockStatus = {
  active: true,
  subscribers: 1,
  buffer_size: 2,
  bridge_state: 'live_debates_available' as const,
  last_event_at: '2026-04-03T05:00:01Z',
  activity_age_seconds: 1,
  recent_activity_window_seconds: 120,
  recent_event_count: 2,
  live_debate_count: 1,
  live_debate_ids: ['debate-1'],
  live_debates: [
    {
      debate_id: 'debate-1',
      recent_event_count: 2,
      last_event_at: '2026-04-03T05:00:01Z',
      event_types: ['proposal'],
    },
  ],
  unattributed_recent_event_count: 0,
};

function buildEvent(overrides: Partial<SpectateEvent> = {}): SpectateEvent {
  return {
    event_type: 'proposal',
    timestamp: '2026-04-03T05:00:00Z',
    data: { details: 'Initial event' },
    debate_id: 'debate-1',
    pipeline_id: null,
    agent_name: 'claude',
    round_number: 1,
    ...overrides,
  };
}

describe('useSpectate', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    MockEventSource.reset();
    global.EventSource = MockEventSource as unknown as typeof EventSource;
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('bootstraps recent events and appends live SSE updates', async () => {
    const initialEvent = buildEvent();
    const liveEvent = buildEvent({
      timestamp: '2026-04-03T05:00:02Z',
      data: { details: 'Live event' },
      agent_name: 'gpt-5',
    });

    mockFetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/v1/spectate/recent')) {
        return {
          ok: true,
          json: async () => ({ events: [initialEvent] }),
        };
      }
      if (url.endsWith('/api/v1/spectate/status')) {
        return {
          ok: true,
          json: async () => mockStatus,
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    const { result } = renderHook(() => useSpectate());

    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.events).toEqual([initialEvent]);
    expect(MockEventSource.instances).toHaveLength(1);

    const stream = MockEventSource.instances[0];
    expect(stream.url).toContain('/api/v1/spectate/stream?count=50&format=sse');

    act(() => {
      stream.open();
      stream.emit('proposal', liveEvent);
    });

    await waitFor(() =>
      expect(result.current.events).toEqual([initialEvent, liveEvent]),
    );
    expect(result.current.connected).toBe(true);
  });

  it('falls back to polling when the live stream errors', async () => {
    jest.useFakeTimers();

    let recentCalls = 0;
    mockFetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/v1/spectate/recent')) {
        recentCalls += 1;
        return {
          ok: true,
          json: async () => ({
            events: [buildEvent({ timestamp: `2026-04-03T05:00:0${recentCalls}Z` })],
          }),
        };
      }
      if (url.endsWith('/api/v1/spectate/status')) {
        return {
          ok: true,
          json: async () => mockStatus,
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    renderHook(() => useSpectate(undefined, undefined, { pollInterval: 2000 }));

    await waitFor(() => expect(recentCalls).toBe(1));
    expect(MockEventSource.instances).toHaveLength(1);

    const stream = MockEventSource.instances[0];
    act(() => {
      stream.fail();
    });

    await waitFor(() => expect(stream.close).toHaveBeenCalled());
    await waitFor(() => expect(recentCalls).toBe(2));

    act(() => {
      jest.advanceTimersByTime(2000);
    });

    await waitFor(() => expect(recentCalls).toBeGreaterThanOrEqual(3));
  });
});
