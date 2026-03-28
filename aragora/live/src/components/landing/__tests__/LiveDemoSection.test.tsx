import type { ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import { LiveDemoSection } from '../LiveDemoSection';
import { useSpectate } from '@/hooks/useSpectate';
import { useDebateWebSocket } from '@/hooks/useDebateWebSocket';

jest.mock('next/link', () => {
  return function MockLink({
    children,
    href,
  }: {
    children: ReactNode;
    href: string;
  }) {
    return <a href={href}>{children}</a>;
  };
});

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

jest.mock('@/hooks/useScrollReveal', () => ({
  useScrollReveal: () => ({ current: null }),
}));

jest.mock('@/hooks/useSpectate', () => ({
  useSpectate: jest.fn(),
}));

jest.mock('@/hooks/useDebateWebSocket', () => ({
  useDebateWebSocket: jest.fn(),
}));

const mockUseSpectate = useSpectate as jest.MockedFunction<typeof useSpectate>;
const mockUseDebateWebSocket = useDebateWebSocket as jest.MockedFunction<typeof useDebateWebSocket>;

function buildDebateState(overrides: Partial<ReturnType<typeof useDebateWebSocket>> = {}) {
  return {
    status: 'streaming' as const,
    error: null,
    errorDetails: null,
    isConnected: true,
    isPolling: false,
    reconnectAttempt: 0,
    connectionQuality: { reconnectCount: 0, avgLatencyMs: 84, uptimeSeconds: 17, lastSeq: 12, bufferSize: 0, oldestSeq: 12 },
    task: 'Should we consolidate vendors or build the platform in-house?',
    agents: ['claude', 'gpt-4', 'grok'],
    debateMode: 'judge',
    settlement: null,
    messages: [
      {
        agent: 'claude',
        role: 'proposal',
        content: 'Consolidating vendors lowers implementation risk and gets the team moving this quarter.',
        round: 1,
        timestamp: 1711540800,
      },
      {
        agent: 'gpt-4',
        role: 'critique',
        content: 'Vendor consolidation creates lock-in unless procurement keeps an exit plan and data portability requirements.',
        round: 1,
        timestamp: 1711540812,
      },
    ],
    streamingMessages: new Map([
      [
        'grok:stream',
        {
          agent: 'grok',
          taskId: 'stream',
          content: 'The internal build only works if the roadmap can tolerate a slower first release.',
          isComplete: false,
          startTime: Date.now() - 1000,
          expectedSeq: 4,
          pendingTokens: new Map(),
          reasoning: [],
          evidence: [],
          confidence: 0.61,
          reasoningPhase: 'counterargument',
        },
      ],
    ]),
    streamEvents: [],
    hasCitations: false,
    sendVote: jest.fn(),
    sendSuggestion: jest.fn(),
    registerAckCallback: jest.fn(() => jest.fn()),
    registerErrorCallback: jest.fn(() => jest.fn()),
    reconnect: jest.fn(),
    sendPing: jest.fn(),
    ...overrides,
  };
}

describe('LiveDemoSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseSpectate.mockReturnValue({
      events: [],
      connected: true,
      loaded: true,
      status: {
        active: true,
        subscribers: 2,
        buffer_size: 8,
        bridge_state: 'live_debates_available',
        last_event_at: '2026-03-27T15:00:00Z',
        activity_age_seconds: 3,
        recent_activity_window_seconds: 120,
        recent_event_count: 8,
        live_debate_count: 1,
        live_debate_ids: ['adhoc_live_1'],
        live_debates: [
          {
            debate_id: 'adhoc_live_1',
            recent_event_count: 8,
            last_event_at: '2026-03-27T15:00:00Z',
            event_types: ['agent_message', 'consensus'],
          },
        ],
        unattributed_recent_event_count: 0,
      },
      refresh: jest.fn(),
    });

    mockUseDebateWebSocket.mockReturnValue(buildDebateState());
  });

  it('renders a live debate transcript and subscribes to the discovered debate', () => {
    render(<LiveDemoSection />);

    expect(screen.getByTestId('live-debate-section')).toBeInTheDocument();
    expect(screen.getByText('Should we consolidate vendors or build the platform in-house?')).toBeInTheDocument();
    expect(screen.getByText('Consolidating vendors lowers implementation risk and gets the team moving this quarter.')).toBeInTheDocument();
    expect(screen.getByText('The internal build only works if the roadmap can tolerate a slower first release.')).toBeInTheDocument();
    expect(screen.getByTestId('live-debate-agent-claude')).toBeInTheDocument();
    expect(mockUseDebateWebSocket).toHaveBeenCalledWith(
      expect.objectContaining({
        debateId: 'adhoc_live_1',
        enabled: true,
      }),
    );
  });

  it('shows an honest empty state when no live debate is discoverable', () => {
    mockUseSpectate.mockReturnValue({
      events: [],
      connected: true,
      loaded: true,
      status: {
        active: true,
        subscribers: 0,
        buffer_size: 0,
        bridge_state: 'idle',
        last_event_at: null,
        activity_age_seconds: null,
        recent_activity_window_seconds: 120,
        recent_event_count: 0,
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        unattributed_recent_event_count: 0,
      },
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(screen.getByTestId('live-debate-empty')).toBeInTheDocument();
    expect(screen.getByText('No live debate is discoverable right now.')).toBeInTheDocument();
    expect(mockUseDebateWebSocket).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: false,
      }),
    );
  });

  it('falls back to recent spectate events when status lacks live debate summaries', () => {
    mockUseSpectate.mockReturnValue({
      events: [
        {
          event_type: 'agent_message',
          timestamp: new Date().toISOString(),
          data: {},
          debate_id: 'adhoc_recent_2',
          pipeline_id: null,
          agent_name: 'claude',
          round_number: 1,
        },
      ],
      connected: true,
      loaded: true,
      status: null,
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(mockUseDebateWebSocket).toHaveBeenCalledWith(
      expect.objectContaining({
        debateId: 'adhoc_recent_2',
        enabled: true,
      }),
    );
  });
});
