import { render, screen } from '@testing-library/react';
import { LiveDemoSection } from '../LiveDemoSection';
import { useSpectate } from '@/hooks/useSpectate';
import { useDebateWebSocket } from '@/hooks/useDebateWebSocket';

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

jest.mock('@/hooks/useSpectate', () => ({
  useSpectate: jest.fn(),
}));

jest.mock('@/hooks/useDebateWebSocket', () => ({
  useDebateWebSocket: jest.fn(),
}));

const mockUseSpectate = useSpectate as jest.MockedFunction<typeof useSpectate>;
const mockUseDebateWebSocket = useDebateWebSocket as jest.MockedFunction<typeof useDebateWebSocket>;

describe('LiveDemoSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseSpectate.mockReturnValue({
      events: [],
      connected: true,
      loaded: true,
      status: null,
      refresh: jest.fn(),
    });

    mockUseDebateWebSocket.mockReturnValue({
      status: 'connecting',
      error: null,
      errorDetails: null,
      isConnected: false,
      isPolling: false,
      reconnectAttempt: 0,
      connectionQuality: null,
      task: '',
      agents: [],
      debateMode: null,
      settlement: null,
      messages: [],
      streamingMessages: new Map(),
      streamEvents: [],
      hasCitations: false,
      sendVote: jest.fn(),
      sendSuggestion: jest.fn(),
      registerAckCallback: jest.fn(() => jest.fn()),
      registerErrorCallback: jest.fn(() => jest.fn()),
      reconnect: jest.fn(),
      sendPing: jest.fn(),
    });
  });

  it('shows a truthful standby state when no live debate is discoverable', () => {
    render(<LiveDemoSection />);

    expect(screen.getByTestId('landing-live-standby')).toBeInTheDocument();
    expect(screen.getByText(/No public live debate is discoverable right now/i)).toBeInTheDocument();
    expect(mockUseDebateWebSocket).toHaveBeenCalledWith({
      debateId: '',
      enabled: false,
    });
  });

  it('connects to the freshest debate websocket and renders live transcript messages', () => {
    const now = new Date().toISOString();

    mockUseSpectate.mockReturnValue({
      events: [
        {
          event_type: 'proposal',
          timestamp: now,
          data: { details: 'Opening position landed' },
          debate_id: 'adhoc_live_42',
          pipeline_id: null,
          agent_name: 'claude',
          round_number: 1,
        },
      ],
      connected: true,
      loaded: true,
      status: {
        active: true,
        subscribers: 3,
        buffer_size: 12,
        bridge_state: 'activity_unattributed',
        last_event_at: now,
        activity_age_seconds: 1,
        recent_activity_window_seconds: 120,
        recent_event_count: 1,
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        unattributed_recent_event_count: 1,
      },
      refresh: jest.fn(),
    });

    mockUseDebateWebSocket.mockReturnValue({
      status: 'streaming',
      error: null,
      errorDetails: null,
      isConnected: true,
      isPolling: false,
      reconnectAttempt: 0,
      connectionQuality: null,
      task: 'Should we split the monolith into services?',
      agents: ['claude', 'gpt-4'],
      debateMode: null,
      settlement: null,
      messages: [
        {
          agent: 'claude',
          content: 'The monolith is not the bottleneck yet. Optimize delivery first.',
          round: 1,
          timestamp: 1710000000,
          role: 'proposal',
        },
      ],
      streamingMessages: new Map([
        [
          'gpt-4-task-1',
          {
            agent: 'gpt-4',
            taskId: 'task-1',
            content: 'Counterpoint: service boundaries are already visible in the org chart.',
            isComplete: false,
            startTime: Date.now(),
            expectedSeq: 0,
            pendingTokens: new Map(),
            reasoning: [],
            evidence: [],
            confidence: 0.72,
            reasoningPhase: 'CRITIQUING',
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
    });

    render(<LiveDemoSection />);

    expect(screen.getByText(/Should we split the monolith into services/i)).toBeInTheDocument();
    expect(screen.getByText(/The monolith is not the bottleneck yet/i)).toBeInTheDocument();
    expect(screen.getByText(/Counterpoint: service boundaries are already visible/i)).toBeInTheDocument();
    expect(screen.getByText(/Open full live viewer/i)).toBeInTheDocument();
    expect(mockUseDebateWebSocket).toHaveBeenCalledWith({
      debateId: 'adhoc_live_42',
      enabled: true,
    });
  });

  it('follows the most recent discoverable debate when multiple are present', () => {
    const older = new Date(Date.now() - 60_000).toISOString();
    const newer = new Date().toISOString();

    mockUseSpectate.mockReturnValue({
      events: [
        {
          event_type: 'proposal',
          timestamp: older,
          data: {},
          debate_id: 'older_debate',
          pipeline_id: null,
          agent_name: 'claude',
          round_number: 1,
        },
        {
          event_type: 'critique',
          timestamp: newer,
          data: {},
          debate_id: 'newer_debate',
          pipeline_id: null,
          agent_name: 'gpt-4',
          round_number: 1,
        },
      ],
      connected: true,
      loaded: true,
      status: {
        active: true,
        subscribers: 2,
        buffer_size: 2,
        bridge_state: 'activity_unattributed',
        last_event_at: newer,
        activity_age_seconds: 1,
        recent_activity_window_seconds: 120,
        recent_event_count: 2,
        live_debate_count: 0,
        live_debate_ids: [],
        live_debates: [],
        unattributed_recent_event_count: 2,
      },
      refresh: jest.fn(),
    });

    render(<LiveDemoSection />);

    expect(mockUseDebateWebSocket).toHaveBeenCalledWith({
      debateId: 'newer_debate',
      enabled: true,
    });
  });
});
