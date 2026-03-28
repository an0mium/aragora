import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LiveDemoSection } from '../LiveDemoSection';

const mockBackendConfig = {
  api: 'http://localhost:8080',
  ws: 'wss://live.example/ws',
};

const mockReconnect = jest.fn();
const useDebateWebSocketMock = jest.fn();

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark' }),
}));

jest.mock('../../BackendSelector', () => ({
  useBackend: () => ({ config: mockBackendConfig }),
}));

jest.mock('@/hooks/useDebateWebSocket', () => ({
  useDebateWebSocket: (options: unknown) => useDebateWebSocketMock(options),
}));

jest.mock('@/components/debate/LiveDebateStream', () => ({
  LiveDebateStream: ({
    status,
    task,
    error,
  }: {
    status: string;
    task: string;
    error: string | null;
  }) => (
    <div data-testid="live-debate-stream">
      <span>{status}</span>
      <span>{task}</span>
      {error && <span>{error}</span>}
    </div>
  ),
}));

describe('LiveDemoSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockBackendConfig.api = 'http://localhost:8080';
    mockBackendConfig.ws = 'wss://live.example/ws';
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'adhoc-demo-1',
        topic: 'Should we migrate our monolith to microservices this quarter?',
      }),
    }) as typeof fetch;

    useDebateWebSocketMock.mockReturnValue({
      status: 'streaming',
      error: null,
      errorDetails: null,
      task: 'Should we migrate our monolith to microservices this quarter?',
      agents: ['anthropic-api', 'openai-api', 'gemini'],
      messages: [],
      streamingMessages: new Map(),
      streamEvents: [],
      reconnectAttempt: 0,
      connectionQuality: null,
      isPolling: false,
      reconnect: mockReconnect,
    });
  });

  it('bootstraps a live debate on mount and connects using the selected backend websocket', async () => {
    render(<LiveDemoSection />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/v1/playground/debate',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    await waitFor(() => {
      expect(useDebateWebSocketMock).toHaveBeenLastCalledWith({
        debateId: 'adhoc-demo-1',
        wsUrl: 'wss://live.example/ws',
        enabled: true,
      });
    });

    expect(screen.getByTestId('live-debate-stream')).toBeInTheDocument();
  });

  it('uses the same-origin debate endpoint when the selected backend api base is empty', async () => {
    mockBackendConfig.api = '';

    render(<LiveDemoSection />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/v1/playground/debate/',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });
  });

  it('queues the next landing-page debate while a live stream is active', async () => {
    const user = userEvent.setup();

    render(<LiveDemoSection />);

    await waitFor(() => {
      expect(screen.getByText(/watch agents argue in public/i)).toBeInTheDocument();
    });

    await user.click(
      screen.getByRole('button', {
        name: /should the product team prioritize reliability fixes over new growth experiments\?/i,
      })
    );

    expect(screen.getByText(/queued next:/i)).toBeInTheDocument();
    expect(
      screen.getByText(/should the product team prioritize reliability fixes over new growth experiments\?/i)
    ).toBeInTheDocument();
  });
});
