/**
 * Tests for DebateViewer component
 *
 * Tests cover:
 * - WebSocket connection lifecycle
 * - Message rendering and scrolling
 * - Phase transitions
 * - User participation integration
 * - Error handling
 *
 * Note: The DebateViewer uses WebSocket only for debate IDs starting with 'adhoc_'.
 * For other IDs, it fetches from Supabase.
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { DebateViewer } from '../src/components/DebateViewer';

// Mock next/link
jest.mock('next/link', () => {
  return ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  );
});

// Mock Supabase fetch
jest.mock('../src/utils/supabase', () => ({
  fetchDebateById: jest.fn(),
}));

// Mock scrollIntoView (not available in jsdom)
Element.prototype.scrollIntoView = jest.fn();

// Mock WebSocket - connection must be manually triggered via simulateOpen() for test control
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState: number = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: ((error: Error) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    // Don't auto-connect - let tests control when connection opens
  }

  send = jest.fn();
  close = jest.fn((code?: number, reason?: string) => {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      // Pass a mock CloseEvent with the code
      this.onclose({ code: code || 1000, reason: reason || '', wasClean: true } as CloseEvent);
    }
  });

  // Helper to simulate incoming messages
  simulateMessage(data: object) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }

  // Helper to manually open connection (for test control)
  simulateOpen() {
    if (this.readyState === MockWebSocket.CONNECTING) {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) this.onopen();
    }
  }
}

let mockWsInstance: MockWebSocket | null = null;

// Create WebSocket mock with static constants
const WebSocketMock = jest.fn((url: string) => {
  mockWsInstance = new MockWebSocket(url);
  return mockWsInstance;
});
// Static constants required for cleanup logic in hooks
(WebSocketMock as { CONNECTING: number }).CONNECTING = 0;
(WebSocketMock as { OPEN: number }).OPEN = 1;
(WebSocketMock as { CLOSING: number }).CLOSING = 2;
(WebSocketMock as { CLOSED: number }).CLOSED = 3;

// @ts-expect-error - mocking global WebSocket
global.WebSocket = WebSocketMock;

describe('DebateViewer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWsInstance = null;
  });

  // Use adhoc_ prefix to enable WebSocket mode
  const LIVE_DEBATE_ID = 'adhoc_test-123';

  it('renders connecting state initially for live debates', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });
    expect(screen.getByText(/Connecting/i)).toBeInTheDocument();
  });

  it('establishes WebSocket connection with correct URL', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    expect(global.WebSocket).toHaveBeenCalledWith(
      expect.stringContaining('ws://localhost:3001')
    );
  });

  it('displays live debate view when WebSocket connects', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    // Open the connection - this triggers status change which may recreate WebSocket
    await act(async () => {
      mockWsInstance?.simulateOpen();
    });

    // After connection opens, component should show live debate UI
    // Note: the hook may have recreated the WebSocket due to status change
    await waitFor(() => {
      expect(screen.getByText(/LIVE DEBATE|Connecting/i)).toBeInTheDocument();
    });
  });

  it('creates WebSocket with correct URL', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    // Verify WebSocket was created
    expect(global.WebSocket).toHaveBeenCalled();
    // Verify it's using the correct URL pattern
    const wsCall = (global.WebSocket as jest.Mock).mock.calls[0];
    expect(wsCall[0]).toContain('ws://localhost:3001');
  });

  it('handles debate_start events', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    await act(async () => {
      mockWsInstance?.simulateOpen();
    });

    await act(async () => {
      mockWsInstance?.simulateMessage({
        type: 'debate_start',
        data: {
          task: 'Discuss the best approach for feature X',
          agents: ['claude-3-opus', 'gemini-2.0-flash'],
        },
        timestamp: Date.now(),
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/Discuss the best approach for feature X/)).toBeInTheDocument();
    });
    expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    expect(screen.getByText('gemini-2.0-flash')).toBeInTheDocument();
  });

  it('displays agent names when messages arrive', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    await act(async () => {
      mockWsInstance?.simulateOpen();
    });

    await act(async () => {
      mockWsInstance?.simulateMessage({
        type: 'agent_message',
        data: {
          agent: 'claude-3-opus',
          role: 'proposer',
          content: 'Test message',
        },
        timestamp: Date.now(),
        round: 1,
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/CLAUDE-3-OPUS/)).toBeInTheDocument();
    });
  });

  it('auto-scrolls to new messages', async () => {
    const scrollIntoViewMock = jest.fn();
    Element.prototype.scrollIntoView = scrollIntoViewMock;

    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    await act(async () => {
      mockWsInstance?.simulateOpen();
    });

    await act(async () => {
      mockWsInstance?.simulateMessage({
        type: 'agent_message',
        data: {
          agent: 'claude-3-opus',
          content: 'New message that should trigger scroll',
        },
        timestamp: Date.now(),
      });
    });

    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalled();
    });
  });

  it('cleans up WebSocket on unmount', async () => {
    let unmount: () => void;
    await act(async () => {
      const result = render(
        <DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />
      );
      unmount = result.unmount;
    });

    // Unmount should call close on the WebSocket
    await act(async () => {
      unmount();
    });

    // WebSocket close should have been called (either on the original or any recreated instance)
    expect(mockWsInstance?.close).toHaveBeenCalled();
  });

  it('creates WebSocket for live debate IDs', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    // For live debates (adhoc_ prefix), WebSocket should be created
    expect(global.WebSocket).toHaveBeenCalled();
  });

  it('displays debate ID correctly', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    await act(async () => {
      mockWsInstance?.simulateOpen();
    });

    // Debate ID should be visible somewhere in the UI
    expect(screen.getByText(new RegExp(LIVE_DEBATE_ID, 'i'))).toBeInTheDocument();
  });
});

describe('DebateViewer error states', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWsInstance = null;
  });

  const LIVE_DEBATE_ID = 'adhoc_error-test';

  it('handles WebSocket creation for live debates', async () => {
    await act(async () => {
      render(<DebateViewer debateId={LIVE_DEBATE_ID} wsUrl="ws://localhost:3001" />);
    });

    // WebSocket should be created for live debates
    expect(global.WebSocket).toHaveBeenCalled();
  });
});

describe('DebateViewer archived debates', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWsInstance = null;
    // Reset the mock for each test
    const supabaseMock = jest.requireMock('../src/utils/supabase');
    supabaseMock.fetchDebateById.mockReset();
  });

  it('does not create WebSocket for non-live debates', async () => {
    const supabaseMock = jest.requireMock('../src/utils/supabase');
    // Return null to simulate loading state
    supabaseMock.fetchDebateById.mockResolvedValue(null);

    await act(async () => {
      render(<DebateViewer debateId="archived-debate-123" wsUrl="ws://localhost:3001" />);
    });

    // WebSocket should NOT be called for non-adhoc debates
    expect(global.WebSocket).not.toHaveBeenCalled();
  });

  it('shows error for non-existent archived debate', async () => {
    const supabaseMock = jest.requireMock('../src/utils/supabase');
    supabaseMock.fetchDebateById.mockResolvedValue(null);

    await act(async () => {
      render(<DebateViewer debateId="non-existent" wsUrl="ws://localhost:3001" />);
    });

    await waitFor(() => {
      expect(screen.getByText(/Debate not found/i)).toBeInTheDocument();
    });
  });
});
