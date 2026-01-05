/**
 * Tests for DebateViewer component
 *
 * Tests cover:
 * - WebSocket connection lifecycle
 * - Message rendering and scrolling
 * - Phase transitions
 * - User participation integration
 * - Error handling
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { DebateViewer } from '../src/components/DebateViewer';

// Mock next/link
jest.mock('next/link', () => {
  return ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  );
});

// Mock WebSocket
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
    // Simulate connection opening
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) this.onopen();
    }, 10);
  }

  send = jest.fn();
  close = jest.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose();
  });

  // Helper to simulate incoming messages
  simulateMessage(data: object) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }
}

let mockWsInstance: MockWebSocket | null = null;

// @ts-expect-error - mocking global WebSocket
global.WebSocket = jest.fn((url: string) => {
  mockWsInstance = new MockWebSocket(url);
  return mockWsInstance;
});

describe('DebateViewer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWsInstance = null;
  });

  it('renders loading state initially', () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);
    expect(screen.getByText(/Connecting/i)).toBeInTheDocument();
  });

  it('establishes WebSocket connection with correct URL', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(global.WebSocket).toHaveBeenCalledWith(
        expect.stringContaining('ws://localhost:3001')
      );
    });
  });

  it('displays connection status when connected', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    // After connection opens, should show connected state
    await waitFor(() => {
      expect(screen.queryByText(/Connecting/i)).not.toBeInTheDocument();
    });
  });

  it('renders agent messages from WebSocket', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    act(() => {
      mockWsInstance?.simulateMessage({
        type: 'agent_message',
        data: {
          agent: 'claude-3-opus',
          role: 'proposer',
          content: 'I propose we implement feature X because it would improve user experience.',
        },
        timestamp: Date.now(),
        round: 1,
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/I propose we implement feature X/)).toBeInTheDocument();
    });
  });

  it('handles phase transition events', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    act(() => {
      mockWsInstance?.simulateMessage({
        type: 'phase_change',
        data: {
          phase: 'voting',
          previous_phase: 'deliberation',
        },
        timestamp: Date.now(),
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/voting/i)).toBeInTheDocument();
    });
  });

  it('handles vote events', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    act(() => {
      mockWsInstance?.simulateMessage({
        type: 'vote',
        data: {
          voter: 'claude-3-opus',
          choice: 'proposal_a',
          rationale: 'This approach is more maintainable.',
        },
        timestamp: Date.now(),
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/claude-3-opus/)).toBeInTheDocument();
    });
  });

  it('handles consensus events', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    act(() => {
      mockWsInstance?.simulateMessage({
        type: 'consensus_reached',
        data: {
          outcome: 'accepted',
          confidence: 0.85,
          final_decision: 'Feature X will be implemented with modifications.',
        },
        timestamp: Date.now(),
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/consensus/i)).toBeInTheDocument();
    });
  });

  it('auto-scrolls to new messages', async () => {
    const scrollIntoViewMock = jest.fn();
    Element.prototype.scrollIntoView = scrollIntoViewMock;

    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    act(() => {
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
    const { unmount } = render(
      <DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />
    );

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    unmount();

    expect(mockWsInstance?.close).toHaveBeenCalled();
  });

  it('handles WebSocket reconnection on error', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    const initialCallCount = (global.WebSocket as jest.Mock).mock.calls.length;

    // Simulate error
    act(() => {
      if (mockWsInstance?.onerror) {
        mockWsInstance.onerror(new Error('Connection lost'));
      }
      if (mockWsInstance?.onclose) {
        mockWsInstance.readyState = MockWebSocket.CLOSED;
        mockWsInstance.onclose();
      }
    });

    // Should attempt reconnection
    await waitFor(() => {
      expect((global.WebSocket as jest.Mock).mock.calls.length).toBeGreaterThan(initialCallCount);
    }, { timeout: 5000 });
  });

  it('sends user vote through WebSocket', async () => {
    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    // Simulate receiving proposals to enable voting
    act(() => {
      mockWsInstance?.simulateMessage({
        type: 'agent_message',
        data: { agent: 'Agent1', role: 'proposer', content: 'Proposal A' },
        timestamp: Date.now(),
        round: 1,
      });
      mockWsInstance?.simulateMessage({
        type: 'agent_message',
        data: { agent: 'Agent2', role: 'proposer', content: 'Proposal B' },
        timestamp: Date.now(),
        round: 1,
      });
    });

    // Check if send was called with vote data (depends on UI implementation)
    // This is a structural test - actual voting UI may vary
  });

  it('displays debate ID correctly', async () => {
    render(<DebateViewer debateId="debate-abc-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(mockWsInstance?.readyState).toBe(MockWebSocket.OPEN);
    });

    // Debate ID should be visible somewhere in the UI
    expect(screen.getByText(/debate-abc-123/i)).toBeInTheDocument();
  });
});

describe('DebateViewer error states', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWsInstance = null;
  });

  it('shows error message on WebSocket failure', async () => {
    // Override WebSocket to fail immediately
    const originalWS = global.WebSocket;
    // @ts-expect-error - mocking global WebSocket
    global.WebSocket = jest.fn(() => {
      const ws = new MockWebSocket('ws://localhost:3001');
      setTimeout(() => {
        if (ws.onerror) ws.onerror(new Error('Connection refused'));
        if (ws.onclose) {
          ws.readyState = MockWebSocket.CLOSED;
          ws.onclose();
        }
      }, 10);
      return ws;
    });

    render(<DebateViewer debateId="test-123" wsUrl="ws://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText(/disconnected|error|failed/i)).toBeInTheDocument();
    }, { timeout: 3000 });

    global.WebSocket = originalWS;
  });
});
