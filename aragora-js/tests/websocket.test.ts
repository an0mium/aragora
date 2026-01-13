/**
 * Aragora SDK WebSocket Tests
 *
 * Comprehensive E2E tests for the DebateStream WebSocket client including:
 * - Connection establishment
 * - Event streaming (debate_start, round_start, agent_message, critique, vote, consensus, debate_end)
 * - Reconnection logic with exponential backoff
 * - Heartbeat/ping-pong handling
 * - Error handling and cleanup
 * - Async iterator support (streamDebate)
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { DebateStream, streamDebate } from '../src/websocket';
import type { DebateEvent, DebateEventType } from '../src/types';

// =============================================================================
// Mock WebSocket
// =============================================================================

type WebSocketEventHandler = (event: unknown) => void;

interface MockWebSocketInstance {
  url: string;
  readyState: number;
  onopen: WebSocketEventHandler | null;
  onmessage: WebSocketEventHandler | null;
  onerror: WebSocketEventHandler | null;
  onclose: WebSocketEventHandler | null;
  send: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  triggerOpen: () => void;
  triggerMessage: (data: unknown) => void;
  triggerError: () => void;
  triggerClose: (code?: number, reason?: string) => void;
}

let mockWebSocketInstance: MockWebSocketInstance | null = null;
const mockWebSocketInstances: MockWebSocketInstance[] = [];

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState: number = MockWebSocket.CONNECTING;
  onopen: WebSocketEventHandler | null = null;
  onmessage: WebSocketEventHandler | null = null;
  onerror: WebSocketEventHandler | null = null;
  onclose: WebSocketEventHandler | null = null;

  send = vi.fn();
  close = vi.fn((code?: number, reason?: string) => {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose({ code: code ?? 1000, reason: reason ?? '' });
    }
  });

  constructor(url: string) {
    this.url = url;
    mockWebSocketInstance = this as unknown as MockWebSocketInstance;
    mockWebSocketInstances.push(this as unknown as MockWebSocketInstance);
  }

  triggerOpen() {
    this.readyState = MockWebSocket.OPEN;
    if (this.onopen) {
      this.onopen({});
    }
  }

  triggerMessage(data: unknown) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }

  triggerError() {
    if (this.onerror) {
      this.onerror({});
    }
  }

  triggerClose(code = 1000, reason = '') {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose({ code, reason });
    }
  }
}

// Replace global WebSocket with mock
(global as unknown as { WebSocket: typeof MockWebSocket }).WebSocket = MockWebSocket;

// =============================================================================
// Test Helpers
// =============================================================================

function createDebateEvent(
  type: DebateEventType,
  data: Record<string, unknown> = {}
): DebateEvent {
  return {
    type,
    debate_id: 'debate-123',
    timestamp: new Date().toISOString(),
    data,
  };
}

// =============================================================================
// Tests
// =============================================================================

describe('DebateStream', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockWebSocketInstance = null;
    mockWebSocketInstances.length = 0;
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  // ===========================================================================
  // Connection Establishment
  // ===========================================================================

  describe('connection establishment', () => {
    it('should construct WebSocket URL correctly from http base URL', () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      stream.connect();

      expect(mockWebSocketInstance?.url).toBe('ws://localhost:8080/ws/debates/debate-123');
    });

    it('should construct WebSocket URL correctly from https base URL', () => {
      const stream = new DebateStream('https://api.aragora.ai', 'debate-456');
      stream.connect();

      expect(mockWebSocketInstance?.url).toBe('wss://api.aragora.ai/ws/debates/debate-456');
    });

    it('should strip trailing slash from base URL', () => {
      const stream = new DebateStream('http://localhost:8080/', 'debate-789');
      stream.connect();

      expect(mockWebSocketInstance?.url).toBe('ws://localhost:8080/ws/debates/debate-789');
    });

    it('should resolve connect() promise on successful connection', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();

      // Simulate successful connection
      mockWebSocketInstance?.triggerOpen();

      await expect(connectPromise).resolves.toBeUndefined();
      expect(stream.isConnected).toBe(true);
    });

    it('should reject connect() promise on connection error', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();

      // Simulate connection error while still connecting
      mockWebSocketInstance?.triggerError();

      await expect(connectPromise).rejects.toThrow('WebSocket error');
    });

    it('should expose debateId property', () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-abc');
      expect(stream.debateId).toBe('debate-abc');
    });

    it('should report isConnected as false before connection', () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      expect(stream.isConnected).toBe(false);
    });
  });

  // ===========================================================================
  // Event Streaming
  // ===========================================================================

  describe('event streaming', () => {
    let stream: DebateStream;

    beforeEach(async () => {
      stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;
    });

    it('should emit debate_start event to specific listener', () => {
      const callback = vi.fn();
      stream.on('debate_start', callback);

      const event = createDebateEvent('debate_start', { agents: ['claude', 'gpt4'] });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledTimes(1);
      expect(callback).toHaveBeenCalledWith(event);
    });

    it('should emit round_start event', () => {
      const callback = vi.fn();
      stream.on('round_start', callback);

      const event = createDebateEvent('round_start', { round: 1 });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'round_start',
          data: { round: 1 },
        })
      );
    });

    it('should emit agent_message event', () => {
      const callback = vi.fn();
      stream.on('agent_message', callback);

      const event = createDebateEvent('agent_message', {
        agent_id: 'claude',
        content: 'This is my response',
        round: 1,
      });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'agent_message',
          data: expect.objectContaining({
            agent_id: 'claude',
            content: 'This is my response',
          }),
        })
      );
    });

    it('should emit critique event', () => {
      const callback = vi.fn();
      stream.on('critique', callback);

      const event = createDebateEvent('critique', {
        critic_id: 'gpt4',
        target_agent_id: 'claude',
        content: 'I disagree with this point',
        severity: 'medium',
      });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'critique',
          data: expect.objectContaining({
            critic_id: 'gpt4',
            severity: 'medium',
          }),
        })
      );
    });

    it('should emit vote event', () => {
      const callback = vi.fn();
      stream.on('vote', callback);

      const event = createDebateEvent('vote', {
        voter_id: 'user-123',
        vote: 'agree',
        agent_id: 'claude',
      });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'vote',
          data: expect.objectContaining({
            voter_id: 'user-123',
            vote: 'agree',
          }),
        })
      );
    });

    it('should emit consensus event', () => {
      const callback = vi.fn();
      stream.on('consensus', callback);

      const event = createDebateEvent('consensus', {
        reached: true,
        confidence: 0.85,
        conclusion: 'The agents agree on this point',
        supporting_agents: ['claude', 'gpt4'],
      });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'consensus',
          data: expect.objectContaining({
            reached: true,
            confidence: 0.85,
          }),
        })
      );
    });

    it('should emit debate_end event', () => {
      const callback = vi.fn();
      stream.on('debate_end', callback);

      const event = createDebateEvent('debate_end', {
        status: 'completed',
        total_rounds: 5,
      });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'debate_end',
          data: expect.objectContaining({
            status: 'completed',
          }),
        })
      );
    });

    it('should emit round_end event', () => {
      const callback = vi.fn();
      stream.on('round_end', callback);

      const event = createDebateEvent('round_end', { round: 1 });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'round_end',
          data: { round: 1 },
        })
      );
    });

    it('should emit error event', () => {
      const callback = vi.fn();
      stream.on('error', callback);

      const event = createDebateEvent('error', {
        message: 'Agent timeout',
        code: 'AGENT_TIMEOUT',
      });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          data: expect.objectContaining({
            message: 'Agent timeout',
          }),
        })
      );
    });

    it('should emit to wildcard (*) listeners for all events', () => {
      const callback = vi.fn();
      stream.on('*', callback);

      const events: DebateEvent[] = [
        createDebateEvent('debate_start', {}),
        createDebateEvent('round_start', { round: 1 }),
        createDebateEvent('agent_message', { content: 'test' }),
      ];

      events.forEach((event) => mockWebSocketInstance?.triggerMessage(event));

      expect(callback).toHaveBeenCalledTimes(3);
    });

    it('should emit to both specific and wildcard listeners', () => {
      const specificCallback = vi.fn();
      const wildcardCallback = vi.fn();

      stream.on('agent_message', specificCallback);
      stream.on('*', wildcardCallback);

      const event = createDebateEvent('agent_message', { content: 'test' });
      mockWebSocketInstance?.triggerMessage(event);

      expect(specificCallback).toHaveBeenCalledTimes(1);
      expect(wildcardCallback).toHaveBeenCalledTimes(1);
    });

    it('should support multiple listeners for the same event type', () => {
      const callback1 = vi.fn();
      const callback2 = vi.fn();

      stream.on('consensus', callback1);
      stream.on('consensus', callback2);

      const event = createDebateEvent('consensus', { reached: true });
      mockWebSocketInstance?.triggerMessage(event);

      expect(callback1).toHaveBeenCalledTimes(1);
      expect(callback2).toHaveBeenCalledTimes(1);
    });

    it('should return this for method chaining', () => {
      const result = stream
        .on('debate_start', vi.fn())
        .on('agent_message', vi.fn())
        .on('debate_end', vi.fn());

      expect(result).toBe(stream);
    });
  });

  // ===========================================================================
  // Event Unsubscription
  // ===========================================================================

  describe('event unsubscription', () => {
    let stream: DebateStream;

    beforeEach(async () => {
      stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;
    });

    it('should remove specific event listener with off()', () => {
      const callback = vi.fn();
      stream.on('agent_message', callback);

      // First message should trigger callback
      mockWebSocketInstance?.triggerMessage(createDebateEvent('agent_message', {}));
      expect(callback).toHaveBeenCalledTimes(1);

      // Remove listener
      stream.off('agent_message', callback);

      // Second message should not trigger callback
      mockWebSocketInstance?.triggerMessage(createDebateEvent('agent_message', {}));
      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('should only remove the specified callback', () => {
      const callback1 = vi.fn();
      const callback2 = vi.fn();

      stream.on('agent_message', callback1);
      stream.on('agent_message', callback2);
      stream.off('agent_message', callback1);

      mockWebSocketInstance?.triggerMessage(createDebateEvent('agent_message', {}));

      expect(callback1).not.toHaveBeenCalled();
      expect(callback2).toHaveBeenCalledTimes(1);
    });

    it('should return this for method chaining', () => {
      const callback = vi.fn();
      stream.on('debate_start', callback);
      const result = stream.off('debate_start', callback);

      expect(result).toBe(stream);
    });

    it('should handle removing non-existent callback gracefully', () => {
      const callback = vi.fn();
      // Should not throw
      expect(() => stream.off('debate_start', callback)).not.toThrow();
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('error handling', () => {
    let stream: DebateStream;

    beforeEach(async () => {
      stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;
    });

    it('should emit error when receiving invalid JSON', () => {
      const errorCallback = vi.fn();
      stream.onError(errorCallback);

      // Simulate receiving invalid JSON
      if (mockWebSocketInstance?.onmessage) {
        mockWebSocketInstance.onmessage({ data: 'not valid json' });
      }

      expect(errorCallback).toHaveBeenCalledTimes(1);
      expect(errorCallback).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('Failed to parse message'),
        })
      );
    });

    it('should emit error on WebSocket error event', () => {
      const errorCallback = vi.fn();
      stream.onError(errorCallback);

      mockWebSocketInstance?.triggerError();

      expect(errorCallback).toHaveBeenCalledTimes(1);
      expect(errorCallback).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'WebSocket error',
        })
      );
    });

    it('should emit close event on connection close', () => {
      const closeCallback = vi.fn();
      stream.onClose(closeCallback);

      mockWebSocketInstance?.triggerClose(1000, 'Normal closure');

      expect(closeCallback).toHaveBeenCalledTimes(1);
      expect(closeCallback).toHaveBeenCalledWith(1000, 'Normal closure');
    });

    it('should support multiple error callbacks', () => {
      const callback1 = vi.fn();
      const callback2 = vi.fn();

      stream.onError(callback1);
      stream.onError(callback2);

      mockWebSocketInstance?.triggerError();

      expect(callback1).toHaveBeenCalledTimes(1);
      expect(callback2).toHaveBeenCalledTimes(1);
    });

    it('should return this for error callback chaining', () => {
      const result = stream.onError(vi.fn()).onClose(vi.fn());
      expect(result).toBe(stream);
    });
  });

  // ===========================================================================
  // Reconnection Logic
  // ===========================================================================

  describe('reconnection logic', () => {
    it('should attempt reconnection with default options on unexpected close', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Simulate unexpected close
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');

      // Should schedule reconnect
      expect(mockWebSocketInstances.length).toBe(1);

      // Advance timer by reconnect interval (1000ms default)
      await vi.advanceTimersByTimeAsync(1000);

      // Should have created a new WebSocket connection
      expect(mockWebSocketInstances.length).toBe(2);
    });

    it('should use exponential backoff for reconnection attempts', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        reconnect: true,
        reconnectInterval: 1000,
        maxReconnectAttempts: 5,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // First close - triggers reconnect attempt 1
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      await vi.advanceTimersByTimeAsync(1000); // 1000 * 2^0 = 1000ms
      expect(mockWebSocketInstances.length).toBe(2);

      // Second close - triggers reconnect attempt 2
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      await vi.advanceTimersByTimeAsync(2000); // 1000 * 2^1 = 2000ms
      expect(mockWebSocketInstances.length).toBe(3);

      // Third close - triggers reconnect attempt 3
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      await vi.advanceTimersByTimeAsync(4000); // 1000 * 2^2 = 4000ms
      expect(mockWebSocketInstances.length).toBe(4);
    });

    it('should emit error when max reconnect attempts reached', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        reconnect: true,
        reconnectInterval: 100,
        maxReconnectAttempts: 2,
      });

      const errorCallback = vi.fn();
      stream.onError(errorCallback);

      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // First reconnect attempt
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      await vi.advanceTimersByTimeAsync(100);
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');

      // Second reconnect attempt
      await vi.advanceTimersByTimeAsync(200);
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');

      // Third attempt exceeds max, should emit error
      expect(errorCallback).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Max reconnect attempts reached',
        })
      );
    });

    it('should not reconnect when reconnect option is false', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        reconnect: false,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');

      await vi.advanceTimersByTimeAsync(5000);

      // Should not have created any new connections
      expect(mockWebSocketInstances.length).toBe(1);
    });

    it('should not reconnect after manual disconnect', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        reconnect: true,
        reconnectInterval: 100,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Manual disconnect
      stream.disconnect();

      await vi.advanceTimersByTimeAsync(1000);

      // Should only have the original connection
      expect(mockWebSocketInstances.length).toBe(1);
    });

    it('should reset reconnect attempts on successful connection', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        reconnect: true,
        reconnectInterval: 100,
        maxReconnectAttempts: 3,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // First close and reconnect
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      await vi.advanceTimersByTimeAsync(100);
      mockWebSocketInstance?.triggerOpen(); // Successful reconnect

      // Second close should restart from attempt 1
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      await vi.advanceTimersByTimeAsync(100); // 100 * 2^0 = 100ms (not 200ms)
      expect(mockWebSocketInstances.length).toBe(3);
    });
  });

  // ===========================================================================
  // Heartbeat/Ping-Pong
  // ===========================================================================

  describe('heartbeat handling', () => {
    it('should start heartbeat on connection', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        heartbeatInterval: 5000,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Advance past heartbeat interval
      await vi.advanceTimersByTimeAsync(5000);

      expect(mockWebSocketInstance?.send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'ping' })
      );
    });

    it('should send heartbeat at configured interval', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        heartbeatInterval: 1000,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Advance through multiple intervals
      await vi.advanceTimersByTimeAsync(3500);

      // Should have sent 3 heartbeats (at 1000, 2000, 3000ms)
      expect(mockWebSocketInstance?.send).toHaveBeenCalledTimes(3);
      expect(mockWebSocketInstance?.send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'ping' })
      );
    });

    it('should stop heartbeat on disconnect', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        heartbeatInterval: 1000,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // First heartbeat
      await vi.advanceTimersByTimeAsync(1000);
      expect(mockWebSocketInstance?.send).toHaveBeenCalledTimes(1);

      // Disconnect
      stream.disconnect();

      // No more heartbeats after disconnect
      await vi.advanceTimersByTimeAsync(3000);
      expect(mockWebSocketInstance?.send).toHaveBeenCalledTimes(1);
    });

    it('should stop heartbeat on connection close', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        heartbeatInterval: 1000,
        reconnect: false,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // First heartbeat
      await vi.advanceTimersByTimeAsync(1000);
      expect(mockWebSocketInstance?.send).toHaveBeenCalledTimes(1);

      // Close connection
      mockWebSocketInstance?.triggerClose(1000, 'Normal closure');

      // No more heartbeats
      await vi.advanceTimersByTimeAsync(3000);
      expect(mockWebSocketInstance?.send).toHaveBeenCalledTimes(1);
    });

    it('should not send heartbeat if connection is not open', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        heartbeatInterval: 1000,
        reconnect: false,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Simulate connection becoming not open
      if (mockWebSocketInstance) {
        mockWebSocketInstance.readyState = MockWebSocket.CLOSING;
      }

      await vi.advanceTimersByTimeAsync(1000);

      // Should not send heartbeat when not open
      expect(mockWebSocketInstance?.send).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Send Method
  // ===========================================================================

  describe('send method', () => {
    let stream: DebateStream;

    beforeEach(async () => {
      stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;
    });

    it('should send JSON-stringified data', () => {
      stream.send({ type: 'user_vote', vote: 'agree' });

      expect(mockWebSocketInstance?.send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'user_vote', vote: 'agree' })
      );
    });

    it('should not send when connection is not open', () => {
      if (mockWebSocketInstance) {
        mockWebSocketInstance.readyState = MockWebSocket.CLOSED;
      }

      stream.send({ type: 'user_vote', vote: 'agree' });

      expect(mockWebSocketInstance?.send).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Disconnect/Cleanup
  // ===========================================================================

  describe('disconnect and cleanup', () => {
    it('should close WebSocket with code 1000 on disconnect', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      stream.disconnect();

      expect(mockWebSocketInstance?.close).toHaveBeenCalledWith(1000, 'Client disconnect');
    });

    it('should clear pending reconnect timeout on disconnect', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        reconnect: true,
        reconnectInterval: 1000,
      });
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Trigger close to schedule reconnect
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');

      // Disconnect before reconnect timer fires
      stream.disconnect();

      // Advance past reconnect interval
      await vi.advanceTimersByTimeAsync(2000);

      // Should not have created new connection (only original + failed one)
      expect(mockWebSocketInstances.length).toBe(1);
    });

    it('should report isConnected as false after disconnect', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      expect(stream.isConnected).toBe(true);

      stream.disconnect();

      expect(stream.isConnected).toBe(false);
    });

    it('should handle multiple disconnect calls gracefully', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      stream.disconnect();
      // Should not throw
      expect(() => stream.disconnect()).not.toThrow();
    });
  });

  // ===========================================================================
  // Default Options
  // ===========================================================================

  describe('default options', () => {
    it('should use default reconnect interval of 1000ms', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');

      // Should not reconnect before 1000ms
      await vi.advanceTimersByTimeAsync(500);
      expect(mockWebSocketInstances.length).toBe(1);

      // Should reconnect at 1000ms
      await vi.advanceTimersByTimeAsync(500);
      expect(mockWebSocketInstances.length).toBe(2);
    });

    it('should use default max reconnect attempts of 5', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123', {
        reconnectInterval: 10,
      });
      const errorCallback = vi.fn();
      stream.onError(errorCallback);

      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Exhaust all 5 reconnect attempts
      for (let i = 0; i < 5; i++) {
        mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
        await vi.advanceTimersByTimeAsync(10 * Math.pow(2, i));
      }

      // 6th close should emit max attempts error
      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      expect(errorCallback).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Max reconnect attempts reached',
        })
      );
    });

    it('should use default heartbeat interval of 30000ms', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      // Should not send heartbeat before 30000ms
      await vi.advanceTimersByTimeAsync(29000);
      expect(mockWebSocketInstance?.send).not.toHaveBeenCalled();

      // Should send heartbeat at 30000ms
      await vi.advanceTimersByTimeAsync(1000);
      expect(mockWebSocketInstance?.send).toHaveBeenCalledWith(
        JSON.stringify({ type: 'ping' })
      );
    });

    it('should enable reconnect by default', async () => {
      const stream = new DebateStream('http://localhost:8080', 'debate-123');
      const connectPromise = stream.connect();
      mockWebSocketInstance?.triggerOpen();
      await connectPromise;

      mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
      await vi.advanceTimersByTimeAsync(1000);

      // Should have attempted reconnect
      expect(mockWebSocketInstances.length).toBe(2);
    });
  });
});

// =============================================================================
// Async Iterator Tests (streamDebate)
// =============================================================================

/**
 * Helper to create a generator and open its connection.
 * The streamDebate generator creates the WebSocket and waits for connection
 * when .next() is first called. We need to:
 * 1. Start the .next() call (which creates WebSocket and awaits connect)
 * 2. Open the connection (so connect() resolves)
 * 3. Then the .next() resolves (waiting for first event)
 */
async function setupStreamDebate(options?: Parameters<typeof streamDebate>[2]) {
  const generator = streamDebate('http://localhost:8080', 'debate-123', {
    reconnect: false,
    heartbeatInterval: 60000, // Long interval to avoid interference
    ...options,
  });

  // Start the first .next() call - this will:
  // 1. Create the WebSocket (populating mockWebSocketInstance)
  // 2. Await stream.connect() (which awaits onopen)
  // 3. Then wait for first event
  const firstNextPromise = generator.next();

  // Give microtask queue time to create the WebSocket
  // Use Promise.resolve().then() for better compatibility with fake timers
  await Promise.resolve();
  await Promise.resolve();

  // Now mockWebSocketInstance should be populated
  if (!mockWebSocketInstance) {
    throw new Error('WebSocket was not created');
  }

  // Open the connection - this resolves stream.connect()
  mockWebSocketInstance.triggerOpen();

  return { generator, firstNextPromise };
}

describe('streamDebate', () => {
  beforeEach(() => {
    mockWebSocketInstance = null;
    mockWebSocketInstances.length = 0;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should yield events as async iterator', async () => {
    const { generator, firstNextPromise } = await setupStreamDebate();

    // Send an event
    const event = createDebateEvent('debate_start', { agents: ['claude'] });
    mockWebSocketInstance?.triggerMessage(event);

    // Now the first .next() should resolve with the event
    const result = await firstNextPromise;
    expect(result.done).toBe(false);
    expect(result.value).toEqual(event);

    // Clean up
    await generator.return(undefined);
  });

  it('should yield multiple events in order', async () => {
    const { generator, firstNextPromise } = await setupStreamDebate();

    const events = [
      createDebateEvent('debate_start', {}),
      createDebateEvent('round_start', { round: 1 }),
      createDebateEvent('agent_message', { content: 'Hello' }),
    ];

    // Queue up events
    events.forEach((event) => mockWebSocketInstance?.triggerMessage(event));

    // First event from the initial next
    const first = await firstNextPromise;
    expect(first.value).toEqual(events[0]);

    // Iterate and collect remaining
    const results: DebateEvent[] = [first.value!];
    for (let i = 1; i < events.length; i++) {
      const { value, done } = await generator.next();
      if (!done && value) {
        results.push(value);
      }
    }

    expect(results).toEqual(events);

    await generator.return(undefined);
  });

  it('should end iteration on debate_end event', async () => {
    const { generator, firstNextPromise } = await setupStreamDebate();

    // Send debate_end
    const endEvent = createDebateEvent('debate_end', { status: 'completed' });
    mockWebSocketInstance?.triggerMessage(endEvent);

    const { value, done } = await firstNextPromise;
    expect(value).toEqual(endEvent);
    expect(done).toBe(false);

    // Next iteration should complete (generator ends after debate_end)
    const final = await generator.next();
    expect(final.done).toBe(true);
  });

  it('should end iteration on error event', async () => {
    const { generator, firstNextPromise } = await setupStreamDebate();

    const errorEvent = createDebateEvent('error', { message: 'Something went wrong' });
    mockWebSocketInstance?.triggerMessage(errorEvent);

    const { value } = await firstNextPromise;
    expect(value).toEqual(errorEvent);

    // Next iteration should complete
    const final = await generator.next();
    expect(final.done).toBe(true);
  });

  it('should synthesize debate_end on connection close', async () => {
    const { generator, firstNextPromise } = await setupStreamDebate();

    // Close connection (no events sent yet)
    mockWebSocketInstance?.triggerClose(1000, 'Normal closure');

    const { value, done } = await firstNextPromise;
    expect(done).toBe(false);
    expect(value?.type).toBe('debate_end');
    expect(value?.data).toEqual({ reason: 'connection_closed' });

    await generator.return(undefined);
  });

  it('should disconnect on generator return', async () => {
    const { generator, firstNextPromise } = await setupStreamDebate();

    // Send an event so firstNextPromise can resolve
    mockWebSocketInstance?.triggerMessage(createDebateEvent('debate_start', {}));
    await firstNextPromise;

    // Capture the WebSocket instance before return
    const ws = mockWebSocketInstance;

    // Return early
    await generator.return(undefined);

    // The WebSocket should have been closed
    expect(ws?.readyState).toBe(MockWebSocket.CLOSED);
  });

  it('should pass options to DebateStream', async () => {
    vi.useFakeTimers();

    const generator = streamDebate('http://localhost:8080', 'debate-123', {
      heartbeatInterval: 5000,
      reconnect: false,
    });

    // Start the generator and create WebSocket
    const firstNextPromise = generator.next();
    await vi.advanceTimersByTimeAsync(0);

    // Open connection
    mockWebSocketInstance?.triggerOpen();

    // Verify heartbeat interval is respected
    await vi.advanceTimersByTimeAsync(5000);
    expect(mockWebSocketInstance?.send).toHaveBeenCalledWith(
      JSON.stringify({ type: 'ping' })
    );

    // Send event to allow cleanup
    mockWebSocketInstance?.triggerMessage(createDebateEvent('debate_end', {}));
    await firstNextPromise;
    await generator.return(undefined);

    vi.useRealTimers();
  });

  it('should buffer events while consumer is slow', async () => {
    const { generator, firstNextPromise } = await setupStreamDebate();

    // Send multiple events rapidly
    const events = [
      createDebateEvent('debate_start', {}),
      createDebateEvent('round_start', { round: 1 }),
      createDebateEvent('agent_message', { content: 'Message 1' }),
      createDebateEvent('agent_message', { content: 'Message 2' }),
      createDebateEvent('agent_message', { content: 'Message 3' }),
    ];

    events.forEach((event) => mockWebSocketInstance?.triggerMessage(event));

    // Consume - first event via firstNextPromise
    const first = await firstNextPromise;
    const results: DebateEvent[] = [first.value!];

    // Remaining events
    for (let i = 1; i < events.length; i++) {
      const { value, done } = await generator.next();
      if (!done && value) {
        results.push(value);
      }
    }

    expect(results).toEqual(events);

    await generator.return(undefined);
  });
});

// =============================================================================
// Integration-style Tests
// =============================================================================

describe('DebateStream integration scenarios', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockWebSocketInstance = null;
    mockWebSocketInstances.length = 0;
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('should handle full debate lifecycle', async () => {
    const stream = new DebateStream('http://localhost:8080', 'debate-123');
    const events: DebateEvent[] = [];

    stream.on('*', (event) => events.push(event));

    const connectPromise = stream.connect();
    mockWebSocketInstance?.triggerOpen();
    await connectPromise;

    // Simulate full debate lifecycle
    const lifecycle: DebateEvent[] = [
      createDebateEvent('debate_start', { agents: ['claude', 'gpt4'] }),
      createDebateEvent('round_start', { round: 1 }),
      createDebateEvent('agent_message', { agent_id: 'claude', content: 'First response' }),
      createDebateEvent('agent_message', { agent_id: 'gpt4', content: 'Second response' }),
      createDebateEvent('critique', { critic_id: 'claude', target_agent_id: 'gpt4' }),
      createDebateEvent('critique', { critic_id: 'gpt4', target_agent_id: 'claude' }),
      createDebateEvent('round_end', { round: 1 }),
      createDebateEvent('round_start', { round: 2 }),
      createDebateEvent('agent_message', { agent_id: 'claude', content: 'Revised response' }),
      createDebateEvent('agent_message', { agent_id: 'gpt4', content: 'Revised response' }),
      createDebateEvent('vote', { voter_id: 'user-1', vote: 'agree', agent_id: 'claude' }),
      createDebateEvent('consensus', { reached: true, confidence: 0.9 }),
      createDebateEvent('round_end', { round: 2 }),
      createDebateEvent('debate_end', { status: 'completed', total_rounds: 2 }),
    ];

    lifecycle.forEach((event) => mockWebSocketInstance?.triggerMessage(event));

    expect(events).toEqual(lifecycle);

    stream.disconnect();
  });

  it('should handle reconnection during debate', async () => {
    const stream = new DebateStream('http://localhost:8080', 'debate-123', {
      reconnect: true,
      reconnectInterval: 100,
    });

    const events: DebateEvent[] = [];
    const closeCallback = vi.fn();

    stream.on('*', (event) => events.push(event));
    stream.onClose(closeCallback);

    const connectPromise = stream.connect();
    mockWebSocketInstance?.triggerOpen();
    await connectPromise;

    // Receive some events
    mockWebSocketInstance?.triggerMessage(createDebateEvent('debate_start', {}));
    mockWebSocketInstance?.triggerMessage(createDebateEvent('round_start', { round: 1 }));

    expect(events).toHaveLength(2);

    // Connection drops
    mockWebSocketInstance?.triggerClose(1006, 'Connection lost');
    expect(closeCallback).toHaveBeenCalled();

    // Wait for reconnect
    await vi.advanceTimersByTimeAsync(100);
    expect(mockWebSocketInstances.length).toBe(2);

    // New connection opens
    mockWebSocketInstance?.triggerOpen();

    // Continue receiving events on new connection
    mockWebSocketInstance?.triggerMessage(createDebateEvent('agent_message', { round: 1 }));
    expect(events).toHaveLength(3);

    stream.disconnect();
  });

  it('should track event counts by type', async () => {
    const stream = new DebateStream('http://localhost:8080', 'debate-123');
    const eventCounts: Record<string, number> = {};

    stream.on('*', (event) => {
      eventCounts[event.type] = (eventCounts[event.type] || 0) + 1;
    });

    const connectPromise = stream.connect();
    mockWebSocketInstance?.triggerOpen();
    await connectPromise;

    // Send various events
    const events: DebateEvent[] = [
      createDebateEvent('debate_start', {}),
      createDebateEvent('round_start', { round: 1 }),
      createDebateEvent('agent_message', {}),
      createDebateEvent('agent_message', {}),
      createDebateEvent('agent_message', {}),
      createDebateEvent('critique', {}),
      createDebateEvent('critique', {}),
      createDebateEvent('round_end', { round: 1 }),
      createDebateEvent('consensus', {}),
      createDebateEvent('debate_end', {}),
    ];

    events.forEach((event) => mockWebSocketInstance?.triggerMessage(event));

    expect(eventCounts).toEqual({
      debate_start: 1,
      round_start: 1,
      agent_message: 3,
      critique: 2,
      round_end: 1,
      consensus: 1,
      debate_end: 1,
    });

    stream.disconnect();
  });
});
