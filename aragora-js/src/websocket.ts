/**
 * Aragora WebSocket Client
 *
 * Real-time streaming for debate events.
 */

import {
  DebateEvent,
  DebateEventType,
  WebSocketOptions,
} from './types';

// =============================================================================
// WebSocket Client
// =============================================================================

type EventCallback = (event: DebateEvent) => void;
type ErrorCallback = (error: Error) => void;
type CloseCallback = (code: number, reason: string) => void;

export class DebateStream {
  private ws: WebSocket | null = null;
  private url: string;
  readonly debateId: string;
  private options: Required<WebSocketOptions>;
  private reconnectAttempts = 0;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private eventCallbacks: Map<DebateEventType | '*', EventCallback[]> = new Map();
  private errorCallbacks: ErrorCallback[] = [];
  private closeCallbacks: CloseCallback[] = [];
  private isClosing = false;

  constructor(
    baseUrl: string,
    debateId: string,
    options: WebSocketOptions = {}
  ) {
    this.debateId = debateId;
    this.url = this.buildUrl(baseUrl, debateId);
    this.options = {
      reconnect: options.reconnect ?? true,
      reconnectInterval: options.reconnectInterval ?? 1000,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 5,
      heartbeatInterval: options.heartbeatInterval ?? 30000,
    };
  }

  private buildUrl(baseUrl: string, debateId: string): string {
    const wsUrl = baseUrl
      .replace(/^http:/, 'ws:')
      .replace(/^https:/, 'wss:')
      .replace(/\/$/, '');
    if (wsUrl.endsWith('/ws')) {
      return wsUrl;
    }
    return `${wsUrl}/ws`;
  }

  /**
   * Connect to the debate stream.
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);
        this.isClosing = false;

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as DebateEvent;
            if (!this.shouldEmit(data)) {
              return;
            }
            this.emitEvent(data);
          } catch (error) {
            this.emitError(new Error(`Failed to parse message: ${event.data}`));
          }
        };

        this.ws.onerror = () => {
          const error = new Error('WebSocket error');
          this.emitError(error);
          if (this.ws?.readyState === WebSocket.CONNECTING) {
            reject(error);
          }
        };

        this.ws.onclose = (event) => {
          this.stopHeartbeat();
          this.emitClose(event.code, event.reason);

          if (!this.isClosing && this.options.reconnect) {
            this.attemptReconnect();
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the debate stream.
   */
  disconnect(): void {
    this.isClosing = true;
    this.stopHeartbeat();

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Subscribe to specific event types.
   *
   * @param eventType - Event type to subscribe to, or '*' for all events
   * @param callback - Callback function for events
   */
  on(eventType: DebateEventType | '*', callback: EventCallback): this {
    const callbacks = this.eventCallbacks.get(eventType) ?? [];
    callbacks.push(callback);
    this.eventCallbacks.set(eventType, callbacks);
    return this;
  }

  /**
   * Subscribe to errors.
   */
  onError(callback: ErrorCallback): this {
    this.errorCallbacks.push(callback);
    return this;
  }

  /**
   * Subscribe to connection close.
   */
  onClose(callback: CloseCallback): this {
    this.closeCallbacks.push(callback);
    return this;
  }

  /**
   * Remove event subscription.
   */
  off(eventType: DebateEventType | '*', callback: EventCallback): this {
    const callbacks = this.eventCallbacks.get(eventType);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index !== -1) {
        callbacks.splice(index, 1);
      }
    }
    return this;
  }

  /**
   * Send a message to the server.
   */
  send(data: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  /**
   * Check if connected.
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private emitEvent(event: DebateEvent): void {
    // Emit to specific type listeners
    const typeCallbacks = this.eventCallbacks.get(event.type);
    if (typeCallbacks) {
      typeCallbacks.forEach((callback) => callback(event));
    }

    // Emit to wildcard listeners
    const allCallbacks = this.eventCallbacks.get('*');
    if (allCallbacks) {
      allCallbacks.forEach((callback) => callback(event));
    }
  }

  private emitError(error: Error): void {
    this.errorCallbacks.forEach((callback) => callback(error));
  }

  private emitClose(code: number, reason: string): void {
    this.closeCallbacks.forEach((callback) => callback(code, reason));
  }

  private getEventLoopId(event: DebateEvent): string | undefined {
    const data = event.data as Record<string, unknown> | undefined;
    const dataDebateId =
      typeof data?.['debate_id'] === 'string' ? (data['debate_id'] as string) : undefined;
    const dataLoopId =
      typeof data?.['loop_id'] === 'string' ? (data['loop_id'] as string) : undefined;
    return event.loop_id || event.debate_id || dataDebateId || dataLoopId;
  }

  private shouldEmit(event: DebateEvent): boolean {
    if (!this.debateId) {
      return true;
    }
    const eventLoopId = this.getEventLoopId(event);
    return !eventLoopId || eventLoopId === this.debateId;
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      this.emitError(new Error('Max reconnect attempts reached'));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);

    this.reconnectTimeout = setTimeout(() => {
      this.connect().catch((error) => {
        this.emitError(error);
      });
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, this.options.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

// =============================================================================
// Async Iterator Support
// =============================================================================

/**
 * Create an async iterable stream for debate events.
 *
 * @example
 * ```typescript
 * const stream = streamDebate('http://localhost:8765', 'debate-123');
 *
 * for await (const event of stream) {
 *   console.log(event.type, event.data);
 *
 *   if (event.type === 'debate_end') {
 *     break;
 *   }
 * }
 * ```
 */
export async function* streamDebate(
  baseUrl: string,
  debateId: string,
  options?: WebSocketOptions
): AsyncGenerator<DebateEvent, void, unknown> {
  const stream = new DebateStream(baseUrl, debateId, options);
  const eventQueue: DebateEvent[] = [];
  let resolveNext: ((event: DebateEvent) => void) | null = null;
  let rejectNext: ((error: Error) => void) | null = null;
  let ended = false;

  stream.on('*', (event) => {
    if (resolveNext) {
      resolveNext(event);
      resolveNext = null;
    } else {
      eventQueue.push(event);
    }

    if (event.type === 'debate_end' || event.type === 'error') {
      ended = true;
    }
  });

  stream.onError((error) => {
    if (rejectNext) {
      rejectNext(error);
      rejectNext = null;
    }
    ended = true;
  });

  stream.onClose(() => {
    ended = true;
    if (resolveNext) {
      // Create a synthetic close event
      resolveNext({
        type: 'debate_end',
        data: { reason: 'connection_closed' },
        timestamp: Date.now() / 1000,
        loop_id: debateId,
        debate_id: debateId,
      });
    }
  });

  await stream.connect();

  try {
    while (!ended) {
      if (eventQueue.length > 0) {
        yield eventQueue.shift()!;
      } else {
        const event = await new Promise<DebateEvent>((resolve, reject) => {
          resolveNext = resolve;
          rejectNext = reject;
        });
        yield event;
      }
    }
  } finally {
    stream.disconnect();
  }
}

export default DebateStream;
