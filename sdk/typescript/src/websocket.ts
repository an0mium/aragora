/**
 * Aragora WebSocket Client
 *
 * Provides real-time streaming for debate events, agent messages, and consensus.
 */

import type {
  AragoraConfig,
  WebSocketEvent,
  WebSocketEventType,
  DebateStartEvent,
  RoundStartEvent,
  AgentMessageEvent,
  CritiqueEvent,
  VoteEvent,
  ConsensusEvent,
  DebateEndEvent,
} from './types';

export type WebSocketState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

export interface WebSocketOptions {
  /** Auto-reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Maximum reconnection attempts (default: 5) */
  maxReconnectAttempts?: number;
  /** Base delay between reconnection attempts in ms (default: 1000) */
  reconnectDelay?: number;
  /** Heartbeat interval in ms (default: 30000) */
  heartbeatInterval?: number;
}

type EventHandler<T> = (event: T) => void;

interface EventHandlers {
  connected: EventHandler<void>[];
  disconnected: EventHandler<{ code: number; reason: string }>[];
  error: EventHandler<Error>[];
  debate_start: EventHandler<DebateStartEvent>[];
  round_start: EventHandler<RoundStartEvent>[];
  agent_message: EventHandler<AgentMessageEvent>[];
  critique: EventHandler<CritiqueEvent>[];
  vote: EventHandler<VoteEvent>[];
  consensus: EventHandler<ConsensusEvent>[];
  debate_end: EventHandler<DebateEndEvent>[];
  message: EventHandler<WebSocketEvent>[];
}

export class AragoraWebSocket {
  private ws: WebSocket | null = null;
  private config: AragoraConfig;
  private options: Required<WebSocketOptions>;
  private state: WebSocketState = 'disconnected';
  private reconnectAttempts = 0;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private handlers: EventHandlers = {
    connected: [],
    disconnected: [],
    error: [],
    debate_start: [],
    round_start: [],
    agent_message: [],
    critique: [],
    vote: [],
    consensus: [],
    debate_end: [],
    message: [],
  };

  constructor(config: AragoraConfig, options: WebSocketOptions = {}) {
    this.config = config;
    this.options = {
      autoReconnect: options.autoReconnect ?? true,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 5,
      reconnectDelay: options.reconnectDelay ?? 1000,
      heartbeatInterval: options.heartbeatInterval ?? 30000,
    };
  }

  /**
   * Get the current connection state.
   */
  getState(): WebSocketState {
    return this.state;
  }

  /**
   * Connect to the WebSocket server.
   */
  async connect(debateId?: string): Promise<void> {
    if (this.state === 'connected' || this.state === 'connecting') {
      return;
    }

    this.state = 'connecting';

    const wsUrl = this.buildWsUrl(debateId);

    return new Promise((resolve, reject) => {
      try {
        // Use native WebSocket in browser, or require ws for Node.js
        if (typeof WebSocket !== 'undefined') {
          this.ws = new WebSocket(wsUrl);
        } else {
          // Node.js environment - dynamically require ws
          // eslint-disable-next-line @typescript-eslint/no-var-requires
          const WS = require('ws');
          this.ws = new WS(wsUrl);
        }

        this.ws!.onopen = () => {
          this.state = 'connected';
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.emit('connected', undefined);
          resolve();
        };

        this.ws!.onclose = (event) => {
          this.handleDisconnect(event.code, event.reason);
        };

        this.ws!.onerror = (event) => {
          const error = new Error('WebSocket error');
          this.emit('error', error);
          if (this.state === 'connecting') {
            reject(error);
          }
        };

        this.ws!.onmessage = (event) => {
          this.handleMessage(event.data as string);
        };
      } catch (error) {
        this.state = 'disconnected';
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server.
   */
  disconnect(): void {
    this.options.autoReconnect = false;
    this.cleanup();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.state = 'disconnected';
  }

  /**
   * Subscribe to a specific debate's events.
   */
  subscribe(debateId: string): void {
    this.send({
      type: 'subscribe',
      debate_id: debateId,
    });
  }

  /**
   * Unsubscribe from a debate's events.
   */
  unsubscribe(debateId: string): void {
    this.send({
      type: 'unsubscribe',
      debate_id: debateId,
    });
  }

  /**
   * Send a message to the server.
   */
  send(data: Record<string, unknown>): void {
    if (this.ws && this.state === 'connected') {
      this.ws.send(JSON.stringify(data));
    }
  }

  // Event handlers
  on(event: 'connected', handler: EventHandler<void>): () => void;
  on(event: 'disconnected', handler: EventHandler<{ code: number; reason: string }>): () => void;
  on(event: 'error', handler: EventHandler<Error>): () => void;
  on(event: 'debate_start', handler: EventHandler<DebateStartEvent>): () => void;
  on(event: 'round_start', handler: EventHandler<RoundStartEvent>): () => void;
  on(event: 'agent_message', handler: EventHandler<AgentMessageEvent>): () => void;
  on(event: 'critique', handler: EventHandler<CritiqueEvent>): () => void;
  on(event: 'vote', handler: EventHandler<VoteEvent>): () => void;
  on(event: 'consensus', handler: EventHandler<ConsensusEvent>): () => void;
  on(event: 'debate_end', handler: EventHandler<DebateEndEvent>): () => void;
  on(event: 'message', handler: EventHandler<WebSocketEvent>): () => void;
  on<K extends keyof EventHandlers>(event: K, handler: EventHandlers[K][number]): () => void {
    (this.handlers[event] as EventHandlers[K]).push(handler as EventHandlers[K][number]);
    return () => this.off(event, handler);
  }

  /**
   * Remove an event handler.
   */
  off<K extends keyof EventHandlers>(event: K, handler: EventHandlers[K][number]): void {
    const handlers = this.handlers[event] as EventHandlers[K];
    const index = handlers.indexOf(handler as EventHandlers[K][number]);
    if (index !== -1) {
      handlers.splice(index, 1);
    }
  }

  /**
   * Wait for a specific event to occur.
   */
  once<T>(event: keyof EventHandlers): Promise<T> {
    return new Promise((resolve) => {
      const handler = (data: T) => {
        this.off(event, handler as EventHandlers[typeof event][number]);
        resolve(data);
      };
      this.on(event as 'message', handler as EventHandler<WebSocketEvent>);
    });
  }

  private buildWsUrl(debateId?: string): string {
    let wsUrl = this.config.wsUrl;

    if (!wsUrl) {
      // Convert HTTP URL to WebSocket URL
      wsUrl = this.config.baseUrl
        .replace(/^http:/, 'ws:')
        .replace(/^https:/, 'wss:');
    }

    // Ensure path ends with /ws
    if (!wsUrl.endsWith('/ws')) {
      wsUrl = wsUrl.replace(/\/?$/, '/ws');
    }

    // Add debate ID if provided
    if (debateId) {
      wsUrl += `?debate_id=${encodeURIComponent(debateId)}`;
    }

    // Add auth token if available
    if (this.config.apiKey) {
      const separator = debateId ? '&' : '?';
      wsUrl += `${separator}token=${encodeURIComponent(this.config.apiKey)}`;
    }

    return wsUrl;
  }

  private handleMessage(data: string): void {
    try {
      const event = JSON.parse(data) as WebSocketEvent;

      // Emit to generic message handlers
      this.emit('message', event);

      // Emit to specific event type handlers
      const eventType = event.type as keyof EventHandlers;
      if (eventType in this.handlers) {
        this.emit(eventType, event.data);
      }
    } catch (error) {
      this.emit('error', new Error(`Failed to parse message: ${data}`));
    }
  }

  private handleDisconnect(code: number, reason: string): void {
    this.cleanup();
    this.state = 'disconnected';
    this.emit('disconnected', { code, reason });

    if (this.options.autoReconnect && this.reconnectAttempts < this.options.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    this.state = 'reconnecting';
    const delay = this.options.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(() => {
        // Reconnect failed, will be handled by onclose
      });
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.send({ type: 'ping' });
    }, this.options.heartbeatInterval);
  }

  private cleanup(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private emit<K extends keyof EventHandlers>(
    event: K,
    data: Parameters<EventHandlers[K][number]>[0]
  ): void {
    for (const handler of this.handlers[event]) {
      try {
        (handler as (data: unknown) => void)(data);
      } catch (error) {
        console.error(`Error in ${event} handler:`, error);
      }
    }
  }
}

/**
 * Create a WebSocket client for streaming debate events.
 */
export function createWebSocket(
  config: AragoraConfig,
  options?: WebSocketOptions
): AragoraWebSocket {
  return new AragoraWebSocket(config, options);
}
