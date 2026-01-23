/**
 * WebSocket Stream Client for Aragora Control Plane.
 *
 * Provides real-time updates for:
 * - Agent status changes
 * - Task lifecycle events
 * - System health metrics
 * - Deliberation progress
 */

import * as vscode from 'vscode';

export type ControlPlaneEventType =
  | 'connected'
  | 'disconnected'
  | 'agent_registered'
  | 'agent_unregistered'
  | 'agent_status_changed'
  | 'task_submitted'
  | 'task_claimed'
  | 'task_completed'
  | 'task_failed'
  | 'deliberation_started'
  | 'deliberation_consensus'
  | 'deliberation_failed'
  | 'scheduler_stats'
  | 'health_update'
  | 'error';

export interface ControlPlaneEvent {
  type: ControlPlaneEventType;
  timestamp: number;
  data: Record<string, unknown>;
}

export interface AgentState {
  id: string;
  status: 'idle' | 'busy' | 'offline' | 'error';
  model: string;
  current_task_id?: string;
}

export interface SchedulerStats {
  pending_tasks: number;
  running_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  agents_registered: number;
  agents_idle: number;
  agents_busy: number;
}

export interface StreamClientOptions {
  /** WebSocket URL */
  wsUrl: string;
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Reconnect delay in ms */
  reconnectDelay?: number;
  /** Max reconnect attempts */
  maxReconnectAttempts?: number;
}

export interface StreamClientEvents {
  onConnected?: () => void;
  onDisconnected?: (reason: string) => void;
  onAgentRegistered?: (agentId: string, data: Record<string, unknown>) => void;
  onAgentStatusChanged?: (agentId: string, oldStatus: string, newStatus: string) => void;
  onTaskCompleted?: (taskId: string, agentId: string) => void;
  onTaskFailed?: (taskId: string, error: string) => void;
  onDeliberationStarted?: (taskId: string, question: string) => void;
  onDeliberationConsensus?: (taskId: string, answer: string) => void;
  onSchedulerStats?: (stats: SchedulerStats) => void;
  onEvent?: (event: ControlPlaneEvent) => void;
  onError?: (error: string) => void;
}

/**
 * WebSocket client for Control Plane streaming updates.
 */
export class ControlPlaneStreamClient {
  private ws: WebSocket | null = null;
  private options: Required<StreamClientOptions>;
  private events: StreamClientEvents;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private isConnecting = false;
  private shouldReconnect = true;

  // State
  private agents: Map<string, AgentState> = new Map();
  private schedulerStats: SchedulerStats | null = null;

  constructor(options: StreamClientOptions, events: StreamClientEvents = {}) {
    this.options = {
      wsUrl: options.wsUrl,
      autoReconnect: options.autoReconnect ?? true,
      reconnectDelay: options.reconnectDelay ?? 3000,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 10,
    };
    this.events = events;
  }

  /**
   * Connect to the control plane WebSocket stream.
   */
  connect(): void {
    if (this.ws || this.isConnecting) {
      return;
    }

    this.isConnecting = true;
    this.shouldReconnect = true;

    try {
      // Use global WebSocket in Node.js 18+ or ws package
      const WebSocketImpl = typeof WebSocket !== 'undefined' ? WebSocket : require('ws');
      const ws: WebSocket = new WebSocketImpl(this.options.wsUrl);
      this.ws = ws;

      ws.onopen = () => {
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.events.onConnected?.();
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data as string) as ControlPlaneEvent;
          this.handleEvent(data);
        } catch (e) {
          this.events.onError?.(`Failed to parse message: ${e}`);
        }
      };

      ws.onclose = (event: CloseEvent) => {
        this.ws = null;
        this.isConnecting = false;
        this.events.onDisconnected?.(event.reason || 'Connection closed');

        if (this.shouldReconnect && this.options.autoReconnect) {
          this.scheduleReconnect();
        }
      };

      ws.onerror = (event: Event) => {
        this.events.onError?.(`WebSocket error: ${event}`);
      };
    } catch (e) {
      this.isConnecting = false;
      this.events.onError?.(`Failed to connect: ${e}`);

      if (this.shouldReconnect && this.options.autoReconnect) {
        this.scheduleReconnect();
      }
    }
  }

  /**
   * Disconnect from the WebSocket stream.
   */
  disconnect(): void {
    this.shouldReconnect = false;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Check if connected.
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get current agents state.
   */
  getAgents(): Map<string, AgentState> {
    return new Map(this.agents);
  }

  /**
   * Get current scheduler stats.
   */
  getSchedulerStats(): SchedulerStats | null {
    return this.schedulerStats;
  }

  /**
   * Send a ping to keep connection alive.
   */
  sendPing(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ping' }));
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      this.events.onError?.('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.options.reconnectDelay * Math.min(this.reconnectAttempts, 5);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private handleEvent(event: ControlPlaneEvent): void {
    // Call generic handler
    this.events.onEvent?.(event);

    const { data } = event;

    switch (event.type) {
      case 'connected':
        // Initial connection confirmed
        break;

      case 'agent_registered': {
        const agentId = data.agent_id as string;
        const agent: AgentState = {
          id: agentId,
          status: 'idle',
          model: (data.model as string) || 'unknown',
        };
        this.agents.set(agentId, agent);
        this.events.onAgentRegistered?.(agentId, data);
        break;
      }

      case 'agent_unregistered': {
        const agentId = data.agent_id as string;
        this.agents.delete(agentId);
        break;
      }

      case 'agent_status_changed': {
        const agentId = data.agent_id as string;
        const oldStatus = data.old_status as string;
        const newStatus = data.new_status as string;
        const existing = this.agents.get(agentId);
        if (existing) {
          existing.status = newStatus as AgentState['status'];
          existing.current_task_id = data.current_task_id as string | undefined;
        }
        this.events.onAgentStatusChanged?.(agentId, oldStatus, newStatus);
        break;
      }

      case 'task_completed': {
        const taskId = data.task_id as string;
        const agentId = (data.agent_id as string) || 'unknown';
        this.events.onTaskCompleted?.(taskId, agentId);
        break;
      }

      case 'task_failed': {
        const taskId = data.task_id as string;
        const error = (data.error as string) || 'Unknown error';
        this.events.onTaskFailed?.(taskId, error);
        break;
      }

      case 'deliberation_started': {
        const taskId = data.task_id as string;
        const question = (data.question as string) || '';
        this.events.onDeliberationStarted?.(taskId, question);
        break;
      }

      case 'deliberation_consensus': {
        const taskId = data.task_id as string;
        const answer = (data.answer as string) || '';
        this.events.onDeliberationConsensus?.(taskId, answer);
        break;
      }

      case 'scheduler_stats': {
        this.schedulerStats = data as unknown as SchedulerStats;
        this.events.onSchedulerStats?.(this.schedulerStats);
        break;
      }

      case 'error':
        this.events.onError?.((data.error as string) || 'Unknown error');
        break;
    }
  }
}

/**
 * Create and manage a stream client integrated with VS Code.
 */
export function createStreamClient(
  context: vscode.ExtensionContext,
  apiUrl: string,
  callbacks: StreamClientEvents = {}
): ControlPlaneStreamClient {
  // Convert HTTP URL to WebSocket URL
  const wsUrl = apiUrl
    .replace('https://', 'wss://')
    .replace('http://', 'ws://')
    .replace(/\/api\/?$/, '/api/control-plane/stream');

  const client = new ControlPlaneStreamClient(
    {
      wsUrl,
      autoReconnect: true,
      reconnectDelay: 3000,
      maxReconnectAttempts: 10,
    },
    {
      ...callbacks,
      onConnected: () => {
        callbacks.onConnected?.();
      },
      onDisconnected: (reason) => {
        callbacks.onDisconnected?.(reason);
      },
      onError: (error) => {
        console.error('[Aragora Stream]', error);
        callbacks.onError?.(error);
      },
    }
  );

  // Cleanup on deactivation
  context.subscriptions.push({
    dispose: () => client.disconnect(),
  });

  return client;
}

export default ControlPlaneStreamClient;
