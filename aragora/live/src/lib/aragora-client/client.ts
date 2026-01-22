/**
 * Aragora SDK Client (Modular)
 *
 * Unified client that provides access to all API modules.
 * This is the recommended entry point for the SDK.
 *
 * Usage:
 * ```typescript
 * import { createClient } from '@/lib/aragora-client/client';
 *
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'xxx' });
 *
 * // Access API modules
 * const debates = await client.debates.list();
 * const agents = await client.agents.list();
 * const health = await client.health();
 *
 * // WebSocket streaming
 * await client.ws.connect();
 * client.ws.subscribe('debate-123');
 * client.ws.on('agent_message', (event) => console.log(event));
 * ```
 */

import { HttpClient, AragoraClientConfig, AragoraError } from './apis/base';
import { DebatesAPI } from './apis/debates';
import { AgentsAPI } from './apis/agents';
import { AnalyticsAPI } from './apis/analytics';
import { WorkflowsAPI } from './apis/workflows';
import { AragoraWebSocket, createWebSocket, WebSocketOptions } from './apis/websocket';

// =============================================================================
// Client Configuration
// =============================================================================

export interface ClientConfig extends AragoraClientConfig {
  /** WebSocket URL (defaults to baseUrl with ws:// protocol) */
  wsUrl?: string;
}

// =============================================================================
// Unified Client
// =============================================================================

export class AragoraClient {
  private http: HttpClient;
  private _ws: AragoraWebSocket | null = null;
  private wsOptions: WebSocketOptions;

  // API Modules
  readonly debates: DebatesAPI;
  readonly agents: AgentsAPI;
  readonly analytics: AnalyticsAPI;
  readonly workflows: WorkflowsAPI;

  constructor(config: ClientConfig) {
    this.http = new HttpClient(config);

    // Initialize API modules
    this.debates = new DebatesAPI(this.http);
    this.agents = new AgentsAPI(this.http);
    this.analytics = new AnalyticsAPI(this.http);
    this.workflows = new WorkflowsAPI(this.http);

    // Store WebSocket options for lazy initialization
    this.wsOptions = {
      wsUrl: config.wsUrl || config.baseUrl.replace(/^http/, 'ws') + '/ws',
      apiKey: config.apiKey,
    };
  }

  /**
   * Get WebSocket client (lazy initialized)
   */
  get ws(): AragoraWebSocket {
    if (!this._ws) {
      this._ws = createWebSocket(this.wsOptions);
    }
    return this._ws;
  }

  // ==========================================================================
  // Health & System
  // ==========================================================================

  /**
   * Health check
   */
  async health(): Promise<{ status: string; version?: string }> {
    return this.http.get('/health');
  }

  /**
   * Readiness check
   */
  async ready(): Promise<{ ready: boolean; checks: Record<string, boolean> }> {
    return this.http.get('/health/ready');
  }

  /**
   * Get API version
   */
  async version(): Promise<{ version: string; build?: string }> {
    return this.http.get('/api/version');
  }

  // ==========================================================================
  // User & Auth (convenience methods)
  // ==========================================================================

  /**
   * Get current user profile
   */
  async me(): Promise<unknown> {
    return this.http.get('/api/users/me');
  }

  /**
   * Update current user profile
   */
  async updateProfile(updates: Record<string, unknown>): Promise<unknown> {
    return this.http.patch('/api/users/me', updates);
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  /**
   * Disconnect WebSocket and cleanup resources
   */
  disconnect(): void {
    this._ws?.disconnect();
    this._ws = null;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a new Aragora client instance
 */
export function createClient(config: ClientConfig): AragoraClient {
  return new AragoraClient(config);
}

// Singleton instance for frontend usage
let _clientInstance: AragoraClient | null = null;

/**
 * Get or create a singleton client instance
 * Primarily for frontend usage with React context
 */
export function getClient(apiKey?: string, baseUrl?: string): AragoraClient {
  if (!_clientInstance || apiKey) {
    _clientInstance = createClient({
      baseUrl: baseUrl || process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai',
      apiKey,
    });
  }
  return _clientInstance;
}

/**
 * Clear the singleton client instance
 */
export function clearClient(): void {
  _clientInstance?.disconnect();
  _clientInstance = null;
}

// =============================================================================
// Re-exports
// =============================================================================

export { AragoraError };

// Re-export types from modules
export type {
  Debate,
  DebateMessage,
  DebateRound,
  DebateCreateRequest,
  DebateCreateResponse,
  ConsensusResult,
} from './apis/debates';

export type {
  AgentProfile,
  LeaderboardEntry,
} from './apis/agents';

export type {
  Workflow,
  WorkflowTemplate,
  WorkflowExecution,
} from './apis/workflows';

export type {
  WebSocketState,
  DebateEvent,
} from './apis/websocket';
