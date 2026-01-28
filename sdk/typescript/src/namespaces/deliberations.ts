/**
 * Deliberations Namespace API
 *
 * Provides visibility into active vetted decisionmaking sessions across the system.
 * Deliberations are multi-agent debates that produce consensus decisions.
 */

/**
 * Deliberation status types.
 */
export type DeliberationStatus =
  | 'initializing'
  | 'active'
  | 'consensus_forming'
  | 'complete'
  | 'failed';

/**
 * A single deliberation session.
 */
export interface Deliberation {
  id: string;
  task: string;
  status: DeliberationStatus;
  agents: string[];
  current_round: number;
  total_rounds: number;
  consensus_score: number;
  started_at: string;
  updated_at: string;
  message_count: number;
  votes: Record<string, unknown>;
}

/**
 * Statistics for deliberations.
 */
export interface DeliberationStats {
  active_count: number;
  completed_today: number;
  average_consensus_time: number;
  average_rounds: number;
  top_agents: string[];
  timestamp: string;
}

/**
 * Response for listing active deliberations.
 */
export interface ActiveDeliberationsResponse {
  deliberations: Deliberation[];
  count: number;
  timestamp: string;
}

/**
 * Stream configuration for real-time updates.
 */
export interface DeliberationStreamConfig {
  type: 'websocket';
  path: string;
  events: string[];
}

/**
 * Interface for the internal client methods used by DeliberationsAPI.
 */
interface DeliberationsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Deliberations API namespace.
 *
 * Provides visibility into multi-agent vetted decisionmaking sessions.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get active deliberations
 * const active = await client.deliberations.listActive();
 * console.log(`${active.count} active deliberations`);
 *
 * // Get statistics
 * const stats = await client.deliberations.getStats();
 * console.log(`Completed today: ${stats.completed_today}`);
 *
 * // Get specific deliberation
 * const delib = await client.deliberations.get('delib-123');
 * console.log(`Round ${delib.current_round}/${delib.total_rounds}`);
 * ```
 */
export class DeliberationsAPI {
  constructor(private client: DeliberationsClientInterface) {}

  /**
   * List all active deliberation sessions.
   */
  async listActive(): Promise<ActiveDeliberationsResponse> {
    return this.client.request('GET', '/api/v2/deliberations/active');
  }

  /**
   * Get deliberation statistics.
   */
  async getStats(): Promise<DeliberationStats> {
    return this.client.request('GET', '/api/v2/deliberations/stats');
  }

  /**
   * Get a specific deliberation by ID.
   */
  async get(deliberationId: string): Promise<Deliberation> {
    return this.client.request('GET', `/api/v2/deliberations/${deliberationId}`);
  }

  /**
   * Get WebSocket stream configuration for real-time updates.
   */
  async getStreamConfig(): Promise<DeliberationStreamConfig> {
    return this.client.request('GET', '/api/v2/deliberations/stream/config');
  }
}
