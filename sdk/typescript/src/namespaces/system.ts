/**
 * System Namespace API
 *
 * Provides a namespaced interface for system administration and monitoring.
 * Includes database maintenance, history tracking, and circuit breaker management.
 */

/**
 * Maintenance task types.
 */
export type MaintenanceTask = 'status' | 'vacuum' | 'analyze' | 'checkpoint' | 'full';

/**
 * Circuit breaker status.
 */
export type CircuitBreakerStatus = 'closed' | 'open' | 'half_open';

/**
 * Cycle history entry.
 */
export interface CycleEntry {
  id: string;
  loop_id: string;
  cycle_number: number;
  started_at: string;
  completed_at?: string;
  status: string;
  debates_count: number;
  result?: unknown;
}

/**
 * Event history entry.
 */
export interface EventEntry {
  id: string;
  loop_id: string;
  event_type: string;
  timestamp: string;
  details?: unknown;
}

/**
 * Debate history entry.
 */
export interface DebateHistoryEntry {
  id: string;
  loop_id: string;
  task: string;
  status: string;
  created_at: string;
  completed_at?: string;
  consensus?: string;
  agents: string[];
}

/**
 * History summary statistics.
 */
export interface HistorySummary {
  total_debates: number;
  total_agents: number;
  total_matches: number;
}

/**
 * Circuit breaker metrics.
 */
export interface CircuitBreakerMetrics {
  status: CircuitBreakerStatus;
  failures: number;
  success_rate: number;
  last_failure?: string;
  last_success?: string;
  open_since?: string;
}

/**
 * Authentication statistics.
 */
export interface AuthStats {
  total_users: number;
  active_sessions: number;
  failed_attempts: number;
  tokens_issued: number;
  tokens_revoked: number;
}

/**
 * Maintenance result.
 */
export interface MaintenanceResult {
  task: MaintenanceTask;
  success: boolean;
  message: string;
  duration_ms?: number;
  details?: Record<string, unknown>;
}

/**
 * Debug test response.
 */
export interface DebugTestResponse {
  status: string;
  method: string;
  message: string;
}

/**
 * Response for listing cycles.
 */
export interface CyclesResponse {
  cycles: CycleEntry[];
}

/**
 * Response for listing events.
 */
export interface EventsResponse {
  events: EventEntry[];
}

/**
 * Response for listing debate history.
 */
export interface DebateHistoryResponse {
  debates: DebateHistoryEntry[];
}

/**
 * Response for circuit breakers.
 */
export type CircuitBreakersResponse = Record<string, CircuitBreakerMetrics>;

/**
 * Response for token revocation.
 */
export interface RevokeTokenResponse {
  success: boolean;
  message: string;
}

/**
 * History query options.
 */
export interface HistoryOptions {
  loop_id?: string;
  limit?: number;
}

/**
 * Interface for the internal client methods used by SystemAPI.
 */
interface SystemClientInterface {
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown>; body?: unknown }): Promise<T>;
}

/**
 * System API namespace.
 *
 * Provides methods for system administration:
 * - Database maintenance tasks
 * - History tracking (cycles, events, debates)
 * - Circuit breaker monitoring
 * - Authentication statistics
 * - Debug endpoints
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Run database maintenance
 * const result = await client.system.runMaintenance('vacuum');
 *
 * // Get circuit breaker status
 * const breakers = await client.system.getCircuitBreakers();
 *
 * // Get debate history
 * const { debates } = await client.system.getDebateHistory({ limit: 50 });
 *
 * // Get auth statistics
 * const stats = await client.system.getAuthStats();
 * ```
 */
export class SystemAPI {
  constructor(private client: SystemClientInterface) {}

  /**
   * Run a debug test endpoint.
   * Useful for verifying API connectivity.
   */
  async debugTest(): Promise<DebugTestResponse> {
    return this.client.request<DebugTestResponse>('GET', '/api/debug/test');
  }

  /**
   * Get cycle history.
   * @param options - Filter by loop_id and limit results
   */
  async getCycles(options?: HistoryOptions): Promise<CyclesResponse> {
    return this.client.request<CyclesResponse>('GET', '/api/history/cycles', {
      params: {
        loop_id: options?.loop_id,
        limit: options?.limit ?? 50,
      },
    });
  }

  /**
   * Get event history.
   * @param options - Filter by loop_id and limit results
   */
  async getEvents(options?: HistoryOptions): Promise<EventsResponse> {
    return this.client.request<EventsResponse>('GET', '/api/history/events', {
      params: {
        loop_id: options?.loop_id,
        limit: options?.limit ?? 100,
      },
    });
  }

  /**
   * Get debate history.
   * @param options - Filter by loop_id and limit results
   */
  async getDebateHistory(options?: HistoryOptions): Promise<DebateHistoryResponse> {
    return this.client.request<DebateHistoryResponse>('GET', '/api/history/debates', {
      params: {
        loop_id: options?.loop_id,
        limit: options?.limit ?? 50,
      },
    });
  }

  /**
   * Get history summary statistics.
   * @param loop_id - Optional filter by loop ID
   */
  async getHistorySummary(loop_id?: string): Promise<HistorySummary> {
    return this.client.request<HistorySummary>('GET', '/api/history/summary', {
      params: {
        loop_id,
      },
    });
  }

  /**
   * Run database maintenance task.
   * @param task - Type of maintenance to run
   */
  async runMaintenance(task: MaintenanceTask = 'status'): Promise<MaintenanceResult> {
    return this.client.request<MaintenanceResult>('GET', '/api/system/maintenance', {
      params: {
        task,
      },
    });
  }

  /**
   * Get authentication statistics.
   */
  async getAuthStats(): Promise<AuthStats> {
    return this.client.request<AuthStats>('GET', '/api/auth/stats');
  }

  /**
   * Revoke a token.
   * @param tokenData - Token identification data
   */
  async revokeToken(tokenData: { token_id?: string; user_id?: string }): Promise<RevokeTokenResponse> {
    return this.client.request<RevokeTokenResponse>('POST', '/api/auth/revoke', {
      body: tokenData,
    });
  }

  /**
   * Get circuit breaker metrics.
   * Returns status and metrics for all circuit breakers.
   */
  async getCircuitBreakers(): Promise<CircuitBreakersResponse> {
    return this.client.request<CircuitBreakersResponse>('GET', '/api/circuit-breakers');
  }

  /**
   * Get Prometheus metrics.
   * Returns raw Prometheus format metrics.
   */
  async getPrometheusMetrics(): Promise<string> {
    return this.client.request<string>('GET', '/metrics');
  }
}
