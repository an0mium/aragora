/**
 * History Namespace API
 *
 * Provides access to historical data including debate history,
 * nomic cycles, and system events.
 */

import type { AragoraClient } from '../client';

/**
 * Historical debate entry
 */
export interface HistoricalDebate {
  id: string;
  question: string;
  started_at: string;
  ended_at?: string;
  status: 'completed' | 'active' | 'cancelled';
  winner?: string;
  consensus_reached: boolean;
  participants: string[];
  domain?: string;
  rounds: number;
}

/**
 * Nomic cycle entry
 */
export interface NomicCycle {
  cycle_id: string;
  phase: string;
  started_at: string;
  ended_at?: string;
  status: 'completed' | 'active' | 'failed';
  improvements_proposed: number;
  improvements_implemented: number;
  files_modified: string[];
  summary?: string;
}

/**
 * System event entry
 */
export interface HistoricalEvent {
  id: string;
  event_type: string;
  timestamp: string;
  actor?: string;
  resource_type?: string;
  resource_id?: string;
  details: Record<string, unknown>;
  severity: 'info' | 'warning' | 'error';
}

/**
 * History summary
 */
export interface HistorySummary {
  total_debates: number;
  debates_this_week: number;
  debates_this_month: number;
  total_nomic_cycles: number;
  active_agents: number;
  most_active_agent: string;
  top_domains: Array<{ domain: string; count: number }>;
  consensus_rate: number;
  average_debate_duration_minutes: number;
}

/**
 * History query options
 */
export interface HistoryQueryOptions {
  limit?: number;
  offset?: number;
  since?: string;
  until?: string;
  status?: string;
  agent?: string;
  domain?: string;
}

/**
 * History namespace for accessing historical data.
 *
 * @example
 * ```typescript
 * // Get recent debates
 * const debates = await client.history.listDebates({ limit: 10 });
 *
 * // Get history summary
 * const summary = await client.history.getSummary();
 * console.log(`Total debates: ${summary.total_debates}`);
 * ```
 */
export class HistoryNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * List historical debates.
   *
   * @param options - Query options for filtering and pagination
   */
  async listDebates(options?: HistoryQueryOptions): Promise<HistoricalDebate[]> {
    const response = await this.client.request<{ debates: HistoricalDebate[] }>(
      'GET',
      '/api/v1/history/debates',
      { params: options as Record<string, unknown> }
    );
    return response.debates;
  }

  /**
   * List nomic improvement cycles.
   *
   * @param options - Query options for filtering and pagination
   */
  async listCycles(options?: HistoryQueryOptions): Promise<NomicCycle[]> {
    const response = await this.client.request<{ cycles: NomicCycle[] }>(
      'GET',
      '/api/v1/history/cycles',
      { params: options as Record<string, unknown> }
    );
    return response.cycles;
  }

  /**
   * List system events.
   *
   * @param options - Query options for filtering and pagination
   */
  async listEvents(options?: HistoryQueryOptions & {
    event_type?: string;
    severity?: 'info' | 'warning' | 'error';
  }): Promise<HistoricalEvent[]> {
    const response = await this.client.request<{ events: HistoricalEvent[] }>(
      'GET',
      '/api/v1/history/events',
      { params: options as Record<string, unknown> }
    );
    return response.events;
  }

  /**
   * Get a summary of historical activity.
   *
   * @param options.period - Time period for summary ('week', 'month', 'year', 'all')
   */
  async getSummary(options?: { period?: 'week' | 'month' | 'year' | 'all' }): Promise<HistorySummary> {
    return this.client.request<HistorySummary>('GET', '/api/v1/history/summary', {
      params: options,
    });
  }

  /**
   * Get history for a specific agent.
   *
   * @param agentName - The agent name
   * @param options - Query options
   */
  async getAgentHistory(
    agentName: string,
    options?: HistoryQueryOptions
  ): Promise<HistoricalDebate[]> {
    const response = await this.client.request<{ debates: HistoricalDebate[] }>(
      'GET',
      `/api/v1/agent/${encodeURIComponent(agentName)}/history`,
      { params: options as Record<string, unknown> }
    );
    return response.debates;
  }
}
