/**
 * Agent Dashboard Namespace API
 *
 * Provides monitoring and management of agents via the dashboard:
 * - List agents with metrics
 * - Pause and resume agents
 * - Dashboard health and queue status
 * - Real-time event stream info
 */

import type { AragoraClient } from '../client';

/**
 * Agent Dashboard API namespace.
 */
export class AgentDashboardAPI {
  constructor(private client: AragoraClient) {}

  // =========================================================================
  // Agents
  // =========================================================================

  /**
   * List all agents visible in the dashboard.
   * @route GET /api/agent-dashboard/agents
   */
  async listAgents(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/agent-dashboard/agents') as Promise<Record<string, unknown>>;
  }

  /**
   * Get detailed metrics for a specific agent.
   * @route GET /api/agent-dashboard/agents/{agentId}/metrics
   */
  async getAgentMetrics(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/agent-dashboard/agents/${encodeURIComponent(agentId)}/metrics`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Pause an agent so it stops accepting new tasks.
   * @route POST /api/agent-dashboard/agents/{agentId}/pause
   */
  async pauseAgent(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/agent-dashboard/agents/${encodeURIComponent(agentId)}/pause`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Resume a paused agent.
   * @route POST /api/agent-dashboard/agents/{agentId}/resume
   */
  async resumeAgent(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/agent-dashboard/agents/${encodeURIComponent(agentId)}/resume`
    ) as Promise<Record<string, unknown>>;
  }

  // =========================================================================
  // Health & Metrics
  // =========================================================================

  /**
   * Get overall agent dashboard health status.
   * @route GET /api/agent-dashboard/health
   */
  async getHealth(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/agent-dashboard/health') as Promise<Record<string, unknown>>;
  }

  /**
   * Get aggregate dashboard metrics.
   * @route GET /api/agent-dashboard/metrics
   */
  async getMetrics(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/agent-dashboard/metrics') as Promise<Record<string, unknown>>;
  }

  // =========================================================================
  // Queue
  // =========================================================================

  /**
   * Get the current task queue state.
   * @route GET /api/agent-dashboard/queue
   */
  async getQueue(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/agent-dashboard/queue') as Promise<Record<string, unknown>>;
  }

  /**
   * Reprioritize tasks in the queue.
   * @route POST /api/agent-dashboard/queue/prioritize
   */
  async prioritizeQueue(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/agent-dashboard/queue/prioritize', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  // =========================================================================
  // Stream
  // =========================================================================

  /**
   * Get SSE/WebSocket stream connection info for real-time agent events.
   * @route GET /api/agent-dashboard/stream
   */
  async getStreamInfo(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/agent-dashboard/stream') as Promise<Record<string, unknown>>;
  }
}
