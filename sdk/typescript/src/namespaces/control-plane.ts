/**
 * Control Plane Namespace API
 *
 * Provides a namespaced interface for control plane operations including
 * agent management, task scheduling, policy governance, and health monitoring.
 */

import type { AragoraClient } from '../client';

/**
 * Control Plane API namespace.
 *
 * Provides enterprise-grade orchestration capabilities:
 * - Agent lifecycle management (registration, heartbeats, status)
 * - Task scheduling and distribution
 * - Health monitoring and metrics
 * - Policy violations and deliberations
 * - Audit logs and notifications
 */
export class ControlPlaneAPI {
  constructor(private client: AragoraClient) {}

  // ===========================================================================
  // Agents
  // ===========================================================================

  /**
   * List registered agents.
   * @route GET /api/control-plane/agents
   */
  async listAgents(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/agents') as Promise<Record<string, unknown>>;
  }

  /**
   * Register an agent.
   * @route POST /api/control-plane/agents
   */
  async registerAgent(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/control-plane/agents', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get an agent by ID.
   * @route GET /api/control-plane/agents/{agent_id}
   */
  async getAgent(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/agents/${encodeURIComponent(agentId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Deregister an agent.
   * @route DELETE /api/control-plane/agents/{agent_id}
   */
  async deregisterAgent(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'DELETE',
      `/api/control-plane/agents/${encodeURIComponent(agentId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Send agent heartbeat.
   * @route POST /api/control-plane/agents/{agent_id}/heartbeat
   */
  async heartbeat(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/control-plane/agents/${encodeURIComponent(agentId)}/heartbeat`
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Health
  // ===========================================================================

  /**
   * Get control plane health.
   * @route GET /api/control-plane/health
   */
  async getHealth(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/health') as Promise<Record<string, unknown>>;
  }

  /**
   * Get detailed health information.
   * @route GET /api/control-plane/health/detailed
   */
  async getHealthDetailed(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/health/detailed') as Promise<Record<string, unknown>>;
  }

  /**
   * Get health of a specific agent.
   * @route GET /api/control-plane/health/{agent_id}
   */
  async getAgentHealth(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/health/${encodeURIComponent(agentId)}`
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Tasks
  // ===========================================================================

  /**
   * Create a task.
   * @route POST /api/control-plane/tasks
   */
  async createTask(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/control-plane/tasks', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Claim a task.
   * @route POST /api/control-plane/tasks/claim
   */
  async claimTask(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/control-plane/tasks/claim', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get task history.
   * @route GET /api/control-plane/tasks/history
   */
  async getTaskHistory(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/tasks/history') as Promise<Record<string, unknown>>;
  }

  /**
   * Get a task by ID.
   * @route GET /api/control-plane/tasks/{task_id}
   */
  async getTask(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/tasks/${encodeURIComponent(taskId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Cancel a task.
   * @route POST /api/control-plane/tasks/{task_id}/cancel
   */
  async cancelTask(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/control-plane/tasks/${encodeURIComponent(taskId)}/cancel`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Complete a task.
   * @route POST /api/control-plane/tasks/{task_id}/complete
   */
  async completeTask(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/control-plane/tasks/${encodeURIComponent(taskId)}/complete`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Mark a task as failed.
   * @route POST /api/control-plane/tasks/{task_id}/fail
   */
  async failTask(taskId: string, body?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/control-plane/tasks/${encodeURIComponent(taskId)}/fail`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Metrics & Stats
  // ===========================================================================

  /**
   * Get control plane metrics.
   * @route GET /api/control-plane/metrics
   */
  async getMetrics(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/metrics') as Promise<Record<string, unknown>>;
  }

  /**
   * Get control plane statistics.
   * @route GET /api/control-plane/stats
   */
  async getStats(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/stats') as Promise<Record<string, unknown>>;
  }

  /**
   * Get circuit breaker status.
   * @route GET /api/control-plane/breakers
   */
  async getBreakers(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/breakers') as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Queue
  // ===========================================================================

  /**
   * Get task queue.
   * @route GET /api/control-plane/queue
   */
  async getQueue(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/queue') as Promise<Record<string, unknown>>;
  }

  /**
   * Get queue metrics.
   * @route GET /api/control-plane/queue/metrics
   */
  async getQueueMetrics(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/queue/metrics') as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Audit
  // ===========================================================================

  /**
   * Get control plane audit log.
   * @route GET /api/control-plane/audit
   */
  async getAudit(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/audit') as Promise<Record<string, unknown>>;
  }

  /**
   * Get audit statistics.
   * @route GET /api/control-plane/audit/stats
   */
  async getAuditStats(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/audit/stats') as Promise<Record<string, unknown>>;
  }

  /**
   * Verify audit integrity.
   * @route GET /api/control-plane/audit/verify
   */
  async verifyAudit(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/audit/verify') as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Policy Violations
  // ===========================================================================

  /**
   * List policy violations.
   * @route GET /api/control-plane/policies/violations
   */
  async listViolations(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/policies/violations') as Promise<Record<string, unknown>>;
  }

  /**
   * Get policy violations statistics.
   * @route GET /api/control-plane/policies/violations/stats
   */
  async getViolationsStats(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/policies/violations/stats') as Promise<Record<string, unknown>>;
  }

  /**
   * Get a policy violation by ID.
   * @route GET /api/control-plane/policies/violations/{violation_id}
   */
  async getViolation(violationId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/policies/violations/${encodeURIComponent(violationId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Update a policy violation.
   * @route PATCH /api/control-plane/policies/violations/{violation_id}
   */
  async updateViolation(violationId: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'PATCH',
      `/api/control-plane/policies/violations/${encodeURIComponent(violationId)}`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Deliberations
  // ===========================================================================

  /**
   * Create a deliberation.
   * @route POST /api/control-plane/deliberations
   */
  async createDeliberation(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/control-plane/deliberations', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a deliberation.
   * @route GET /api/control-plane/deliberations/{request_id}
   */
  async getDeliberation(requestId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/deliberations/${encodeURIComponent(requestId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get deliberation status.
   * @route GET /api/control-plane/deliberations/{request_id}/status
   */
  async getDeliberationStatus(requestId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/deliberations/${encodeURIComponent(requestId)}/status`
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Notifications
  // ===========================================================================

  /**
   * List control plane notifications.
   * @route GET /api/control-plane/notifications
   */
  async listNotifications(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/notifications') as Promise<Record<string, unknown>>;
  }

  /**
   * Get notification statistics.
   * @route GET /api/control-plane/notifications/stats
   */
  async getNotificationStats(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/notifications/stats') as Promise<Record<string, unknown>>;
  }
}
