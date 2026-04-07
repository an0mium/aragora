/**
 * Control Plane Namespace API
 *
 * Provides a namespaced interface for control plane operations including
 * agent management, task scheduling, policy governance, and health monitoring.
 */

interface ControlPlaneClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
  registerAgent(body: Record<string, unknown>): Promise<Record<string, unknown>>;
  unregisterAgent(agentId: string): Promise<Record<string, unknown>>;
  sendHeartbeat(body: Record<string, unknown>): Promise<Record<string, unknown>>;
  getAgentStatus(agentId: string): Promise<Record<string, unknown>>;
  listRegisteredAgents(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  submitTask(body: Record<string, unknown>): Promise<Record<string, unknown>>;
  getTaskStatus(taskId: string): Promise<Record<string, unknown>>;
  listTasks(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  claimTask(body: Record<string, unknown>): Promise<Record<string, unknown> | null>;
  completeTask(taskId: string, body: Record<string, unknown>): Promise<Record<string, unknown>>;
  failTask(taskId: string, body: Record<string, unknown>): Promise<Record<string, unknown>>;
  cancelTask(taskId: string): Promise<Record<string, unknown>>;
  getControlPlaneHealth(): Promise<Record<string, unknown>>;
  createPolicy(body: Record<string, unknown>): Promise<Record<string, unknown>>;
  getPolicy(policyId: string): Promise<Record<string, unknown>>;
  updatePolicy(policyId: string, body: Record<string, unknown>): Promise<Record<string, unknown>>;
  deletePolicy(policyId: string): Promise<Record<string, unknown>>;
  listPolicies(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  scheduleTask(body: Record<string, unknown>): Promise<Record<string, unknown>>;
  getScheduledTask(scheduleId: string): Promise<Record<string, unknown>>;
  listScheduledTasks(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  cancelScheduledTask(scheduleId: string): Promise<Record<string, unknown>>;
  createDeliberation(body: Record<string, unknown>): Promise<Record<string, unknown>>;
  getDeliberation(deliberationId: string): Promise<Record<string, unknown>>;
  listDeliberations(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  voteOnDeliberation(deliberationId: string, body: Record<string, unknown>): Promise<Record<string, unknown>>;
  closeDeliberation(deliberationId: string, body?: Record<string, unknown>): Promise<Record<string, unknown>>;
  getDeliberationTranscript(deliberationId: string): Promise<Record<string, unknown>>;
  listAuditLogs(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  getAuditLog(logId: string): Promise<Record<string, unknown>>;
  listPolicyViolations(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  acknowledgePolicyViolation(violationId: string, body?: Record<string, unknown>): Promise<Record<string, unknown>>;
  escalatePolicyViolation(violationId: string, body: Record<string, unknown>): Promise<Record<string, unknown>>;
  getAgentMetrics(agentId: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  getTaskMetrics(params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  getControlPlaneSystemMetrics(): Promise<Record<string, unknown>>;
}

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
  readonly agents: {
    register: (body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    unregister: (agentId: string) => Promise<Record<string, unknown>>;
    heartbeat: (body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    getStatus: (agentId: string) => Promise<Record<string, unknown>>;
    list: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
  };

  readonly tasks: {
    submit: (body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    getStatus: (taskId: string) => Promise<Record<string, unknown>>;
    list: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    claim: (body: Record<string, unknown>) => Promise<Record<string, unknown> | null>;
    complete: (taskId: string, body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    fail: (taskId: string, body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    cancel: (taskId: string) => Promise<Record<string, unknown>>;
  };

  readonly policies: {
    create: (body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    get: (policyId: string) => Promise<Record<string, unknown>>;
    update: (policyId: string, body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    delete: (policyId: string) => Promise<Record<string, unknown>>;
    list: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
  };

  readonly schedules: {
    create: (body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    get: (scheduleId: string) => Promise<Record<string, unknown>>;
    list: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    cancel: (scheduleId: string) => Promise<Record<string, unknown>>;
  };

  readonly deliberations: {
    create: (body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    get: (deliberationId: string) => Promise<Record<string, unknown>>;
    list: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    vote: (deliberationId: string, body: Record<string, unknown>) => Promise<Record<string, unknown>>;
    close: (deliberationId: string, body?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    getTranscript: (deliberationId: string) => Promise<Record<string, unknown>>;
  };

  readonly auditLogs: {
    list: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    get: (logId: string) => Promise<Record<string, unknown>>;
  };

  readonly violations: {
    list: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    acknowledge: (violationId: string, body?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    escalate: (violationId: string, body: Record<string, unknown>) => Promise<Record<string, unknown>>;
  };

  readonly metrics: {
    getAgentMetrics: (agentId: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    getTaskMetrics: (params?: Record<string, unknown>) => Promise<Record<string, unknown>>;
    getSystemMetrics: () => Promise<Record<string, unknown>>;
  };

  constructor(private client: ControlPlaneClientInterface) {
    this.agents = {
      register: (body) => this.client.registerAgent(body),
      unregister: (agentId) => this.client.unregisterAgent(agentId),
      heartbeat: (body) => this.client.sendHeartbeat(body),
      getStatus: (agentId) => this.client.getAgentStatus(agentId),
      list: (params) => this.client.listRegisteredAgents(params),
    };
    this.tasks = {
      submit: (body) => this.client.submitTask(body),
      getStatus: (taskId) => this.client.getTaskStatus(taskId),
      list: (params) => this.client.listTasks(params),
      claim: (body) => this.client.claimTask(body),
      complete: (taskId, body) => this.client.completeTask(taskId, body),
      fail: (taskId, body) => this.client.failTask(taskId, body),
      cancel: (taskId) => this.client.cancelTask(taskId),
    };
    this.policies = {
      create: (body) => this.client.createPolicy(body),
      get: (policyId) => this.client.getPolicy(policyId),
      update: (policyId, body) => this.client.updatePolicy(policyId, body),
      delete: (policyId) => this.client.deletePolicy(policyId),
      list: (params) => this.client.listPolicies(params),
    };
    this.schedules = {
      create: (body) => this.client.scheduleTask(body),
      get: (scheduleId) => this.client.getScheduledTask(scheduleId),
      list: (params) => this.client.listScheduledTasks(params),
      cancel: (scheduleId) => this.client.cancelScheduledTask(scheduleId),
    };
    this.deliberations = {
      create: (body) => this.client.createDeliberation(body),
      get: (deliberationId) => this.client.getDeliberation(deliberationId),
      list: (params) => this.client.listDeliberations(params),
      vote: (deliberationId, body) => this.client.voteOnDeliberation(deliberationId, body),
      close: (deliberationId, body) => this.client.closeDeliberation(deliberationId, body),
      getTranscript: (deliberationId) => this.client.getDeliberationTranscript(deliberationId),
    };
    this.auditLogs = {
      list: (params) => this.client.listAuditLogs(params),
      get: (logId) => this.client.getAuditLog(logId),
    };
    this.violations = {
      list: (params) => this.client.listPolicyViolations(params),
      acknowledge: (violationId, body) => this.client.acknowledgePolicyViolation(violationId, body),
      escalate: (violationId, body) => this.client.escalatePolicyViolation(violationId, body),
    };
    this.metrics = {
      getAgentMetrics: (agentId, params) => this.client.getAgentMetrics(agentId, params),
      getTaskMetrics: (params) => this.client.getTaskMetrics(params),
      getSystemMetrics: () => this.client.getControlPlaneSystemMetrics(),
    };
  }

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

  /**
   * Get detailed metrics for a specific agent.
   * @route GET /api/control-plane/agents/{agent_id}/metrics
   */
  async getAgentMetrics(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/agents/${encodeURIComponent(agentId)}/metrics`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Pause an agent.
   * @route POST /api/control-plane/agents/{agent_id}/pause
   */
  async pauseAgent(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/control-plane/agents/${encodeURIComponent(agentId)}/pause`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Resume a paused agent.
   * @route POST /api/control-plane/agents/{agent_id}/resume
   */
  async resumeAgent(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/control-plane/agents/${encodeURIComponent(agentId)}/resume`
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
    if ('getControlPlaneHealth' in this.client && typeof this.client.getControlPlaneHealth === 'function') {
      return this.client.getControlPlaneHealth();
    }
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
   * Get system-level metrics.
   * @route GET /api/control-plane/metrics/system
   */
  async getSystemMetrics(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/metrics/system') as Promise<Record<string, unknown>>;
  }

  /**
   * Get task-level metrics.
   * @route GET /api/control-plane/metrics/tasks
   */
  async getTaskMetrics(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/metrics/tasks') as Promise<Record<string, unknown>>;
  }

  /**
   * Get metrics for a specific agent by agent ID.
   * @route GET /api/control-plane/metrics/agents/{agent_id}
   */
  async getAgentMetricsById(agentId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/control-plane/metrics/agents/${encodeURIComponent(agentId)}`
    ) as Promise<Record<string, unknown>>;
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

  /**
   * Prioritize queued work.
   * @route POST /api/control-plane/queue/prioritize
   */
  async prioritizeQueue(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/control-plane/queue/prioritize', {
      body,
    }) as Promise<Record<string, unknown>>;
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
   * Get control plane audit logs.
   * @route GET /api/control-plane/audit-logs
   */
  async getAuditLogs(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/audit-logs') as Promise<Record<string, unknown>>;
  }

  /**
   * Get a specific audit log by ID.
   * @route GET /api/control-plane/audit-logs/{log_id}
   */
  async getAuditLog(logId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/control-plane/audit-logs/${encodeURIComponent(logId)}`
    ) as Promise<Record<string, unknown>>;
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

  /**
   * List control plane policies.
   * @route GET /api/control-plane/policies
   */
  async listPolicies(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/policies') as Promise<Record<string, unknown>>;
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

  /**
   * Get deliberation transcript.
   * @route GET /api/control-plane/deliberations/{request_id}/transcript
   */
  async getDeliberationTranscript(requestId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/control-plane/deliberations/${encodeURIComponent(requestId)}/transcript`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * List control plane schedules.
   * @route GET /api/control-plane/schedules
   */
  async listSchedules(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/schedules') as Promise<Record<string, unknown>>;
  }

  /**
   * Get a specific schedule by ID.
   * @route GET /api/control-plane/schedules/{schedule_id}
   */
  async getSchedule(scheduleId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/control-plane/schedules/${encodeURIComponent(scheduleId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Update a control plane task schedule.
   * @route PUT /api/control-plane/schedules/{schedule_id}
   */
  async updateSchedule(scheduleId: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'PUT',
      `/api/v1/control-plane/schedules/${encodeURIComponent(scheduleId)}`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete a control plane task schedule.
   * @route DELETE /api/control-plane/schedules/{schedule_id}
   */
  async deleteSchedule(scheduleId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'DELETE',
      `/api/v1/control-plane/schedules/${encodeURIComponent(scheduleId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get stream connection info for control plane events.
   * @route GET /api/control-plane/stream
   */
  async getStreamInfo(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/stream') as Promise<Record<string, unknown>>;
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
