/**
 * Control Plane Namespace API
 *
 * Provides a namespaced interface for control plane operations including
 * agent management, task scheduling, policy governance, and health monitoring.
 */

/**
 * Interface for the internal client methods used by ControlPlaneAPI.
 */
interface ControlPlaneClientInterface {
  // Agent Management
  registerAgent(body: {
    agent_id: string;
    name?: string;
    capabilities?: string[];
    metadata?: Record<string, unknown>;
  }): Promise<{ registered: boolean; agent_id: string }>;

  unregisterAgent(agentId: string): Promise<{ unregistered: boolean }>;

  sendHeartbeat(body: {
    agent_id: string;
    status?: 'idle' | 'busy' | 'offline' | 'draining';
    current_task?: string;
    metrics?: Record<string, number>;
  }): Promise<{ acknowledged: boolean }>;

  getAgentStatus(agentId: string): Promise<{
    agent_id: string;
    status: string;
    last_heartbeat?: string;
    current_task?: string;
  }>;

  listRegisteredAgents(params?: {
    status?: string;
    capability?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    agents: Array<{
      agent_id: string;
      name?: string;
      status: string;
      capabilities?: string[];
      last_heartbeat?: string;
    }>;
  }>;

  // Task Management
  submitTask(body: {
    task_type: string;
    payload: Record<string, unknown>;
    priority?: 'low' | 'normal' | 'high' | 'critical';
    agent_hint?: string;
    timeout_seconds?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ task_id: string; status: string }>;

  getTaskStatus(taskId: string): Promise<{
    task_id: string;
    status: string;
    assigned_agent?: string;
    result?: unknown;
    error?: string;
    submitted_at: string;
    completed_at?: string;
  }>;

  listTasks(params?: {
    status?: string;
    task_type?: string;
    agent_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    tasks: Array<{
      task_id: string;
      task_type: string;
      status: string;
      priority: string;
      assigned_agent?: string;
      submitted_at: string;
    }>;
  }>;

  claimTask(body: {
    agent_id: string;
    task_type?: string;
    capabilities?: string[];
  }): Promise<{ task_id: string; payload: Record<string, unknown> } | null>;

  completeTask(
    taskId: string,
    body: {
      result: unknown;
      metrics?: Record<string, number>;
    }
  ): Promise<{ completed: boolean }>;

  failTask(
    taskId: string,
    body: {
      error: string;
      retry?: boolean;
    }
  ): Promise<{ failed: boolean }>;

  cancelTask(taskId: string): Promise<{ cancelled: boolean }>;

  // Health Monitoring
  getControlPlaneHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    agents_total: number;
    agents_active: number;
    tasks_pending: number;
    tasks_running: number;
  }>;

  // Policy Management
  createPolicy(body: {
    name: string;
    description?: string;
    rules: Array<{
      condition: string;
      action: 'allow' | 'deny' | 'require_approval';
      priority?: number;
    }>;
    enabled?: boolean;
    scope?: 'global' | 'tenant' | 'user';
  }): Promise<{ policy_id: string; created: boolean }>;

  getPolicy(policyId: string): Promise<{
    policy_id: string;
    name: string;
    description?: string;
    rules: Array<{
      condition: string;
      action: string;
      priority: number;
    }>;
    enabled: boolean;
    scope: string;
    created_at: string;
    updated_at: string;
  }>;

  updatePolicy(
    policyId: string,
    body: {
      name?: string;
      description?: string;
      rules?: Array<{
        condition: string;
        action: 'allow' | 'deny' | 'require_approval';
        priority?: number;
      }>;
      enabled?: boolean;
    }
  ): Promise<{ updated: boolean }>;

  deletePolicy(policyId: string): Promise<{ deleted: boolean }>;

  listPolicies(params?: {
    scope?: string;
    enabled?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<{
    policies: Array<{
      policy_id: string;
      name: string;
      enabled: boolean;
      scope: string;
      rules_count: number;
    }>;
  }>;

  // Scheduling
  scheduleTask(body: {
    task_type: string;
    payload: Record<string, unknown>;
    schedule_at?: string;
    cron?: string;
    priority?: 'low' | 'normal' | 'high' | 'critical';
    max_retries?: number;
    timeout_seconds?: number;
  }): Promise<{ schedule_id: string; next_run_at: string }>;

  getScheduledTask(scheduleId: string): Promise<{
    schedule_id: string;
    task_type: string;
    status: 'active' | 'paused' | 'completed' | 'failed';
    schedule_at?: string;
    cron?: string;
    next_run_at?: string;
    last_run_at?: string;
    run_count: number;
  }>;

  listScheduledTasks(params?: {
    status?: string;
    task_type?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    schedules: Array<{
      schedule_id: string;
      task_type: string;
      status: string;
      next_run_at?: string;
      cron?: string;
    }>;
  }>;

  cancelScheduledTask(scheduleId: string): Promise<{ cancelled: boolean }>;
}

/**
 * Control Plane API namespace.
 *
 * Provides enterprise-grade orchestration capabilities:
 * - Agent lifecycle management (registration, heartbeats, status)
 * - Task scheduling and distribution
 * - Policy governance and access control
 * - Health monitoring and metrics
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Register an agent
 * await client.controlPlane.agents.register({
 *   agent_id: 'worker-1',
 *   name: 'Analysis Worker',
 *   capabilities: ['code-review', 'security-scan']
 * });
 *
 * // Submit a task
 * const { task_id } = await client.controlPlane.tasks.submit({
 *   task_type: 'code-review',
 *   payload: { repo: 'my-repo', pr: 123 },
 *   priority: 'high'
 * });
 *
 * // Check health
 * const health = await client.controlPlane.getHealth();
 * ```
 */
export class ControlPlaneAPI {
  public readonly agents: AgentsSubAPI;
  public readonly tasks: TasksSubAPI;
  public readonly policies: PoliciesSubAPI;
  public readonly schedules: SchedulesSubAPI;

  constructor(private client: ControlPlaneClientInterface) {
    this.agents = new AgentsSubAPI(client);
    this.tasks = new TasksSubAPI(client);
    this.policies = new PoliciesSubAPI(client);
    this.schedules = new SchedulesSubAPI(client);
  }

  /**
   * Get control plane health status.
   */
  async getHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    agents_total: number;
    agents_active: number;
    tasks_pending: number;
    tasks_running: number;
  }> {
    return this.client.getControlPlaneHealth();
  }
}

/**
 * Agent management sub-API.
 */
class AgentsSubAPI {
  constructor(private client: ControlPlaneClientInterface) {}

  /**
   * Register an agent with the control plane.
   */
  async register(body: {
    agent_id: string;
    name?: string;
    capabilities?: string[];
    metadata?: Record<string, unknown>;
  }): Promise<{ registered: boolean; agent_id: string }> {
    return this.client.registerAgent(body);
  }

  /**
   * Unregister an agent from the control plane.
   */
  async unregister(agentId: string): Promise<{ unregistered: boolean }> {
    return this.client.unregisterAgent(agentId);
  }

  /**
   * Send a heartbeat for an agent.
   */
  async heartbeat(body: {
    agent_id: string;
    status?: 'idle' | 'busy' | 'offline' | 'draining';
    current_task?: string;
    metrics?: Record<string, number>;
  }): Promise<{ acknowledged: boolean }> {
    return this.client.sendHeartbeat(body);
  }

  /**
   * Get an agent's status.
   */
  async getStatus(agentId: string): Promise<{
    agent_id: string;
    status: string;
    last_heartbeat?: string;
    current_task?: string;
  }> {
    return this.client.getAgentStatus(agentId);
  }

  /**
   * List registered agents.
   */
  async list(params?: {
    status?: string;
    capability?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    agents: Array<{
      agent_id: string;
      name?: string;
      status: string;
      capabilities?: string[];
      last_heartbeat?: string;
    }>;
  }> {
    return this.client.listRegisteredAgents(params);
  }
}

/**
 * Task management sub-API.
 */
class TasksSubAPI {
  constructor(private client: ControlPlaneClientInterface) {}

  /**
   * Submit a task for execution.
   */
  async submit(body: {
    task_type: string;
    payload: Record<string, unknown>;
    priority?: 'low' | 'normal' | 'high' | 'critical';
    agent_hint?: string;
    timeout_seconds?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ task_id: string; status: string }> {
    return this.client.submitTask(body);
  }

  /**
   * Get task status.
   */
  async getStatus(taskId: string): Promise<{
    task_id: string;
    status: string;
    assigned_agent?: string;
    result?: unknown;
    error?: string;
    submitted_at: string;
    completed_at?: string;
  }> {
    return this.client.getTaskStatus(taskId);
  }

  /**
   * List tasks.
   */
  async list(params?: {
    status?: string;
    task_type?: string;
    agent_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    tasks: Array<{
      task_id: string;
      task_type: string;
      status: string;
      priority: string;
      assigned_agent?: string;
      submitted_at: string;
    }>;
  }> {
    return this.client.listTasks(params);
  }

  /**
   * Claim a pending task for an agent.
   */
  async claim(body: {
    agent_id: string;
    task_type?: string;
    capabilities?: string[];
  }): Promise<{ task_id: string; payload: Record<string, unknown> } | null> {
    return this.client.claimTask(body);
  }

  /**
   * Complete a task with a result.
   */
  async complete(
    taskId: string,
    body: {
      result: unknown;
      metrics?: Record<string, number>;
    }
  ): Promise<{ completed: boolean }> {
    return this.client.completeTask(taskId, body);
  }

  /**
   * Fail a task with an error.
   */
  async fail(
    taskId: string,
    body: {
      error: string;
      retry?: boolean;
    }
  ): Promise<{ failed: boolean }> {
    return this.client.failTask(taskId, body);
  }

  /**
   * Cancel a task.
   */
  async cancel(taskId: string): Promise<{ cancelled: boolean }> {
    return this.client.cancelTask(taskId);
  }
}

/**
 * Policy management sub-API.
 */
class PoliciesSubAPI {
  constructor(private client: ControlPlaneClientInterface) {}

  /**
   * Create a new policy.
   */
  async create(body: {
    name: string;
    description?: string;
    rules: Array<{
      condition: string;
      action: 'allow' | 'deny' | 'require_approval';
      priority?: number;
    }>;
    enabled?: boolean;
    scope?: 'global' | 'tenant' | 'user';
  }): Promise<{ policy_id: string; created: boolean }> {
    return this.client.createPolicy(body);
  }

  /**
   * Get a policy by ID.
   */
  async get(policyId: string): Promise<{
    policy_id: string;
    name: string;
    description?: string;
    rules: Array<{
      condition: string;
      action: string;
      priority: number;
    }>;
    enabled: boolean;
    scope: string;
    created_at: string;
    updated_at: string;
  }> {
    return this.client.getPolicy(policyId);
  }

  /**
   * Update a policy.
   */
  async update(
    policyId: string,
    body: {
      name?: string;
      description?: string;
      rules?: Array<{
        condition: string;
        action: 'allow' | 'deny' | 'require_approval';
        priority?: number;
      }>;
      enabled?: boolean;
    }
  ): Promise<{ updated: boolean }> {
    return this.client.updatePolicy(policyId, body);
  }

  /**
   * Delete a policy.
   */
  async delete(policyId: string): Promise<{ deleted: boolean }> {
    return this.client.deletePolicy(policyId);
  }

  /**
   * List policies.
   */
  async list(params?: {
    scope?: string;
    enabled?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<{
    policies: Array<{
      policy_id: string;
      name: string;
      enabled: boolean;
      scope: string;
      rules_count: number;
    }>;
  }> {
    return this.client.listPolicies(params);
  }
}

/**
 * Schedule management sub-API.
 */
class SchedulesSubAPI {
  constructor(private client: ControlPlaneClientInterface) {}

  /**
   * Schedule a task for future execution.
   */
  async create(body: {
    task_type: string;
    payload: Record<string, unknown>;
    schedule_at?: string;
    cron?: string;
    priority?: 'low' | 'normal' | 'high' | 'critical';
    max_retries?: number;
    timeout_seconds?: number;
  }): Promise<{ schedule_id: string; next_run_at: string }> {
    return this.client.scheduleTask(body);
  }

  /**
   * Get a scheduled task.
   */
  async get(scheduleId: string): Promise<{
    schedule_id: string;
    task_type: string;
    status: 'active' | 'paused' | 'completed' | 'failed';
    schedule_at?: string;
    cron?: string;
    next_run_at?: string;
    last_run_at?: string;
    run_count: number;
  }> {
    return this.client.getScheduledTask(scheduleId);
  }

  /**
   * List scheduled tasks.
   */
  async list(params?: {
    status?: string;
    task_type?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    schedules: Array<{
      schedule_id: string;
      task_type: string;
      status: string;
      next_run_at?: string;
      cron?: string;
    }>;
  }> {
    return this.client.listScheduledTasks(params);
  }

  /**
   * Cancel a scheduled task.
   */
  async cancel(scheduleId: string): Promise<{ cancelled: boolean }> {
    return this.client.cancelScheduledTask(scheduleId);
  }
}
