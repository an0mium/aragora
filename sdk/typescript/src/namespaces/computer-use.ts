/**
 * Computer Use Namespace API
 *
 * Provides methods for computer use orchestration:
 * - Task creation, execution, cancellation, and deletion
 * - Action execution, listing, and deletion
 * - Action statistics
 * - Policy CRUD (create, read, update, delete)
 * - Approval workflow (list, get, approve, deny)
 *
 * Endpoints:
 *   POST   /api/v1/computer-use/tasks                       - Create and run task
 *   GET    /api/v1/computer-use/tasks                       - List tasks
 *   GET    /api/v1/computer-use/tasks/{id}                  - Get task status
 *   POST   /api/v1/computer-use/tasks/{id}/cancel           - Cancel task
 *   DELETE /api/v1/computer-use/tasks/{id}                  - Delete task record
 *   GET    /api/v1/computer-use/actions                     - List actions
 *   POST   /api/v1/computer-use/actions                     - Execute an action
 *   GET    /api/v1/computer-use/actions/{id}                - Get action details
 *   DELETE /api/v1/computer-use/actions/{id}                - Delete action record
 *   GET    /api/v1/computer-use/actions/stats               - Get action stats
 *   GET    /api/v1/computer-use/policies                    - List policies
 *   POST   /api/v1/computer-use/policies                    - Create policy
 *   GET    /api/v1/computer-use/policies/{id}               - Get policy details
 *   PUT    /api/v1/computer-use/policies/{id}               - Update policy
 *   DELETE /api/v1/computer-use/policies/{id}               - Delete policy
 *   GET    /api/v1/computer-use/approvals                   - List approvals
 *   GET    /api/v1/computer-use/approvals/{id}              - Get approval details
 *   POST   /api/v1/computer-use/approvals/{id}/approve      - Approve request
 *   POST   /api/v1/computer-use/approvals/{id}/deny         - Deny request
 */

/**
 * Task status values.
 */
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Computer use task information.
 */
export interface ComputerUseTask {
  task_id: string;
  goal: string;
  status: TaskStatus;
  max_steps: number;
  dry_run: boolean;
  steps?: ComputerUseStep[];
  result?: string;
  error?: string;
  created_at: string;
  updated_at?: string;
  completed_at?: string;
}

/**
 * A single step in a computer use task.
 */
export interface ComputerUseStep {
  step_number: number;
  action_type: string;
  description: string;
  status: 'pending' | 'completed' | 'failed';
  screenshot_url?: string;
  error?: string;
}

/**
 * Action statistics by type.
 */
export interface ActionStats {
  click: number;
  type: number;
  screenshot: number;
  scroll: number;
  wait: number;
  total: number;
}

/**
 * Computer use policy.
 */
export interface ComputerUsePolicy {
  policy_id: string;
  name: string;
  description?: string;
  allowed_actions?: string[];
  blocked_domains?: string[];
  created_at: string;
  updated_at?: string;
}

/**
 * Options for creating a computer use task.
 */
export interface CreateTaskOptions {
  /** The goal to accomplish */
  goal: string;
  /** Maximum steps to take (default 10) */
  maxSteps?: number;
  /** If true, simulate without actually performing actions */
  dryRun?: boolean;
}

/**
 * Options for listing tasks.
 */
export interface ListTasksOptions {
  /** Maximum tasks to return (1-100) */
  limit?: number;
  /** Filter by task status */
  status?: TaskStatus;
}

/**
 * Options for creating a policy.
 */
export interface CreatePolicyOptions {
  /** Policy name */
  name: string;
  /** Optional description */
  description?: string;
  /** List of allowed action types */
  allowedActions?: string[];
  /** List of blocked domains */
  blockedDomains?: string[];
}

/**
 * Options for updating a policy.
 */
export interface UpdatePolicyOptions {
  /** Updated policy name */
  name?: string;
  /** Updated description */
  description?: string;
  /** Updated list of allowed action types */
  allowedActions?: string[];
  /** Updated list of blocked domains */
  blockedDomains?: string[];
  /** Maximum number of steps per task */
  maxSteps?: number;
  /** Timeout in seconds for task execution */
  timeoutSeconds?: number;
}

/**
 * Action type values.
 */
export type ActionType = 'click' | 'type' | 'screenshot' | 'scroll' | 'key';

/**
 * Options for executing a single computer use action.
 */
export interface ExecuteActionOptions {
  /** Type of action to perform */
  actionType: ActionType;
  /** Action-specific parameters (e.g., coordinates for click, text for type) */
  parameters?: Record<string, unknown>;
  /** Optional associated task ID */
  taskId?: string;
}

/**
 * Client interface for making HTTP requests.
 */
interface ComputerUseClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Computer Use API namespace.
 *
 * Provides methods for orchestrating computer use tasks with
 * screenshot analysis, clicking, typing, and scrolling.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Create a task
 * const task = await client.computerUse.createTask({
 *   goal: 'Open the settings page and enable dark mode',
 *   maxSteps: 10,
 * });
 *
 * // Check task status
 * const status = await client.computerUse.getTask(task.task_id);
 * console.log(`Status: ${status.task.status}`);
 *
 * // Cancel a running task
 * await client.computerUse.cancelTask(task.task_id);
 *
 * // Get action statistics
 * const stats = await client.computerUse.getActionStats();
 * console.log(`Total actions: ${stats.stats.total}`);
 * ```
 */
export class ComputerUseAPI {
  constructor(private client: ComputerUseClientInterface) {}

  // =========================================================================
  // Tasks
  // =========================================================================

  /**
   * Create and run a computer use task.
   *
   * @param options - Task creation options
   * @returns Task creation result with task_id and status
   */
  async createTask(options: CreateTaskOptions): Promise<{
    task_id: string;
    status: TaskStatus;
    message: string;
  }> {
    return this.client.request('POST', '/api/v1/computer-use/tasks', {
      json: {
        goal: options.goal,
        max_steps: options.maxSteps ?? 10,
        dry_run: options.dryRun ?? false,
      },
    });
  }

  /**
   * List recent tasks.
   *
   * @param options - Filtering and pagination options
   * @returns List of tasks with total count
   */
  async listTasks(options?: ListTasksOptions): Promise<{
    tasks: ComputerUseTask[];
    total: number;
  }> {
    const params: Record<string, unknown> = {};
    if (options?.limit !== undefined) params.limit = options.limit;
    if (options?.status) params.status = options.status;

    return this.client.request('GET', '/api/v1/computer-use/tasks', {
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }

  /**
   * Get task status and details.
   *
   * @param taskId - Task ID
   * @returns Task details including status, steps, and result
   */
  async getTask(taskId: string): Promise<{
    task: ComputerUseTask;
  }> {
    return this.client.request('GET', `/api/v1/computer-use/tasks/${taskId}`);
  }

  /**
   * Cancel a running task.
   *
   * @param taskId - Task ID to cancel
   * @returns Success message
   */
  async cancelTask(taskId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    return this.client.request('POST', `/api/v1/computer-use/tasks/${taskId}/cancel`);
  }

  /**
   * Delete a task record.
   *
   * Only completed, failed, or cancelled tasks can be deleted.
   * Running tasks must be cancelled first.
   *
   * @param taskId - Task ID to delete
   * @returns Deletion confirmation
   */
  async deleteTask(taskId: string): Promise<{
    deleted: boolean;
    task_id: string;
  }> {
    return this.client.request('DELETE', `/api/v1/computer-use/tasks/${taskId}`);
  }

  // =========================================================================
  // Actions
  // =========================================================================

  /**
   * Get action statistics.
   *
   * @returns Statistics by action type (click, type, screenshot, etc.)
   */
  async getActionStats(): Promise<{
    stats: ActionStats;
  }> {
    return this.client.request('GET', '/api/v1/computer-use/actions/stats');
  }

  // =========================================================================
  // Policies
  // =========================================================================

  /**
   * List active policies.
   *
   * @returns List of policies with total count
   */
  async listPolicies(): Promise<{
    policies: ComputerUsePolicy[];
    total: number;
  }> {
    return this.client.request('GET', '/api/v1/computer-use/policies');
  }

  /**
   * Create a computer use policy.
   *
   * @param options - Policy creation options
   * @returns Created policy information
   */
  async createPolicy(options: CreatePolicyOptions): Promise<{
    policy_id: string;
    message: string;
  }> {
    const data: Record<string, unknown> = { name: options.name };
    if (options.description) data.description = options.description;
    if (options.allowedActions) data.allowed_actions = options.allowedActions;
    if (options.blockedDomains) data.blocked_domains = options.blockedDomains;

    return this.client.request('POST', '/api/v1/computer-use/policies', {
      json: data,
    });
  }

  /**
   * Get a specific policy by ID.
   *
   * @param policyId - Policy ID
   * @returns Policy details
   */
  async getPolicy(policyId: string): Promise<ComputerUsePolicy> {
    return this.client.request('GET', `/api/v1/computer-use/policies/${policyId}`);
  }

  /**
   * Update a computer use policy.
   *
   * @param policyId - Policy ID to update
   * @param options - Updated policy fields
   * @returns Updated policy confirmation
   */
  async updatePolicy(policyId: string, options: UpdatePolicyOptions): Promise<{
    policy_id: string;
    message: string;
  }> {
    const data: Record<string, unknown> = {};
    if (options.name !== undefined) data.name = options.name;
    if (options.description !== undefined) data.description = options.description;
    if (options.allowedActions !== undefined) data.allowed_actions = options.allowedActions;
    if (options.blockedDomains !== undefined) data.blocked_domains = options.blockedDomains;
    if (options.maxSteps !== undefined) data.max_steps = options.maxSteps;
    if (options.timeoutSeconds !== undefined) data.timeout_seconds = options.timeoutSeconds;

    return this.client.request('PUT', `/api/v1/computer-use/policies/${policyId}`, {
      json: data,
    });
  }

  /**
   * Delete a computer use policy.
   *
   * The default policy cannot be deleted.
   *
   * @param policyId - Policy ID to delete
   * @returns Deletion confirmation
   */
  async deletePolicy(policyId: string): Promise<{
    deleted: boolean;
    policy_id: string;
  }> {
    return this.client.request('DELETE', `/api/v1/computer-use/policies/${policyId}`);
  }

  // =========================================================================
  // Actions (additional)
  // =========================================================================

  /**
   * List available computer use actions.
   *
   * @returns Available action types
   */
  async listActions(): Promise<{
    actions: Record<string, unknown>[];
  }> {
    return this.client.request('GET', '/api/v1/computer-use/actions');
  }

  /**
   * Get details for a specific action.
   *
   * @param actionId - Action ID
   * @returns Action details
   */
  async getAction(actionId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/computer-use/actions/${actionId}`);
  }

  /**
   * Execute a single computer use action.
   *
   * The action is validated against the active policy before execution.
   *
   * @param options - Action execution options
   * @returns Execution result with action_id and success status
   */
  async executeAction(options: ExecuteActionOptions): Promise<{
    action_id: string;
    action_type: string;
    success: boolean;
    message: string;
  }> {
    const data: Record<string, unknown> = { action_type: options.actionType };
    if (options.parameters !== undefined) data.parameters = options.parameters;
    if (options.taskId !== undefined) data.task_id = options.taskId;

    return this.client.request('POST', '/api/v1/computer-use/actions', {
      json: data,
    });
  }

  /**
   * Delete a computer use action record.
   *
   * Only completed or failed actions can be deleted.
   *
   * @param actionId - Action ID to delete
   * @returns Deletion confirmation
   */
  async deleteAction(actionId: string): Promise<{
    deleted: boolean;
    action_id: string;
  }> {
    return this.client.request('DELETE', `/api/v1/computer-use/actions/${actionId}`);
  }

  // =========================================================================
  // Approvals
  // =========================================================================

  /**
   * List approval requests.
   *
   * @param options - Filtering options
   * @returns List of approval requests
   */
  async listApprovals(options?: {
    status?: string;
    limit?: number;
  }): Promise<{
    approvals: Record<string, unknown>[];
    count: number;
  }> {
    const params: Record<string, unknown> = {};
    if (options?.limit !== undefined) params.limit = options.limit;
    if (options?.status) params.status = options.status;

    return this.client.request('GET', '/api/v1/computer-use/approvals', {
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }

  /**
   * Get a specific approval request.
   *
   * @param requestId - Approval request ID
   * @returns Approval request details
   */
  async getApproval(requestId: string): Promise<{
    approval: Record<string, unknown>;
  }> {
    return this.client.request('GET', `/api/v1/computer-use/approvals/${requestId}`);
  }

  /**
   * Approve a pending approval request.
   *
   * @param requestId - Approval request ID
   * @param reason - Optional reason for approval
   * @returns Approval confirmation
   */
  async approveApproval(requestId: string, reason?: string): Promise<{
    approved: boolean;
    request_id: string;
  }> {
    return this.client.request('POST', `/api/v1/computer-use/approvals/${requestId}/approve`, {
      json: reason ? { reason } : undefined,
    });
  }

  /**
   * Deny a pending approval request.
   *
   * @param requestId - Approval request ID
   * @param reason - Optional reason for denial
   * @returns Denial confirmation
   */
  async denyApproval(requestId: string, reason?: string): Promise<{
    denied: boolean;
    request_id: string;
  }> {
    return this.client.request('POST', `/api/v1/computer-use/approvals/${requestId}/deny`, {
      json: reason ? { reason } : undefined,
    });
  }
}
