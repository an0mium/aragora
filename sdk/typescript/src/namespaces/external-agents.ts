/**
 * External Agents API
 *
 * Submit tasks to external AI agent frameworks (OpenHands, AutoGPT, CrewAI)
 * through Aragora's enterprise security gateway.
 */

/**
 * External adapter descriptor.
 */
export interface ExternalAdapter {
  name: string;
  description: string;
  config_class: string;
}

/**
 * Response from listing available adapters.
 */
export interface AdapterListResponse {
  adapters: ExternalAdapter[];
  total: number;
}

/**
 * Health status for a single adapter.
 */
export interface AdapterHealth {
  adapter_name: string;
  healthy: boolean;
  last_check: string;
  response_time_ms: number;
  error?: string;
  framework_version?: string;
}

/**
 * Response from the health endpoint.
 */
export interface HealthResponse {
  health: AdapterHealth[];
  total: number;
}

/**
 * Options for submitting a task to an external agent.
 */
export interface TaskSubmitOptions {
  adapter?: string;
  toolPermissions?: string[];
  timeoutSeconds?: number;
  maxSteps?: number;
  context?: Record<string, unknown>;
  workspaceId?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Response after submitting a task.
 */
export interface TaskSubmitResponse {
  task_id: string;
  status: string;
  adapter: string;
}

/**
 * Information about a submitted task.
 */
export interface TaskInfo {
  task_id: string;
  status: string;
  result?: Record<string, unknown>;
}

/**
 * Response after cancelling a task.
 */
export interface CancelResponse {
  task_id: string;
  cancelled: boolean;
}

/**
 * Client interface for external agents operations.
 */
interface ExternalAgentsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * External Agents API namespace.
 *
 * Provides methods for submitting tasks to external AI agent frameworks
 * (OpenHands, AutoGPT, CrewAI) through Aragora's enterprise security gateway:
 * - Listing available adapters
 * - Checking adapter health
 * - Submitting, polling, and cancelling tasks
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List available adapters
 * const { adapters } = await client.externalAgents.listAdapters();
 *
 * // Check adapter health
 * const { health } = await client.externalAgents.getHealth();
 *
 * // Submit a task
 * const task = await client.externalAgents.submitTask('code_review', 'Review auth module', {
 *   adapter: 'openhands',
 *   timeoutSeconds: 300,
 * });
 *
 * // Poll for results
 * const info = await client.externalAgents.getTask(task.task_id);
 *
 * // Cancel a running task
 * const result = await client.externalAgents.cancelTask(task.task_id);
 * ```
 */
export class ExternalAgentsAPI {
  constructor(private client: ExternalAgentsClientInterface) {}

  /**
   * List all registered external agent adapters.
   *
   * Returns the set of adapters available for task submission,
   * including their names, descriptions, and configuration classes.
   */
  async listAdapters(): Promise<AdapterListResponse> {
    return this.client.request(
      'GET',
      '/api/v1/external-agents/adapters',
    ) as Promise<AdapterListResponse>;
  }

  /**
   * Get health status of external agent adapters.
   *
   * Returns health information including connectivity, response time,
   * and framework version for each adapter. Optionally filter by adapter name.
   *
   * @param adapterName - Optional adapter name to check health for a specific adapter.
   */
  async getHealth(adapterName?: string): Promise<HealthResponse> {
    const params: Record<string, unknown> = {};
    if (adapterName !== undefined) {
      params.adapter = adapterName;
    }
    return this.client.request(
      'GET',
      '/api/v1/external-agents/health',
      { params },
    ) as Promise<HealthResponse>;
  }

  /**
   * Submit a task to an external agent framework.
   *
   * Sends the task through Aragora's security gateway, which enforces
   * tool permissions, timeout limits, and workspace isolation before
   * delegating to the target adapter.
   *
   * @param taskType - The type of task to execute (e.g. 'code_review', 'analysis').
   * @param prompt - The natural-language prompt describing the task.
   * @param options - Optional configuration including adapter selection,
   *   tool permissions, timeout, max steps, context, workspace, and metadata.
   */
  async submitTask(
    taskType: string,
    prompt: string,
    options?: TaskSubmitOptions,
  ): Promise<TaskSubmitResponse> {
    return this.client.request(
      'POST',
      '/api/v1/external-agents/tasks',
      {
        json: {
          task_type: taskType,
          prompt,
          adapter: options?.adapter,
          tool_permissions: options?.toolPermissions,
          timeout_seconds: options?.timeoutSeconds,
          max_steps: options?.maxSteps,
          context: options?.context,
          workspace_id: options?.workspaceId,
          metadata: options?.metadata,
        },
      },
    );
  }

  /**
   * Get the current status and result of a submitted task.
   *
   * @param taskId - The ID of the task to retrieve.
   */
  async getTask(taskId: string): Promise<TaskInfo> {
    return this.client.request(
      'GET',
      `/api/v1/external-agents/tasks/${taskId}`,
    ) as Promise<TaskInfo>;
  }

  /**
   * Cancel a running external agent task.
   *
   * Sends a cancellation request to the adapter. The task may not
   * stop immediately depending on the external framework's behavior.
   *
   * @param taskId - The ID of the task to cancel.
   */
  async cancelTask(taskId: string): Promise<CancelResponse> {
    return this.client.request(
      'DELETE',
      `/api/v1/external-agents/tasks/${taskId}`,
    ) as Promise<CancelResponse>;
  }
}
