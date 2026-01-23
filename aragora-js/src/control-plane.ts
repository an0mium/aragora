/**
 * Control Plane API for the Aragora TypeScript SDK.
 *
 * Provides agent management, task submission, and health monitoring
 * for the Aragora control plane.
 */

import type { AragoraClient } from './client';
import type {
  RegisteredAgent,
  AgentHealth,
  Task,
  ControlPlaneStatus,
  ResourceUtilization,
  TaskPriority,
} from './types';

export class ControlPlaneAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  // =========================================================================
  // Agent Operations
  // =========================================================================

  /**
   * Register an agent with the control plane.
   */
  async registerAgent(
    agentId: string,
    capabilities: string[] = ['debate'],
    options: { model?: string; provider?: string } = {}
  ): Promise<RegisteredAgent> {
    const data = await this.client.post<{ agent: RegisteredAgent }>('/api/v1/control-plane/agents', {
      agent_id: agentId,
      capabilities,
      model: options.model ?? 'unknown',
      provider: options.provider ?? 'unknown',
    });
    return data.agent;
  }

  /**
   * Unregister an agent from the control plane.
   */
  async unregisterAgent(agentId: string): Promise<boolean> {
    const data = await this.client.post<{ success: boolean }>(
      `/api/v1/control-plane/agents/${agentId}/unregister`,
      {}
    );
    return data.success ?? false;
  }

  /**
   * List all registered agents.
   */
  async listAgents(options: {
    capability?: string;
    onlyAvailable?: boolean;
  } = {}): Promise<RegisteredAgent[]> {
    const params: Record<string, string | boolean> = {
      only_available: options.onlyAvailable ?? true,
    };
    if (options.capability) {
      params.capability = options.capability;
    }
    const data = await this.client.get<{ agents: RegisteredAgent[] }>('/api/v1/control-plane/agents', params);
    return data.agents ?? [];
  }

  /**
   * Get detailed health status for an agent.
   */
  async getAgentHealth(agentId: string): Promise<AgentHealth> {
    return this.client.get<AgentHealth>(`/api/v1/control-plane/agents/${agentId}/health`);
  }

  // =========================================================================
  // Task Operations
  // =========================================================================

  /**
   * Submit a task to the control plane.
   */
  async submitTask(
    taskType: string,
    payload: Record<string, unknown>,
    options: {
      requiredCapabilities?: string[];
      priority?: TaskPriority;
      timeoutSeconds?: number;
    } = {}
  ): Promise<string> {
    const data = await this.client.post<{ task_id: string }>('/api/v1/control-plane/tasks', {
      task_type: taskType,
      payload,
      required_capabilities: options.requiredCapabilities,
      priority: options.priority ?? 'normal',
      timeout_seconds: options.timeoutSeconds ?? 300,
    });
    return data.task_id;
  }

  /**
   * Get task status.
   */
  async getTaskStatus(taskId: string): Promise<Task> {
    return this.client.get<Task>(`/api/v1/control-plane/tasks/${taskId}`);
  }

  /**
   * Cancel a pending or running task.
   */
  async cancelTask(taskId: string): Promise<boolean> {
    const data = await this.client.post<{ success: boolean }>(
      `/api/v1/control-plane/tasks/${taskId}/cancel`,
      {}
    );
    return data.success ?? false;
  }

  /**
   * List tasks in the pending queue.
   */
  async listPendingTasks(options: {
    taskType?: string;
    limit?: number;
  } = {}): Promise<Task[]> {
    const params: Record<string, string | number> = {
      status: 'pending',
      limit: options.limit ?? 20,
    };
    if (options.taskType) {
      params.task_type = options.taskType;
    }
    const data = await this.client.get<{ tasks: Task[] }>('/api/v1/control-plane/tasks', params);
    return data.tasks ?? [];
  }

  /**
   * Wait for a task to complete.
   */
  async waitForTask(
    taskId: string,
    options: { pollInterval?: number; timeout?: number } = {}
  ): Promise<Task> {
    const pollInterval = options.pollInterval ?? 1000;
    const timeout = options.timeout ?? 300000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const task = await this.getTaskStatus(taskId);
      if (['completed', 'failed', 'cancelled', 'timeout'].includes(task.status)) {
        return task;
      }
      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Task ${taskId} did not complete within ${timeout}ms`);
  }

  // =========================================================================
  // Health & Status
  // =========================================================================

  /**
   * Get overall control plane status.
   */
  async getStatus(): Promise<ControlPlaneStatus> {
    return this.client.get<ControlPlaneStatus>('/api/v1/control-plane/status');
  }

  /**
   * Trigger a health check.
   */
  async triggerHealthCheck(agentId?: string): Promise<Record<string, unknown>> {
    if (agentId) {
      return this.client.post<Record<string, unknown>>(
        `/api/v1/control-plane/agents/${agentId}/health-check`,
        {}
      );
    }
    return this.client.post<Record<string, unknown>>('/api/v1/control-plane/health-check', {});
  }

  /**
   * Get resource utilization metrics.
   */
  async getResourceUtilization(): Promise<ResourceUtilization> {
    return this.client.get<ResourceUtilization>('/api/v1/control-plane/utilization');
  }
}
