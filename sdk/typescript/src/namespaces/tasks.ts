/**
 * Tasks Namespace API
 *
 * Provides methods for task management.
 */

interface TasksClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class TasksAPI {
  constructor(private client: TasksClientInterface) {}

  /** Create a new task. */
  async create(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v2/tasks', { body: data });
  }

  /** Get a task by ID. */
  async get(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v2/tasks/${encodeURIComponent(taskId)}`);
  }

  /** List task history with optional filters. */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/control-plane/tasks/history', { params });
  }

  /** Approve task checkpoint data for an in-flight task. */
  async update(taskId: string, data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v2/tasks/${encodeURIComponent(taskId)}`, { body: data });
  }

  /** Cancel a task by ID. */
  async delete(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/control-plane/tasks/${encodeURIComponent(taskId)}/cancel`);
  }
}
