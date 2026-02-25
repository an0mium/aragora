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
    return this.client.request('POST', '/api/v1/tasks', { body: data });
  }

  /** Get a task by ID. */
  async get(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v2/tasks/${encodeURIComponent(taskId)}`);
  }

  /** List tasks with optional filters. */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v2/tasks', { params });
  }

  /** Update a task. */
  async update(taskId: string, data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PUT', `/api/v2/tasks/${encodeURIComponent(taskId)}`, { body: data });
  }

  /** Delete a task. */
  async delete(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request('DELETE', `/api/v2/tasks/${encodeURIComponent(taskId)}`);
  }
}
