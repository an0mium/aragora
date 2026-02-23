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
}
