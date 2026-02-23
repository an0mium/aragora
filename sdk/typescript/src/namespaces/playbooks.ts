/**
 * Playbooks Namespace API
 *
 * Provides methods for playbook management and execution.
 */

interface PlaybooksClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class PlaybooksAPI {
  constructor(private client: PlaybooksClientInterface) {}

  /** List available playbooks. */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/playbooks', { params });
  }

  /** Get a playbook by ID. */
  async get(playbookId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/playbooks/${playbookId}`);
  }

  /** Get run status for a playbook. */
  async getRunStatus(playbookId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/playbooks/${playbookId}/run`);
  }

  /** Execute a playbook. */
  async run(playbookId: string, data?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/playbooks/${playbookId}/run`, { body: data });
  }
}
