/**
 * Moderation Namespace API
 *
 * Content moderation configuration, queue management, and statistics.
 */

interface ModerationClientInterface {
  request<T = unknown>(method: string, path: string, options?: {
    params?: Record<string, unknown>;
    json?: Record<string, unknown>;
    body?: Record<string, unknown>;
  }): Promise<T>;
}

export class ModerationAPI {
  constructor(private client: ModerationClientInterface) {}

  async getConfig(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/moderation/config');
  }

  async updateConfig(config: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PUT', '/api/v1/moderation/config', { body: config });
  }

  async getStats(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/moderation/stats');
  }

  async getQueue(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/moderation/queue');
  }

  async approveItem(itemId: string): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/moderation/queue/${encodeURIComponent(itemId)}/approve`);
  }

  async rejectItem(itemId: string): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/moderation/queue/${encodeURIComponent(itemId)}/reject`);
  }
}
