/**
 * Spectate Namespace API
 *
 * Debate spectating via the canonical public stream endpoints.
 */

interface SpectateClientInterface {
  request<T = unknown>(method: string, path: string, options?: {
    params?: Record<string, unknown>;
    json?: Record<string, unknown>;
    body?: Record<string, unknown>;
  }): Promise<T>;
}

export class SpectateAPI {
  constructor(private client: SpectateClientInterface) {}

  /**
   * Request the spectate stream endpoint scoped to a single debate.
   */
  async connectSSE(debateId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/spectate/stream', {
      params: { debate_id: debateId },
    });
  }

  async getRecent(options?: { count?: number; debateId?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/spectate/recent', {
      params: { count: options?.count ?? 50, ...(options?.debateId ? { debate_id: options.debateId } : {}) },
    });
  }

  async getStatus(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/spectate/status');
  }

  async getStream(options?: { count?: number; debateId?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/spectate/stream', {
      params: { count: options?.count ?? 50, ...(options?.debateId ? { debate_id: options.debateId } : {}) },
    });
  }
}
