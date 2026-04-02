/**
 * Spectate Namespace API
 *
 * Real-time debate observation via Server-Sent Events (SSE).
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
   * Request the debate-filtered spectate stream endpoint.
   *
   * The server returns a buffered JSON preview by default and can emit a
   * finite SSE snapshot when requested at the HTTP layer.
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
