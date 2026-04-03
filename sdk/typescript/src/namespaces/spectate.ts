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

function buildSSEStreamInfo(debateId: string): Record<string, unknown> {
  return {
    debate_id: debateId,
    stream_url: `/api/v1/debates/${encodeURIComponent(debateId)}/spectate`,
  };
}

export class SpectateAPI {
  constructor(private client: SpectateClientInterface) {}

  /**
   * Return connection details for a debate's live SSE stream.
   *
   * The shared HTTP client assumes JSON responses, so this helper returns
   * the canonical stream URL instead of issuing the SSE request directly.
   */
  async connectSSE(debateId: string): Promise<Record<string, unknown>> {
    return buildSSEStreamInfo(debateId);
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
