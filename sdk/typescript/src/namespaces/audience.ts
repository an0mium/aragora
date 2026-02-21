/**
 * Audience Namespace API
 *
 * Audience suggestion submission and retrieval for debates.
 */

interface AudienceClientInterface {
  request<T = unknown>(method: string, path: string, options?: {
    params?: Record<string, unknown>;
    json?: Record<string, unknown>;
    body?: Record<string, unknown>;
  }): Promise<T>;
}

export class AudienceAPI {
  constructor(private client: AudienceClientInterface) {}

  async getSuggestions(debateId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/debates/${encodeURIComponent(debateId)}/audience/suggestions`);
  }

  async submitSuggestion(debateId: string, suggestion: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/debates/${encodeURIComponent(debateId)}/audience/suggestions`, { body: suggestion });
  }

  async listSuggestions(debateId: string, options?: { maxClusters?: number; threshold?: number }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/audience/suggestions', {
      params: { debate_id: debateId, max_clusters: options?.maxClusters ?? 5, threshold: options?.threshold ?? 0.6 },
    });
  }

  async createSuggestion(debateId: string, suggestion: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/audience/suggestions', { body: { ...suggestion, debate_id: debateId } });
  }
}
