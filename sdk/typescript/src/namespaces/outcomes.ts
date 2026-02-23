/**
 * Outcomes Namespace API
 *
 * Provides methods for querying decision outcomes.
 */

interface OutcomesClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class OutcomesAPI {
  constructor(private client: OutcomesClientInterface) {}

  /** Get outcome impact analysis. */
  async getImpact(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/outcomes/impact', { params });
  }

  /** Search outcomes. */
  async search(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/outcomes/search', { params });
  }
}
