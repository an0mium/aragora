/**
 * Context Namespace API
 *
 * Provides methods for context budget management and estimation.
 */

interface ContextClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class ContextAPI {
  constructor(private client: ContextClientInterface) {}

  /** Get current context budget usage. */
  async getBudget(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/context/budget');
  }

  /** Estimate context budget for a planned operation. */
  async estimateBudget(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/context/budget/estimate', { params });
  }
}
