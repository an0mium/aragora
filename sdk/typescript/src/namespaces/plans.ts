/**
 * Plans Namespace API
 *
 * Provides methods for plan management.
 */

interface PlansClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class PlansAPI {
  constructor(private client: PlansClientInterface) {}

  /** Update a plan. */
  async update(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PUT', '/api/v1/plans', { body: data });
  }
}
