/**
 * Plans Namespace API
 *
 * Provides methods for decision plan management.
 */

interface PlansClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class PlansAPI {
  constructor(private client: PlansClientInterface) {}

  /** List all decision plans. */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/plans', { params });
  }

  /** Get a specific plan by ID. */
  async get(planId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/plans/${planId}`);
  }

  /** Create a new decision plan. */
  async create(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/plans', { body: data });
  }

  /** Update a plan. */
  async update(planId: string, data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PUT', `/api/v1/plans/${planId}`, { body: data });
  }

  /** Approve a decision plan for execution. */
  async approve(planId: string): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/plans/${planId}/approve`);
  }

  /** Reject a decision plan. */
  async reject(planId: string, reason = ''): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/plans/${planId}/reject`, { body: { reason } });
  }

  /** Execute an approved decision plan. */
  async execute(planId: string): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/plans/${planId}/execute`);
  }
}
