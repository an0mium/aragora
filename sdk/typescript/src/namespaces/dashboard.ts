/**
 * Dashboard Namespace API
 *
 * Provides endpoints used by dashboard views.
 */

export interface DashboardDebateEntry {
  debate_id: string;
  task?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
}

interface DashboardClientInterface {
  request<T = unknown>(method: string, path: string, options?: { params?: Record<string, unknown> }): Promise<T>;
}

export class DashboardAPI {
  constructor(private client: DashboardClientInterface) {}

  async listDebates(params?: { limit?: number; offset?: number }): Promise<{ debates: DashboardDebateEntry[]; total?: number }> {
    return this.client.request('GET', '/api/v1/dashboard/debates', { params: params as Record<string, unknown> });
  }
}
