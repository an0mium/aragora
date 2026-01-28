/**
 * Insights Namespace API
 *
 * Provides access to insight extraction and recent insight feeds.
 */

export interface InsightEntry {
  insight_id: string;
  source?: string;
  summary: string;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

export interface ExtractDetailedRequest {
  content: string;
  metadata?: Record<string, unknown>;
}

interface InsightsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown; json?: Record<string, unknown> }
  ): Promise<T>;
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
}

export class InsightsAPI {
  constructor(private client: InsightsClientInterface) {}

  async getRecent(params?: { limit?: number; offset?: number }): Promise<{ insights: InsightEntry[]; total?: number }> {
    return this.client.request('GET', '/api/v1/insights/recent', { params: params as Record<string, unknown> });
  }

  async extractDetailed(body: ExtractDetailedRequest): Promise<{ insight?: InsightEntry; details?: Record<string, unknown> }> {
    return this.client.post('/api/v1/insights/extract-detailed', body);
  }
}
