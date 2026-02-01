/**
 * Insights Namespace API
 *
 * Provides access to insight extraction, recent insight feeds, and agent flip tracking.
 */

export interface InsightEntry {
  insight_id: string;
  source?: string;
  summary: string;
  confidence?: number;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

export interface ExtractDetailedRequest {
  content: string;
  context?: string;
  metadata?: Record<string, unknown>;
}

export interface DetailedInsight {
  insight?: InsightEntry;
  details?: Record<string, unknown>;
  key_findings?: string[];
  confidence?: number;
}

export interface FlipEntry {
  flip_id: string;
  agent: string;
  debate_id?: string;
  from_position: string;
  to_position: string;
  reason?: string;
  round?: number;
  timestamp: string;
}

export interface FlipSummary {
  total_flips: number;
  agents: Record<string, number>;
  most_common_reasons?: string[];
  period?: string;
}

interface InsightsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

export class InsightsAPI {
  constructor(private client: InsightsClientInterface) {}

  /**
   * Get recent insights from the InsightStore.
   */
  async getRecent(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ insights: InsightEntry[]; total?: number }> {
    return this.client.request('GET', '/api/v1/insights/recent', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Extract detailed insights from content.
   */
  async extractDetailed(body: ExtractDetailedRequest): Promise<DetailedInsight> {
    return this.client.request('POST', '/api/v1/insights/extract-detailed', { body });
  }

  /**
   * Get recent position flips across all agents.
   */
  async getRecentFlips(params?: {
    limit?: number;
    agent?: string;
  }): Promise<{ flips: FlipEntry[]; total?: number }> {
    return this.client.request('GET', '/api/v1/flips/recent', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get flip summary for dashboard display.
   */
  async getFlipSummary(): Promise<FlipSummary> {
    return this.client.request('GET', '/api/v1/flips/summary');
  }
}
