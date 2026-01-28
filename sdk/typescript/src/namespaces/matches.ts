/**
 * Matches Namespace API
 *
 * Provides access to recent match data.
 */

export interface MatchEntry {
  match_id: string;
  agents: string[];
  winner?: string;
  created_at?: string;
}

interface MatchesClientInterface {
  request<T = unknown>(method: string, path: string, options?: { params?: Record<string, unknown> }): Promise<T>;
}

export class MatchesAPI {
  constructor(private client: MatchesClientInterface) {}

  async listRecent(params?: { limit?: number; offset?: number }): Promise<{ matches: MatchEntry[]; total?: number }> {
    return this.client.request('GET', '/api/v1/matches/recent', { params: params as Record<string, unknown> });
  }
}
