/**
 * Matches Namespace API
 *
 * Provides access to ELO match records and results.
 */

export interface MatchEntry {
  match_id: string;
  agents: string[];
  winner?: string;
  loser?: string;
  elo_changes?: Record<string, number>;
  debate_id?: string;
  domain?: string;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

interface MatchesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown> }
  ): Promise<T>;
}

export class MatchesAPI {
  constructor(private client: MatchesClientInterface) {}

  /**
   * List recent ELO matches across all agents.
   */
  async listRecent(params?: {
    limit?: number;
    offset?: number;
    agent?: string;
    domain?: string;
  }): Promise<{ matches: MatchEntry[]; total?: number }> {
    return this.client.request('GET', '/api/v1/matches/recent', {
      params: params as Record<string, unknown>,
    });
  }
}
