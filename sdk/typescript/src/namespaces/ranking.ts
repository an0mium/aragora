/**
 * Ranking Namespace API
 *
 * Provides access to agent ELO rankings and performance statistics.
 */

import type { AragoraClient } from '../client';

/**
 * Agent ranking entry
 */
export interface AgentRanking {
  rank: number;
  agent: string;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  total_debates: number;
  win_rate: number;
  streak: number;
  streak_type: 'win' | 'loss' | 'none';
  last_active?: string;
  elo_change_24h?: number;
  trend: 'up' | 'down' | 'stable';
}

/**
 * Ranking statistics
 */
export interface RankingStats {
  total_ranked_agents: number;
  total_debates: number;
  average_elo: number;
  elo_range: { min: number; max: number };
  most_improved: { agent: string; elo_gain: number };
  most_active: { agent: string; debates: number };
  highest_win_rate: { agent: string; rate: number };
  longest_streak: { agent: string; streak: number; type: 'win' | 'loss' };
  last_updated: string;
}

/**
 * Ranking query options
 */
export interface RankingQueryOptions {
  limit?: number;
  offset?: number;
  domain?: string;
  min_debates?: number;
  sort_by?: 'elo' | 'wins' | 'win_rate' | 'recent_activity';
  order?: 'asc' | 'desc';
}

/**
 * Ranking namespace for agent performance rankings.
 *
 * @example
 * ```typescript
 * // Get top 10 agents
 * const rankings = await client.ranking.list({ limit: 10 });
 * rankings.forEach((r, i) => {
 *   console.log(`${i + 1}. ${r.agent}: ${r.elo} ELO`);
 * });
 *
 * // Get ranking statistics
 * const stats = await client.ranking.getStats();
 * console.log(`Most improved: ${stats.most_improved.agent}`);
 * ```
 */
export class RankingNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * List agent rankings.
   *
   * @param options - Query options for filtering and sorting
   */
  async list(options?: RankingQueryOptions): Promise<AgentRanking[]> {
    const response = await this.client.request<{ rankings: AgentRanking[] }>(
      'GET',
      '/api/v1/rankings',
      { params: options }
    );
    return response.rankings;
  }

  /**
   * Get a specific agent's ranking.
   *
   * @param agentName - The agent name
   */
  async get(agentName: string): Promise<AgentRanking> {
    const response = await this.client.request<{ ranking: AgentRanking }>(
      'GET',
      `/api/v1/rankings/${encodeURIComponent(agentName)}`
    );
    return response.ranking;
  }

  /**
   * Get aggregate ranking statistics.
   */
  async getStats(): Promise<RankingStats> {
    return this.client.request<RankingStats>('GET', '/api/v1/ranking/stats');
  }

  /**
   * Get rankings for a specific domain.
   *
   * @param domain - The domain to filter by (e.g., 'technology', 'finance')
   * @param options - Additional query options
   */
  async listByDomain(
    domain: string,
    options?: Omit<RankingQueryOptions, 'domain'>
  ): Promise<AgentRanking[]> {
    return this.list({ ...options, domain });
  }

  /**
   * Get the top N agents by ELO.
   *
   * @param n - Number of top agents to return (default: 10)
   */
  async getTop(n: number = 10): Promise<AgentRanking[]> {
    return this.list({ limit: n, sort_by: 'elo', order: 'desc' });
  }

  /**
   * Get recently active agents' rankings.
   *
   * @param options - Query options
   */
  async getRecentlyActive(options?: Omit<RankingQueryOptions, 'sort_by'>): Promise<AgentRanking[]> {
    return this.list({ ...options, sort_by: 'recent_activity', order: 'desc' });
  }
}
