/**
 * Reputation Namespace API
 *
 * Provides access to agent reputation scores and profiles.
 */

export interface ReputationEntry {
  agent: string;
  reputation: number | null;
  trustworthiness?: number;
  expertise?: Record<string, number>;
  community_standing?: number;
  debate_count?: number;
  updated_at?: string;
}

export interface AgentReputation {
  agent: string;
  overall_reputation: number;
  trustworthiness: number;
  expertise: Record<string, number>;
  community_standing: number;
  history?: Array<{
    timestamp: string;
    reputation: number;
    event?: string;
  }>;
}

interface ReputationClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown> }
  ): Promise<T>;
}

export class ReputationAPI {
  constructor(private client: ReputationClientInterface) {}

  /**
   * List all agent reputations.
   */
  async listAll(params?: {
    limit?: number;
    sort_by?: string;
  }): Promise<{ reputations: ReputationEntry[]; count?: number }> {
    return this.client.request('GET', '/api/v1/reputation/all', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get reputation for a specific agent.
   */
  async getAgent(agentName: string): Promise<AgentReputation> {
    return this.client.request('GET', `/api/v1/agent/${agentName}/reputation`);
  }
}
