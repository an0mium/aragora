/**
 * Agents Namespace API
 *
 * Provides a namespaced interface for agent-related operations.
 */

import type {
  Agent,
  AgentComparison,
  AgentConsistency,
  AgentFlip,
  AgentMoment,
  AgentNetwork,
  AgentPosition,
  AgentProfile,
  AgentRelationship,
  DomainRating,
  HeadToHeadStats,
  OpponentBriefing,
  PaginationParams,
} from '../types';

// =============================================================================
// Agent-specific Types (matching Python SDK)
// =============================================================================

/**
 * Agent persona configuration.
 */
export interface AgentPersona {
  agent_name: string;
  description?: string;
  traits?: string[];
  expertise?: string[];
  model?: string;
  temperature?: number;
  created_at?: string;
  updated_at?: string;
}

/**
 * Grounded persona with performance metrics.
 */
export interface GroundedPersona {
  agent_name: string;
  elo: number;
  domain_elos: Record<string, number>;
  win_rate: number;
  calibration_score: number;
  position_accuracy: number;
  debates_count: number;
  last_active?: string;
}

/**
 * Identity prompt response.
 */
export interface IdentityPrompt {
  agent_name: string;
  prompt: string;
  sections_included?: string[];
  token_count?: number;
}

/**
 * Agent accuracy metrics.
 */
export interface AgentAccuracy {
  agent_name: string;
  total_positions: number;
  verified_positions: number;
  correct_positions: number;
  accuracy_rate: number;
  by_domain?: Record<string, {
    total: number;
    correct: number;
    accuracy: number;
  }>;
}

/**
 * Interface for the internal client methods used by AgentsAPI.
 */
interface AgentsClientInterface {
  // Core listing methods
  listAgents(): Promise<{ agents: Agent[] }>;
  listAgentsAvailability(): Promise<{ available: string[]; missing?: string[] }>;
  listAgentsHealth(): Promise<Record<string, unknown>>;
  listLocalAgents(): Promise<Record<string, unknown>>;
  getLocalAgentsStatus(): Promise<Record<string, unknown>>;

  // Basic agent info
  getAgentProfile(name: string): Promise<AgentProfile>;

  // Calibration
  getAgentCalibrationCurve(name: string): Promise<Record<string, unknown>>;
  getAgentCalibrationSummary(name: string): Promise<Record<string, unknown>>;

  // Performance
  getAgentHeadToHead(name: string, opponent: string): Promise<HeadToHeadStats>;
  getAgentOpponentBriefing(name: string, opponent: string): Promise<OpponentBriefing>;
  getAgentConsistency(name: string): Promise<AgentConsistency>;

  // Flips and positions
  getAgentFlips(name: string, params?: { limit?: number } & PaginationParams): Promise<{ flips: AgentFlip[] }>;
  getAgentPositions(name: string, params?: { topic?: string; limit?: number } & PaginationParams): Promise<{ positions: AgentPosition[] }>;

  // Network and relationships
  getAgentNetwork(name: string): Promise<AgentNetwork>;
  getAgentAllies(name: string): Promise<{ allies: AgentRelationship[] }>;
  getAgentRivals(name: string): Promise<{ rivals: AgentRelationship[] }>;

  // Other metrics
  getAgentMoments(name: string, params?: { type?: string; limit?: number } & PaginationParams): Promise<{ moments: AgentMoment[] }>;
  getAgentDomains(name: string): Promise<{ domains: DomainRating[] }>;

  // Comparison and leaderboards
  getLeaderboard(): Promise<{ agents: Agent[] }>;
  compareAgents(agents: string[]): Promise<AgentComparison>;
  getAgentMetadata(name: string): Promise<Record<string, unknown>>;
  getAgentIntrospection(name: string): Promise<Record<string, unknown>>;
  getLeaderboardView(): Promise<Record<string, unknown>>;
  getRecentMatches(params?: { limit?: number }): Promise<{ matches: unknown[] }>;
  getRecentFlips(params?: { limit?: number }): Promise<{ flips: unknown[] }>;
  getFlipsSummary(): Promise<Record<string, unknown>>;
  getCalibrationLeaderboard(): Promise<{ agents: Array<{ name: string; score: number }> }>;

  // Generic request method (for persona/identity/accuracy endpoints)
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

/**
 * Agents API namespace.
 *
 * Provides methods for managing and analyzing AI agents:
 * - Listing and retrieving agents
 * - Viewing agent profiles and performance
 * - Analyzing head-to-head matchups
 * - Tracking ELO ratings and calibration
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List all agents
 * const { agents } = await client.agents.list();
 *
 * // Get agent profile
 * const profile = await client.agents.getProfile('claude');
 *
 * // Get head-to-head stats
 * const h2h = await client.agents.getHeadToHead('claude', 'gpt-4');
 * ```
 */
export class AgentsAPI {
  constructor(private client: AgentsClientInterface) {}

  /**
   * List all available agents.
   */
  async list(): Promise<{ agents: Agent[] }> {
    return this.client.listAgents();
  }

  /**
   * Get availability information for configured agents.
   */
  async getAvailability(): Promise<{ available: string[]; missing?: string[] }> {
    return this.client.listAgentsAvailability();
  }

  /**
   * Get health status for configured agents.
   */
  async getHealth(): Promise<Record<string, unknown>> {
    return this.client.listAgentsHealth();
  }

  /**
   * List local agents on the current host.
   */
  async listLocal(): Promise<Record<string, unknown>> {
    return this.client.listLocalAgents();
  }

  /**
   * Get local agent status.
   */
  async getLocalStatus(): Promise<Record<string, unknown>> {
    return this.client.getLocalAgentsStatus();
  }

  /**
   * Get an agent's detailed profile.
   */
  async getProfile(name: string): Promise<AgentProfile> {
    return this.client.getAgentProfile(name);
  }

  /**
   * Get an agent's calibration curve data.
   */
  async getCalibrationCurve(name: string): Promise<Record<string, unknown>> {
    return this.client.getAgentCalibrationCurve(name);
  }

  /**
   * Get an agent's calibration summary data.
   */
  async getCalibrationSummary(name: string): Promise<Record<string, unknown>> {
    return this.client.getAgentCalibrationSummary(name);
  }

  /**
   * Get head-to-head statistics against another agent.
   */
  async getHeadToHead(name: string, opponent: string): Promise<HeadToHeadStats> {
    return this.client.getAgentHeadToHead(name, opponent);
  }

  /**
   * Get a briefing on an opponent agent.
   */
  async getOpponentBriefing(name: string, opponent: string): Promise<OpponentBriefing> {
    return this.client.getAgentOpponentBriefing(name, opponent);
  }

  /**
   * Get an agent's consistency metrics.
   */
  async getConsistency(name: string): Promise<AgentConsistency> {
    return this.client.getAgentConsistency(name);
  }

  /**
   * Get an agent's position flips (changes in stance).
   */
  async getFlips(name: string, params?: { limit?: number } & PaginationParams): Promise<{ flips: AgentFlip[] }> {
    return this.client.getAgentFlips(name, params);
  }

  /**
   * Get an agent's network of relationships.
   */
  async getNetwork(name: string): Promise<AgentNetwork> {
    return this.client.getAgentNetwork(name);
  }

  /**
   * Get an agent's allies.
   */
  async getAllies(name: string): Promise<{ allies: AgentRelationship[] }> {
    return this.client.getAgentAllies(name);
  }

  /**
   * Get an agent's rivals.
   */
  async getRivals(name: string): Promise<{ rivals: AgentRelationship[] }> {
    return this.client.getAgentRivals(name);
  }

  /**
   * Get notable moments from an agent's history.
   */
  async getMoments(name: string, params?: { type?: string; limit?: number } & PaginationParams): Promise<{ moments: AgentMoment[] }> {
    return this.client.getAgentMoments(name, params);
  }

  /**
   * Get an agent's positions on various topics.
   */
  async getPositions(name: string, params?: { topic?: string; limit?: number } & PaginationParams): Promise<{ positions: AgentPosition[] }> {
    return this.client.getAgentPositions(name, params);
  }

  /**
   * Get an agent's domain expertise ratings.
   */
  async getDomains(name: string): Promise<{ domains: DomainRating[] }> {
    return this.client.getAgentDomains(name);
  }

  /**
   * Get agent rankings leaderboard.
   */
  async getLeaderboard(): Promise<{ agents: Agent[] }> {
    return this.client.getLeaderboard();
  }

  /**
   * Compare two agents' performance.
   *
   * @param agent1 - First agent name
   * @param agent2 - Second agent name
   * @returns Comparison data
   */
  async compareAgents(agent1: string, agent2: string): Promise<AgentComparison> {
    return this.client.compareAgents([agent1, agent2]);
  }

  /**
   * Get agent metadata (model info, capabilities).
   */
  async getMetadata(name: string): Promise<Record<string, unknown>> {
    return this.client.getAgentMetadata(name);
  }

  /**
   * Get agent introspection data (self-awareness metrics).
   */
  async getIntrospection(name: string): Promise<Record<string, unknown>> {
    return this.client.getAgentIntrospection(name);
  }

  /**
   * Get consolidated leaderboard view with all tabs.
   */
  async getLeaderboardView(): Promise<Record<string, unknown>> {
    return this.client.getLeaderboardView();
  }

  /**
   * Get recent matches across all agents.
   */
  async getRecentMatches(params?: { limit?: number }): Promise<{ matches: unknown[] }> {
    return this.client.getRecentMatches(params);
  }

  /**
   * Get recent position flips across all agents.
   */
  async getRecentFlips(params?: { limit?: number }): Promise<{ flips: unknown[] }> {
    return this.client.getRecentFlips(params);
  }

  /**
   * Get flip summary data for dashboard display.
   */
  async getFlipsSummary(): Promise<Record<string, unknown>> {
    return this.client.getFlipsSummary();
  }

  /**
   * Get agents ranked by calibration score.
   */
  async getCalibrationLeaderboard(): Promise<{ agents: Array<{ name: string; score: number }> }> {
    return this.client.getCalibrationLeaderboard();
  }

  // ===========================================================================
  // Agent Profile & Identity (matching Python SDK)
  // ===========================================================================

  /**
   * Get agent's persona configuration.
   *
   * @param name - Agent name
   * @returns Persona configuration
   */
  async getPersona(name: string): Promise<AgentPersona> {
    return this.client.request<AgentPersona>(
      'GET',
      `/api/agent/${encodeURIComponent(name)}/persona`
    );
  }

  /**
   * Delete agent's custom persona.
   *
   * @param name - Agent name
   * @returns Deletion result
   */
  async deletePersona(name: string): Promise<{ success: boolean; message: string }> {
    return this.client.request<{ success: boolean; message: string }>(
      'DELETE',
      `/api/agent/${encodeURIComponent(name)}/persona`
    );
  }

  /**
   * Get agent's grounded (evidence-based) persona with performance metrics.
   *
   * @param name - Agent name
   * @returns Grounded persona with ELO and performance data
   */
  async getGroundedPersona(name: string): Promise<GroundedPersona> {
    return this.client.request<GroundedPersona>(
      'GET',
      `/api/agent/${encodeURIComponent(name)}/grounded-persona`
    );
  }

  /**
   * Get agent's identity prompt.
   *
   * @param name - Agent name
   * @returns Identity prompt for the agent
   */
  async getIdentityPrompt(name: string): Promise<IdentityPrompt> {
    return this.client.request<IdentityPrompt>(
      'GET',
      `/api/agent/${encodeURIComponent(name)}/identity-prompt`
    );
  }

  // ===========================================================================
  // Agent Analytics (matching Python SDK)
  // ===========================================================================

  /**
   * Get agent's accuracy metrics.
   *
   * @param name - Agent name
   * @returns Accuracy metrics including position accuracy by domain
   */
  async getAccuracy(name: string): Promise<AgentAccuracy> {
    return this.client.request<AgentAccuracy>(
      'GET',
      `/api/agent/${encodeURIComponent(name)}/accuracy`
    );
  }

  // ===========================================================================
  // Agent Rankings
  // ===========================================================================

  /**
   * Get agent rankings.
   *
   * @param options - Ranking filter options
   * @returns List of agent rankings
   */
  async getRankings(options: {
    limit?: number;
    offset?: number;
    domain?: string;
    minDebates?: number;
    sortBy?: 'elo' | 'wins' | 'win_rate' | 'recent_activity';
    order?: 'asc' | 'desc';
  } = {}): Promise<{ rankings: Array<{ agent: string; elo: number; rank: number }> }> {
    const params: Record<string, unknown> = {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      sort_by: options.sortBy ?? 'elo',
      order: options.order ?? 'desc',
    };
    if (options.domain) {
      params.domain = options.domain;
    }
    if (options.minDebates !== undefined) {
      params.min_debates = options.minDebates;
    }
    return this.client.request<{ rankings: Array<{ agent: string; elo: number; rank: number }> }>(
      'GET',
      '/api/v1/rankings',
      { params }
    );
  }

  /**
   * Get head-to-head comparison between two agents.
   */
  async getHeadToHeadStats(agentId: string, opponentId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/agent/${agentId}/head-to-head/${opponentId}`) as Promise<Record<string, unknown>>;
  }

  /**
   * Get opponent briefing for an agent matchup.
   */
  async getOpponentBriefingReport(agentId: string, opponentId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/agent/${agentId}/opponent-briefing/${opponentId}`) as Promise<Record<string, unknown>>;
  }

}
