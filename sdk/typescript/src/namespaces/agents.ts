/**
 * Agents Namespace API
 *
 * Provides a namespaced interface for agent-related operations.
 */

import type {
  Agent,
  AgentCalibration,
  AgentComparison,
  AgentConsistency,
  AgentFlip,
  AgentMoment,
  AgentNetwork,
  AgentPerformance,
  AgentPosition,
  AgentProfile,
  AgentRelationship,
  DomainRating,
  HeadToHeadStats,
  OpponentBriefing,
  PaginationParams,
  TeamSelection,
  TeamSelectionRequest,
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
 * Team selection strategy.
 */
export type TeamSelectionStrategy = 'balanced' | 'competitive' | 'diverse' | 'specialized';

/**
 * Team selection request for agent selection endpoint.
 */
export interface SelectTeamRequest {
  task: string;
  team_size?: number;
  strategy?: TeamSelectionStrategy;
}

/**
 * Team selection response.
 */
export interface SelectTeamResponse {
  agents: string[];
  rationale?: string;
  diversity_score?: number;
  total_score?: number;
  coverage?: Record<string, string[]>;
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
  getAgent(name: string): Promise<Agent>;
  getAgentProfile(name: string): Promise<AgentProfile>;
  getAgentHistory(name: string, params?: PaginationParams): Promise<{ matches: unknown[] }>;

  // Calibration
  getAgentCalibration(name: string): Promise<AgentCalibration>;
  getAgentCalibrationCurve(name: string): Promise<Record<string, unknown>>;
  getAgentCalibrationSummary(name: string): Promise<Record<string, unknown>>;

  // Performance
  getAgentPerformance(name: string): Promise<AgentPerformance>;
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
  getAgentRelationship(agentA: string, agentB: string): Promise<AgentRelationship>;

  // Other metrics
  getAgentReputation(name: string): Promise<{ reputation: number | null }>;
  getAgentMoments(name: string, params?: { type?: string; limit?: number } & PaginationParams): Promise<{ moments: AgentMoment[] }>;
  getAgentDomains(name: string): Promise<{ domains: DomainRating[] }>;
  getAgentElo(agentName: string): Promise<{ agent: string; elo: number; history: Array<{ date: string; elo: number }> }>;

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

  // Team selection
  selectTeam(request: TeamSelectionRequest): Promise<TeamSelection>;

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
 * // Get agent details
 * const agent = await client.agents.get('claude');
 *
 * // Get agent performance stats
 * const perf = await client.agents.getPerformance('claude');
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
   * Get an agent by name.
   */
  async get(name: string): Promise<Agent> {
    return this.client.getAgent(name);
  }

  /**
   * Get an agent's detailed profile.
   */
  async getProfile(name: string): Promise<AgentProfile> {
    return this.client.getAgentProfile(name);
  }

  /**
   * Get an agent's match history.
   */
  async getHistory(name: string, params?: PaginationParams): Promise<{ matches: unknown[] }> {
    return this.client.getAgentHistory(name, params);
  }

  /**
   * Get an agent's calibration data.
   */
  async getCalibration(name: string): Promise<AgentCalibration> {
    return this.client.getAgentCalibration(name);
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
   * Get an agent's performance statistics.
   */
  async getPerformance(name: string): Promise<AgentPerformance> {
    return this.client.getAgentPerformance(name);
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
   * Get an agent's reputation score.
   */
  async getReputation(name: string): Promise<{ reputation: number | null }> {
    return this.client.getAgentReputation(name);
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
   * Get an agent's ELO rating and history.
   */
  async getElo(name: string): Promise<{ agent: string; elo: number; history: Array<{ date: string; elo: number }> }> {
    return this.client.getAgentElo(name);
  }

  /**
   * Get the relationship between two agents.
   */
  async getRelationship(agentA: string, agentB: string): Promise<AgentRelationship> {
    return this.client.getAgentRelationship(agentA, agentB);
  }

  /**
   * Get agent rankings leaderboard.
   */
  async getLeaderboard(): Promise<{ agents: Agent[] }> {
    return this.client.getLeaderboard();
  }

  /**
   * Compare multiple agents side-by-side.
   */
  async compare(agents: string[]): Promise<AgentComparison> {
    return this.client.compareAgents(agents);
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
  // Team Selection (matching Python SDK)
  // ===========================================================================

  /**
   * Select an optimal team of agents for a task.
   *
   * @param task - The task description
   * @param teamSize - Number of agents to select (default: 3)
   * @param strategy - Selection strategy (default: 'balanced')
   * @returns Selected team with rationale
   *
   * @example
   * ```typescript
   * const team = await client.agents.selectTeam(
   *   'Design a distributed caching system',
   *   3,
   *   'balanced'
   * );
   * console.log(`Selected agents: ${team.agents.join(', ')}`);
   * ```
   */
  async selectTeam(
    task: string,
    teamSize: number = 3,
    strategy: TeamSelectionStrategy = 'balanced'
  ): Promise<SelectTeamResponse> {
    const request: TeamSelectionRequest = {
      task_type: task,
      team_size: teamSize,
    };

    // Map strategy to request parameters
    if (strategy === 'diverse') {
      request.diversity_weight = 0.8;
      request.quality_weight = 0.2;
    } else if (strategy === 'competitive') {
      request.diversity_weight = 0.2;
      request.quality_weight = 0.8;
    } else if (strategy === 'specialized') {
      request.diversity_weight = 0.0;
      request.quality_weight = 1.0;
    } else {
      // balanced
      request.diversity_weight = 0.5;
      request.quality_weight = 0.5;
    }

    const result = await this.client.selectTeam(request);
    return {
      agents: result.agents.map(a => a.agent_id),
      rationale: undefined, // Not provided by current API
      diversity_score: result.diversity_score,
      total_score: result.total_score,
      coverage: result.coverage,
    };
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
  // Agent Relationships (matching Python SDK)
  // ===========================================================================

  /**
   * Get relationship data between this agent and others.
   *
   * Same as getNetwork but named to match Python SDK.
   *
   * @param name - Agent name
   * @returns Relationship data with other agents
   */
  async getRelationships(name: string): Promise<AgentNetwork> {
    return this.client.getAgentNetwork(name);
  }

  // ===========================================================================
  // Alternate Compare Methods (matching Python SDK)
  // ===========================================================================

  /**
   * Compare two agents' performance.
   *
   * Alternative method name matching Python SDK's `compare_agents`.
   *
   * @param agent1 - First agent name
   * @param agent2 - Second agent name
   * @returns Comparison data
   */
  async compareAgents(agent1: string, agent2: string): Promise<AgentComparison> {
    return this.client.compareAgents([agent1, agent2]);
  }

  // ===========================================================================
  // Agent Registration (matching Python SDK control_plane)
  // ===========================================================================

  /**
   * Register a new agent with the system.
   *
   * @param agentId - Unique agent identifier
   * @param options - Registration options
   * @returns Registered agent info
   *
   * @example
   * ```typescript
   * const agent = await client.agents.register('my-agent', {
   *   capabilities: ['debate', 'analysis'],
   *   model: 'gpt-4',
   *   provider: 'openai'
   * });
   * ```
   */
  async register(
    agentId: string,
    options: {
      capabilities?: string[];
      model?: string;
      provider?: string;
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/control-plane/agents',
      {
        body: {
          agent_id: agentId,
          capabilities: options.capabilities || [],
          model: options.model || 'unknown',
          provider: options.provider || 'unknown',
          metadata: options.metadata || {},
        },
      }
    );
  }

  /**
   * Unregister an agent from the system.
   *
   * @param agentId - Agent ID to unregister
   * @returns Unregistration result
   */
  async unregister(agentId: string): Promise<{ success: boolean; message: string }> {
    return this.client.request<{ success: boolean; message: string }>(
      'DELETE',
      `/api/v1/control-plane/agents/${encodeURIComponent(agentId)}`
    );
  }

  // ===========================================================================
  // Agent Statistics & Analytics
  // ===========================================================================

  /**
   * Get aggregate statistics for all agents.
   *
   * @returns Statistics including total agents, average ELO, etc.
   */
  async getStats(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/agents/stats'
    );
  }

  // ===========================================================================
  // Agent Calibration
  // ===========================================================================

  /**
   * Trigger calibration for an agent.
   *
   * @param name - Agent name
   * @param options - Calibration options
   * @returns Calibration result
   */
  async calibrate(
    name: string,
    options: {
      domains?: string[];
      sampleSize?: number;
    } = {}
  ): Promise<AgentCalibration> {
    return this.client.request<AgentCalibration>(
      'POST',
      `/api/v1/agents/${encodeURIComponent(name)}/calibrate`,
      {
        body: {
          domains: options.domains,
          sample_size: options.sampleSize,
        },
      }
    );
  }

  // ===========================================================================
  // Agent Rankings & ELO
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
   * Update an agent's ELO rating.
   *
   * @param name - Agent name
   * @param eloChange - ELO points to add (can be negative)
   * @param options - Update options
   * @returns Updated ELO info
   */
  async updateElo(
    name: string,
    eloChange: number,
    options: {
      debateId?: string;
      reason?: string;
    } = {}
  ): Promise<{ agent: string; elo: number; previous_elo: number; change: number }> {
    return this.client.request<{ agent: string; elo: number; previous_elo: number; change: number }>(
      'POST',
      `/api/v1/agents/${encodeURIComponent(name)}/elo`,
      {
        body: {
          elo_change: eloChange,
          debate_id: options.debateId,
          reason: options.reason,
        },
      }
    );
  }

  // ===========================================================================
  // Agent Enable/Disable
  // ===========================================================================

  /**
   * Enable an agent for participation in debates.
   *
   * @param name - Agent name
   * @returns Enable result
   */
  async enable(name: string): Promise<{ success: boolean; agent: string; enabled: boolean }> {
    return this.client.request<{ success: boolean; agent: string; enabled: boolean }>(
      'POST',
      `/api/v1/agents/${encodeURIComponent(name)}/enable`
    );
  }

  /**
   * Disable an agent from participating in debates.
   *
   * @param name - Agent name
   * @param reason - Optional reason for disabling
   * @returns Disable result
   */
  async disable(
    name: string,
    reason?: string
  ): Promise<{ success: boolean; agent: string; enabled: boolean }> {
    return this.client.request<{ success: boolean; agent: string; enabled: boolean }>(
      'POST',
      `/api/v1/agents/${encodeURIComponent(name)}/disable`,
      {
        body: reason ? { reason } : undefined,
      }
    );
  }

  // ===========================================================================
  // Agent Quota Management
  // ===========================================================================

  /**
   * Get quota information for an agent.
   *
   * @param name - Agent name
   * @returns Quota details
   */
  async getQuota(name: string): Promise<{
    agent: string;
    debates_limit: number;
    debates_used: number;
    tokens_limit: number;
    tokens_used: number;
    reset_at: string;
  }> {
    return this.client.request<{
      agent: string;
      debates_limit: number;
      debates_used: number;
      tokens_limit: number;
      tokens_used: number;
      reset_at: string;
    }>(
      'GET',
      `/api/v1/agents/${encodeURIComponent(name)}/quota`
    );
  }

  /**
   * Set quota limits for an agent.
   *
   * @param name - Agent name
   * @param quota - Quota settings
   * @returns Updated quota info
   */
  async setQuota(
    name: string,
    quota: {
      debatesLimit?: number;
      tokensLimit?: number;
    }
  ): Promise<{
    agent: string;
    debates_limit: number;
    tokens_limit: number;
    updated_at: string;
  }> {
    return this.client.request<{
      agent: string;
      debates_limit: number;
      tokens_limit: number;
      updated_at: string;
    }>(
      'PUT',
      `/api/v1/agents/${encodeURIComponent(name)}/quota`,
      {
        body: {
          debates_limit: quota.debatesLimit,
          tokens_limit: quota.tokensLimit,
        },
      }
    );
  }

  // ===========================================================================
  // Agent Heartbeat (for long-running agents)
  // ===========================================================================

  /**
   * Send a heartbeat for an agent to indicate it's still active.
   *
   * @param agentId - Agent ID
   * @param status - Optional status update
   * @returns Heartbeat acknowledgment
   */
  async heartbeat(
    agentId: string,
    status?: string
  ): Promise<{ acknowledged: boolean; timestamp: string }> {
    return this.client.request<{ acknowledged: boolean; timestamp: string }>(
      'POST',
      `/api/v1/control-plane/agents/${encodeURIComponent(agentId)}/heartbeat`,
      {
        body: status ? { status } : undefined,
      }
    );
  }
}
