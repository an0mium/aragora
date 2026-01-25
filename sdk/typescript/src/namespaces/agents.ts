/**
 * Agents Namespace API
 *
 * Provides a namespaced interface for agent-related operations.
 */

import type {
  Agent,
  AgentCalibration,
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
} from '../types';

/**
 * Interface for the internal client methods used by AgentsAPI.
 */
interface AgentsClientInterface {
  listAgents(): Promise<{ agents: Agent[] }>;
  getAgent(name: string): Promise<Agent>;
  getAgentProfile(name: string): Promise<AgentProfile>;
  getAgentHistory(name: string, params?: PaginationParams): Promise<{ matches: unknown[] }>;
  getAgentCalibration(name: string): Promise<AgentCalibration>;
  getAgentPerformance(name: string): Promise<AgentPerformance>;
  getAgentHeadToHead(name: string, opponent: string): Promise<HeadToHeadStats>;
  getAgentOpponentBriefing(name: string, opponent: string): Promise<OpponentBriefing>;
  getAgentConsistency(name: string): Promise<AgentConsistency>;
  getAgentFlips(name: string, params?: { limit?: number } & PaginationParams): Promise<{ flips: AgentFlip[] }>;
  getAgentNetwork(name: string): Promise<AgentNetwork>;
  getAgentMoments(name: string, params?: { type?: string; limit?: number } & PaginationParams): Promise<{ moments: AgentMoment[] }>;
  getAgentPositions(name: string, params?: { topic?: string; limit?: number } & PaginationParams): Promise<{ positions: AgentPosition[] }>;
  getAgentDomains(name: string): Promise<{ domains: DomainRating[] }>;
  getAgentElo(agentName: string): Promise<{ agent: string; elo: number; history: Array<{ date: string; elo: number }> }>;
  getAgentRelationship(agentA: string, agentB: string): Promise<AgentRelationship>;
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
}
