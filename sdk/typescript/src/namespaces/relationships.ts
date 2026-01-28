/**
 * Relationships Namespace API
 *
 * Provides access to agent-to-agent relationship data, including
 * collaboration patterns, rivalry analysis, and network graphs.
 */

import type { AragoraClient } from '../client';

/**
 * Individual agent relationship
 */
export interface AgentRelationship {
  agent_a: string;
  agent_b: string;
  relationship_type: 'ally' | 'rival' | 'neutral' | 'unknown';
  agreement_rate: number;
  total_interactions: number;
  last_interaction?: string;
  collaboration_score: number;
  conflict_score: number;
}

/**
 * Relationship graph node
 */
export interface RelationshipNode {
  agent: string;
  connections: number;
  centrality: number;
  cluster?: string;
}

/**
 * Relationship graph edge
 */
export interface RelationshipEdge {
  source: string;
  target: string;
  weight: number;
  type: 'positive' | 'negative' | 'neutral';
}

/**
 * Full relationship graph
 */
export interface RelationshipGraph {
  nodes: RelationshipNode[];
  edges: RelationshipEdge[];
  clusters: Array<{
    id: string;
    agents: string[];
    cohesion: number;
  }>;
  metadata: {
    total_agents: number;
    total_relationships: number;
    average_connections: number;
    generated_at: string;
  };
}

/**
 * Relationship statistics
 */
export interface RelationshipStats {
  total_relationships: number;
  ally_count: number;
  rival_count: number;
  neutral_count: number;
  average_agreement_rate: number;
  most_connected_agent: string;
  strongest_alliance: { agents: [string, string]; score: number };
  strongest_rivalry: { agents: [string, string]; score: number };
}

/**
 * Relationship summary
 */
export interface RelationshipSummary {
  agent: string;
  allies: string[];
  rivals: string[];
  neutral: string[];
  top_collaborators: Array<{ agent: string; score: number }>;
  summary_text: string;
}

/**
 * Relationships namespace for agent relationship analysis.
 *
 * @example
 * ```typescript
 * // Get relationship between two agents
 * const relationship = await client.relationships.get('gpt-4', 'claude');
 * console.log(`Agreement rate: ${relationship.agreement_rate}%`);
 *
 * // Get full relationship graph
 * const graph = await client.relationships.getGraph();
 * console.log(`Network has ${graph.nodes.length} agents`);
 * ```
 */
export class RelationshipsNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * Get the relationship between two specific agents.
   *
   * @param agentA - First agent name
   * @param agentB - Second agent name
   * @returns Relationship details between the two agents
   */
  async get(agentA: string, agentB: string): Promise<AgentRelationship> {
    return this.client.request<AgentRelationship>(
      'GET',
      `/api/v1/relationship/${encodeURIComponent(agentA)}/${encodeURIComponent(agentB)}`
    );
  }

  /**
   * Get the full agent relationship graph.
   *
   * Useful for visualizing agent networks and identifying clusters.
   *
   * @param options.min_interactions - Minimum interactions to include edge
   * @param options.include_inactive - Include agents with no recent activity
   */
  async getGraph(options?: {
    min_interactions?: number;
    include_inactive?: boolean;
  }): Promise<RelationshipGraph> {
    return this.client.request<RelationshipGraph>('GET', '/api/v1/relationships/graph', {
      params: options,
    });
  }

  /**
   * Get aggregate relationship statistics.
   */
  async getStats(): Promise<RelationshipStats> {
    return this.client.request<RelationshipStats>('GET', '/api/v1/relationships/stats');
  }

  /**
   * Get a summary of relationships for a specific agent or all agents.
   *
   * @param agent - Optional agent name for agent-specific summary
   */
  async getSummary(agent?: string): Promise<RelationshipSummary | RelationshipSummary[]> {
    const params = agent ? { agent } : undefined;
    return this.client.request<RelationshipSummary | RelationshipSummary[]>(
      'GET',
      '/api/v1/relationships/summary',
      { params }
    );
  }
}
