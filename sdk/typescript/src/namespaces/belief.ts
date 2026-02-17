/**
 * Belief Network Namespace API
 *
 * Provides access to belief network analysis, cruxes, and provenance tracking.
 * The belief network represents claims and their supporting/opposing relationships.
 */

import type { AragoraClient } from '../client';

/**
 * Crux analysis result from CruxDetector.
 */
export interface CruxAnalysis {
  debate_id: string;
  cruxes: Array<{
    claim_id: string;
    statement: string;
    influence_score: number;
    disagreement_score: number;
    uncertainty_score: number;
    centrality_score: number;
    composite_score: number;
  }>;
  count: number;
}

/**
 * Belief crux representing a key claim that would impact debate outcome.
 */
export interface BeliefCrux {
  claim_id: string;
  claim_text: string;
  impact_score: number;
  current_confidence: number;
  supporting_agents: string[];
  opposing_agents: string[];
}

/**
 * Response for cruxes endpoint.
 */
export interface BeliefCruxes {
  debate_id: string;
  cruxes: BeliefCrux[];
  analysis_timestamp: string;
}

/**
 * Load-bearing claim with centrality score.
 */
export interface LoadBearingClaim {
  claim_id: string;
  claim_text: string;
  centrality_score: number;
  dependents_count: number;
}

/**
 * Response for load-bearing claims.
 */
export interface LoadBearingClaims {
  debate_id: string;
  claims: LoadBearingClaim[];
}

/**
 * Node in the belief graph.
 */
export interface BeliefGraphNode {
  id: string;
  text: string;
  type: 'claim' | 'evidence' | 'inference';
  confidence: number;
  is_crux?: boolean;
}

/**
 * Link between nodes in the belief graph.
 */
export interface BeliefGraphLink {
  source: string;
  target: string;
  type: 'supports' | 'opposes' | 'depends_on';
  strength: number;
}

/**
 * Complete belief graph structure.
 */
export interface BeliefGraph {
  debate_id: string;
  nodes: BeliefGraphNode[];
  links: BeliefGraphLink[];
}

/**
 * Export format options.
 */
export type ExportFormat = 'json' | 'graphml' | 'csv';

/**
 * Exported belief network.
 */
export interface BeliefExport {
  format: ExportFormat;
  content: string;
  metadata: {
    debate_id: string;
    exported_at: string;
    node_count: number;
    link_count: number;
  };
}

/**
 * Belief Network API.
 *
 * Provides methods for analyzing belief networks in debates:
 * - Finding cruxes (key claims that would change outcomes)
 * - Identifying load-bearing claims (high centrality)
 * - Exporting graph structures
 */
export class BeliefAPI {
  constructor(private client: AragoraClient) {}

  /**
   * Get key claims (cruxes) that would most impact the debate outcome.
   * @route GET /api/v1/belief-network/{debate_id}/cruxes
   */
  async getCruxes(
    debateId: string,
    options?: { top_k?: number }
  ): Promise<BeliefCruxes> {
    return this.client.request(
      'GET',
      `/api/v1/belief-network/${encodeURIComponent(debateId)}/cruxes`,
      { params: options }
    ) as Promise<BeliefCruxes>;
  }

  /**
   * Get claims with highest centrality (most load-bearing).
   * @route GET /api/v1/belief-network/{debate_id}/load-bearing-claims
   */
  async getLoadBearingClaims(
    debateId: string,
    options?: { limit?: number }
  ): Promise<LoadBearingClaims> {
    return this.client.request(
      'GET',
      `/api/v1/belief-network/${encodeURIComponent(debateId)}/load-bearing-claims`,
      { params: options }
    ) as Promise<LoadBearingClaims>;
  }

  /**
   * Get belief network as a graph structure for visualization.
   * @route GET /api/v1/belief-network/{debate_id}/graph
   */
  async getGraph(
    debateId: string,
    options?: { include_cruxes?: boolean }
  ): Promise<BeliefGraph> {
    return this.client.request(
      'GET',
      `/api/v1/belief-network/${encodeURIComponent(debateId)}/graph`,
      { params: options }
    ) as Promise<BeliefGraph>;
  }

  /**
   * Export belief network in various formats.
   * @route GET /api/v1/belief-network/{debate_id}/export
   */
  async export(
    debateId: string,
    options?: { format?: ExportFormat }
  ): Promise<BeliefExport> {
    return this.client.request(
      'GET',
      `/api/v1/belief-network/${encodeURIComponent(debateId)}/export`,
      { params: options }
    ) as Promise<BeliefExport>;
  }
}
