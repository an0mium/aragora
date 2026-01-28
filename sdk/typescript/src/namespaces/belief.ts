/**
 * Belief Network Namespace API
 *
 * Provides access to belief network analysis, cruxes, and provenance tracking.
 * The belief network represents claims and their supporting/opposing relationships.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get key cruxes (claims that would most impact outcome)
 * const cruxes = await client.belief.getCruxes('debate-123');
 *
 * // Get belief network as a graph
 * const graph = await client.belief.getGraph('debate-123');
 *
 * // Export belief network
 * const exported = await client.belief.export('debate-123', { format: 'graphml' });
 * ```
 */

/**
 * Interface for belief client methods.
 */
interface BeliefClientInterface {
  getBeliefCruxes(debateId: string, params?: { top_k?: number }): Promise<BeliefCruxes>;
  getLoadBearingClaims(debateId: string, params?: { limit?: number }): Promise<LoadBearingClaims>;
  getBeliefGraph(debateId: string, params?: { include_cruxes?: boolean }): Promise<BeliefGraph>;
  exportBeliefNetwork(debateId: string, params?: { format?: ExportFormat }): Promise<BeliefExport>;
  getClaimSupport(debateId: string, claimId: string): Promise<ClaimSupport>;
  getClaimProvenance(debateId: string, claimId: string): Promise<ClaimProvenance>;
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
 * Support/opposition for a claim.
 */
export interface ClaimSupport {
  claim_id: string;
  supporting_claims: Array<{ id: string; text: string; strength: number }>;
  opposing_claims: Array<{ id: string; text: string; strength: number }>;
  net_support: number;
}

/**
 * Provenance chain for a claim.
 */
export interface ClaimProvenance {
  claim_id: string;
  origin: {
    agent: string;
    round: number;
    timestamp: string;
  };
  modifications: Array<{
    agent: string;
    round: number;
    action: 'refined' | 'challenged' | 'supported';
    timestamp: string;
  }>;
}

/**
 * Belief Network API.
 *
 * Provides methods for analyzing belief networks in debates:
 * - Finding cruxes (key claims that would change outcomes)
 * - Identifying load-bearing claims (high centrality)
 * - Exporting graph structures
 * - Tracking claim provenance
 */
export class BeliefAPI {
  constructor(private client: BeliefClientInterface) {}

  /**
   * Get key claims that would most impact the debate outcome.
   *
   * @param debateId - The debate ID
   * @param options - Query options
   * @param options.topK - Number of top cruxes to return (1-10, default: 3)
   */
  async getCruxes(
    debateId: string,
    options?: { topK?: number }
  ): Promise<BeliefCruxes> {
    const params = options?.topK ? { top_k: options.topK } : undefined;
    return this.client.getBeliefCruxes(debateId, params);
  }

  /**
   * Get claims with highest centrality (most load-bearing).
   *
   * @param debateId - The debate ID
   * @param options - Query options
   * @param options.limit - Maximum claims to return (1-20, default: 5)
   */
  async getLoadBearingClaims(
    debateId: string,
    options?: { limit?: number }
  ): Promise<LoadBearingClaims> {
    const params = options?.limit ? { limit: options.limit } : undefined;
    return this.client.getLoadBearingClaims(debateId, params);
  }

  /**
   * Get belief network as a graph structure for visualization.
   *
   * @param debateId - The debate ID
   * @param options - Query options
   * @param options.includeCruxes - Whether to include crux detection (default: true)
   */
  async getGraph(
    debateId: string,
    options?: { includeCruxes?: boolean }
  ): Promise<BeliefGraph> {
    const params = options?.includeCruxes !== undefined
      ? { include_cruxes: options.includeCruxes }
      : undefined;
    return this.client.getBeliefGraph(debateId, params);
  }

  /**
   * Export belief network in various formats.
   *
   * @param debateId - The debate ID
   * @param options - Export options
   * @param options.format - Export format (json, graphml, csv)
   */
  async export(
    debateId: string,
    options?: { format?: ExportFormat }
  ): Promise<BeliefExport> {
    return this.client.exportBeliefNetwork(debateId, options);
  }

  /**
   * Get support and opposition for a specific claim.
   *
   * @param debateId - The debate ID
   * @param claimId - The claim ID
   */
  async getClaimSupport(debateId: string, claimId: string): Promise<ClaimSupport> {
    return this.client.getClaimSupport(debateId, claimId);
  }

  /**
   * Get provenance chain for a claim (origin and modifications).
   *
   * @param debateId - The debate ID
   * @param claimId - The claim ID
   */
  async getClaimProvenance(debateId: string, claimId: string): Promise<ClaimProvenance> {
    return this.client.getClaimProvenance(debateId, claimId);
  }
}
