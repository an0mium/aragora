/**
 * Belief Network Namespace API
 *
 * Provides access to debate belief networks, cruxes, and load-bearing claims.
 * These represent the epistemic structure of arguments within debates.
 */

/**
 * A crux - a key disagreement point that, if resolved, would change positions
 */
export interface Crux {
  id: string;
  debate_id: string;
  description: string;
  agents_pro: string[];
  agents_con: string[];
  importance: number;
  resolved: boolean;
  resolution?: string;
  identified_at: string;
}

/**
 * A load-bearing claim - a claim that many other claims depend on
 */
export interface LoadBearingClaim {
  id: string;
  debate_id: string;
  claim: string;
  agent_name: string;
  confidence: number;
  dependents_count: number;
  dependent_claims: string[];
  supporting_evidence: string[];
  created_at: string;
}

/**
 * Belief network node
 */
export interface BeliefNode {
  id: string;
  claim: string;
  agent_name: string;
  confidence: number;
  type: 'premise' | 'inference' | 'conclusion';
  supporting_nodes: string[];
  opposing_nodes: string[];
}

/**
 * Belief network edge
 */
export interface BeliefEdge {
  source_id: string;
  target_id: string;
  relationship: 'supports' | 'contradicts' | 'qualifies' | 'depends_on';
  strength: number;
}

/**
 * Belief network graph node for visualization
 */
export interface BeliefGraphNode {
  id: string;
  claim_id: string;
  statement: string;
  author: string;
  centrality: number;
  is_crux: boolean;
  crux_score: number | null;
  entropy: number;
  belief: {
    true_prob: number;
    false_prob: number;
    uncertain_prob: number;
  } | null;
}

/**
 * Belief network graph edge for visualization
 */
export interface BeliefGraphEdge {
  source: string;
  target: string;
  weight: number;
  type: 'supports' | 'contradicts' | 'qualifies' | 'depends_on' | 'influences';
}

/**
 * Belief network graph response
 */
export interface BeliefGraphResponse {
  nodes: BeliefGraphNode[];
  links: BeliefGraphEdge[];
  metadata: {
    debate_id: string;
    total_claims: number;
    crux_count: number;
  };
}

/**
 * Export format types
 */
export type ExportFormat = 'json' | 'graphml' | 'csv';

/**
 * JSON export response
 */
export interface JsonExportResponse {
  format: 'json';
  debate_id: string;
  nodes: Array<{
    id: string;
    statement: string;
    author: string;
    centrality: number;
    is_crux: boolean;
  }>;
  edges: Array<{
    source: string;
    target: string;
    weight: number;
    type: string;
  }>;
  summary: {
    total_nodes: number;
    total_edges: number;
    crux_count: number;
  };
}

/**
 * GraphML export response
 */
export interface GraphMLExportResponse {
  format: 'graphml';
  debate_id: string;
  content: string;
  content_type: 'application/xml';
}

/**
 * CSV export response
 */
export interface CSVExportResponse {
  format: 'csv';
  debate_id: string;
  nodes_csv: Array<Record<string, unknown>>;
  edges_csv: Array<Record<string, unknown>>;
  headers: {
    nodes: string[];
    edges: string[];
  };
}

/**
 * Union type for export responses
 */
export type ExportResponse = JsonExportResponse | GraphMLExportResponse | CSVExportResponse;

/**
 * Interface for the internal client used by BeliefNetworkAPI.
 */
interface BeliefNetworkClientInterface {
  get<T>(path: string): Promise<T>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<T>;
}

/**
 * Belief Network API namespace.
 *
 * Provides methods for analyzing debate belief structures:
 * - Identify cruxes (key disagreements)
 * - Find load-bearing claims
 * - Explore argument dependencies
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get cruxes for a debate
 * const { cruxes } = await client.beliefNetwork.getCruxes('debate-123');
 *
 * // Get load-bearing claims
 * const { claims } = await client.beliefNetwork.getLoadBearingClaims('debate-123');
 * ```
 */
export class BeliefNetworkAPI {
  constructor(private client: BeliefNetworkClientInterface) {}

  /**
   * Get cruxes (key disagreements) for a debate.
   *
   * Cruxes are points where, if one party changed their mind,
   * it would significantly affect their overall position.
   */
  async getCruxes(debateId: string, options?: { resolved?: boolean; min_importance?: number }): Promise<{ cruxes: Crux[]; total: number }> {
    return this.client.request('GET', `/api/v1/belief-network/${debateId}/cruxes`, { params: options });
  }

  /**
   * Get load-bearing claims for a debate.
   *
   * Load-bearing claims are claims that many other claims depend on.
   * If these are invalidated, the overall argument structure may collapse.
   */
  async getLoadBearingClaims(debateId: string, options?: { min_dependents?: number; agent?: string }): Promise<{ claims: LoadBearingClaim[]; total: number }> {
    return this.client.request('GET', `/api/v1/belief-network/${debateId}/load-bearing-claims`, { params: options });
  }

  /**
   * Get belief network as a graph structure for visualization.
   *
   * Returns nodes (claims) and links (influence relationships) suitable
   * for force-directed graph rendering.
   *
   * @param debateId - The debate ID
   * @param options.include_cruxes - Include crux analysis in nodes (default: true)
   *
   * @example
   * ```typescript
   * const graph = await client.beliefNetwork.getGraph('debate-123');
   * console.log(`${graph.nodes.length} claims, ${graph.links.length} relationships`);
   * console.log(`${graph.metadata.crux_count} cruxes identified`);
   * ```
   */
  async getGraph(debateId: string, options?: { include_cruxes?: boolean }): Promise<BeliefGraphResponse> {
    return this.client.request('GET', `/api/v1/belief-network/${debateId}/graph`, { params: options });
  }

  /**
   * Export belief network in various formats.
   *
   * Supported formats:
   * - json: Full JSON structure with nodes, edges, and summary
   * - graphml: GraphML format for Gephi/yEd visualization tools
   * - csv: CSV-friendly structure with separate nodes and edges arrays
   *
   * @param debateId - The debate ID
   * @param format - Export format ('json' | 'graphml' | 'csv')
   *
   * @example
   * ```typescript
   * // Export as GraphML for Gephi
   * const graphml = await client.beliefNetwork.export('debate-123', 'graphml');
   * console.log(graphml.content); // XML content
   *
   * // Export as JSON
   * const json = await client.beliefNetwork.export('debate-123', 'json');
   * console.log(`${json.summary.total_nodes} nodes`);
   * ```
   */
  async export(debateId: string, format: ExportFormat = 'json'): Promise<ExportResponse> {
    return this.client.request('GET', `/api/v1/belief-network/${debateId}/export`, { params: { format } });
  }
}
