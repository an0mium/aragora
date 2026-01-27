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
}
