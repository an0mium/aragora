/**
 * Laboratory Namespace API
 *
 * Provides access to the persona laboratory for detecting emergent traits
 * and suggesting beneficial cross-pollinations between agents.
 */

/**
 * An emergent trait detected from agent performance.
 */
export interface EmergentTrait {
  agent: string;
  trait: string;
  domain: string;
  confidence: number;
  evidence: string[];
  detected_at: string;
}

/**
 * A suggested trait transfer between agents.
 */
export interface CrossPollinationSuggestion {
  source_agent: string;
  trait_or_domain: string;
  reason: string;
}

/**
 * Response for emergent traits listing.
 */
export interface EmergentTraitsResponse {
  emergent_traits: EmergentTrait[];
  count: number;
  min_confidence: number;
}

/**
 * Response for cross-pollination suggestions.
 */
export interface CrossPollinationResponse {
  target_agent: string;
  suggestions: CrossPollinationSuggestion[];
  count: number;
}

/**
 * Options for listing emergent traits.
 */
export interface EmergentTraitsOptions {
  min_confidence?: number;
  limit?: number;
}

/**
 * Request for cross-pollination suggestions.
 */
export interface CrossPollinationRequest {
  target_agent: string;
}

/**
 * Interface for the internal client methods used by LaboratoryAPI.
 */
interface LaboratoryClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Laboratory API namespace.
 *
 * Provides access to persona laboratory features for detecting
 * emergent traits and suggesting beneficial cross-pollinations.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get emergent traits with high confidence
 * const traits = await client.laboratory.getEmergentTraits({
 *   min_confidence: 0.8,
 *   limit: 10
 * });
 * for (const trait of traits.emergent_traits) {
 *   console.log(`${trait.agent}: ${trait.trait} (${trait.confidence})`);
 * }
 *
 * // Get cross-pollination suggestions
 * const suggestions = await client.laboratory.suggestCrossPollinations({
 *   target_agent: 'claude'
 * });
 * for (const s of suggestions.suggestions) {
 *   console.log(`Transfer ${s.trait_or_domain} from ${s.source_agent}`);
 * }
 * ```
 */
export class LaboratoryAPI {
  constructor(private client: LaboratoryClientInterface) {}

  /**
   * Get emergent traits detected from agent performance.
   */
  async getEmergentTraits(options?: EmergentTraitsOptions): Promise<EmergentTraitsResponse> {
    return this.client.request('GET', '/api/v2/laboratory/emergent-traits', {
      params: options as Record<string, unknown>,
    });
  }

  /**
   * Suggest beneficial trait transfers for a target agent.
   */
  async suggestCrossPollinations(request: CrossPollinationRequest): Promise<CrossPollinationResponse> {
    return this.client.request('POST', '/api/v2/laboratory/cross-pollinations', {
      json: request as unknown as Record<string, unknown>,
    });
  }
}
