/**
 * Evaluation Namespace API
 *
 * Provides a namespaced interface for LLM-as-Judge evaluation operations.
 * This wraps the flat client methods for quality scoring and comparison.
 */

/**
 * Evaluation dimension for scoring.
 */
export interface EvaluationDimension {
  id: string;
  name: string;
  description: string;
  weight: number;
  criteria: string[];
}

/**
 * Weight profile for evaluation.
 */
export interface EvaluationProfile {
  id: string;
  name: string;
  description: string;
  dimensions: Record<string, number>;
  default: boolean;
}

/**
 * Single evaluation request.
 */
export interface EvaluateRequest {
  response: string;
  prompt?: string;
  context?: string;
  dimensions?: string[];
  profile?: string;
  reference?: string;
}

/**
 * Evaluation result with scores.
 */
export interface EvaluationResult {
  overall_score: number;
  dimension_scores: Record<string, number>;
  feedback: string;
  strengths: string[];
  weaknesses: string[];
  suggestions: string[];
  profile_used: string;
}

/**
 * Comparison request between two responses.
 */
export interface CompareRequest {
  response_a: string;
  response_b: string;
  prompt?: string;
  context?: string;
  dimensions?: string[];
  profile?: string;
}

/**
 * Comparison result.
 */
export interface ComparisonResult {
  winner: 'A' | 'B' | 'tie';
  margin: number;
  response_a_score: number;
  response_b_score: number;
  dimension_comparison: Record<string, { a: number; b: number; winner: string }>;
  reasoning: string;
}

/**
 * Interface for the internal client methods used by EvaluationAPI.
 */
interface EvaluationClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: unknown }
  ): Promise<T>;
}

/**
 * Evaluation API namespace.
 *
 * Provides methods for LLM-as-Judge evaluation:
 * - Evaluating responses against quality dimensions
 * - Comparing two responses head-to-head
 * - Managing evaluation dimensions and profiles
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Evaluate a response
 * const result = await client.evaluation.evaluate({
 *   response: 'The answer to your question is...',
 *   prompt: 'What is the capital of France?',
 *   dimensions: ['accuracy', 'clarity', 'completeness'],
 * });
 *
 * // Compare two responses
 * const comparison = await client.evaluation.compare({
 *   response_a: 'First response...',
 *   response_b: 'Second response...',
 *   prompt: 'Original prompt',
 * });
 *
 * // List available dimensions
 * const dimensions = await client.evaluation.listDimensions();
 * ```
 */
export class EvaluationAPI {
  constructor(private client: EvaluationClientInterface) {}

  /**
   * Evaluate a single response for quality.
   */
  async evaluate(request: EvaluateRequest): Promise<EvaluationResult> {
    return this.client.request<EvaluationResult>('POST', '/api/v1/evaluate', {
      json: request,
    });
  }

  /**
   * Compare two responses head-to-head.
   */
  async compare(request: CompareRequest): Promise<ComparisonResult> {
    return this.client.request<ComparisonResult>('POST', '/api/v1/evaluate/compare', {
      json: request,
    });
  }

  /**
   * List available evaluation dimensions.
   */
  async listDimensions(): Promise<{ dimensions: EvaluationDimension[] }> {
    return this.client.request<{ dimensions: EvaluationDimension[] }>('POST', '/api/v1/evaluate/dimensions');
  }

  /**
   * List available evaluation profiles.
   */
  async listProfiles(): Promise<{ profiles: EvaluationProfile[] }> {
    return this.client.request<{ profiles: EvaluationProfile[] }>('POST', '/api/v1/evaluate/profiles');
  }

  /**
   * Get a specific evaluation dimension by ID.
   */
  async getDimension(dimensionId: string): Promise<EvaluationDimension> {
    const { dimensions } = await this.listDimensions();
    const dimension = dimensions.find(d => d.id === dimensionId);
    if (!dimension) {
      throw new Error(`Dimension not found: ${dimensionId}`);
    }
    return dimension;
  }

  /**
   * Get a specific evaluation profile by ID.
   */
  async getProfile(profileId: string): Promise<EvaluationProfile> {
    const { profiles } = await this.listProfiles();
    const profile = profiles.find(p => p.id === profileId);
    if (!profile) {
      throw new Error(`Profile not found: ${profileId}`);
    }
    return profile;
  }
}

export default EvaluationAPI;
