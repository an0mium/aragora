/**
 * Explainability Namespace API
 *
 * Provides a namespaced interface for decision explainability.
 * Essential for understanding AI decisions and building trust.
 */

import type {
  ExplainabilityResult,
  ExplanationFactor,
  CounterfactualScenario,
  CounterfactualList,
  CounterfactualGeneration,
  Narrative,
  Provenance,
} from '../types';

/**
 * Interface for the internal client methods used by ExplainabilityAPI.
 */
interface ExplainabilityClientInterface {
  getExplanation(debateId: string, options?: {
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
  }): Promise<ExplainabilityResult>;
  getExplanationFactors(debateId: string, options?: {
    min_contribution?: number;
  }): Promise<{ factors: ExplanationFactor[] }>;
  getCounterfactuals(debateId: string, options?: {
    max_scenarios?: number;
  }): Promise<CounterfactualList>;
  generateCounterfactual(debateId: string, body: {
    hypothesis: string;
    affected_agents?: string[];
  }): Promise<CounterfactualGeneration>;
  getProvenance(debateId: string): Promise<Provenance>;
  getNarrative(debateId: string, options?: {
    format?: 'brief' | 'detailed' | 'executive_summary';
  }): Promise<Narrative>;
  createBatchExplanation(body: {
    debate_ids: string[];
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
    include_narrative?: boolean;
  }): Promise<{ batch_id: string; status_url: string }>;
  getBatchExplanationStatus(batchId: string): Promise<{
    batch_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress_pct: number;
  }>;
  getBatchExplanationResults(batchId: string, params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ results: Array<{ debate_id: string; status: string; explanation?: unknown }> }>;
}

/**
 * Explainability API namespace.
 *
 * Provides methods for understanding AI decisions:
 * - Get decision factors and their weights
 * - Generate counterfactual scenarios ("what if")
 * - Trace decision provenance
 * - Get human-readable narratives
 * - Batch process multiple debates
 *
 * Essential for building trust, compliance, and informed decision-making.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get full explanation for a debate
 * const explanation = await client.explainability.get('debate-123', {
 *   include_factors: true,
 *   include_counterfactuals: true
 * });
 *
 * // Get just the key factors
 * const { factors } = await client.explainability.getFactors('debate-123', {
 *   min_weight: 0.1
 * });
 *
 * // Get executive-friendly narrative
 * const narrative = await client.explainability.getNarrative('debate-123', {
 *   audience: 'executive'
 * });
 * console.log(narrative);
 *
 * // Generate "what if" scenario
 * const counterfactual = await client.explainability.generateCounterfactual('debate-123', {
 *   changes: ['Remove security agent', 'Add compliance requirement']
 * });
 * ```
 */
export class ExplainabilityAPI {
  constructor(private client: ExplainabilityClientInterface) {}

  // ===========================================================================
  // Core Explainability
  // ===========================================================================

  /**
   * Get full explanation for a debate decision.
   *
   * @param debateId - The debate ID to explain
   * @param options - What to include in the explanation
   */
  async get(debateId: string, options?: {
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
  }): Promise<ExplainabilityResult> {
    return this.client.getExplanation(debateId, options);
  }

  /**
   * Get the factors that influenced a decision.
   *
   * Factors represent the key considerations that affected the outcome,
   * with contributions indicating their relative importance.
   *
   * @example
   * ```typescript
   * const { factors } = await client.explainability.getFactors('debate-123');
   * for (const factor of factors) {
   *   console.log(`${factor.name}: ${factor.contribution > 0 ? '+' : ''}${factor.contribution}`);
   * }
   * ```
   */
  async getFactors(debateId: string, options?: {
    min_contribution?: number;
  }): Promise<{ factors: ExplanationFactor[] }> {
    return this.client.getExplanationFactors(debateId, options);
  }

  // ===========================================================================
  // Counterfactuals
  // ===========================================================================

  /**
   * Get existing counterfactual scenarios for a debate.
   *
   * @param options.max_scenarios - Maximum number of scenarios to return
   */
  async getCounterfactuals(debateId: string, options?: {
    max_scenarios?: number;
  }): Promise<CounterfactualList> {
    return this.client.getCounterfactuals(debateId, options);
  }

  /**
   * Generate a new counterfactual scenario.
   *
   * Counterfactuals help understand how changes would affect the decision.
   *
   * @example
   * ```typescript
   * const counterfactual = await client.explainability.generateCounterfactual('debate-123', {
   *   hypothesis: 'What if we had a larger budget?',
   *   affected_agents: ['gpt-4', 'claude']
   * });
   * console.log('Predicted outcome:', counterfactual.predicted_outcome);
   * ```
   */
  async generateCounterfactual(debateId: string, body: {
    hypothesis: string;
    affected_agents?: string[];
  }): Promise<CounterfactualGeneration> {
    return this.client.generateCounterfactual(debateId, body);
  }

  // ===========================================================================
  // Provenance
  // ===========================================================================

  /**
   * Get the provenance (source tracking) for a decision.
   *
   * Traces where information came from and how reasoning progressed.
   */
  async getProvenance(debateId: string): Promise<Provenance> {
    return this.client.getProvenance(debateId);
  }

  // ===========================================================================
  // Narrative
  // ===========================================================================

  /**
   * Get a human-readable narrative explanation.
   *
   * @param debateId - The debate ID
   * @param options.format - Format of the narrative:
   *   - brief: Short summary
   *   - detailed: Full explanation
   *   - executive_summary: High-level executive summary
   */
  async getNarrative(debateId: string, options?: {
    format?: 'brief' | 'detailed' | 'executive_summary';
  }): Promise<Narrative> {
    return this.client.getNarrative(debateId, options);
  }

  // ===========================================================================
  // Batch Processing
  // ===========================================================================

  /**
   * Create a batch explanation request for multiple debates.
   *
   * Useful for generating reports across many decisions.
   *
   * @example
   * ```typescript
   * const { batch_id } = await client.explainability.createBatch({
   *   debate_ids: ['debate-1', 'debate-2', 'debate-3'],
   *   include_factors: true,
   *   include_narrative: true
   * });
   *
   * // Poll for completion
   * let status = await client.explainability.getBatchStatus(batch_id);
   * while (status.status === 'processing') {
   *   await sleep(1000);
   *   status = await client.explainability.getBatchStatus(batch_id);
   * }
   *
   * // Get results
   * const { results } = await client.explainability.getBatchResults(batch_id);
   * ```
   */
  async createBatch(request: {
    debate_ids: string[];
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
    include_narrative?: boolean;
  }): Promise<{ batch_id: string; status_url: string }> {
    return this.client.createBatchExplanation(request);
  }

  /**
   * Get the status of a batch explanation request.
   */
  async getBatchStatus(batchId: string): Promise<{
    batch_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress_pct: number;
  }> {
    return this.client.getBatchExplanationStatus(batchId);
  }

  /**
   * Get the results of a completed batch explanation.
   */
  async getBatchResults(batchId: string, params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ results: Array<{ debate_id: string; status: string; explanation?: unknown }> }> {
    return this.client.getBatchExplanationResults(batchId, params);
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  /**
   * Get the top N positive factors from an explanation.
   */
  getTopPositiveFactors(result: ExplainabilityResult, n: number = 5): ExplanationFactor[] {
    return (result.factors ?? [])
      .filter(f => f.contribution > 0)
      .sort((a, b) => b.contribution - a.contribution)
      .slice(0, n);
  }

  /**
   * Get the top N negative factors from an explanation.
   */
  getTopNegativeFactors(result: ExplainabilityResult, n: number = 5): ExplanationFactor[] {
    return (result.factors ?? [])
      .filter(f => f.contribution < 0)
      .sort((a, b) => a.contribution - b.contribution)
      .slice(0, n);
  }
}

// Re-export types for convenience
export type { ExplainabilityResult, ExplanationFactor, CounterfactualScenario, Narrative, Provenance };
