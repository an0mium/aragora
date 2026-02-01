/**
 * Uncertainty Namespace API
 *
 * Provides uncertainty estimation, agent calibration profiles, and follow-up generation.
 */

export interface UncertaintyEstimateRequest {
  content: string;
  context?: string;
  debate_id?: string;
  config?: Record<string, unknown>;
}

export interface UncertaintyEstimate {
  overall_uncertainty: number;
  confidence_interval?: [number, number];
  cruxes?: Array<{
    description: string;
    importance: number;
    uncertainty: number;
  }>;
  methodology?: string;
  metadata?: Record<string, unknown>;
}

export interface DebateUncertaintyMetrics {
  debate_id: string;
  overall_uncertainty: number;
  round_uncertainties?: Array<{
    round: number;
    uncertainty: number;
  }>;
  agent_uncertainties?: Record<string, number>;
  convergence_trend?: number[];
  cruxes?: Array<{
    description: string;
    importance: number;
  }>;
}

export interface AgentCalibrationProfile {
  agent_id: string;
  calibration_score: number;
  overconfidence_bias?: number;
  accuracy_by_confidence?: Array<{
    confidence_bucket: number;
    actual_accuracy: number;
    sample_count: number;
  }>;
  total_predictions?: number;
}

export interface FollowUpRequest {
  debate_id?: string;
  cruxes?: Array<{ description: string; importance: number }>;
  max_suggestions?: number;
}

export interface FollowUpSuggestion {
  question: string;
  rationale: string;
  priority: number;
  related_crux?: string;
}

interface UncertaintyClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

export class UncertaintyAPI {
  constructor(private client: UncertaintyClientInterface) {}

  /**
   * Estimate uncertainty for content or a debate response.
   */
  async estimate(body: UncertaintyEstimateRequest): Promise<UncertaintyEstimate> {
    return this.client.request('POST', '/api/v1/uncertainty/estimate', { body });
  }

  /**
   * Get uncertainty metrics for a debate.
   */
  async getDebateMetrics(debateId: string): Promise<DebateUncertaintyMetrics> {
    return this.client.request('GET', `/api/v1/uncertainty/debate/${debateId}`);
  }

  /**
   * Get calibration profile for an agent.
   */
  async getAgentProfile(agentId: string): Promise<AgentCalibrationProfile> {
    return this.client.request('GET', `/api/v1/uncertainty/agent/${agentId}`);
  }

  /**
   * Generate follow-up suggestions from uncertainty cruxes.
   */
  async generateFollowUps(body: FollowUpRequest): Promise<{ suggestions: FollowUpSuggestion[] }> {
    return this.client.request('POST', '/api/v1/uncertainty/followups', { body });
  }
}
