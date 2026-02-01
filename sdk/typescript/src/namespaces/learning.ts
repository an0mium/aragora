/**
 * Learning Namespace API
 *
 * Provides access to learning cycles, patterns, agent evolution, and insights.
 */

export interface LearningCycle {
  cycle_id: string;
  started_at: string;
  completed_at?: string;
  debates_analyzed: number;
  patterns_found: number;
  summary?: string;
}

export interface LearnedPattern {
  id: string;
  pattern_type: string;
  description: string;
  confidence: number;
  frequency: number;
  first_seen: string;
  last_seen: string;
  metadata?: Record<string, unknown>;
}

export interface AgentEvolution {
  agent: string;
  timeline: Array<{
    timestamp: string;
    elo_rating?: number;
    calibration_score?: number;
    debate_count?: number;
    win_rate?: number;
  }>;
  trend?: 'improving' | 'stable' | 'declining';
}

export interface LearningInsight {
  id: string;
  type: string;
  description: string;
  confidence: number;
  source_cycles: string[];
  created_at: string;
}

export interface LearningEvolutionResponse {
  summary?: Record<string, unknown>;
  timeline?: Array<Record<string, unknown>>;
  patterns?: LearnedPattern[];
}

interface LearningClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown> }
  ): Promise<T>;
}

export class LearningAPI {
  constructor(private client: LearningClientInterface) {}

  /**
   * Get all learning cycle summaries.
   */
  async getCycles(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ cycles: LearningCycle[]; total?: number }> {
    return this.client.request('GET', '/api/v1/learning/cycles', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get learned patterns across cycles.
   */
  async getPatterns(params?: {
    pattern_type?: string;
    min_confidence?: number;
    limit?: number;
  }): Promise<{ patterns: LearnedPattern[]; total?: number }> {
    return this.client.request('GET', '/api/v1/learning/patterns', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get agent performance evolution over time.
   */
  async getAgentEvolution(params?: {
    agents?: string[];
    period?: string;
  }): Promise<{ agents: AgentEvolution[] }> {
    return this.client.request('GET', '/api/v1/learning/agent-evolution', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get aggregated insights from learning cycles.
   */
  async getInsights(params?: {
    limit?: number;
  }): Promise<{ insights: LearningInsight[] }> {
    return this.client.request('GET', '/api/v1/learning/insights', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get meta-learning evolution patterns.
   */
  async getEvolution(): Promise<LearningEvolutionResponse> {
    return this.client.request('GET', '/api/v1/learning/evolution');
  }
}
