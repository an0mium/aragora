/**
 * Evolution Namespace API
 *
 * Provides access to agent prompt evolution, patterns, A/B testing, and summaries.
 */

export interface EvolutionPattern {
  id: string;
  pattern_type: string;
  description: string;
  frequency: number;
  impact_score?: number;
  agents?: string[];
  metadata?: Record<string, unknown>;
}

export interface EvolutionSummary {
  total_agents: number;
  total_evolutions: number;
  top_patterns: EvolutionPattern[];
  improvement_rate?: number;
  metadata?: Record<string, unknown>;
}

export interface EvolutionHistoryEntry {
  timestamp: string;
  version?: number;
  score?: number;
  changes?: Record<string, unknown>;
  trigger?: string;
}

export interface EvolutionHistoryResponse {
  agent: string;
  history: EvolutionHistoryEntry[];
  total: number;
}

export interface PromptVersion {
  agent: string;
  version: number;
  prompt: string;
  created_at: string;
  performance_score?: number;
  metadata?: Record<string, unknown>;
}

export interface ABTest {
  id: string;
  agent: string;
  variant_a: string;
  variant_b: string;
  status: 'active' | 'completed' | 'cancelled';
  started_at: string;
  completed_at?: string;
  results?: {
    variant_a_score: number;
    variant_b_score: number;
    winner?: 'a' | 'b' | 'tie';
    sample_size: number;
    confidence: number;
  };
}

export interface CreateABTestRequest {
  agent: string;
  variant_a: string;
  variant_b: string;
  config?: Record<string, unknown>;
}

interface EvolutionClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

export class EvolutionAPI {
  constructor(private client: EvolutionClientInterface) {}

  /**
   * Get top evolution patterns across all agents.
   */
  async getPatterns(params?: {
    limit?: number;
  }): Promise<{ patterns: EvolutionPattern[] }> {
    return this.client.request('GET', '/api/v1/evolution/patterns', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get evolution summary statistics.
   */
  async getSummary(): Promise<EvolutionSummary> {
    return this.client.request('GET', '/api/v1/evolution/summary');
  }

  /**
   * Get prompt evolution history for an agent.
   */
  async getHistory(agent: string): Promise<EvolutionHistoryResponse> {
    return this.client.request('GET', `/api/v1/evolution/${agent}/history`);
  }

  /**
   * Get the current or a specific prompt version for an agent.
   */
  async getPrompt(agent: string, params?: {
    version?: number;
  }): Promise<PromptVersion> {
    return this.client.request('GET', `/api/v1/evolution/${agent}/prompt`, {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * List all A/B tests.
   */
  async listABTests(params?: {
    limit?: number;
    status?: string;
  }): Promise<{ tests: ABTest[]; total?: number }> {
    return this.client.request('POST', '/api/v1/evolution/ab-tests', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get the active A/B test for an agent.
   */
  async getActiveABTest(agent: string): Promise<ABTest | null> {
    return this.client.request('GET', `/api/v1/evolution/ab-tests/${agent}/active`);
  }

  /**
   * Start a new A/B test.
   */
  async createABTest(body: CreateABTestRequest): Promise<ABTest> {
    return this.client.request('POST', '/api/v1/evolution/ab-tests', { body });
  }

  /**
   * Get a specific A/B test by ID.
   */
  async getABTest(testId: string): Promise<ABTest> {
    return this.client.request('GET', `/api/v1/evolution/ab-tests/${testId}`);
  }

  /** Get evolution overview. */
  async getOverview(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/evolution');
  }
}
