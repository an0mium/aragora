/**
 * Learning Namespace API
 *
 * Provides access to autonomous learning and meta-learning analytics,
 * including learning patterns, efficiency metrics, and agent evolution.
 */

/**
 * Learning session summary.
 */
export interface LearningSession {
  session_id: string;
  agent_name: string;
  domain: string;
  started_at: string;
  completed_at?: string;
  status: 'active' | 'completed' | 'failed';
  examples_processed: number;
  accuracy_improvement: number;
  insights_generated: number;
}

/**
 * Meta-learning statistics.
 */
export interface MetaLearningStats {
  total_sessions: number;
  active_sessions: number;
  completed_sessions: number;
  average_accuracy_improvement: number;
  top_learning_agents: Array<{
    agent_name: string;
    sessions: number;
    avg_improvement: number;
  }>;
  learning_trends: Array<{
    date: string;
    sessions: number;
    avg_improvement: number;
  }>;
}

/**
 * Learning pattern detected across agents.
 */
export interface LearningPattern {
  pattern_id: string;
  name: string;
  description: string;
  agents: string[];
  domains: string[];
  confidence: number;
  discovered_at: string;
}

/**
 * Learning efficiency metrics.
 */
export interface LearningEfficiency {
  agent_name: string;
  domain: string;
  examples_per_insight: number;
  time_to_improvement: number;
  retention_score: number;
  transfer_capability: number;
}

/**
 * Options for listing learning sessions.
 */
export interface ListSessionsOptions {
  agent?: string;
  domain?: string;
  status?: 'active' | 'completed' | 'failed';
  limit?: number;
  offset?: number;
}

/**
 * Options for listing learning patterns.
 */
export interface ListPatternsOptions {
  min_confidence?: number;
  limit?: number;
}

/**
 * Interface for the internal client methods used by LearningAPI.
 */
interface LearningClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Learning API namespace.
 *
 * Provides access to autonomous learning and meta-learning analytics.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get meta-learning statistics
 * const stats = await client.learning.getStats();
 * console.log(`${stats.active_sessions} active learning sessions`);
 *
 * // Get learning patterns
 * const patterns = await client.learning.listPatterns({
 *   min_confidence: 0.8
 * });
 * for (const pattern of patterns.patterns) {
 *   console.log(`${pattern.name}: ${pattern.confidence}`);
 * }
 *
 * // Get agent efficiency
 * const efficiency = await client.learning.getEfficiency('claude', 'coding');
 * console.log(`Examples per insight: ${efficiency[0].examples_per_insight}`);
 * ```
 */
export class LearningAPI {
  constructor(private client: LearningClientInterface) {}

  /**
   * Get meta-learning statistics across all agents.
   */
  async getStats(): Promise<MetaLearningStats> {
    return this.client.request('GET', '/api/v2/learning/stats');
  }

  /**
   * List learning sessions with optional filtering.
   */
  async listSessions(options?: ListSessionsOptions): Promise<{ sessions: LearningSession[]; total: number }> {
    return this.client.request('GET', '/api/v2/learning/sessions', {
      params: options as Record<string, unknown>,
    });
  }

  /**
   * Get a specific learning session by ID.
   */
  async getSession(sessionId: string): Promise<LearningSession> {
    return this.client.request('GET', `/api/v2/learning/sessions/${sessionId}`);
  }

  /**
   * List detected learning patterns.
   */
  async listPatterns(options?: ListPatternsOptions): Promise<{ patterns: LearningPattern[]; count: number }> {
    return this.client.request('GET', '/api/v2/learning/patterns', {
      params: options as Record<string, unknown>,
    });
  }

  /**
   * Get learning efficiency metrics for an agent.
   */
  async getEfficiency(agentName: string, domain?: string): Promise<LearningEfficiency[]> {
    return this.client.request('GET', `/api/v2/learning/efficiency/${agentName}`, {
      params: domain !== undefined ? { domain } : undefined,
    });
  }
}
