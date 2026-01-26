/**
 * Consensus Namespace API
 *
 * Provides a namespaced interface for consensus detection and management.
 * Consensus tracks settled topics, dissenting views, and historical debate outcomes.
 */

/**
 * Similar debate result.
 */
export interface SimilarDebate {
  topic: string;
  conclusion: string;
  strength: string;
  confidence: number;
  similarity: number;
  agents: string[];
  dissent_count: number;
  timestamp: string;
}

/**
 * Response for finding similar debates.
 */
export interface SimilarDebatesResponse {
  query: string;
  similar: SimilarDebate[];
  count: number;
}

/**
 * Settled topic.
 */
export interface SettledTopic {
  topic: string;
  conclusion: string;
  confidence: number;
  strength: string;
  timestamp: string;
}

/**
 * Response for settled topics.
 */
export interface SettledTopicsResponse {
  min_confidence: number;
  topics: SettledTopic[];
  count: number;
}

/**
 * Consensus statistics.
 */
export interface ConsensusStats {
  total_topics: number;
  high_confidence_count: number;
  domains: string[];
  avg_confidence: number;
  total_dissents: number;
  by_strength: Record<string, number>;
  by_domain: Record<string, number>;
}

/**
 * Dissenting view.
 */
export interface DissentView {
  id: string;
  topic: string;
  agent: string;
  position: string;
  reasoning: string;
  timestamp: string;
}

/**
 * Response for dissenting views.
 */
export interface DissentsResponse {
  dissents: DissentView[];
  count: number;
}

/**
 * Contrarian view.
 */
export interface ContrarianView {
  id: string;
  topic: string;
  agent: string;
  perspective: string;
  confidence: number;
  timestamp: string;
}

/**
 * Response for contrarian views.
 */
export interface ContrariansResponse {
  views: ContrarianView[];
  count: number;
}

/**
 * Risk warning.
 */
export interface RiskWarning {
  id: string;
  topic: string;
  warning: string;
  severity: 'low' | 'medium' | 'high';
  source_agent: string;
  timestamp: string;
}

/**
 * Response for risk warnings.
 */
export interface WarningsResponse {
  warnings: RiskWarning[];
  count: number;
}

/**
 * Domain history entry.
 */
export interface DomainHistoryEntry {
  topic: string;
  conclusion: string;
  confidence: number;
  timestamp: string;
}

/**
 * Response for domain history.
 */
export interface DomainHistoryResponse {
  domain: string;
  history: DomainHistoryEntry[];
  count: number;
}

/**
 * Options for finding similar debates.
 */
export interface FindSimilarOptions {
  topic: string;
  limit?: number;
}

/**
 * Options for getting settled topics.
 */
export interface SettledOptions {
  minConfidence?: number;
  limit?: number;
}

/**
 * Options for filtering by topic/domain.
 */
export interface FilterOptions {
  topic?: string;
  domain?: string;
  limit?: number;
}

/**
 * Interface for the internal client methods used by ConsensusAPI.
 */
interface ConsensusClientInterface {
  request<T>(method: string, path: string, options?: { params?: Record<string, string | number | boolean | undefined> }): Promise<T>;
}

/**
 * Consensus API namespace.
 *
 * Provides methods for querying and managing consensus data:
 * - Finding debates similar to a topic
 * - Getting high-confidence settled topics
 * - Accessing dissenting views and contrarian perspectives
 * - Viewing risk warnings and edge cases
 * - Domain-specific history
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Find similar debates
 * const { similar } = await client.consensus.findSimilar({
 *   topic: 'Should we adopt microservices?',
 *   limit: 10
 * });
 *
 * // Get settled topics with high confidence
 * const { topics } = await client.consensus.getSettled({ minConfidence: 0.9 });
 *
 * // Get contrarian views
 * const { views } = await client.consensus.getContrarian({ domain: 'architecture' });
 * ```
 */
export class ConsensusAPI {
  constructor(private client: ConsensusClientInterface) {}

  /**
   * Find debates similar to a topic.
   * Uses semantic similarity to find related discussions.
   */
  async findSimilar(options: FindSimilarOptions): Promise<SimilarDebatesResponse> {
    return this.client.request<SimilarDebatesResponse>('GET', '/api/consensus/similar', {
      params: {
        topic: options.topic,
        limit: options.limit ?? 5,
      },
    });
  }

  /**
   * Get high-confidence settled topics.
   * Returns topics where debates reached strong consensus.
   */
  async getSettled(options?: SettledOptions): Promise<SettledTopicsResponse> {
    return this.client.request<SettledTopicsResponse>('GET', '/api/consensus/settled', {
      params: {
        min_confidence: options?.minConfidence ?? 0.8,
        limit: options?.limit ?? 20,
      },
    });
  }

  /**
   * Get consensus memory statistics.
   */
  async getStats(): Promise<ConsensusStats> {
    return this.client.request<ConsensusStats>('GET', '/api/consensus/stats');
  }

  /**
   * Get recent dissenting views.
   * Dissents are minority opinions that disagreed with consensus.
   */
  async getDissents(options?: FilterOptions): Promise<DissentsResponse> {
    return this.client.request<DissentsResponse>('GET', '/api/consensus/dissents', {
      params: {
        topic: options?.topic,
        domain: options?.domain,
        limit: options?.limit ?? 10,
      },
    });
  }

  /**
   * Get contrarian perspectives.
   * Views that challenge mainstream thinking on topics.
   */
  async getContrarian(options?: FilterOptions): Promise<ContrariansResponse> {
    return this.client.request<ContrariansResponse>('GET', '/api/consensus/contrarian-views', {
      params: {
        topic: options?.topic,
        domain: options?.domain,
        limit: options?.limit ?? 10,
      },
    });
  }

  /**
   * Get risk warnings and edge cases.
   * Identified risks and caveats from debates.
   */
  async getRiskWarnings(options?: FilterOptions): Promise<WarningsResponse> {
    return this.client.request<WarningsResponse>('GET', '/api/consensus/risk-warnings', {
      params: {
        topic: options?.topic,
        domain: options?.domain,
        limit: options?.limit ?? 10,
      },
    });
  }

  /**
   * Get domain-specific history.
   * Historical debate outcomes for a specific domain.
   */
  async getDomainHistory(domain: string, limit?: number): Promise<DomainHistoryResponse> {
    return this.client.request<DomainHistoryResponse>('GET', `/api/consensus/domain/${encodeURIComponent(domain)}`, {
      params: {
        limit: limit ?? 50,
      },
    });
  }

  /**
   * Seed demo consensus data.
   * Useful for development and testing.
   */
  async seedDemo(): Promise<{ success: boolean; message: string }> {
    return this.client.request<{ success: boolean; message: string }>('GET', '/api/consensus/seed-demo');
  }
}
