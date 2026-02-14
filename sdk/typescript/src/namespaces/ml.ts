/**
 * ML (Machine Learning) Namespace API
 *
 * Provides methods for ML capabilities:
 * - Agent routing recommendations
 * - Response quality scoring
 * - Consensus prediction
 * - Text embeddings and semantic search
 * - Training data export
 *
 * Endpoints:
 *   POST /api/v1/ml/route          - Get ML-based agent routing
 *   POST /api/v1/ml/score          - Score response quality
 *   POST /api/v1/ml/score-batch    - Score multiple responses
 *   POST /api/v1/ml/consensus      - Predict consensus likelihood
 *   POST /api/v1/ml/embed          - Generate text embeddings
 *   POST /api/v1/ml/search         - Semantic search
 *   POST /api/v1/ml/export-training - Export debate data for training
 *   GET  /api/v1/ml/models         - List available ML models
 *   GET  /api/v1/ml/stats          - Get ML module statistics
 */

/**
 * Export format options.
 */
export type ExportFormat = 'json' | 'jsonl';

/**
 * Quality score breakdown.
 */
export interface QualityScore {
  overall: number;
  coherence: number;
  completeness: number;
  relevance: number;
  clarity: number;
}

/**
 * Agent routing result.
 */
export interface RoutingResult {
  selected_agents: string[];
  task_type: string;
  confidence: number;
  reasoning: string;
}

/**
 * Consensus prediction result.
 */
export interface ConsensusPrediction {
  probability: number;
  confidence: number;
  convergence_trend: 'increasing' | 'stable' | 'decreasing';
  predicted_rounds_remaining: number;
  key_disagreements?: string[];
}

/**
 * Embedding result.
 */
export interface EmbeddingResult {
  embeddings: number[][];
  dimension: number;
}

/**
 * Semantic search result item.
 */
export interface SearchResult {
  text: string;
  score: number;
  index: number;
}

/**
 * ML model information.
 */
export interface MLModel {
  name: string;
  version: string;
  capabilities: string[];
  status: 'active' | 'deprecated' | 'experimental';
}

/**
 * ML system statistics.
 */
export interface MLStats {
  routing: {
    total_requests: number;
    average_latency_ms: number;
    accuracy: number;
  };
  consensus_calibration: {
    predictions: number;
    brier_score: number;
    calibration_curve: Array<{ predicted: number; actual: number }>;
  };
}

/**
 * Training data debate entry.
 */
export interface TrainingDebate {
  task: string;
  consensus: string;
  rejected?: string[];
}

/**
 * Options for agent routing.
 */
export interface RouteOptions {
  /** The task description */
  task: string;
  /** List of available agent names */
  availableAgents: string[];
  /** Number of agents to select (default 3) */
  teamSize?: number;
  /** Optional constraints (e.g., { require_code: true }) */
  constraints?: Record<string, unknown>;
}

/**
 * Options for response scoring.
 */
export interface ScoreOptions {
  /** The response text to score */
  text: string;
  /** Optional task context for relevance scoring */
  context?: string;
}

/**
 * Options for batch scoring.
 */
export interface ScoreBatchOptions {
  /** List of response texts (max 100) */
  texts: string[];
  /** Optional list of contexts (same length as texts) */
  contexts?: string[];
}

/**
 * Options for consensus prediction.
 */
export interface PredictConsensusOptions {
  /** List of [agent_name, response_text] tuples */
  responses: Array<[string, string]>;
  /** Task context */
  context?: string;
  /** Current debate round (default 1) */
  currentRound?: number;
  /** Total planned rounds (default 3) */
  totalRounds?: number;
}

/**
 * Options for text embedding.
 */
export interface EmbedOptions {
  /** Single text to embed */
  text?: string;
  /** List of texts to embed (max 100) */
  texts?: string[];
}

/**
 * Options for semantic search.
 */
export interface SearchOptions {
  /** Search query */
  query: string;
  /** List of documents to search (max 1000) */
  documents: string[];
  /** Number of results to return (default 5) */
  topK?: number;
  /** Minimum similarity threshold (default 0.0) */
  threshold?: number;
}

/**
 * Options for training data export.
 */
export interface ExportTrainingOptions {
  /** List of debate data */
  debates: TrainingDebate[];
  /** Output format (default 'json') */
  format?: ExportFormat;
}

/**
 * Client interface for making HTTP requests.
 */
interface MLClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * ML API namespace.
 *
 * Provides methods for ML-based agent routing, quality scoring,
 * consensus prediction, and embeddings.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get agent routing recommendation
 * const routing = await client.ml.route({
 *   task: 'Implement a caching layer',
 *   availableAgents: ['claude', 'gpt-4', 'codex'],
 *   teamSize: 3,
 * });
 * console.log(routing.selected_agents);
 *
 * // Score response quality
 * const score = await client.ml.score({
 *   text: 'The caching layer should use Redis...',
 * });
 * console.log(`Quality: ${score.overall}`);
 *
 * // Generate embeddings
 * const embeddings = await client.ml.embed({
 *   texts: ['Hello world', 'Machine learning'],
 * });
 *
 * // Semantic search
 * const results = await client.ml.search({
 *   query: 'caching strategies',
 *   documents: ['Use Redis', 'Use Memcached', 'Use in-memory'],
 *   topK: 2,
 * });
 * ```
 */
export class MLAPI {
  constructor(private client: MLClientInterface) {}

  // =========================================================================
  // Routing
  // =========================================================================

  /**
   * Get ML-based agent routing for a task.
   *
   * @param options - Routing options
   * @returns Routing result with selected agents and reasoning
   */
  async route(options: RouteOptions): Promise<RoutingResult> {
    const data: Record<string, unknown> = {
      task: options.task,
      available_agents: options.availableAgents,
      team_size: options.teamSize ?? 3,
    };
    if (options.constraints) data.constraints = options.constraints;

    return this.client.request('POST', '/api/v1/ml/route', {
      json: data,
    });
  }

  // =========================================================================
  // Scoring
  // =========================================================================

  /**
   * Score response quality.
   *
   * @param options - Scoring options
   * @returns Quality scores
   */
  async score(options: ScoreOptions): Promise<QualityScore> {
    const data: Record<string, unknown> = { text: options.text };
    if (options.context) data.context = options.context;

    return this.client.request('POST', '/api/v1/ml/score', {
      json: data,
    });
  }

  /**
   * Score multiple responses in batch.
   *
   * @param options - Batch scoring options
   * @returns Array of quality scores
   */
  async scoreBatch(options: ScoreBatchOptions): Promise<{
    scores: QualityScore[];
  }> {
    const data: Record<string, unknown> = { texts: options.texts };
    if (options.contexts) data.contexts = options.contexts;

    return this.client.request('POST', '/api/v1/ml/score-batch', {
      json: data,
    });
  }

  // =========================================================================
  // Consensus Prediction
  // =========================================================================

  /**
   * Predict consensus likelihood.
   *
   * @param options - Prediction options
   * @returns Consensus prediction result
   */
  async predictConsensus(options: PredictConsensusOptions): Promise<ConsensusPrediction> {
    const data: Record<string, unknown> = {
      responses: options.responses.map(([agent, text]) => [agent, text]),
      current_round: options.currentRound ?? 1,
      total_rounds: options.totalRounds ?? 3,
    };
    if (options.context) data.context = options.context;

    return this.client.request('POST', '/api/v1/ml/consensus', {
      json: data,
    });
  }

  // =========================================================================
  // Embeddings
  // =========================================================================

  /**
   * Generate text embeddings.
   *
   * @param options - Embedding options
   * @returns Embeddings with dimension
   */
  async embed(options: EmbedOptions): Promise<EmbeddingResult> {
    const data: Record<string, unknown> = {};
    if (options.text) data.text = options.text;
    if (options.texts) data.texts = options.texts;

    return this.client.request('POST', '/api/v1/ml/embed', {
      json: data,
    });
  }

  /**
   * Semantic search over documents.
   *
   * @param options - Search options
   * @returns Search results with scores
   */
  async search(options: SearchOptions): Promise<{
    results: SearchResult[];
  }> {
    return this.client.request('POST', '/api/v1/ml/search', {
      json: {
        query: options.query,
        documents: options.documents,
        top_k: options.topK ?? 5,
        threshold: options.threshold ?? 0.0,
      },
    });
  }

  // =========================================================================
  // Training Data Export
  // =========================================================================

  /**
   * Export debate data for training.
   *
   * Requires ml:train permission.
   *
   * @param options - Export options
   * @returns Export result with examples count and data
   */
  async exportTraining(options: ExportTrainingOptions): Promise<{
    examples: number;
    data: string;
    format: ExportFormat;
  }> {
    return this.client.request('POST', '/api/v1/ml/export-training', {
      json: {
        debates: options.debates,
        format: options.format ?? 'json',
      },
    });
  }

  // =========================================================================
  // Models & Stats
  // =========================================================================

  /**
   * List available ML models and capabilities.
   *
   * @returns Available models and capabilities
   */
  async listModels(): Promise<{
    capabilities: string[];
    models: MLModel[];
  }> {
    return this.client.request('POST', '/api/v1/ml/models');
  }

  /**
   * Get ML module statistics.
   *
   * @returns ML system statistics
   */
  async getStats(): Promise<MLStats> {
    return this.client.request('POST', '/api/v1/ml/stats');
  }
}
