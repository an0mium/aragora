/**
 * Decisions Namespace API
 *
 * Provides a namespaced interface for unified decision-making.
 * Routes decisions through debate, workflow, or gauntlet based on
 * the content and configuration.
 */

/**
 * Decision type
 */
export type DecisionType = 'debate' | 'workflow' | 'gauntlet' | 'quick' | 'auto';

/**
 * Decision priority
 */
export type DecisionPriority = 'high' | 'normal' | 'low';

/**
 * Decision status
 */
export type DecisionStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'timeout';

/**
 * Response channel configuration
 */
export interface ResponseChannel {
  platform: 'http_api' | 'slack' | 'teams' | 'discord' | 'email' | 'webhook';
  config?: Record<string, unknown>;
}

/**
 * Decision context
 */
export interface DecisionContext {
  user_id?: string;
  workspace_id?: string;
  org_id?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Decision configuration
 */
export interface DecisionConfig {
  agents?: string[];
  rounds?: number;
  consensus?: 'majority' | 'unanimous' | 'supermajority';
  timeout_seconds?: number;
  required_capabilities?: string[];
}

/**
 * Decision request
 */
export interface DecisionRequest {
  content: string;
  decision_type?: DecisionType;
  config?: DecisionConfig;
  context?: DecisionContext;
  priority?: DecisionPriority;
  response_channels?: ResponseChannel[];
  async?: boolean;
}

/**
 * Decision result
 */
export interface DecisionResult {
  request_id: string;
  status: DecisionStatus;
  decision_type: DecisionType;
  answer?: string;
  confidence?: number;
  consensus_reached?: boolean;
  reasoning?: string;
  evidence_used?: string[];
  duration_seconds?: number;
  error?: string;
  completed_at?: string;
}

/**
 * Decision status response
 */
export interface DecisionStatusResponse {
  request_id: string;
  status: DecisionStatus;
  completed_at?: string;
}

/**
 * Decision summary
 */
export interface DecisionSummary {
  request_id: string;
  status: DecisionStatus;
  decision_type?: DecisionType;
  completed_at?: string;
}

/**
 * Decision list response
 */
export interface DecisionListResponse {
  decisions: DecisionSummary[];
  total: number;
}

/**
 * Interface for the internal client used by DecisionsAPI.
 */
interface DecisionsClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown> }): Promise<T>;
}

/**
 * Decisions API namespace.
 *
 * Provides methods for unified decision-making:
 * - Submit decisions for debate, workflow, or gauntlet routing
 * - Poll for decision status and results
 * - List recent decisions
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Submit a decision request
 * const result = await client.decisions.create({
 *   content: 'Should we migrate to TypeScript?',
 *   decision_type: 'debate',
 *   config: {
 *     agents: ['anthropic-api', 'openai-api', 'gemini-api'],
 *     rounds: 3,
 *     consensus: 'majority',
 *   },
 *   context: {
 *     workspace_id: 'ws-123',
 *   },
 * });
 *
 * console.log(`Decision: ${result.answer} (confidence: ${result.confidence})`);
 *
 * // Poll for async decision status
 * const status = await client.decisions.getStatus(result.request_id);
 *
 * // List recent decisions
 * const { decisions } = await client.decisions.list({ limit: 20 });
 * ```
 */
export class DecisionsAPI {
  constructor(private client: DecisionsClientInterface) {}

  /**
   * Create a new decision request.
   * Routes to debate, workflow, or gauntlet based on content and config.
   * @param body - Decision request configuration
   */
  async create(body: DecisionRequest): Promise<DecisionResult> {
    return this.client.post('/api/v1/decisions', body);
  }

  /**
   * Get a decision result by request ID.
   * @param requestId - The decision request ID
   */
  async get(requestId: string): Promise<DecisionResult> {
    return this.client.get(`/api/v1/decisions/${requestId}`);
  }

  /**
   * Get decision status for polling.
   * @param requestId - The decision request ID
   */
  async getStatus(requestId: string): Promise<DecisionStatusResponse> {
    return this.client.get(`/api/v1/decisions/${requestId}/status`);
  }

  /**
   * List recent decisions.
   * @param options - Filter options
   */
  async list(options?: { limit?: number }): Promise<DecisionListResponse> {
    return this.client.request('GET', '/api/v1/decisions', { params: options });
  }

  /**
   * Wait for a decision to complete (polling helper).
   * @param requestId - The decision request ID
   * @param options - Polling configuration
   */
  async waitForCompletion(
    requestId: string,
    options?: { intervalMs?: number; timeoutMs?: number }
  ): Promise<DecisionResult> {
    const { intervalMs = 1000, timeoutMs = 300000 } = options || {};
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const status = await this.getStatus(requestId);

      if (status.status === 'completed' || status.status === 'failed' || status.status === 'timeout') {
        return this.get(requestId);
      }

      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }

    throw new Error(`Decision ${requestId} did not complete within ${timeoutMs}ms`);
  }
}
