/**
 * Hybrid Debates Namespace API
 *
 * Provides methods for managing hybrid debates that combine external
 * and internal agents for consensus-driven decisions.
 *
 * Endpoints:
 *   POST /api/v1/debates/hybrid          - Start a hybrid debate
 *   GET  /api/v1/debates/hybrid          - List hybrid debates
 *   GET  /api/v1/debates/hybrid/{id}     - Get hybrid debate result
 */

/**
 * Hybrid debate status.
 */
export type HybridDebateStatus = 'pending' | 'running' | 'completed' | 'failed';

/**
 * Hybrid debate configuration.
 */
export interface HybridDebateConfig {
  /** Maximum duration in seconds */
  max_duration_seconds?: number;
  /** Enable streaming updates */
  enable_streaming?: boolean;
  /** Additional agent parameters */
  agent_params?: Record<string, unknown>;
}

/**
 * Hybrid debate result.
 */
export interface HybridDebateResult {
  debate_id: string;
  status: HybridDebateStatus;
  task: string;
  external_agent: string;
  verification_agents: string[];
  consensus_threshold: number;
  domain: string;
  created_at: string;
  completed_at?: string;
  result?: {
    consensus_reached: boolean;
    consensus_score: number;
    final_answer: string;
    rounds_completed: number;
    agent_responses: Array<{
      agent: string;
      response: string;
      confidence: number;
    }>;
  };
  error?: string;
}

/**
 * Hybrid debate list response.
 */
export interface HybridDebateListResponse {
  debates: HybridDebateResult[];
  total: number;
}

/**
 * Create hybrid debate request.
 */
export interface CreateHybridDebateRequest {
  /** The question or topic to debate */
  task: string;
  /** Name of the registered external agent */
  external_agent: string;
  /** Consensus threshold between 0.0 and 1.0 */
  consensus_threshold?: number;
  /** Maximum number of debate rounds (1-10) */
  max_rounds?: number;
  /** Internal verification agent names */
  verification_agents?: string[];
  /** Domain context for the debate */
  domain?: string;
  /** Additional configuration */
  config?: HybridDebateConfig;
}

/**
 * List hybrid debates options.
 */
export interface ListHybridDebatesOptions {
  /** Filter by debate status */
  status?: HybridDebateStatus;
  /** Maximum number of results (1-100) */
  limit?: number;
}

/**
 * Client interface for hybrid debates operations.
 */
interface HybridDebatesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

/**
 * Hybrid Debates API namespace.
 *
 * Provides methods for starting and managing hybrid debates that
 * coordinate external agents (e.g., CrewAI, LangGraph) with internal
 * verification agents to produce consensus-driven decisions.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Start a hybrid debate
 * const debate = await client.hybridDebates.create({
 *   task: 'Should we migrate to Kubernetes?',
 *   external_agent: 'crewai-infra-team',
 *   consensus_threshold: 0.8,
 *   max_rounds: 5,
 * });
 *
 * // Get debate result
 * const result = await client.hybridDebates.get(debate.debate_id);
 *
 * // List hybrid debates
 * const { debates } = await client.hybridDebates.list({ status: 'completed' });
 * ```
 */
export class HybridDebatesAPI {
  constructor(private client: HybridDebatesClientInterface) {}

  /**
   * Start a hybrid debate.
   *
   * Creates a new hybrid debate combining an external agent with
   * internal verification agents for consensus-driven decisions.
   *
   * @param request - The hybrid debate configuration
   * @returns The created hybrid debate with ID and initial status
   *
   * @example
   * ```typescript
   * const debate = await client.hybridDebates.create({
   *   task: 'Design a rate limiter',
   *   external_agent: 'crewai-arch-team',
   *   consensus_threshold: 0.75,
   *   verification_agents: ['claude', 'gpt-4'],
   * });
   * console.log(`Started debate: ${debate.debate_id}`);
   * ```
   */
  async create(request: CreateHybridDebateRequest): Promise<HybridDebateResult> {
    const body: Record<string, unknown> = {
      task: request.task,
      external_agent: request.external_agent,
      consensus_threshold: request.consensus_threshold ?? 0.7,
      max_rounds: request.max_rounds ?? 3,
      domain: request.domain ?? 'general',
    };

    if (request.verification_agents) {
      body.verification_agents = request.verification_agents;
    }
    if (request.config) {
      body.config = request.config;
    }

    return this.client.request('POST', '/api/v1/debates/hybrid', { body });
  }

  /**
   * Get a hybrid debate result.
   *
   * @param debateId - The hybrid debate ID
   * @returns Full debate details including result when completed
   *
   * @example
   * ```typescript
   * const debate = await client.hybridDebates.get('hd_abc123');
   * if (debate.status === 'completed' && debate.result) {
   *   console.log(`Consensus: ${debate.result.consensus_reached}`);
   *   console.log(`Answer: ${debate.result.final_answer}`);
   * }
   * ```
   */
  async get(debateId: string): Promise<HybridDebateResult> {
    return this.client.request('GET', `/api/v1/debates/hybrid/${encodeURIComponent(debateId)}`);
  }

  /**
   * List hybrid debates.
   *
   * @param options - Filter and pagination options
   * @returns List of hybrid debates with total count
   *
   * @example
   * ```typescript
   * // List all completed hybrid debates
   * const { debates, total } = await client.hybridDebates.list({
   *   status: 'completed',
   *   limit: 50,
   * });
   * console.log(`Found ${total} completed debates`);
   * ```
   */
  async list(options?: ListHybridDebatesOptions): Promise<HybridDebateListResponse> {
    const params: Record<string, unknown> = {};

    if (options?.status) {
      params.status = options.status;
    }
    if (options?.limit !== undefined && options.limit !== 20) {
      params.limit = options.limit;
    }

    return this.client.request('POST', '/api/v1/debates/hybrid', {
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }
}
