/**
 * Orchestration Namespace API
 *
 * Provides methods for unified multi-agent deliberation orchestration:
 * - Async and sync deliberation endpoints
 * - Knowledge context integration
 * - Output channel routing
 * - Template-based workflows
 *
 * Endpoints:
 *   POST /api/v1/orchestration/deliberate      - Async deliberation
 *   POST /api/v1/orchestration/deliberate/sync - Sync deliberation
 *   GET  /api/v1/orchestration/status/:id      - Get status
 *   GET  /api/v1/orchestration/templates       - List templates
 */

/**
 * Team selection strategy options.
 */
export type TeamStrategy = 'specified' | 'best_for_domain' | 'diverse' | 'fast' | 'random';

/**
 * Output format options.
 */
export type OutputFormat = 'standard' | 'decision_receipt' | 'summary' | 'github_review' | 'slack_message';

/**
 * Deliberation status values.
 */
export type DeliberationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Knowledge source configuration.
 */
export interface KnowledgeSource {
  type: string;
  id: string;
  filters?: Record<string, unknown>;
}

/**
 * Output channel configuration.
 */
export interface OutputChannel {
  type: string;
  id: string;
  options?: Record<string, unknown>;
}

/**
 * Deliberation template.
 */
export interface DeliberationTemplate {
  name: string;
  description: string;
  default_agents?: string[];
  default_rounds?: number;
  output_format?: OutputFormat;
  knowledge_sources?: string[];
}

/**
 * Deliberation result.
 */
export interface DeliberationResult {
  request_id: string;
  status: DeliberationStatus;
  question: string;
  consensus?: string;
  dissents?: string[];
  agents_used: string[];
  rounds_completed: number;
  output_format: OutputFormat;
  output_channels_notified?: string[];
  created_at: string;
  completed_at?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Options for deliberation requests.
 */
export interface DeliberateOptions {
  /** The question or decision to deliberate on */
  question: string;
  /** Sources for context (e.g., "slack:C123", "confluence:12345") */
  knowledgeSources?: Array<string | KnowledgeSource>;
  /** Workspace IDs to include in context */
  workspaces?: string[];
  /** Strategy for agent team selection */
  teamStrategy?: TeamStrategy;
  /** Explicit list of agents to use */
  agents?: string[];
  /** Channels to route results to (e.g., "slack:C789") */
  outputChannels?: Array<string | OutputChannel>;
  /** Format for the output */
  outputFormat?: OutputFormat;
  /** Whether consensus is required (default true) */
  requireConsensus?: boolean;
  /** Priority level (low, normal, high, critical) */
  priority?: string;
  /** Maximum deliberation rounds (default 3) */
  maxRounds?: number;
  /** Timeout for the deliberation in seconds (default 300) */
  timeoutSeconds?: number;
  /** Template name to use */
  template?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Client interface for making HTTP requests.
 */
interface OrchestrationClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Orchestration API namespace.
 *
 * Provides methods for unified multi-agent deliberation:
 * - Submit deliberations (async or sync)
 * - Check deliberation status
 * - List available templates
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Async deliberation
 * const result = await client.orchestration.deliberate({
 *   question: 'Should we migrate to Kubernetes?',
 *   knowledgeSources: ['confluence:12345', 'slack:C123456'],
 *   outputChannels: ['slack:C789'],
 * });
 *
 * // Check status
 * const status = await client.orchestration.getStatus(result.request_id);
 * console.log(`Status: ${status.status}`);
 *
 * // Sync deliberation (waits for completion)
 * const syncResult = await client.orchestration.deliberateSync({
 *   question: 'Which testing framework should we use?',
 *   agents: ['claude', 'gpt-4'],
 *   maxRounds: 3,
 * });
 * console.log(`Consensus: ${syncResult.consensus}`);
 *
 * // List available templates
 * const templates = await client.orchestration.listTemplates();
 * ```
 */
export class OrchestrationAPI {
  constructor(private client: OrchestrationClientInterface) {}

  // =========================================================================
  // Deliberation
  // =========================================================================

  /**
   * Submit an async deliberation request.
   *
   * Returns immediately with a request_id that can be used to check status.
   *
   * @param options - Deliberation options
   * @returns Request ID and initial status
   */
  async deliberate(options: DeliberateOptions): Promise<{
    request_id: string;
    status: DeliberationStatus;
    message: string;
  }> {
    const data = this.buildDeliberatePayload(options);

    return this.client.request('POST', '/api/v1/orchestration/deliberate', {
      json: data,
    });
  }

  /**
   * Submit a synchronous deliberation request.
   *
   * Blocks until deliberation completes and returns the full result.
   *
   * @param options - Deliberation options
   * @returns Full deliberation result
   */
  async deliberateSync(options: DeliberateOptions): Promise<DeliberationResult> {
    const data = this.buildDeliberatePayload(options);

    return this.client.request('POST', '/api/v1/orchestration/deliberate/sync', {
      json: data,
    });
  }

  /**
   * Build the request payload for deliberation.
   */
  private buildDeliberatePayload(options: DeliberateOptions): Record<string, unknown> {
    const data: Record<string, unknown> = { question: options.question };

    if (options.knowledgeSources) data.knowledge_sources = options.knowledgeSources;
    if (options.workspaces) data.workspaces = options.workspaces;
    if (options.teamStrategy && options.teamStrategy !== 'best_for_domain') {
      data.team_strategy = options.teamStrategy;
    }
    if (options.agents) data.agents = options.agents;
    if (options.outputChannels) data.output_channels = options.outputChannels;
    if (options.outputFormat && options.outputFormat !== 'standard') {
      data.output_format = options.outputFormat;
    }
    if (options.requireConsensus === false) {
      data.require_consensus = options.requireConsensus;
    }
    if (options.priority && options.priority !== 'normal') {
      data.priority = options.priority;
    }
    if (options.maxRounds !== undefined && options.maxRounds !== 3) {
      data.max_rounds = options.maxRounds;
    }
    if (options.timeoutSeconds !== undefined && options.timeoutSeconds !== 300) {
      data.timeout_seconds = options.timeoutSeconds;
    }
    if (options.template) data.template = options.template;
    if (options.metadata) data.metadata = options.metadata;

    return data;
  }

  // =========================================================================
  // Status
  // =========================================================================

  /**
   * Get the status of a deliberation request.
   *
   * @param requestId - The deliberation request ID
   * @returns Deliberation status and result (if completed)
   */
  async getStatus(requestId: string): Promise<DeliberationResult> {
    return this.client.request('GET', `/api/v1/orchestration/status/${requestId}`);
  }

  // =========================================================================
  // Templates
  // =========================================================================

  /**
   * List available deliberation templates.
   *
   * @returns List of templates with count
   */
  async listTemplates(): Promise<{
    templates: DeliberationTemplate[];
    count: number;
  }> {
    return this.client.request('GET', '/api/v1/orchestration/templates');
  }
}
