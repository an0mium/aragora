/**
 * Debates Namespace API
 *
 * Provides a namespaced interface for debate-related operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,
  DebateConvergence,
  DebateCitations,
  DebateEvidence,
  DebateExport,
  DebateUpdateRequest,
  GraphStats,
  Message,
  PaginationParams,
  WebSocketEvent,
} from '../types';
import type { StreamOptions } from '../websocket';

/**
 * Interface for the internal client methods used by DebatesAPI.
 * This allows the namespace to work without circular imports.
 */
interface DebatesClientInterface {
  listDebates(params?: PaginationParams & { status?: string }): Promise<{ debates: Debate[] }>;
  getDebate(debateId: string): Promise<Debate>;
  getDebateBySlug(slug: string): Promise<Debate>;
  createDebate(request: DebateCreateRequest): Promise<DebateCreateResponse>;
  getDebateMessages(debateId: string): Promise<{ messages: Message[] }>;
  getDebateConvergence(debateId: string): Promise<DebateConvergence>;
  getDebateCitations(debateId: string): Promise<DebateCitations>;
  getDebateEvidence(debateId: string): Promise<DebateEvidence>;
  forkDebate(debateId: string, options?: { branch_point?: number }): Promise<{ debate_id: string }>;
  exportDebate(debateId: string, format: 'json' | 'markdown' | 'html' | 'pdf'): Promise<DebateExport>;
  updateDebate(debateId: string, updates: DebateUpdateRequest): Promise<Debate>;
  getDebateGraphStats(debateId: string): Promise<GraphStats>;
  createDebateAndStream(
    request: DebateCreateRequest,
    streamOptions?: Omit<StreamOptions, 'debateId'>
  ): Promise<{ debate: DebateCreateResponse; stream: AsyncGenerator<WebSocketEvent, void, unknown> }>;
  runDebate(
    request: DebateCreateRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<Debate>;
}

/**
 * Debates API namespace.
 *
 * Provides methods for managing debates:
 * - Creating and running debates
 * - Retrieving debate details and messages
 * - Exporting debates in various formats
 * - Analyzing debate convergence and evidence
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Create a debate
 * const response = await client.debates.create({
 *   task: 'Should we use microservices?',
 *   agents: ['claude', 'gpt-4'],
 * });
 *
 * // Get debate details
 * const debate = await client.debates.get(response.debate_id);
 *
 * // List all debates
 * const { debates } = await client.debates.list({ limit: 10 });
 *
 * // Export as markdown
 * const export = await client.debates.export(debateId, 'markdown');
 * ```
 */
export class DebatesAPI {
  constructor(private client: DebatesClientInterface) {}

  /**
   * List all debates with optional filtering and pagination.
   */
  async list(params?: PaginationParams & { status?: string }): Promise<{ debates: Debate[] }> {
    return this.client.listDebates(params);
  }

  /**
   * Get a debate by ID.
   */
  async get(debateId: string): Promise<Debate> {
    return this.client.getDebate(debateId);
  }

  /**
   * Get a debate by its URL slug.
   */
  async getBySlug(slug: string): Promise<Debate> {
    return this.client.getDebateBySlug(slug);
  }

  /**
   * Create a new debate.
   */
  async create(request: DebateCreateRequest): Promise<DebateCreateResponse> {
    return this.client.createDebate(request);
  }

  /**
   * Update an existing debate.
   */
  async update(debateId: string, updates: DebateUpdateRequest): Promise<Debate> {
    return this.client.updateDebate(debateId, updates);
  }

  /**
   * Get all messages from a debate.
   */
  async getMessages(debateId: string): Promise<{ messages: Message[] }> {
    return this.client.getDebateMessages(debateId);
  }

  /**
   * Get convergence analysis for a debate.
   */
  async getConvergence(debateId: string): Promise<DebateConvergence> {
    return this.client.getDebateConvergence(debateId);
  }

  /**
   * Get citations from a debate.
   */
  async getCitations(debateId: string): Promise<DebateCitations> {
    return this.client.getDebateCitations(debateId);
  }

  /**
   * Get evidence gathered during a debate.
   */
  async getEvidence(debateId: string): Promise<DebateEvidence> {
    return this.client.getDebateEvidence(debateId);
  }

  /**
   * Fork a debate from a specific point.
   */
  async fork(debateId: string, options?: { branch_point?: number }): Promise<{ debate_id: string }> {
    return this.client.forkDebate(debateId, options);
  }

  /**
   * Export a debate in a specific format.
   */
  async export(debateId: string, format: 'json' | 'markdown' | 'html' | 'pdf'): Promise<DebateExport> {
    return this.client.exportDebate(debateId, format);
  }

  /**
   * Get graph statistics for a debate.
   */
  async getGraphStats(debateId: string): Promise<GraphStats> {
    return this.client.getDebateGraphStats(debateId);
  }

  /**
   * Create a debate and return a stream of events.
   */
  async createAndStream(
    request: DebateCreateRequest,
    streamOptions?: Omit<StreamOptions, 'debateId'>
  ): Promise<{ debate: DebateCreateResponse; stream: AsyncGenerator<WebSocketEvent, void, unknown> }> {
    return this.client.createDebateAndStream(request, streamOptions);
  }

  /**
   * Create a debate and wait for it to complete.
   * Polls the debate status until it reaches a terminal state.
   */
  async run(
    request: DebateCreateRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<Debate> {
    return this.client.runDebate(request, options);
  }

  /**
   * Alias for run() - creates a debate and waits for completion.
   */
  async waitForCompletion(debateId: string, options?: { pollIntervalMs?: number; timeoutMs?: number }): Promise<Debate> {
    const pollInterval = options?.pollIntervalMs ?? 1000;
    const timeout = options?.timeoutMs ?? 300000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const debate = await this.client.getDebate(debateId);
      if (['completed', 'failed', 'cancelled'].includes(debate.status)) {
        return debate;
      }
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Debate ${debateId} did not complete within ${timeout}ms`);
  }
}
