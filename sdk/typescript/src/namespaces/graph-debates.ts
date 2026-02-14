/**
 * Graph Debates Namespace API
 *
 * Provides endpoints for graph-based debate management.
 * Graph debates enable structured argumentation with
 * branching discussion threads.
 */

interface GraphDebatesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Graph Debates API namespace.
 *
 * @example
 * ```typescript
 * const debates = await client.graphDebates.list();
 * const debate = await client.graphDebates.get(debates[0].id);
 * ```
 */
export class GraphDebatesAPI {
  constructor(private client: GraphDebatesClientInterface) {}

  /**
   * List graph debates.
   * @route GET /api/v1/graph-debates
   */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/graph-debates', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a specific graph debate by ID.
   * @route GET /api/v1/graph-debates/:id
   */
  async get(debateId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/graph-debates/${debateId}`) as Promise<Record<string, unknown>>;
  }

  /**
   * Create a new graph debate.
   * @route POST /api/v1/graph-debates
   */
  async create(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/graph-debates', { json: data }) as Promise<Record<string, unknown>>;
  }
}
