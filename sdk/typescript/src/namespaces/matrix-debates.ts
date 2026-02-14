/**
 * Matrix Debates Namespace API
 *
 * Provides endpoints for matrix debate management.
 * Matrix debates enable multi-dimensional analysis across
 * multiple perspectives simultaneously.
 */

interface MatrixDebatesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Matrix Debates API namespace.
 *
 * @example
 * ```typescript
 * const debates = await client.matrixDebates.list();
 * const debate = await client.matrixDebates.get(debates[0].id);
 * ```
 */
export class MatrixDebatesAPI {
  constructor(private client: MatrixDebatesClientInterface) {}

  /**
   * List matrix debates.
   * @route GET /api/v1/matrix-debates
   */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/matrix-debates', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a specific matrix debate.
   * @route GET /api/v1/matrix-debates/:id
   */
  async get(debateId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/matrix-debates/${debateId}`) as Promise<Record<string, unknown>>;
  }

  /**
   * Create a new matrix debate.
   * @route POST /api/v1/matrix-debates
   */
  async create(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/matrix-debates', { json: data }) as Promise<Record<string, unknown>>;
  }
}
