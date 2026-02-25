/**
 * Templates Namespace API
 *
 * Provides methods for template management.
 */

interface TemplatesClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class TemplatesAPI {
  constructor(private client: TemplatesClientInterface) {}

  /** List available templates. */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/templates', { params });
  }

  /** Get template categories. */
  async getCategories(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/templates/categories');
  }

  /** Get template recommendations. */
  async recommend(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/templates/recommend', { params });
  }

  /** Register a custom template. */
  async register(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/templates/registry', { body: data });
  }

  /** Get a registered template by ID. */
  async getRegistered(templateId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/templates/registry/${encodeURIComponent(templateId)}`);
  }

  /**
   * Update is currently not exposed by the public template registry API contract.
   * Use register() to submit a new listing revision.
   */
  async updateRegistered(templateId: string, data: Record<string, unknown>): Promise<Record<string, unknown>> {
    void templateId;
    return this.register(data);
  }

  /** Delete is currently not exposed by the public template registry API contract. */
  async deleteRegistered(templateId: string): Promise<Record<string, unknown>> {
    void templateId;
    throw new Error("Template registry deletion is not currently exposed by the API contract.");
  }
}
