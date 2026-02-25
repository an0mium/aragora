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

  /** Update a registered template. */
  async updateRegistered(templateId: string, data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PUT', `/api/v1/templates/registry/${encodeURIComponent(templateId)}`, { body: data });
  }

  /** Delete a registered template. */
  async deleteRegistered(templateId: string): Promise<Record<string, unknown>> {
    return this.client.request('DELETE', `/api/v1/templates/registry/${encodeURIComponent(templateId)}`);
  }
}
