/**
 * OpenAPI Namespace API
 *
 * Provides access to the published OpenAPI spec.
 */

interface OpenApiClientInterface {
  get<T>(path: string): Promise<T>;
}

export class OpenApiAPI {
  constructor(private client: OpenApiClientInterface) {}

  async getSpec(): Promise<Record<string, unknown>> {
    return this.client.get('/api/v1/openapi');
  }
}
