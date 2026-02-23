/**
 * Feature Flags Namespace API
 *
 * Provides methods for reading feature flag configuration.
 */

interface FeatureFlagsClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class FeatureFlagsAPI {
  constructor(private client: FeatureFlagsClientInterface) {}

  /** List all feature flags and their current states. */
  async list(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/feature-flags');
  }
}
